"""MIT licensed evaluation-only controls for B04 readout/learning attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity
from vprtempo.src.task_feature import TaskFeature
from vprtempo.src.benchmark2700_common import sha_file, write_json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_locked2700 import evaluate


def state_digest(state):
    """Fingerprint every named tensor, including threshold and connection masks."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def assert_same_states(left, right):
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError('Full/Anchor epoch0 states differ: evaluate separately, do not merge')


def load_checked(path, expected):
    if sha_file(path) != expected:
        raise ValueError('frozen source changed: ' + str(path))
    return torch.load(path, map_location='cpu', weights_only=True)


def official_weights(state, input_dim=784, feature_dim=1568, output_dim=1100):
    """Extract the exact two linear maps used by official inference, without clamp."""
    if set(state) != {'model_0', 'model_1', 'model_2'}:
        raise ValueError('expected three official expert modules')
    weights = []
    for i in range(3):
        module = state[f'model_{i}']
        w = module['feature_layer.w.weight']
        o = module['output_layer.w.weight']
        if tuple(w.shape) != (feature_dim, input_dim) or tuple(o.shape) != (output_dim, feature_dim):
            raise ValueError('unexpected official weight shape')
        if not torch.isfinite(w).all() or not torch.isfinite(o).all():
            raise ValueError('nonfinite official weights')
        weights.append((w, o))
    return weights


@torch.no_grad()
def feature_linear(x, weights):
    """Concatenate raw feature activations; no threshold, clipping or per-expert norm."""
    return torch.cat([F.linear(x, w) for w, _ in weights], dim=1)


@torch.no_grad()
def output_linear(x, weights):
    return torch.cat([F.linear(F.linear(x, w), o) for w, o in weights], dim=1)


def check_original_output(weights, summer, previous, device):
    """Recompose official output at batch1 and compare all saved top5 and scores."""
    if len(previous) != len(summer):
        raise ValueError('original prediction count mismatch')
    moved = [(w.to(device), o.to(device)) for w, o in weights]
    max_diff = 0.0
    for x, row in zip(summer.split(1), previous):
        scores = output_linear(x.to(device), moved)[0].cpu().numpy()
        order = scores.argsort()[::-1][:5]
        if order.tolist() != [v['place'] for v in row['place_top5']]:
            raise ValueError('recomposed original output ranking differs from B04')
        diff = max(abs(float(scores[v['place']])-v['score']) for v in row['place_top5'])
        max_diff = max(max_diff, diff)
    if max_diff > 1e-6:
        raise ValueError('recomposed original output scores differ from B04')
    return {'queries_checked': len(previous), 'top5_exact_match': True, 'max_abs_score_diff': max_diff}


def checked_json(item, path_key, sha_key):
    p = Path(item[path_key])
    if sha_file(p) != item[sha_key]:
        raise ValueError('comparison source changed: ' + str(p))
    return json.loads(p.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--models-sha', required=True)
    p.add_argument('--prepared-inputs', type=Path, required=True)
    p.add_argument('--prepared-inputs-sha', required=True)
    for key in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest', 'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--'+key)
    a = p.parse_args()
    cfg = json.loads(a.config.read_bytes())
    for key in ('data_dir','cache_dir','run_root','deployment_manifest','deployment_manifest_sha','environment_manifest'):
        if getattr(a,key): cfg[key] = getattr(a,key)
    cfg['run_id'] = a.run_id
    if (cfg['n_places'], cfg['query_start'], cfg['query_count']) != (3300,600,2700):
        raise ValueError('B05 requires the fixed3300/2700 protocol')
    code = code_identity(ROOT,cfg.get('deployment_manifest'),cfg.get('deployment_manifest_sha'))
    if code['dirty']: raise RuntimeError('dirty evaluation source')
    deployed = code['source']=='verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'),ROOT/'pixi.lock',required=deployed)
    provenance = {'code':code,'environment':env,'runtime':runtime_identity(torch,np)}
    if sha_file(a.models)!=a.models_sha or sha_file(a.prepared_inputs)!=a.prepared_inputs_sha:
        raise ValueError('control identity mismatch')
    tasks = json.loads(a.models.read_bytes())['tasks']
    item = next(v for v in tasks if v['task_id']==a.task_id)
    pointer = json.loads(a.prepared_inputs.read_bytes()); identity = pointer['identity']
    ids = list(range(600,3300))
    if identity['query_ids']!=ids or identity['n_places']!=3300:
        raise ValueError('prepared query identity mismatch')
    cache_name = 'gabor_cache' if item.get('frontend')=='gabor' else 'pixel_cache'
    source = Path(pointer[cache_name])
    data = load_checked(source,pointer[cache_name+'_sha256'])
    inputs = data['features'] if cache_name=='gabor_cache' else data['pixels']
    out,_ = resolve_locations(cfg,ROOT,deployed); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'effective_config.json',{'config':cfg,'task':item,**provenance})
    started = time.monotonic(); device = torch.device(cfg['device'])
    if item['kind']=='random_feature':
        states = []
        for model in item['sources']:
            state = load_checked(Path(model['checkpoint']),model['checkpoint_sha256'])
            meta = state['metadata']; mc = meta['config']
            if meta['epoch']!=0 or meta['git']['commit']!=model['training_commit']:
                raise ValueError('not the locked original epoch0 checkpoint')
            if (mc['seed'],mc['frontend'],mc['n_places'])!=(item['seed'],item['frontend'],3300):
                raise ValueError('epoch0 configuration mismatch')
            if mc['lr']!=0.001: raise ValueError('wrong paired training run')
            if item['frontend']=='gabor' and model['gabor_state_sha256']!=identity['gabor_state_sha256']:
                raise ValueError('Gabor state mismatch')
            states.append(state['model'])
        assert_same_states(*states)
        model = TaskFeature(1152 if item['frontend']=='gabor' else 784,2304,item['seed'],device)
        model.load_state_dict(states[0]); model.requires_grad_(False);model.eval()
        before = state_digest(model.state_dict())
        r = evaluate(model,inputs,ids)
        if state_digest(model.state_dict())!=before: raise RuntimeError('random model mutated')
        details = {'paired_full_anchor_epoch0_identical':True,'model_state_unchanged':True,
                   'model_state_sha256':before,'feature_dim':2304}
    elif item['kind']=='official_feature':
        original = checked_json(item,'original_summary','original_summary_sha256')
        if original['code']['commit']!='dfaf3451d238d3cb39f64b06eaa601916f49adaa':
            raise ValueError('wrong official model producer')
        if original['data_manifest_sha256']!=identity['manifest_sha256']:
            raise ValueError('original/new sample identity mismatch')
        state = load_checked(Path(item['checkpoint']),item['checkpoint_sha256'])
        before = hashlib.sha256(''.join(state_digest(state[k]) for k in sorted(state)).encode()).hexdigest()
        weights = official_weights(state)
        previous = checked_json(item,'original_predictions','original_predictions_sha256')
        parity = check_original_output(weights,inputs['summer'],previous,device)
        moved = [(w.to(device),o.to(device)) for w,o in weights]
        features = {s:torch.cat([feature_linear(x.to(device),moved).cpu() for x in values.split(256)])
                    for s,values in inputs.items()}
        r = evaluate(None,features,ids)
        after = hashlib.sha256(''.join(state_digest(state[k]) for k in sorted(state)).encode()).hexdigest()
        if before!=after: raise RuntimeError('official model mutated')
        details = {'model_state_unchanged':True,'model_state_sha256':before,'feature_dim':4704,
                   'readout':'concat_three_raw_linear_feature_maps_then_global_L2_and_cosine',
                   'extra_clamp':False,'threshold_applied':False,'per_expert_normalization':False,
                   'official_output_parity':parity,'official_core_identity':original['official_core_identity']}
    else: raise ValueError('unknown evaluation kind')
    write_json(out/'predictions.json',r.pop('predictions'))
    r.update({'status':'complete','task':item,'details':details,
              'data_manifest_sha256':identity['manifest_sha256'],'query_role':'previously_observed_benchmark',
              'prepared_inputs_sha256':a.prepared_inputs_sha,'elapsed_s':time.monotonic()-started,**provenance})
    write_json(out/'summary.json',r)
    print(json.dumps({'task':a.task_id,'R@1':r['subsets']['overall']['R@1'],'details':details}),flush=True)


if __name__=='__main__':main()
