"""MIT licensed fixed preselected checkpoint diagnosis on observed query seasons."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from vprtempo.src.benchmark2700_common import sha_file, write_json
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity
from vprtempo.src.expert_finetune import tensor_digest
from run_expert_finetune import load_source, extract, checked_json
from eval_locked2700 import evaluate


def read(p):
    return json.loads(Path(p).read_bytes())


def restore(lock, device):
    model = load_source(lock['source'], device)
    cp = lock['checkpoint']
    if sha_file(cp['path']) != cp['sha256']:
        raise ValueError('checkpoint hash differs')
    payload = torch.load(cp['path'], map_location='cpu', weights_only=True)
    md = payload['metadata']
    expected = {'epoch': lock['epoch'], 'method': lock['method'], 'seed': lock['seed'],
                'code_commit': lock['training_commit'],
                'source_checkpoint_sha256': lock['source']['checkpoint_sha256'],
                'initial_state_sha256': lock['initial_state_sha256'],
                'schedule_sha256': lock['schedule_identity']['tensor_sha256'],
                'calibration_sha256': lock['calibration_sha256']}
    if any(md.get(k) != v for k, v in expected.items()):
        raise ValueError('checkpoint metadata differs from preselected manifest')
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def score_season(features, season):
    # Existing fixed evaluator names its query tensor "summer". Only its argument
    # key is adapted here; source identities and saved season labels stay explicit.
    result = evaluate(None, {'spring': features['spring'], 'fall': features['fall'],
                            'summer': features[season]}, list(range(600, 3300)))
    if season == 'winter':
        names = {'overall': 'overall', 'development': 'region_600_1099',
                 'buffer': 'region_1100_1149', 'retained': 'region_1150_3299'}
        result['subsets'] = {names[k]: v for k, v in result['subsets'].items()}
    result['query_season'] = season
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sources-sha', required=True)
    p.add_argument('--method', choices=['full', 'anchor'], required=True)
    p.add_argument('--run-id', required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest',
              'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--' + k)
    a = p.parse_args()
    cfg = read(a.config)
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest',
              'deployment_manifest_sha', 'environment_manifest'):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    cfg['run_id'] = a.run_id
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']:
        raise RuntimeError('dirty source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    source = checked_json(a.sources, a.sources_sha)
    summer_pointer = checked_json(source['summer_inputs'], source['summer_inputs_sha256'])
    winter_pointer = checked_json(source['winter_inputs'], source['winter_inputs_sha256'])
    for pointer in (summer_pointer, winter_pointer):
        if pointer['identity']['query_ids'] != list(range(600, 3300)):
            raise ValueError('query identity differs')
        if sha_file(pointer['pixel_cache']) != pointer['pixel_cache_sha256']:
            raise ValueError('cache changed')
    summer = torch.load(summer_pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
    winter = torch.load(winter_pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
    if any(not torch.equal(summer[s], winter[s]) for s in ('spring', 'fall')):
        raise ValueError('reference pixels differ across seasons')
    pixels = {**summer, 'winter': winter['winter']}
    out, _ = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'effective_config.json', {'config': cfg, 'method': a.method,
               'sources_sha256': a.sources_sha, **provenance})
    write_json(out / 'sources.json', source)
    device = torch.device(cfg['device'])
    locks = [m for m in source['models'] if m['method'] == a.method]
    if len(locks) != 2:
        raise ValueError('requires the two predeclared seed0 checkpoints')
    teacher = load_source(locks[0]['source'], device)
    teacher.requires_grad_(False)
    teacher_features = extract(teacher, {s: pixels[s] for s in ('spring', 'fall')}, device)
    rows = []
    started = time.monotonic()
    for lock in locks:
        model = restore(lock, device)
        before = tensor_digest(model.state_dict())
        features = extract(model, pixels, device)
        if any(not torch.isfinite(v).all() for v in features.values()):
            raise FloatingPointError('nonfinite descriptors')
        angular = []
        for season in ('spring', 'fall'):
            own = torch.nn.functional.normalize(features[season], dim=1)
            old = torch.nn.functional.normalize(teacher_features[season], dim=1)
            angular.extend((1 - (own * old).sum(1)).tolist())
        for season in ('summer', 'winter'):
            result = score_season(features, season)
            if season == 'summer':
                old = checked_json(lock['development_predictions'], lock['development_predictions_sha256'])
                for u, v in zip(result['predictions'][:500], old):
                    if [p['place'] for p in u['place_top5']] != [p['place'] for p in v['place_top5']]:
                        raise ValueError('saved development top5 differs')
                    if any(abs(x['score'] - y['score']) > 1e-6 for x, y in zip(u['place_top5'], v['place_top5'])):
                        raise ValueError('saved development scores differ')
            file = f'{lock["model_id"]}_{season}_predictions.json'
            write_json(out / file, result.pop('predictions'))
            rows.append({'model_id': lock['model_id'], 'method': a.method, 'seed': 0,
                         'epoch': lock['epoch'], 'learning_rate': lock['learning_rate'],
                         'relative_weight_drift': lock['relative_weight_drift'],
                         'train_descriptor_cosine_drift_mean': sum(angular) / len(angular),
                         'checkpoint_sha256': lock['checkpoint']['sha256'],
                         'predictions_file': file, **result})
        if tensor_digest(model.state_dict()) != before:
            raise RuntimeError('model changed during evaluation')
    write_json(out / 'summary.json', {'status': 'complete', 'training_updates': 0,
               'results': rows, 'model_state_unchanged': True, 'development_parity_queries': 1000,
               'scope': 'seed0 diagnostic only; no winter-based checkpoint selection',
               'elapsed_s': time.monotonic() - started, **provenance})
    print(json.dumps([{'model': r['model_id'], 'season': r['query_season'],
                      'overall': r['subsets']['overall']} for r in rows]), flush=True)


if __name__ == '__main__':
    main()
