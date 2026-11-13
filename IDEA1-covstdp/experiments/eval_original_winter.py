"""MIT licensed original VPRTempo place-output winter evaluation, no training.

Recompose the official two linear maps, verify all B04 summer top5 scores,
then apply the same batch1 forward to the hash-bound B09 winter pixels.
"""
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
from vprtempo.src.benchmark2700_common import sha_file, write_json, score_report
from vprtempo.src.metrics import recallAtK
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity
from eval_attribution2700 import official_weights, output_linear, check_original_output, state_digest, load_checked


def read(path):
    return json.loads(Path(path).read_bytes())


def checked_json(path, expected):
    if sha_file(path) != expected:
        raise ValueError('JSON identity changed: ' + str(path))
    return read(path)


@torch.no_grad()
def original_scores(pixels, weights, device):
    """Preserve original batch1, unnormalized signed place-output scores."""
    moved = [(w.to(device), o.to(device)) for w, o in weights]
    rows = [output_linear(x.to(device), moved)[0].cpu() for x in pixels.split(1)]
    scores = torch.stack(rows)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('nonfinite original output')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--sources-sha', required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--run-id', required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest',
              'deployment-manifest-sha', 'environment-manifest'):
        parser.add_argument('--' + k)
    args = parser.parse_args()
    cfg = read(args.config)
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest',
              'deployment_manifest_sha', 'environment_manifest'):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    cfg['run_id'] = args.run_id
    if (cfg['n_places'], cfg['query_start'], cfg['query_count'], cfg['GT_tolerance'],
            cfg['query_season'], cfg['inference_batch_size']) != (3300, 600, 2700, 0, 'winter', 1):
        raise ValueError('fixed original winter protocol differs')
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']:
        raise RuntimeError('dirty evaluation source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    runtime = runtime_identity(torch, np)
    runtime.update({'device': cfg['device'], 'device_name': torch.cuda.get_device_name(torch.device(cfg['device']))
                    if cfg['device'].startswith('cuda') else 'cpu'})
    provenance = {'code': code, 'environment': env, 'runtime': runtime}
    sources = checked_json(args.sources, args.sources_sha)
    if len(sources['models']) != 3 or {m['seed'] for m in sources['models']} != {0, 1, 2}:
        raise ValueError('three original models required')
    source = next(m for m in sources['models'] if m['seed'] == args.seed)
    winter_pointer = checked_json(sources['winter_inputs'], sources['winter_inputs_sha256'])
    summer_pointer = checked_json(sources['summer_inputs'], sources['summer_inputs_sha256'])
    identity = winter_pointer['identity']
    if (identity['query_ids'] != list(range(600, 3300)) or identity['query_season'] != 'winter'
            or identity['n_places'] != 3300 or identity['GT_tolerance'] != 0):
        raise ValueError('winter input protocol differs')
    if summer_pointer['identity']['query_ids'] != list(range(600, 3300)):
        raise ValueError('summer parity query protocol differs')
    for season in ('spring', 'fall', 'summer', 'winter'):
        if sha_file(ROOT / f'vprtempo/dataset/nordland-{season}.csv') != identity['csv_sha256'][season]:
            raise ValueError('official CSV version differs')
    out, _ = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'effective_config.json', {'config': cfg, 'seed': args.seed,
               'sources_sha256': args.sources_sha, 'source_model': source, **provenance})
    write_json(out / 'sources.json', sources)
    started = time.monotonic()
    state = load_checked(Path(source['checkpoint']), source['checkpoint_sha256'])
    initial = {name: state_digest(value) for name, value in state.items()}
    weights = official_weights(state)
    summer = load_checked(Path(summer_pointer['pixel_cache']), summer_pointer['pixel_cache_sha256'])['pixels']['summer']
    previous = checked_json(source['original_predictions'], source['original_predictions_sha256'])
    parity = check_original_output(weights, summer, previous, torch.device(cfg['device']))
    old_summary = checked_json(source['original_summary'], source['original_summary_sha256'])
    for k in (1, 5):
        value = sum(any(v['place'] == p['gt'] for v in p['place_top5'][:k]) for p in previous) / 2700
        if abs(value - old_summary['subsets']['overall'][f'R@{k}']) > 1e-12:
            raise ValueError('B04 parity metrics differ')
    write_json(out / 'summer_original_parity.json', parity)
    print('B04 summer parity passed', json.dumps(parity), flush=True)
    winter = load_checked(Path(winter_pointer['pixel_cache']), winter_pointer['pixel_cache_sha256'])['pixels']['winter']
    if winter.shape != (2700, 784) or not torch.isfinite(winter).all():
        raise ValueError('invalid winter pixels')
    scores = original_scores(winter, weights, torch.device(cfg['device']))
    if scores.shape != (2700, 3300):
        raise ValueError('original place-output shape differs')
    valid = (winter.norm(dim=1) > 1e-8) & (scores.norm(dim=1) > 1e-8)
    result = score_report(scores.T.numpy(), list(range(600, 3300)), recallAtK, valid.numpy())
    names = {'overall': 'overall', 'development': 'region_600_1099',
             'buffer': 'region_1100_1149', 'retained': 'region_1150_3299'}
    result['subsets'] = {names[k]: v for k, v in result['subsets'].items()}
    if {name: state_digest(value) for name, value in state.items()} != initial:
        raise RuntimeError('original model state changed during evaluation')
    write_json(out / 'predictions.json', result.pop('predictions'))
    result.update({'status': 'complete', 'method': 'original_output', 'seed': args.seed,
                   'query_season': 'winter', 'training_updates': 0,
                   'readout': 'three official feature->place linear maps, raw outputs concatenated; no cosine or feature retrieval',
                   'source_checkpoint_sha256': source['checkpoint_sha256'], 'source': source,
                   'model_state_unchanged': True, 'module_state_sha256': initial,
                   'summer_original_parity': parity, 'winter_inputs_sha256': sources['winter_inputs_sha256'],
                   'winter_data_identity_sha256': identity['manifest_sha256'],
                   'query_role': 'all2700 evaluation only; no winter tuning',
                   'elapsed_s': time.monotonic() - started, **provenance})
    write_json(out / 'summary.json', result)
    print(json.dumps({'seed': args.seed, 'subsets': result['subsets']}), flush=True)


if __name__ == '__main__':
    main()
