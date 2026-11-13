"""MIT licensed winter evaluation of B08 models locked using summer only.

No optimizer, learning, calibration, model selection or offset search is exposed.
"""
import argparse
import csv
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
from vprtempo.src.dataset import CustomImageDataset, ProcessImage
from vprtempo.src.expert_finetune import tensor_digest
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity
from vprtempo.src.task_protocol import cached_tensor
from run_expert_finetune import load_source, extract, checked_json, json_sha


def read(path):
    return json.loads(Path(path).read_bytes())


def prepare(cfg, sources, cache):
    """Reuse hash-bound B04 reference pixels; decode winter queries once."""
    old = checked_json(sources['prepared_inputs'], sources['prepared_inputs_sha256'])
    ident = old['identity']
    ids = list(range(600, 3300))
    if ident['query_ids'] != ids or ident['n_places'] != 3300:
        raise ValueError('B04 input protocol differs')
    if sha_file(old['pixel_cache']) != old['pixel_cache_sha256']:
        raise ValueError('B04 reference cache changed')
    csv_hashes, rows = {}, {}
    for season in ('spring', 'fall', 'summer', 'winter'):
        path = ROOT / 'vprtempo/dataset' / f'nordland-{season}.csv'
        csv_hashes[season] = sha_file(path)
        with path.open(newline='') as stream:
            rows[season] = list(csv.reader(stream))
        if season != 'winter' and csv_hashes[season] != ident['csv_sha256'][season]:
            raise ValueError('CSV version differs from B04')
    if any(r != rows['summer'] for r in rows.values()):
        raise ValueError('seasonal CSV row correspondence differs; no automatic offset repair')
    reference_records = [r for r in ident['manifest'] if r['season'] in ('spring', 'fall')]
    by_place = {(r['season'], r['place']): r for r in reference_records}
    if len(by_place) != 6600:
        raise ValueError('reference identity incomplete')
    for r in reference_records:
        path = Path(cfg['data_dir']) / r['season'] / r['file']
        stat = path.stat()
        if stat.st_size != r['size'] or stat.st_mtime_ns != r['mtime_ns']:
            raise ValueError('reference image metadata changed')
    accepted = checked_json(sources['alignment_acceptance'], sources['alignment_acceptance_sha256'])
    if not accepted['visual_checked'] or accepted['offset_search'] or accepted['tolerance'] != 0:
        raise ValueError('winter alignment acceptance missing')
    for r in accepted['samples']:
        path = Path(cfg['data_dir']) / r['season'] / r['file']
        if sha_file(path) != r['sha256']:
            raise ValueError('server image differs from visually checked local sample')
    dataset = CustomImageDataset(
        str(ROOT / 'vprtempo/dataset/nordland-winter.csv'), cfg['data_dir'], ['winter'],
        transform=ProcessImage([28, 28], 7, patch_norm=True), filter=8,
        skip=0, max_samples=3300, test=True)
    if len(dataset) != 3300:
        raise ValueError('winter place count differs')
    dataset.img_labels = dataset.img_labels.iloc[ids].reset_index(drop=True)
    manifest = list(reference_records)
    for place, (_, row) in zip(ids, dataset.img_labels.iterrows()):
        path = Path(row['file_path'])
        if (path.name != by_place['spring', place]['file']
                or int(row.iloc[1]) != by_place['spring', place]['csv_index']):
            raise ValueError('winter query identity mismatch')
        stat = path.stat()
        manifest.append({'place': place, 'season': 'winter', 'file': path.name,
                         'csv_index': int(row.iloc[1]), 'size': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns, 'sha256': sha_file(path)})
    identity = {'dataset': 'Nordland', 'n_places': 3300, 'reference_images': 6600,
                'query_ids': ids, 'query_season': 'winter', 'query_role': 'frozen_condition_evaluation_only',
                'dims': [28, 28], 'patches': 7, 'patch_norm': True, 'filter': 8,
                'GT_tolerance': 0, 'csv_sha256': csv_hashes, 'manifest': manifest,
                'manifest_sha256': json_sha(manifest), 'data_dir': cfg['data_dir'],
                'reused_reference_cache_sha256': old['pixel_cache_sha256'],
                'alignment_acceptance_sha256': sources['alignment_acceptance_sha256']}
    key = json_sha({'identity': identity, 'preprocess': sha_file(ROOT / 'vprtempo/src/dataset.py')})
    path = cache / f'winter_pixels_{key[:16]}.pt'
    def build():
        previous = torch.load(old['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
        winter = []
        for i in range(2700):
            winter.append(dataset[i][0].float())
            if (i + 1) % 300 == 0:
                print('decoded winter', i + 1, flush=True)
        return {'pixels': {'spring': previous['spring'], 'fall': previous['fall'],
                           'winter': torch.stack(winter)}, 'identity_sha256': key}
    payload = cached_tensor(path, build)
    if payload['identity_sha256'] != key:
        raise ValueError('winter cache identity mismatch')
    for season, x in payload['pixels'].items():
        n = 2700 if season == 'winter' else 3300
        if x.shape != (n, 784) or not torch.isfinite(x).all():
            raise ValueError('nonfinite pixels or wrong count')
    return {'identity': identity, 'pixel_cache': str(path), 'pixel_cache_sha256': sha_file(path),
            'pixel_tensor_sha256': tensor_digest(payload['pixels'])}


def restore(lock, source, device):
    """Retain original producer identity and verify each selected B08 epoch."""
    model = load_source(source, device)
    if lock['method'] != 'frozen':
        cp = lock['checkpoint']
        if sha_file(cp['path']) != cp['sha256']:
            raise ValueError('selected checkpoint hash differs')
        payload = torch.load(cp['path'], map_location='cpu', weights_only=True)
        md = payload['metadata']
        if (md['epoch'], md['seed'], md['method'], md['code_commit']) != (
                2, lock['seed'], lock['method'], lock['training_commit']):
            raise ValueError('selected checkpoint metadata differs')
        expected = {'source_checkpoint_sha256': source['checkpoint_sha256'],
                    'initial_state_sha256': lock['initial_state_sha256'],
                    'schedule_sha256': lock['schedule_identity']['tensor_sha256'],
                    'calibration_sha256': lock['calibration_sha256'],
                    'probe_sha256': lock['probe_sha256']}
        if any(md.get(k) != v for k, v in expected.items()):
            raise ValueError('selected checkpoint preparation differs')
        model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    model.eval()
    if any(not torch.isfinite(w).all() for w in model.parameters()):
        raise ValueError('nonfinite frozen weights')
    return model


@torch.no_grad()
def evaluate(model, pixels, device):
    before = tensor_digest(model.state_dict())
    feats = extract(model, pixels, device)
    if any(not torch.isfinite(x).all() for x in feats.values()):
        raise FloatingPointError('nonfinite frozen descriptors')
    db = torch.cat((feats['spring'], feats['fall'])).float()
    q = feats['winter'].float()
    valid_db = db.norm(dim=1) > 1e-8
    valid_q = q.norm(dim=1) > 1e-8
    db = torch.nn.functional.normalize(db, dim=1)
    q = torch.nn.functional.normalize(q, dim=1)
    scores = (db @ q.T).numpy()
    scores[~valid_db.numpy()] = -np.inf
    n = len(pixels['spring'])
    places = np.maximum(scores[:n], scores[n:])
    from vprtempo.src.metrics import recallAtK
    r = score_report(places, list(range(600, 3300)), recallAtK, valid_q.numpy())
    # Geographic bins are inherited for diagnosis; winter has NO development split.
    names = {'development': 'region_600_1099', 'buffer': 'region_1100_1149',
             'retained': 'region_1150_3299', 'overall': 'overall'}
    r['subsets'] = {names[k]: v for k, v in r['subsets'].items()}
    r['invalid_reference_images'] = int((~valid_db).sum())
    if tensor_digest(model.state_dict()) != before:
        raise RuntimeError('frozen evaluation mutated model state')
    r.update({'model_state_unchanged': True, 'model_tensor_sha256': before})
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--sources-sha', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=['prepare', 'evaluate'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    parser.add_argument('--prepared-inputs', type=Path)
    parser.add_argument('--prepared-inputs-sha')
    for key in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest',
                'deployment-manifest-sha', 'environment-manifest'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    cfg = read(args.config)
    for key in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest',
                'deployment_manifest_sha', 'environment_manifest'):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    cfg['run_id'] = args.run_id
    if (cfg['n_places'], cfg['query_start'], cfg['query_count'], cfg['GT_tolerance'],
            cfg['query_season'], cfg['locked_epoch']) != (3300, 600, 2700, 0, 'winter', 2):
        raise ValueError('fixed winter protocol differs')
    sources = checked_json(args.sources, args.sources_sha)
    if len(sources['models']) != 9 or {(m['method'], m['seed']) for m in sources['models']} != {
            (m, s) for m in ('frozen', 'full', 'anchor') for s in range(3)}:
        raise ValueError('nine locked models required')
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']:
        raise RuntimeError('dirty evaluation source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    provenance['runtime']['device'] = cfg['device']
    provenance['runtime']['device_name'] = torch.cuda.get_device_name(torch.device(cfg['device'])) if cfg['device'].startswith('cuda') else 'cpu'
    out, cache = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'effective_config.json', {'config': cfg, 'mode': args.mode, 'seed': args.seed,
               'sources_sha256': args.sources_sha, **provenance})
    write_json(out / 'locked_models.json', sources)
    start = time.monotonic()
    if args.mode == 'prepare':
        pointer = prepare(cfg, sources, cache)
        write_json(out / 'prepared_inputs.json', {**pointer, **provenance})
        result = {'mode': 'prepare', 'winter_queries': 2700, 'reference_images': 6600,
                  'no_performance_computed': True}
    else:
        if args.seed is None or args.prepared_inputs is None:
            raise ValueError('evaluation requires seed and fixed inputs')
        pointer = checked_json(args.prepared_inputs, args.prepared_inputs_sha)
        ident = pointer['identity']
        if (ident['query_ids'] != list(range(600, 3300)) or ident['query_season'] != 'winter'
                or ident['query_role'] != 'frozen_condition_evaluation_only'):
            raise ValueError('winter cache protocol differs')
        if sha_file(pointer['pixel_cache']) != pointer['pixel_cache_sha256']:
            raise ValueError('winter pixels changed')
        pixels = torch.load(pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
        if tensor_digest(pixels) != pointer['pixel_tensor_sha256']:
            raise ValueError('winter tensor identity differs')
        rows = []
        for lock in [m for m in sources['models'] if m['seed'] == args.seed]:
            model = restore(lock, lock['source'], torch.device(cfg['device']))
            r = evaluate(model, pixels, torch.device(cfg['device']))
            write_json(out / (lock['method'] + '_predictions.json'), r.pop('predictions'))
            rows.append({'method': lock['method'], 'seed': args.seed, 'locked_model': lock, **r})
            print(json.dumps({'method': lock['method'], 'seed': args.seed, 'overall': r['subsets']['overall']}), flush=True)
        result = {'mode': 'evaluate', 'results': rows, 'prepared_inputs_sha256': args.prepared_inputs_sha,
                  'data_identity_sha256': ident['manifest_sha256']}
    write_json(out / 'summary.json', {'status': 'complete', 'query_season': 'winter',
               'scope': 'same map, held-out condition; no winter-based tuning or selection',
               'training_updates': 0, 'elapsed_s': time.monotonic() - start, **result, **provenance})


if __name__ == '__main__':
    main()
