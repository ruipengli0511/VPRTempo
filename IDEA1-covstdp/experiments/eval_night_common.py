"""MIT licensed common-development-epoch evaluation for mixed-producer night runs."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from vprtempo.src.benchmark2700_common import sha_file, write_json
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity
from vprtempo.src.expert_finetune import tensor_digest
from run_expert_finetune import read, checked_json, load_source, load_inputs, extract, json_sha
from eval_night_checkpoints import score_season


def choose_common(summaries):
    groups = defaultdict(list)
    for s in summaries:
        if s['status'] != 'complete' or s['phase'] != 'night_finetune' or s['method'] not in ('full', 'anchor'):
            raise ValueError('invalid training summary')
        groups[s['method']].append(s)
    if not groups:
        raise ValueError('no groups to select')
    choices = []
    for method, rows in groups.items():
        rows.sort(key=lambda s: s['seed'])
        if [s['seed'] for s in rows] != [0, 1, 2]:
            raise ValueError('exactly three distinct seeds required per method')
        if (len({s['arm'] for s in rows}) != 1 or len({s['retention_coefficient'] for s in rows}) != 1
                or len({s['calibration']['lr'] for s in rows}) != 1
                or len({s['probe_identity']['tensor_sha256'] for s in rows}) != 1):
            raise ValueError('candidate settings or probe differ across seeds')
        eligible = set.intersection(*(set(e['epoch'] for e in s['epoch_results'] if e['eligible_for_selection']) for s in rows))
        if not eligible:
            raise ValueError('no common complete epoch')
        hits = {e: sum(round(next(x['development']['R@1'] for x in s['epoch_results'] if x['epoch'] == e) * 500)
                       for s in rows) for e in eligible}
        epoch = min(eligible, key=lambda e: (-hits[e], e))
        choices.append({'method': method, 'epoch': epoch, 'development_total_correct': hits,
                        'selection_uses': 'mean development500 R@1 across3, earliest tie; R5 not selection'})
    return choices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sources-sha', required=True)
    p.add_argument('--training-runs', type=Path, required=True)
    p.add_argument('--training-runs-sha', required=True)
    p.add_argument('--run-id', required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest', 'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--' + k)
    a = p.parse_args()
    cfg = read(a.config)
    cfg['run_id'] = a.run_id
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest', 'deployment_manifest_sha', 'environment_manifest'):
        if getattr(a, k): cfg[k] = getattr(a, k)
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']: raise RuntimeError('dirty evaluation source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    sources = checked_json(a.sources, a.sources_sha)
    table = checked_json(a.training_runs, a.training_runs_sha)['runs']
    summaries, bindings = [], []
    for item in table:
        d = Path(item['run_dir'])
        for name in ('summary', 'effective_config'):
            if sha_file(d / ('artifacts/' + name + '.json')) != item[name + '_sha256']:
                raise ValueError('training evidence changed')
        if sha_file(d / 'result_manifest.json') != item['result_manifest_sha256']:
            raise ValueError('result manifest changed')
        s = read(d / 'artifacts/summary.json')
        if s['code']['commit'] != item['expected_commit'] or read(d / 'trial_outcome.json')['status'] != 'complete':
            raise ValueError('wrong or failed training producer')
        if (s['method'], s['seed']) != (item['method'], item['seed']):
            raise ValueError('training role mismatch')
        expected_source = next(x for x in sources['models'] if x['seed'] == s['seed'])
        if s['source']['checkpoint_sha256'] != expected_source['checkpoint_sha256']:
            raise ValueError('initial source mismatch')
        pairing = sources['pairing_references'][str(s['seed'])]
        if s['initial_state_sha256'] != pairing['initial_state_sha256'] or s['schedule_identity'] != pairing['schedule_identity']:
            raise ValueError('initial state/mining differs')
        launch = read(d / 'launch_manifest.json')
        cmd = launch['command']
        manifest = Path(cmd[cmd.index('--deployment-manifest') + 1])
        manifest_sha = cmd[cmd.index('--deployment-manifest-sha') + 1]
        producer = code_identity(manifest.parent, manifest, manifest_sha)
        if producer['commit'] != item['expected_commit']:
            raise ValueError('producer deployment identity mismatch')
        for file in ('vprtempo/src/expert_finetune.py', 'vprtempo/src/descriptor_retention.py'):
            if sha_file(manifest.parent / file) != sha_file(ROOT / file):
                raise ValueError('mixed-producer update core differs')
        summaries.append(s)
        bindings.append((item, s, read(d / 'artifacts/effective_config.json')['config']))
    choices = choose_common(summaries)
    out, _ = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'training_runs.input.json').write_bytes(a.training_runs.read_bytes())
    write_json(out / 'training_selection_evidence.json', [
        {'run_id': Path(item['run_dir']).name, 'summary_sha256': item['summary_sha256'],
         **{k: s[k] for k in ('method', 'seed', 'arm', 'retention_coefficient', 'epoch_results')}}
        for item, s, effective in bindings])
    locks = []
    for choice in choices:
        for item, s, effective in bindings:
            if s['method'] != choice['method']: continue
            cp = next(c for c in s['checkpoints'] if c['epoch'] == choice['epoch'])
            if sha_file(cp['path']) != cp['sha256']: raise ValueError('checkpoint changed')
            locks.append({'method': s['method'], 'seed': s['seed'], 'epoch': choice['epoch'],
                'checkpoint': cp, 'training_commit': s['code']['commit'], 'training_run_id': Path(item['run_dir']).name,
                'initial_state_sha256': s['initial_state_sha256'], 'retention_coefficient': s['retention_coefficient'],
                'source': s['source'], 'calibration_sha256': json_sha(s['calibration']),
                'retention_calibration_sha256': json_sha(s['retention_calibration']),
                'config_sha256': json_sha(effective), 'schedule_sha256': s['schedule_identity']['tensor_sha256']})
    # This file is committed to disk before any full-query pixel inference.
    write_json(out / 'locked_models.json', {'selection': choices, 'models': locks,
               'training_runs_sha256': a.training_runs_sha, 'winter_selection': False})
    pointer = checked_json(sources['summer_inputs'], sources['summer_inputs_sha256'])
    pixels = load_inputs(pointer, False)
    winter_pointer = checked_json(sources['winter_inputs'], sources['winter_inputs_sha256'])
    if sha_file(winter_pointer['pixel_cache']) != winter_pointer['pixel_cache_sha256']:
        raise ValueError('winter cache changed')
    winter = torch.load(winter_pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
    if any(not torch.equal(pixels[s], winter[s]) for s in ('spring', 'fall')):
        raise ValueError('reference mismatch')
    pixels['winter'] = winter['winter']
    rows = []
    device = torch.device(cfg['device'])
    for lock in locks:
        payload = torch.load(lock['checkpoint']['path'], map_location='cpu', weights_only=True)
        md = payload['metadata']
        for k in ('epoch', 'seed', 'method', 'initial_state_sha256', 'calibration_sha256',
                  'retention_calibration_sha256', 'config_sha256', 'schedule_sha256'):
            if md[k] != lock[k]: raise ValueError('locked metadata mismatch: ' + k)
        if md['code_commit'] != lock['training_commit']: raise ValueError('locked producer mismatch')
        model = load_source(lock['source'], device)
        model.load_state_dict(payload['model'], strict=True)
        model.requires_grad_(False)
        before = tensor_digest(model.state_dict())
        features = extract(model, pixels, device)
        for season in ('summer', 'winter'):
            r = score_season(features, season)
            name = f'{lock["method"]}_s{lock["seed"]}_{season}_predictions.json'
            write_json(out / name, r.pop('predictions'))
            rows.append({'method': lock['method'], 'seed': lock['seed'], 'epoch': lock['epoch'],
                         'predictions_file': name, **r})
        if tensor_digest(model.state_dict()) != before: raise RuntimeError('evaluation changed state')
    aggregate = []
    for method in sorted({r['method'] for r in rows}):
        for season in ('summer', 'winter'):
            rr = [r for r in rows if r['method'] == method and r['query_season'] == season]
            aggregate.append({'method': method, 'query_season': season, **{
                f'R@{k}': {'mean': statistics.mean(r['subsets']['overall'][f'R@{k}'] for r in rr),
                          'sample_std': statistics.stdev(r['subsets']['overall'][f'R@{k}'] for r in rr)} for k in (1, 5)}})
    write_json(out / 'effective_config.json', {'config': cfg, 'sources_sha256': a.sources_sha,
               'training_runs_sha256': a.training_runs_sha, **provenance})
    write_json(out / 'summary.json', {'status': 'complete', 'phase': 'night_common_evaluation',
               'results': rows, 'aggregate': aggregate, 'selection': choices,
               'locked_models_sha256': sha_file(out / 'locked_models.json'),
               'scope': 'known exposed seasons, common development epoch; no new independent-test claim', **provenance})


if __name__ == '__main__': main()
