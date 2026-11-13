"""MIT licensed paired retention and training-drift-matched plain control.

One physical GPU job, separately identified logical runs. Matching uses only
predeclared epoch2 training drift. Query scores never set its learning rate.
"""
import argparse
import copy
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
from run_expert_finetune import read, checked_json, json_sha, load_inputs, extract
from eval_night_checkpoints import restore, score_season
import run_night_finetune as runner


def matched_rate(target, reference, base_rate, low=1e-4, high=.1):
    if not all(np.isfinite(v) and v > 0 for v in (target, reference, base_rate)):
        raise ValueError('matching requires positive finite training drift')
    return float(f'{min(high, max(low, base_rate * target / reference)):.7g}')


def manifest(directory, run_id):
    files = [{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha_file(p)}
             for p in (directory / 'artifacts').rglob('*') if p.is_file()]
    write_json(directory / 'result_manifest.json', {'run_id': run_id, 'files': files})


def execute(cfg, sources, out, provenance, method, seed):
    started = time.monotonic()
    subruns, summaries, paths = [], {}, {}
    reference = sources['plain_reference_e2'][method]
    reference_summary = checked_json(reference['summary'], reference['summary_sha256'])
    actual_ref = next(e['relative_drift'] for e in reference_summary['epoch_results'] if e['epoch'] == 2)
    if (actual_ref != reference['relative_drift'] or reference_summary['method'] != method
            or reference_summary['seed'] != seed or reference_summary['calibration']['lr'] != reference['learning_rate']
            or reference_summary['initial_state_sha256'] != sources['pairing_references'][str(seed)]['initial_state_sha256']):
        raise ValueError('plain reference training drift differs')
    write_json(out / 'match_rule.prespecified.json', {'epoch': 2, 'reference': reference,
        'parameter_uses': 'epoch2 training drift only, no query scores', 'rate_bounds': [1e-4, .1],
        'relative_drift_matching_tolerance': .1})

    def run(kind, arm, rate):
        child = copy.deepcopy(cfg)
        child.update(run_id=cfg['run_id'] + '-' + kind, run_root=str(out / 'subruns'),
                     max_lr=rate, requested_learning_rate=rate, defer_full_evaluation=False)
        child_out, _ = resolve_locations(child, ROOT, provenance['code']['source'] == 'verified_deployment_snapshot')
        child_out.mkdir(parents=True, exist_ok=False)
        directory = child_out.parent
        write_json(directory / 'launch_manifest.json', {'run_id': child['run_id'], 'logical_subrun': True,
            'parent_run_id': cfg['run_id'], 'invocation': 'run_night_finetune.train function in paired parent',
            'command': [sys.executable, *sys.argv], 'method': method, 'seed': seed, 'arm': arm,
            'source_commit': provenance['code']['commit'], 'environment_id': provenance['environment'].get('environment_id'),
            'slurm_job_id': provenance['runtime']['slurm_job_id']})
        write_json(child_out / 'sources.json', sources)
        runner.train(child, sources, child_out, provenance, method, seed, arm)
        s = read(child_out / 'summary.json')
        if s['status'] != 'complete': raise RuntimeError('logical training run incomplete')
        manifest(directory, child['run_id'])
        write_json(directory / 'trial_outcome.json', {'status': 'complete', 'logical_subrun': True,
                   'slurm_job_id': provenance['runtime']['slurm_job_id'], 'parent_run_id': cfg['run_id']})
        info = {'kind': kind, 'run_id': child['run_id'], 'artifact_subpath': child_out.relative_to(out).as_posix(),
                'run_dir': str(directory), 'summary_sha256': sha_file(child_out / 'summary.json'),
                'counted_training_segments': s['counted_training_segments']}
        subruns.append(info)
        summaries[kind] = s
        paths[kind] = child_out
        return s

    retention = run('retention', 'retention', cfg['requested_learning_rate'])
    if retention['calibration']['margin'] != reference_summary['calibration']['margin']:
        raise ValueError('matching reference uses different relation margin')
    train_stat = read(paths['retention'] / 'training_epoch_2.json')
    target = train_stat['relative_drift']
    if not any(e['epoch'] == 2 and e['eligible_for_selection'] for e in retention['epoch_results']):
        raise RuntimeError('retention has no complete epoch2 for control; preserve evidence and stop')
    rate = matched_rate(target, actual_ref, reference['learning_rate'])
    match = {'epoch': 2, 'target_drift': target, 'target_training_receipt_sha256': sha_file(paths['retention'] / 'training_epoch_2.json'),
             'reference_drift': actual_ref, 'reference_learning_rate': reference['learning_rate'],
             'control_learning_rate': rate, 'uses_query_data': False,
             'relative_drift_matching_tolerance': .1,
             'rule': 'reference lr scaled by ratio of epoch2 training drifts, bounded1e-4..0.1, no score tuning'}
    write_json(out / 'match_rule.json', match)
    if rate in (.1, .01):
        # Both complete plain trajectories already exist in B07. Never silently
        # retrain an existing setting: return a clear reuse request to the Goal.
        rows = []
        for r in retention['locked_results']:
            rows.append({**r, 'predictions_file': (paths['retention'] / r['predictions_file']).relative_to(out).as_posix(),
                         'arm': 'retention', 'comparison': 'development_locked', 'method': method,
                         'seed': seed, 'epoch': retention['selection']['chosen_epoch']})
        return {'status': 'complete', 'phase': 'night_control_reference_reuse', 'method': method, 'seed': seed,
                'control_status': 'pending_existing_model_reuse', 'results': rows, 'subruns': subruns,
                'match_rule': match, 'counted_training_segments': retention['counted_training_segments'],
                'reserved_training_segments': 3, 'query_scores_used_for_matching': False,
                'unique_prediction_files': len({r['predictions_file'] for r in rows}),
                'note': 'Existing B07 plain rate must be reused and verified by Goal; this phase has not completed its control.',
                'elapsed_s': time.monotonic() - started, **provenance}
    plain = run('matched', 'plain', rate)
    actual_control = next(e['relative_drift'] for e in plain['epoch_results'] if e['epoch'] == 2)
    match.update(actual_control_drift=actual_control, relative_drift_mismatch=abs(actual_control / target - 1))
    match['within_declared_matching_tolerance'] = match['relative_drift_mismatch'] <= .1
    write_json(out / 'match_rule.json', match)

    results, missing_fixed = [], []
    for kind, s in summaries.items():
        for r in s['locked_results']:
            file = (paths[kind] / r['predictions_file']).relative_to(out).as_posix()
            results.append({**r, 'predictions_file': file, 'arm': kind, 'comparison': 'development_locked',
                            'method': method, 'seed': seed, 'epoch': s['selection']['chosen_epoch']})
        if s['selection']['chosen_epoch'] == 2:
            for r in s['locked_results']:
                file = (paths[kind] / r['predictions_file']).relative_to(out).as_posix()
                results.append({**r, 'predictions_file': file, 'arm': kind, 'comparison': 'fixed_epoch2',
                                'method': method, 'seed': seed, 'epoch': 2, 'reused_locked_predictions': True})
        else:
            cp = next(c for c in s['checkpoints'] if c['epoch'] == 2)
            missing_fixed.append({'model_id': kind, 'method': method, 'seed': seed, 'epoch': 2,
                'checkpoint': cp, 'source': s['source'], 'initial_state_sha256': s['initial_state_sha256'],
                'training_commit': provenance['code']['commit'], 'schedule_identity': s['schedule_identity'],
                'calibration_sha256': json_sha(s['calibration'])})
    if missing_fixed:
        pointer = checked_json(sources['summer_inputs'], sources['summer_inputs_sha256'])
        pixels = load_inputs(pointer, False)
        winter_pointer = checked_json(sources['winter_inputs'], sources['winter_inputs_sha256'])
        if sha_file(winter_pointer['pixel_cache']) != winter_pointer['pixel_cache_sha256']:
            raise ValueError('winter changed')
        winter = torch.load(winter_pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
        if any(not torch.equal(pixels[s], winter[s]) for s in ('spring', 'fall')):
            raise ValueError('reference pixels changed')
        pixels['winter'] = winter['winter']
        device = torch.device(cfg['device'])
        for lock in missing_fixed:
            model = restore(lock, device)
            before = tensor_digest(model.state_dict())
            features = extract(model, pixels, device)
            for season in ('summer', 'winter'):
                r = score_season(features, season)
                file = f'fixed_e2_{lock["model_id"]}_{season}_predictions.json'
                write_json(out / file, r.pop('predictions'))
                results.append({**r, 'predictions_file': file, 'arm': lock['model_id'], 'comparison': 'fixed_epoch2',
                                'method': method, 'seed': seed, 'epoch': 2, 'reused_locked_predictions': False})
            if tensor_digest(model.state_dict()) != before: raise RuntimeError('fixed evaluation changed state')
    return {'status': 'complete', 'phase': 'night_paired_control', 'method': method, 'seed': seed,
            'results': results, 'subruns': subruns, 'match_rule': match,
            'counted_training_segments': sum(s['counted_training_segments'] for s in summaries.values()),
            'unique_prediction_files': len({r['predictions_file'] for r in results}),
            'query_scores_used_for_matching': False, 'elapsed_s': time.monotonic() - started, **provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sources-sha', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--method', choices=['full', 'anchor'], required=True)
    p.add_argument('--seed', type=int, choices=[0], required=True)
    p.add_argument('--learning-rate', type=float, choices=[.1], required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest', 'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--' + k)
    a = p.parse_args()
    cfg = read(a.config)
    cfg.update(run_id=a.run_id, max_lr=a.learning_rate, requested_learning_rate=a.learning_rate)
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest', 'deployment_manifest_sha', 'environment_manifest'):
        if getattr(a, k): cfg[k] = getattr(a, k)
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']: raise RuntimeError('dirty paired source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    sources = checked_json(a.sources, a.sources_sha)
    out, _ = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'effective_config.json', {'config': cfg, 'method': a.method, 'seed': a.seed,
               'sources_sha256': a.sources_sha, 'paired_training_segments': 3, **provenance})
    write_json(out / 'sources.json', sources)
    result = execute(cfg, sources, out, provenance, a.method, a.seed)
    write_json(out / 'summary.json', result)
    print(json.dumps({'method': a.method, 'matched_rate': result['match_rule']['control_learning_rate'],
                      'drift_mismatch': result['match_rule'].get('relative_drift_mismatch'),
                      'control_status': result.get('control_status', 'new_control_completed')}), flush=True)


if __name__ == '__main__': main()
