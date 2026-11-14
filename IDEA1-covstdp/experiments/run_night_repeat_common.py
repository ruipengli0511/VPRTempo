"""MIT licensed two fixed-seed repeats followed by common evaluation in one job."""
import argparse
import copy
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
from run_expert_finetune import read, checked_json
from run_night_paired import manifest
import run_night_finetune as trainer
import eval_night_common as common


def execute(cfg, sources, out, provenance, method):
    start = time.monotonic()
    declared = sources['seed0_candidate']
    directory = Path(declared['run_dir'])
    for name, file in [('summary', 'artifacts/summary.json'), ('effective_config', 'artifacts/effective_config.json'),
                       ('result_manifest', 'result_manifest.json')]:
        if sha_file(directory / file) != declared[name + '_sha256']:
            raise ValueError('seed0 evidence changed')
    prior = read(directory / 'artifacts/summary.json')
    if (prior['status'] != 'complete' or prior['phase'] != 'night_finetune' or prior['seed'] != 0
            or prior['method'] != method or prior['code']['commit'] != declared['expected_commit']):
        raise ValueError('seed0 is not the registered completed night candidate')
    if read(directory / 'trial_outcome.json')['status'] != 'complete':
        raise ValueError('seed0 logical run is incomplete')
    coefficient, lr, arm = prior['retention_coefficient'], prior['calibration']['lr'], prior['arm']
    if arm == 'retention' and coefficient <= 0: raise ValueError('invalid fixed coefficient')
    base = read(directory / 'artifacts/effective_config.json')['config']
    table = [{**declared, 'method': method, 'seed': 0}]
    training = []
    for seed in (1, 2):
        child = copy.deepcopy(base)
        for key in ('seed', 'method', 'arm', 'lr', 'retention_coefficient', 'source_checkpoint', 'initial_state_sha256'):
            child.pop(key, None)
        child.update(run_id=cfg['run_id'] + '-s' + str(seed), run_root=str(out / 'subruns'),
                     data_dir=cfg['data_dir'], cache_dir=cfg['cache_dir'], max_lr=lr, requested_learning_rate=lr,
                     retention_policy='fixed', fixed_retention_coefficient=coefficient,
                     defer_full_evaluation=True, device=cfg['device'])
        # A reused seed0 config must identify this run's actual deployment.
        for key in ('deployment_manifest', 'deployment_manifest_sha', 'environment_manifest'):
            child[key] = cfg[key]
        child_out, _ = resolve_locations(child, ROOT, provenance['code']['source'] == 'verified_deployment_snapshot')
        child_out.mkdir(parents=True, exist_ok=False)
        d = child_out.parent
        write_json(d / 'launch_manifest.json', {'run_id': child['run_id'], 'logical_subrun': True,
            'parent_run_id': cfg['run_id'], 'invocation': 'deferred train function, no complete queries',
            'command': [sys.executable, *sys.argv], 'method': method, 'seed': seed, 'arm': arm,
            'source_commit': provenance['code']['commit'], 'environment_id': provenance['environment'].get('environment_id'),
            'slurm_job_id': provenance['runtime']['slurm_job_id']})
        write_json(child_out / 'sources.json', sources)
        write_json(child_out / 'sources.json', sources)
        trainer.train(child, sources, child_out, provenance, method, seed, arm)
        s = read(child_out / 'summary.json')
        if (s['status'] != 'complete' or not s['deferred_common_epoch_evaluation'] or s['locked_results']
                or s['counted_training_segments'] != 1 or s['retention_coefficient'] != coefficient):
            raise ValueError('seed repeat violated fixed candidate or deferred query scope')
        manifest(d, child['run_id'])
        write_json(d / 'trial_outcome.json', {'status': 'complete', 'logical_subrun': True,
                   'slurm_job_id': provenance['runtime']['slurm_job_id'], 'parent_run_id': cfg['run_id']})
        item = {'run_dir': str(d), 'method': method, 'seed': seed, 'expected_commit': provenance['code']['commit'],
                'summary_sha256': sha_file(child_out / 'summary.json'),
                'effective_config_sha256': sha_file(child_out / 'effective_config.json'),
                'result_manifest_sha256': sha_file(d / 'result_manifest.json')}
        table.append(item)
        training.append({'artifact_subpath': child_out.relative_to(out).as_posix(), **item})
    # All outputs exist now. Seal their real hashes before any common query score.
    table_path = out / 'training_runs.resolved.json'
    write_json(table_path, {'runs': table})
    write_json(out / 'sources.json', sources)
    common_id = cfg['run_id'] + '-common'
    common_root = out / 'subruns'
    common_argv = ['eval_night_common.py', '--config', str(ROOT / 'IDEA1-covstdp/phase4/configs/night_diagnostic.json'),
        '--sources', str(out / 'sources.json'), '--sources-sha', sha_file(out / 'sources.json'),
        '--training-runs', str(table_path), '--training-runs-sha', sha_file(table_path), '--run-id', common_id,
        '--data-dir', cfg['data_dir'], '--cache-dir', cfg['cache_dir'], '--run-root', str(common_root),
        '--deployment-manifest', cfg['deployment_manifest'], '--deployment-manifest-sha', cfg['deployment_manifest_sha'],
        '--environment-manifest', cfg['environment_manifest']]
    before = sys.argv
    try:
        sys.argv = common_argv
        common.main()
    finally:
        sys.argv = before
    common_out = common_root / common_id / 'artifacts'
    result = read(common_out / 'summary.json')
    if result['status'] != 'complete' or len(result['results']) != 6:
        raise RuntimeError('common evaluation incomplete')
    for name in ('locked_models.json', 'training_selection_evidence.json', 'training_runs.input.json'):
        (out / name).write_bytes((common_out / name).read_bytes())
    rows = []
    for row in result['results']:
        rows.append({**row, 'predictions_file': (common_out / row['predictions_file']).relative_to(out).as_posix()})
    return {**result, 'results': rows, 'orchestration': 'two_deferred_repeats_then_common_evaluation',
            'training_subruns': training, 'common_evaluation_artifact_subpath': common_out.relative_to(out).as_posix(),
            'seed0_reused': declared, 'fixed_coefficient': coefficient, 'fixed_learning_rate': lr,
            'counted_training_segments': 2, 'elapsed_s': time.monotonic() - start,
            'parent_run_id': cfg['run_id'], **provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sources-sha', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--method', choices=['full', 'anchor'], required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest', 'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--' + k)
    a = p.parse_args()
    cfg = read(a.config);cfg['run_id'] = a.run_id
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest', 'deployment_manifest_sha', 'environment_manifest'):
        if getattr(a, k): cfg[k] = getattr(a, k)
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']: raise RuntimeError('dirty repeat source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    sources = checked_json(a.sources, a.sources_sha)
    out, _ = resolve_locations(cfg, ROOT, deployed);out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'effective_config.json', {'config': cfg, 'method': a.method,
               'sources_sha256': a.sources_sha, 'training_segments': 2, **provenance})
    result = execute(cfg, sources, out, provenance, a.method)
    write_json(out / 'summary.json', result)


if __name__ == '__main__': main()
