"""MIT licensed bounded feature adaptation with optional anchor retention.

Training, coefficient calibration and mining use spring/fall only. Summer
development selects a saved epoch before any complete-query inference.
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
from vprtempo.src.expert_finetune import (tensor_digest, make_schedule, triplet_batch, calibrate,
    gradients, apply_gradients, relative_drift, probe_diagnostics)
from vprtempo.src.descriptor_retention import combined_gradient, retention_gradient
from run_expert_finetune import (read, json_sha, checked_json, load_source, load_inputs,
    extract, score, baseline_parity, save_checkpoint, resolve_probe)
from eval_night_checkpoints import score_season


def teacher_batch(teacher, indices, device):
    return teacher[indices[:, 0].to(device), indices[:, 1].to(device)]


def calibrate_retention(model, training, teacher, plan, probe, device, cfg, step_calibration):
    """A train-only Full pilot on a disposable copy; count it as a training segment."""
    before = tensor_digest(model.state_dict())
    pilot = copy.deepcopy(model)
    start = time.monotonic()
    diagnostics = []
    batches = list(plan[0].split(cfg['batch_size']))[:cfg['retention_pilot_batches']]
    for ids in batches:
        a, p, n = triplet_batch(training, ids, device)
        gs, diagnostic = gradients(pilot, a, p, n, step_calibration['margin'], 'full')
        update = apply_gradients(pilot, gs, step_calibration['lr'], cfg['max_relative_step'])
        diagnostics.append({**diagnostic, **update})
    task2 = hold2 = 0.
    probe_rows = []
    for ids in probe.split(cfg['batch_size']):
        a, p, n = triplet_batch(training, ids, device)
        task, td = gradients(pilot, a, p, n, step_calibration['margin'], 'full')
        hold, hd = retention_gradient(pilot, a, teacher_batch(teacher, ids, device))
        task2 += sum(float(g.square().sum()) for g in task)
        hold2 += sum(float(g.square().sum()) for g in hold)
        probe_rows.append({**td, **hd})
    if hold2 <= 1e-24:
        raise RuntimeError('no train-only retention signal after declared pilot')
    ratio = (task2 / hold2) ** 0.5
    raw = cfg['retention_target_gradient_ratio'] * ratio
    coefficient = float(f'{min(cfg["retention_coefficient_max"], max(cfg["retention_coefficient_min"], raw)):.4g}')
    if tensor_digest(model.state_dict()) != before:
        raise RuntimeError('pilot changed original initialization')
    return {'policy': 'common_full_pilot_train_only', 'coefficient': coefficient,
            'raw_coefficient': raw, 'pilot_updates': len(batches), 'pilot_diagnostics': diagnostics,
            'gradient_probe': probe_rows, 'target_weighted_gradient_ratio': cfg['retention_target_gradient_ratio'],
            'achieved_pilot_probe_gradient_ratio': coefficient / ratio,
            'pilot_model_sha256': tensor_digest(pilot.state_dict()), 'original_model_unchanged': True,
            'uses_query_data': False, 'calibration_s': time.monotonic() - start}


def train(cfg, sources, out, provenance, method, seed, arm):
    pointer = checked_json(sources['summer_inputs'], sources['summer_inputs_sha256'])
    source = next(v for v in sources['models'] if v['seed'] == seed)
    device = torch.device(cfg['device'])
    pixels = load_inputs(pointer, True)
    model = load_source(source, device)
    initial = [w.detach().clone() for w in model.weights]
    initial_hash = tensor_digest(model.state_dict())
    initial_result = score(model, pixels, list(range(600, 1100)), device)
    parity = baseline_parity(initial_result, source, True)
    initial_features = extract(model, {s: pixels[s] for s in ('spring', 'fall')}, device, normalized=True)
    plan, schedule_identity = make_schedule(initial_features, cfg['epochs'], seed, cfg['negative_min_gap'])
    reference = sources['pairing_references'][str(seed)]
    if schedule_identity != reference['schedule_identity'] or initial_hash != reference['initial_state_sha256']:
        raise ValueError('initial weights/mining schedule differs from B08')
    training = torch.stack((pixels['spring'], pixels['fall']))
    step = calibrate(model, training, plan, cfg['batch_size'], device, cfg)
    if abs(step['lr'] - cfg['requested_learning_rate']) > 1e-12:
        raise RuntimeError('registered learning rate reduced by safety calibration')
    probe, probe_identity = resolve_probe(cfg, sources, plan, seed)
    teacher = None
    if arm == 'retention':
        teacher = torch.stack((initial_features['spring'], initial_features['fall'])).detach()
        if cfg['retention_policy'] == 'calibrate_train_only':
            hold_calibration = calibrate_retention(model, training, teacher, plan, probe, device, cfg, step)
        else:
            coefficient = cfg['fixed_retention_coefficient']
            if not 0 < coefficient <= cfg['retention_coefficient_max']:
                raise ValueError('invalid predeclared retention coefficient')
            hold_calibration = {'policy': 'predeclared_fixed', 'coefficient': coefficient, 'pilot_updates': 0,
                                'uses_query_data': False, 'calibration_s': 0.}
    else:
        hold_calibration = {'policy': 'disabled_bypass', 'coefficient': 0., 'pilot_updates': 0,
                            'uses_query_data': False, 'calibration_s': 0.}
    del initial_features
    coefficient = hold_calibration['coefficient']
    if tensor_digest(model.state_dict()) != initial_hash:
        raise RuntimeError('preparation changed model')
    torch.save({'plan': plan, 'identity': schedule_identity}, out / 'training_plan.pt')
    write_json(out / 'initial_parity.json', parity)
    write_json(out / 'probe_identity.json', probe_identity)
    write_json(out / 'calibration.json', step)
    write_json(out / 'retention_calibration.json', hold_calibration)
    effective = {**cfg, 'method': method, 'seed': seed, 'arm': arm, 'lr': step['lr'],
                 'margin': step['margin'], 'retention_coefficient': coefficient, 'feature_dim': 4704,
                 'retention_scope': 'anchor_only_for_both_methods', 'source_checkpoint': source,
                 'initial_state_sha256': initial_hash}
    write_json(out / 'effective_config.json', {'config': effective, **provenance})
    metadata = {'seed': seed, 'method': method, 'arm': arm, 'source_checkpoint_sha256': source['checkpoint_sha256'],
                'initial_state_sha256': initial_hash, 'code_commit': provenance['code']['commit'],
                'schedule_sha256': schedule_identity['tensor_sha256'], 'calibration_sha256': json_sha(step),
                'retention_calibration_sha256': json_sha(hold_calibration), 'config_sha256': json_sha(effective),
                'probe_sha256': probe_identity['tensor_sha256']}
    results, checkpoints, history, probes = [], [], [], []
    def probe_record(epoch):
        diagnostic = probe_diagnostics(model, training, probe, step['margin'], method, step['lr'],
                                       cfg['max_relative_step'], cfg['batch_size'], device)
        diagnostic.update(epoch=epoch, epoch_complete=True, probe_sha256=probe_identity['tensor_sha256'])
        diagnostic['gradient_probe_scope'] = 'task_only; combined training updates recorded separately'
        if teacher is not None:
            hold = []
            for ids in probe.split(cfg['batch_size']):
                a, _, _ = triplet_batch(training, ids, device)
                _, h = retention_gradient(model, a, teacher_batch(teacher, ids, device))
                hold.append(h)
            diagnostic['retention_probe_loss'] = sum(h['retention_loss'] * h['retention_valid'] for h in hold) / max(sum(h['retention_valid'] for h in hold), 1)
        write_json(out / f'probe_epoch_{epoch}.json', diagnostic)
        probes.append(diagnostic)
    def record(epoch, result=None, eligible=True):
        r = result or score(model, pixels, list(range(600, 1100)), device)
        write_json(out / f'development_epoch_{epoch}_predictions.json', r.pop('predictions'))
        cp = save_checkpoint(out, model, epoch, {**metadata, 'epoch_complete': eligible})
        checkpoints.append(cp)
        results.append({'epoch': epoch, 'development': r['subsets']['development'],
                        'relative_drift': relative_drift(model, initial), 'eligible_for_selection': eligible})
    probe_record(0)
    record(0, initial_result)
    started = time.monotonic()
    stop = 'budget_complete'
    with (out / 'updates.jsonl').open('x') as log:
        for epoch, rows in enumerate(plan, 1):
            total = {'epoch': epoch, 'batches': 0, 'valid': 0, 'active': 0, 'task_loss_sum': 0.,
                     'retention_loss_sum': 0., 'retention_samples': 0, 'actual_step_sum': 0., 'step_capped': 0,
                     'combined_gradient_norm_sum': 0., 'weighted_retention_gradient_norm_sum': 0.}
            numerical = False
            for batch, ids in enumerate(rows.split(cfg['batch_size'])):
                a, p, n = triplet_batch(training, ids, device)
                target = teacher_batch(teacher, ids, device) if teacher is not None else None
                try:
                    gs, diag = combined_gradient(model, a, p, n, target, step['margin'], method, coefficient)
                    update = apply_gradients(model, gs, step['lr'], cfg['max_relative_step'])
                except FloatingPointError as exc:
                    stop = 'numerical_stop:' + str(exc)
                    numerical = True
                    break
                total['batches'] += 1
                total['valid'] += diag['valid']
                total['active'] += diag['active']
                total['task_loss_sum'] += diag['loss'] * diag['valid']
                total['retention_loss_sum'] += diag.get('retention_loss', 0.) * len(ids)
                total['retention_samples'] += len(ids)
                total['actual_step_sum'] += update['actual_step_norm']
                total['step_capped'] += int(update['step_scale'] < 1)
                total['combined_gradient_norm_sum'] += update['gradient_norm']
                total['weighted_retention_gradient_norm_sum'] += diag.get('weighted_retention_gradient_norm', 0.)
                if batch % 20 == 0:
                    log.write(json.dumps({'epoch': epoch, 'batch': batch, **diag, **update}) + '\n')
                    log.flush()
            total['mean_task_loss'] = total['task_loss_sum'] / max(total['valid'], 1)
            total['mean_retention_loss'] = total['retention_loss_sum'] / max(total['retention_samples'], 1)
            total['relative_drift'] = relative_drift(model, initial)
            history.append(total)
            print(json.dumps(total), flush=True)
            if total['relative_drift'] >= cfg['max_relative_drift']:
                stop = 'drift_guard'
            if not numerical:
                probe_record(epoch)
            if epoch in cfg['evaluation_epochs'] or numerical or stop == 'drift_guard':
                record(epoch, eligible=not numerical)
            if numerical or stop == 'drift_guard':
                break
    training_s = time.monotonic() - started
    eligible = [r for r in results if r['eligible_for_selection']]
    chosen = min(eligible, key=lambda r: (-round(r['development']['R@1'] * 500), r['epoch']))
    cp = next(c for c in checkpoints if c['epoch'] == chosen['epoch'])
    selection = {'rule': 'seed0 development500 R@1 only; earliest tie; epoch0 allowed',
                 'chosen_epoch': chosen['epoch'], 'development': chosen['development'], 'checkpoint': cp,
                 'uses_winter': False, 'uses_retained_summer': False}
    write_json(out / 'selection.json', selection)
    if sha_file(cp['path']) != cp['sha256']:
        raise ValueError('selected checkpoint changed')
    payload = torch.load(cp['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    full_pixels = load_inputs(pointer, False)
    winter_pointer = checked_json(sources['winter_inputs'], sources['winter_inputs_sha256'])
    if sha_file(winter_pointer['pixel_cache']) != winter_pointer['pixel_cache_sha256']:
        raise ValueError('winter input changed')
    winter = torch.load(winter_pointer['pixel_cache'], map_location='cpu', weights_only=True)['pixels']
    if any(not torch.equal(full_pixels[s], winter[s]) for s in ('spring', 'fall')):
        raise ValueError('reference pixels differ')
    before_eval = tensor_digest(model.state_dict())
    features = extract(model, {**full_pixels, 'winter': winter['winter']}, device)
    complete = []
    for season in ('summer', 'winter'):
        r = score_season(features, season)
        file = f'locked_{season}_predictions.json'
        write_json(out / file, r.pop('predictions'))
        complete.append({'query_season': season, 'predictions_file': file, **r})
    if tensor_digest(model.state_dict()) != before_eval:
        raise RuntimeError('final evaluation changed model')
    write_json(out / 'summary.json', {'status': 'complete', 'phase': 'night_finetune', 'method': method,
        'seed': seed, 'arm': arm, 'stop_reason': stop, 'epoch_results': results, 'checkpoints': checkpoints,
        'epoch_diagnostics': history, 'probe_history': probes, 'probe_identity': probe_identity,
        'calibration': step, 'retention_calibration': hold_calibration, 'retention_coefficient': coefficient,
        'schedule_identity': schedule_identity, 'initial_parity': parity, 'initial_state_sha256': initial_hash,
        'selection': selection, 'locked_results': complete, 'final_model_state_unchanged': True,
        'main_training_updates': sum(r['batches'] for r in history),
        'pilot_training_updates': hold_calibration['pilot_updates'],
        'counted_training_segments': 1 + int(hold_calibration['pilot_updates'] > 0),
        'training_loop_with_development_and_probe_s': training_s,
        'peak_allocated_GPU_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'source': source, 'retained_queries_used_for_selection': False, 'winter_used_for_selection': False,
        'scope': 'known-map exposed-condition diagnostic; not independent final test', **provenance})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sources-sha', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--method', choices=['full', 'anchor'], required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--arm', choices=['retention', 'plain'], required=True)
    p.add_argument('--learning-rate', type=float, required=True)
    for k in ('data-dir', 'cache-dir', 'run-root', 'deployment-manifest',
              'deployment-manifest-sha', 'environment-manifest'):
        p.add_argument('--' + k)
    a = p.parse_args()
    cfg = read(a.config)
    cfg.update(run_id=a.run_id, max_lr=a.learning_rate, requested_learning_rate=a.learning_rate)
    for k in ('data_dir', 'cache_dir', 'run_root', 'deployment_manifest',
              'deployment_manifest_sha', 'environment_manifest'):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    if cfg['n_places'] != 3300 or cfg['development_ids'] != [600, 1099] or not 1e-4 <= a.learning_rate <= 0.1:
        raise ValueError('registered protocol or learning rate invalid')
    code = code_identity(ROOT, cfg.get('deployment_manifest'), cfg.get('deployment_manifest_sha'))
    if code['dirty']:
        raise RuntimeError('dirty training source')
    deployed = code['source'] == 'verified_deployment_snapshot'
    env = environment_identity(cfg.get('environment_manifest'), ROOT / 'pixi.lock', required=deployed)
    sources = checked_json(a.sources, a.sources_sha)
    provenance = {'code': code, 'environment': env, 'runtime': runtime_identity(torch, np)}
    provenance['runtime']['device'] = cfg['device']
    provenance['runtime']['device_name'] = torch.cuda.get_device_name(torch.device(cfg['device'])) if str(cfg['device']).startswith('cuda') else 'cpu'
    out, _ = resolve_locations(cfg, ROOT, deployed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'sources.json', sources)
    train(cfg, sources, out, provenance, a.method, a.seed, a.arm)


if __name__ == '__main__':
    main()
