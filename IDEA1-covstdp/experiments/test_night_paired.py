"""Synthetic orchestration test; fake metrics are never VPR experiment results."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_night_paired as paired
from vprtempo.src.benchmark2700_common import write_json, sha_file


def main():
    assert paired.matched_rate(.002, .004, .1) == .05
    for value in (0., -1., float('nan')):
        try: paired.matched_rate(value, .004, .1)
        except ValueError: pass
        else: raise AssertionError('invalid drift accepted')
    original_train, original_inputs = paired.runner.train, paired.load_inputs
    calls = []
    def fake_train(cfg, sources, out, provenance, method, seed, arm):
        calls.append((arm, cfg['requested_learning_rate']))
        drift = .002 if arm == 'retention' else .004 * cfg['requested_learning_rate'] / .1
        checkpoint = out / 'checkpoint_epoch_2.pth'
        checkpoint.write_bytes(b'synthetic-opaque-checkpoint-not-for-loading')
        cp = {'epoch': 2, 'path': str(checkpoint), 'sha256': sha_file(checkpoint)}
        write_json(out / 'training_epoch_2.json', {'relative_drift': drift})
        locked = []
        for season in ('summer', 'winter'):
            file = 'locked_' + season + '_predictions.json'
            write_json(out / file, [])
            locked.append({'query_season': season, 'predictions_file': file,
                           'subsets': {'overall': {'queries': 2700, 'R@1': .8, 'R@5': .9}}})
        summary = {'status': 'complete', 'phase': 'night_finetune', 'method': method, 'seed': seed,
                   'selection': {'chosen_epoch': 2, 'checkpoint': cp}, 'locked_results': locked,
                   'checkpoints': [cp], 'epoch_results': [{'epoch': 2, 'relative_drift': drift,
                        'eligible_for_selection': True, 'development': {'queries': 500, 'R@1': .8}}],
                   'calibration': {'lr': cfg['requested_learning_rate'], 'margin': .1},
                   'counted_training_segments': 2 if arm == 'retention' else 1, **provenance}
        write_json(out / 'summary.json', summary)
    try:
        paired.runner.train = fake_train
        paired.load_inputs = lambda *args: (_ for _ in ()).throw(AssertionError('already locked E2 predictions recomputed'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'parent/artifacts';out.mkdir(parents=True)
            data = root / 'data';data.mkdir()
            reference = root / 'reference.json'
            write_json(reference, {'method': 'anchor', 'seed': 0, 'initial_state_sha256': 'same',
                'calibration': {'lr': .1, 'margin': .1}, 'epoch_results': [{'epoch': 2, 'relative_drift': .004}]})
            source = {'plain_reference_e2': {'anchor': {'summary': str(reference), 'summary_sha256': sha_file(reference),
                      'relative_drift': .004, 'learning_rate': .1}},
                      'pairing_references': {'0': {'initial_state_sha256': 'same'}}}
            cfg = {'run_id': 'synthetic-parent', 'data_dir': str(data), 'cache_dir': str(root/'cache'),
                   'run_root': str(root), 'device': 'cpu', 'requested_learning_rate': .1}
            provenance = {'code': {'commit': 'a'*40, 'source': 'local_git'}, 'environment': {},
                          'runtime': {'slurm_job_id': 'synthetic-no-real-job'}}
            result = paired.execute(cfg, source, out, provenance, 'anchor', 0)
            assert calls == [('retention', .1), ('plain', .05)]
            assert result['counted_training_segments'] == 3 and result['unique_prediction_files'] == 4
            assert len(result['results']) == 8 and result['match_rule']['relative_drift_mismatch'] == 0.
            assert not result['query_scores_used_for_matching']
            assert len({s['run_id'] for s in result['subruns']}) == 2
            for s in result['subruns']:
                d=Path(s['run_dir']);assert d.resolve().is_relative_to(out.resolve())
                assert (d/'artifacts/summary.json').is_file() and (d/'trial_outcome.json').is_file()
            # Known .1 trajectory must not be trained a second time.
            calls.clear()
            write_json(reference, {'method': 'anchor', 'seed': 0, 'initial_state_sha256': 'same',
                'calibration': {'lr': .1, 'margin': .1}, 'epoch_results': [{'epoch': 2, 'relative_drift': .002}]})
            source['plain_reference_e2']['anchor'].update(summary_sha256=sha_file(reference), relative_drift=.002)
            out2=root/'parent2/artifacts';out2.mkdir(parents=True)
            cfg2=copy.deepcopy(cfg);cfg2['run_id']='synthetic-parent2'
            reused=paired.execute(cfg2,source,out2,provenance,'anchor',0)
            assert calls == [('retention', .1)] and reused['counted_training_segments'] == 2
            assert reused['control_status'] == 'pending_existing_model_reuse'
    finally:
        paired.runner.train, paired.load_inputs = original_train, original_inputs
    print('PASS: paired subruns unique/in-project, train-only drift LR matching,3 segments counted, E2 prediction reuse, no real VPR data or GPU')


if __name__ == '__main__': main()
