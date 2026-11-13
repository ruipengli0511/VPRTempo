"""Synthetic repeat orchestration test; no VPR data, GPU or model scoring."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_night_repeat_common as repeat
from vprtempo.src.benchmark2700_common import write_json, sha_file


def summary(seed, commit):
    return {'status': 'complete', 'phase': 'night_finetune', 'method': 'anchor', 'seed': seed,
            'arm': 'retention', 'retention_coefficient': 100., 'calibration': {'lr': .1},
            'probe_identity': {'tensor_sha256': 'same'}, 'epoch_results': [
              {'epoch': 0, 'development': {'R@1': .8}, 'eligible_for_selection': True},
              {'epoch': 1, 'development': {'R@1': [.9,.8,.8][seed]}, 'eligible_for_selection': True},
              {'epoch': 2, 'development': {'R@1': [.8,.9,.8][seed]}, 'eligible_for_selection': True}],
            'code': {'commit': commit}, 'deferred_common_epoch_evaluation': True,
            'locked_results': [], 'counted_training_segments': 1}


def main():
    original_train, original_common = repeat.trainer.train, repeat.common.main
    calls = []
    commit = 'a' * 40
    provenance = {'code': {'commit': commit, 'source': 'local_git'}, 'environment': {},
                  'runtime': {'slurm_job_id': 'synthetic-no-real-job'}}
    def fake_train(cfg, sources, out, prov, method, seed, arm):
        assert cfg['defer_full_evaluation'] and cfg['retention_policy'] == 'fixed'
        assert cfg['fixed_retention_coefficient'] == 100. and cfg['requested_learning_rate'] == .1
        assert seed in (1, 2) and arm == 'retention' and method == 'anchor'
        calls.append(seed)
        write_json(out / 'summary.json', summary(seed, commit))
        write_json(out / 'effective_config.json', {'config': cfg})
    def fake_common():
        def arg(k): return repeat.sys.argv[repeat.sys.argv.index(k) + 1]
        p = Path(arg('--training-runs'));assert sha_file(p) == arg('--training-runs-sha')
        table = json.loads(p.read_bytes())
        assert [x['seed'] for x in table['runs']] == [0, 1, 2]
        all_rows = []
        for item in table['runs']:
            d = Path(item['run_dir'])
            assert sha_file(d / 'artifacts/summary.json') == item['summary_sha256']
            assert json.loads((d / 'trial_outcome.json').read_bytes())['status'] == 'complete'
            all_rows.append(json.loads((d / 'artifacts/summary.json').read_bytes()))
        choices = repeat.common.choose_common(all_rows)
        assert choices[0]['epoch'] == 1
        calls.append('common')
        out = Path(arg('--run-root')) / arg('--run-id') / 'artifacts';out.mkdir(parents=True)
        (out / 'training_runs.input.json').write_bytes(p.read_bytes())
        write_json(out / 'locked_models.json', {'selection': choices, 'models': [], 'training_runs_sha256': sha_file(p)})
        write_json(out / 'training_selection_evidence.json', all_rows)
        rows = []
        for seed in range(3):
            for season in ('summer', 'winter'):
                file = f's{seed}_{season}.json';write_json(out/file, [])
                rows.append({'seed': seed, 'method': 'anchor', 'query_season': season, 'predictions_file': file})
        write_json(out / 'summary.json', {'status': 'complete', 'phase': 'night_common_evaluation',
                   'results': rows, 'selection': choices, 'locked_models_sha256': sha_file(out/'locked_models.json'), **provenance})
    try:
        repeat.trainer.train, repeat.common.main = fake_train, fake_common
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);data=root/'data';data.mkdir();prior=root/'prior';prior_out=prior/'artifacts';prior_out.mkdir(parents=True)
            write_json(prior_out/'summary.json', summary(0, 'b'*40))
            write_json(prior_out/'effective_config.json', {'config': {'device': 'cpu', 'source_checkpoint': 'must_be_removed'}})
            write_json(prior/'result_manifest.json', {'files': []});write_json(prior/'trial_outcome.json', {'status': 'complete'})
            declared = {'run_dir': str(prior), 'expected_commit': 'b'*40,
                        'summary_sha256': sha_file(prior_out/'summary.json'),
                        'effective_config_sha256': sha_file(prior_out/'effective_config.json'),
                        'result_manifest_sha256': sha_file(prior/'result_manifest.json')}
            sources = {'seed0_candidate': declared}
            out=root/'parent/artifacts';out.mkdir(parents=True)
            cfg={'run_id':'synthetic-repeat','data_dir':str(data),'cache_dir':str(root/'cache'),'device':'cpu',
                 'deployment_manifest':'synthetic-manifest','deployment_manifest_sha':'dummy','environment_manifest':'synthetic-env'}
            result=repeat.execute(cfg,sources,out,provenance,'anchor')
            assert calls == [1,2,'common'] and result['counted_training_segments'] == 2
            assert len(result['results']) == 6 and len(result['training_subruns']) == 2
            assert sha_file(out/'training_runs.input.json') == sha_file(out/'training_runs.resolved.json')
            assert all((out/r['predictions_file']).resolve().is_relative_to(out.resolve()) for r in result['results'])
    finally:
        repeat.trainer.train, repeat.common.main = original_train, original_common
    print('PASS: no seed0 retrain, fixed candidate, deferred two seeds, real output hashes sealed before joint selection, common eval in same job; synthetic only')


if __name__ == '__main__': main()
