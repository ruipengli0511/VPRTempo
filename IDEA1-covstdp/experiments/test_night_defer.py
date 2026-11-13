"""Synthetic control-flow test: no complete query load during seed expansion."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_night_finetune as night
from vprtempo.src.expert_finetune import ExpertFeature, make_schedule, tensor_digest


def main():
    torch.set_num_threads(2)
    torch.manual_seed(18)
    model = ExpertFeature([torch.randn(9, 5)])
    pixels = {'spring': torch.randn(128, 5), 'fall': torch.randn(128, 5), 'summer': torch.zeros(1, 5)}
    features = night.extract(model, {s: pixels[s] for s in ('spring', 'fall')}, torch.device('cpu'), normalized=True)
    plan, identity = make_schedule(features, 2, 1, 50)
    source = {'seed': 1, 'checkpoint_sha256': 'synthetic-source'}
    sources = {'models': [source], 'summer_inputs': 'synthetic-summer-pointer', 'summer_inputs_sha256': 'dummy',
               'winter_inputs': 'FORBIDDEN-WINTER', 'winter_inputs_sha256': 'dummy',
               'pairing_references': {'1': {'schedule_identity': identity,
                    'initial_state_sha256': tensor_digest(model.state_dict())}}}
    cfg = {'n_places': 128, 'device': 'cpu', 'epochs': 2, 'negative_min_gap': 50,
           'batch_size': 16, 'calibration_batches': 2, 'margin_candidates': [.1, .3, .5],
           'min_calibration_active_fraction': .01, 'max_lr': .01, 'requested_learning_rate': .01,
           'min_lr': 1e-7, 'max_relative_step': .01, 'max_relative_drift': 1.,
           'probe_triplets': 32, 'evaluation_epochs': [0, 1, 2], 'defer_full_evaluation': True}
    old = {name: getattr(night, name) for name in ('checked_json', 'load_inputs', 'load_source', 'score', 'baseline_parity')}
    accesses = []
    def pointer(path, expected):
        assert path == 'synthetic-summer-pointer', 'complete winter pointer accessed before common lock'
        accesses.append(path)
        return {}
    def inputs(pointer, development):
        assert development is True, 'full summer queries accessed before common lock'
        return pixels
    def fake_development(model, pixels, ids, device):
        assert ids == list(range(600, 1100))
        return {'predictions': [], 'subsets': {'development': {'queries': 500, 'R@1': .8, 'R@5': .9,
                                                               'invalid_queries': 0}}}
    try:
        night.checked_json = pointer
        night.load_inputs = inputs
        night.load_source = lambda item, device: copy.deepcopy(model)
        night.score = fake_development
        night.baseline_parity = lambda *args: {'top5_exact_match': True, 'synthetic_control_flow_only': True}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            night.train(cfg, sources, out, {'code': {'commit': 'a' * 40}, 'environment': {}, 'runtime': {}},
                        'anchor', 1, 'plain')
            result = json.loads((out / 'summary.json').read_bytes())
            assert result['deferred_common_epoch_evaluation'] is True and result['locked_results'] == []
            assert result['selection']['status'] == 'provisional_per_seed_not_common'
            assert result['counted_training_segments'] == 1 and result['pilot_training_updates'] == 0
            assert result['selection']['chosen_epoch'] == 0
            assert accesses == ['synthetic-summer-pointer']
            receipt = json.loads((out / 'training_epoch_2.json').read_bytes())
            assert receipt['epoch'] == 2 and 'development' not in receipt
            assert receipt['relative_drift'] == result['epoch_diagnostics'][1]['relative_drift']
    finally:
        for name, value in old.items(): setattr(night, name, value)
    print('PASS: synthetic training deferred full summer and winter access, no independent seed query selection; not a VPR performance experiment')


if __name__ == '__main__': main()
