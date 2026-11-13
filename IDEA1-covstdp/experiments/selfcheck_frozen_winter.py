"""MIT licensed numerical parity and frozen-state checks for the winter adapter."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frozen_winter import evaluate, restore, sha_file
import eval_frozen_winter as winter
from eval_locked2700 import evaluate as summer_score
from vprtempo.src.expert_finetune import ExpertFeature, tensor_digest


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = ExpertFeature([torch.randn(8, 16), torch.randn(7, 16), torch.randn(9, 16)])
    model.requires_grad_(False)
    spring = torch.randn(3300, 16)
    pixels = {'spring': spring, 'fall': spring.clone(), 'winter': spring[600:].clone()}
    before = tensor_digest(model.state_dict())
    r = evaluate(model, pixels, torch.device('cpu'))
    assert r['subsets']['overall']['R@1'] == r['subsets']['overall']['R@5'] == 1.
    assert 'development' not in r['subsets'] and r['model_state_unchanged']
    old = summer_score(None, winter.extract(model, {'spring': spring, 'fall': spring,
                       'summer': spring[600:]}, torch.device('cpu')), list(range(600, 3300)))
    assert old['predictions'] == r['predictions'], 'winter changed ranking semantics'
    pixels['winter'][0].zero_()
    r = evaluate(model, pixels, torch.device('cpu'))
    assert r['subsets']['overall']['invalid_queries'] == 1
    assert r['subsets']['overall']['R@1'] == 2699 / 2700
    assert tensor_digest(model.state_dict()) == before
    # Validate saved state and metadata using a tiny model, without real checkpoints.
    original_loader = winter.load_source
    winter.load_source = lambda source, device: copy.deepcopy(model).to(device)
    try:
        lock = {'method': 'anchor', 'seed': 2, 'training_commit': 'a' * 40,
                'initial_state_sha256': 'initial', 'schedule_identity': {'tensor_sha256': 'plan'},
                'calibration_sha256': 'calibration', 'probe_sha256': 'probe'}
        source = {'checkpoint_sha256': 'source'}
        metadata = {'epoch': 2, 'seed': 2, 'method': 'anchor', 'code_commit': 'a' * 40,
                    'source_checkpoint_sha256': 'source', 'initial_state_sha256': 'initial',
                    'schedule_sha256': 'plan', 'calibration_sha256': 'calibration', 'probe_sha256': 'probe'}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pth'
            torch.save({'model': model.state_dict(), 'metadata': metadata}, p)
            lock['checkpoint'] = {'path': str(p), 'sha256': sha_file(p)}
            restored = restore(lock, source, torch.device('cpu'))
            assert tensor_digest(restored.state_dict()) == before
            assert not any(w.requires_grad for w in restored.parameters())
            lock['training_commit'] = 'b' * 40
            try:
                restore(lock, source, torch.device('cpu'))
            except ValueError:
                pass
            else:
                raise AssertionError('changed producer identity accepted')
    finally:
        winter.load_source = original_loader
    print('PASS: summer/winter ranking parity, exact GT, invalid query denominator, unchanged state, checkpoint identity rejection')


if __name__ == '__main__':
    main()
