"""MIT licensed checks for original place-output evaluation semantics."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_original_winter import original_scores
from eval_attribution2700 import state_digest, output_linear
from vprtempo.src.benchmark2700_common import score_report
from vprtempo.src.metrics import recallAtK


def main():
    torch.set_num_threads(2)
    weights = [(torch.tensor([[1., -2.], [-3., 1.]]), torch.tensor([[1., -1.], [2., 1.]])),
               (torch.tensor([[-1., 3.], [2., -2.]]), torch.tensor([[2., 1.], [-1., 2.]]))]
    pixels = torch.tensor([[1., 2.], [-2., 1.], [0., 0.]])
    before = [state_digest({'w': w, 'o': o}) for w, o in weights]
    actual = original_scores(pixels, weights, torch.device('cpu'))
    modules = []
    for w, o in weights:
        a = torch.nn.Linear(2, 2, bias=False)
        b = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            a.weight.copy_(w)
            b.weight.copy_(o)
        modules.append(torch.nn.Sequential(a, b))
    expected = torch.cat([m(pixels) for m in modules], 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, 4) and bool((actual < 0).any())
    assert torch.equal(actual[-1], torch.zeros(4))
    assert before == [state_digest({'w': w, 'o': o}) for w, o in weights]
    scores = torch.eye(6).numpy()
    result = score_report(scores, list(range(6)), recallAtK)
    assert result['subsets']['overall']['R@1'] == result['subsets']['overall']['R@5'] == 1.
    print('PASS: two linear maps, expert concatenation, batch1 signed raw scores, zero input, state unchanged, exact-place ranking')


if __name__ == '__main__':
    main()
