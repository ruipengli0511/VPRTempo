"""MIT licensed numerical controls for retention and disabled-module parity."""
import copy
import torch
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vprtempo.src.expert_finetune import ExpertFeature, gradients, tensor_digest, apply_gradients
from vprtempo.src.descriptor_retention import retention_gradient, combined_gradient


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    weights = [torch.randn(7, 5, dtype=torch.float64), torch.randn(6, 5, dtype=torch.float64)]
    weights[0][0, 0] = 0
    model = ExpertFeature(weights)
    a, p, n = [torch.randn(9, 5, dtype=torch.float64) for _ in range(3)]
    teacher = torch.nn.functional.normalize(torch.randn(9, 13, dtype=torch.float64), dim=1)
    before = tensor_digest(model.state_dict())
    gs, _ = retention_gradient(model, a, teacher)
    d = model.encode(a)[0]
    loss = 0.5 * (d - teacher).square().sum(1).mean()
    auto = model.mask_gradient(torch.autograd.grad(loss, tuple(model.weights)))
    for u, v in zip(gs, auto):
        torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-10)
    for method in ('full', 'anchor'):
        combined, _ = combined_gradient(model, a, p, n, teacher, 0.1, method, 3.)
        da = model.encode(a)[0]
        dp = model.encode(p)[0]
        dn = model.encode(n)[0]
        if method == 'anchor':
            dp, dn = dp.detach(), dn.detach()
        task = torch.relu(0.1 - (da * dp).sum(1) + (da * dn).sum(1)).mean()
        total = task + 3. * 0.5 * (da - teacher).square().sum(1).mean()
        expected = model.mask_gradient(torch.autograd.grad(total, tuple(model.weights)))
        for u, v in zip(combined, expected):
            torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-10)
        original, _ = gradients(model, a, p, n, 0.1, method)
        off, diag = combined_gradient(model, a, p, n, None, 0.1, method, 0.)
        assert not diag['retention_enabled']
        assert all(torch.equal(u, v) for u, v in zip(off, original))
        m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
        apply_gradients(m1, original, .01, 1e-4)
        apply_gradients(m2, off, .01, 1e-4)
        assert tensor_digest(m1.state_dict()) == tensor_digest(m2.state_dict())
    same = model.encode(a)[0].detach()
    zero, diag = retention_gradient(model, a, same)
    assert diag['retention_loss'] == 0 and all(torch.count_nonzero(g) == 0 for g in zero)
    assert before == tensor_digest(model.state_dict())
    print('PASS: retention derivative, Full/Anchor combined target, masked zeros, zero initial derivative, disabled bitwise update parity, unchanged state')


if __name__ == '__main__':
    main()
