"""MIT licensed anchor-only functional retention for existing expert features.

The teacher is fixed. This auxiliary derivative is applied to the anchor in
both Full and Anchor; it never opens positive/negative paths in Anchor.
"""
import torch
from vprtempo.src.expert_finetune import gradients


def retention_gradient(model, anchor, teacher_descriptor):
    """Derivative of mean half squared distance between unit descriptors."""
    with torch.no_grad():
        d, h, valid = model.encode(anchor)
        teacher = teacher_descriptor.detach()
        if teacher.shape != d.shape or not torch.isfinite(teacher).all():
            raise ValueError('teacher descriptor shape or finiteness mismatch')
        valid = valid & (teacher.norm(dim=1) > 1e-8)
        count = int(valid.sum())
        error = d - teacher
        loss = 0.5 * error[valid].square().sum(1).mean() if count else anchor.new_tensor(0.)
        feedback = (error - d * (error * d).sum(1, keepdim=True)) / h.norm(dim=1, keepdim=True).clamp_min(1e-8)
        feedback *= valid[:, None]
        all_grad = feedback.T @ anchor / max(count, 1)
        gs = list(all_grad.split([w.shape[0] for w in model.weights], dim=0))
        gs = model.mask_gradient(gs)
        norm = float(torch.sqrt(sum(g.square().sum() for g in gs)))
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gs):
            raise FloatingPointError('nonfinite retention derivative')
        return gs, {'retention_loss': float(loss), 'retention_valid': count,
                    'retention_gradient_norm': norm, 'teacher_requires_grad': False,
                    'retention_scope': 'anchor_only_for_both_methods'}


def combined_gradient(model, anchor, positive, negative, teacher, margin, method, coefficient):
    if coefficient < 0:
        raise ValueError('negative retention coefficient')
    task, diagnostic = gradients(model, anchor, positive, negative, margin, method)
    if coefficient == 0:
        # A true bypass: do not inspect teacher or invoke the disabled module.
        return task, {**diagnostic, 'retention_enabled': False, 'retention_coefficient': 0.}
    hold, detail = retention_gradient(model, anchor, teacher)
    combined = [g + coefficient * r for g, r in zip(task, hold)]
    return combined, {**diagnostic, **detail, 'retention_enabled': True,
                      'retention_coefficient': coefficient,
                      'weighted_retention_gradient_norm': coefficient * detail['retention_gradient_norm'],
                      'combined_gradient_norm': float(torch.sqrt(sum(g.square().sum() for g in combined)))}
