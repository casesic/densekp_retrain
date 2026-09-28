"""DenseKP objectives: explicit annotation weights, optional predicted log sigma.

All arithmetic is float32 even under autocast. Undefined zero-weight targets
are removed before subtraction. Nonfinite model outputs are always errors.
"""
import math
import torch


def _prepare(prediction, target):
    if prediction.ndim != 3 or prediction.shape[-1] != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [B,K,3] prediction and target tensors')
    prediction, target = prediction.float(), target.to(device=prediction.device, dtype=torch.float32)
    weights = target[..., 2]
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite DenseKP prediction')
    if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError('Annotation supervision weights must be finite in [0,1]')
    valid = weights > 0
    if not torch.isfinite(target[..., :2][valid]).all():
        raise ValueError('Nonfinite supervised target xy')
    # Mask both operands before arithmetic: neither NaN*0 nor overflow*0 is safe.
    xy_pred = torch.where(valid[..., None], prediction[..., :2], 0.)
    xy_target = torch.where(valid[..., None], target[..., :2], 0.)
    return prediction, xy_pred - xy_target, weights


def keypoint_loss(prediction, target, with_var):
    prediction, difference, weights = _prepare(prediction, target)
    squared = difference.square().sum(-1)
    normalizer = weights.sum().clamp_min(1)
    if with_var:
        log_sigma = torch.where(weights > 0, prediction[..., 2], 0.).clamp_min(math.log(1e-6))
        keypoints = (1000 * squared * (0.5 * torch.exp(-2 * log_sigma)) * weights).sum() / normalizer
        uncertainty = (2 * log_sigma * weights).sum() / normalizer
    else:
        keypoints = (squared * weights).sum() / normalizer
        uncertainty = prediction.new_zeros(())
    loss = 60 * (keypoints + uncertainty)
    return loss, {'loss/loss_keypoints': keypoints,
                  # Historical logging key retained for checkpoint/tool compatibility.
                  # It names a log-variance regularizer, not calibrated confidence.
                  'loss/loss_confidence': uncertainty, 'loss/total_loss': loss}


def keypoint_error_totals(prediction, target):
    """Sum of person-mean normalized-crop L2 errors, scored count, empty count."""
    _, difference, weights = _prepare(prediction, target)
    counts = weights.sum(-1)
    per_person = (difference.norm(dim=-1) * weights).sum(-1) / torch.where(counts > 0, counts, 1.)
    valid = counts > 0
    return torch.stack((per_person[valid].sum(), valid.sum(), (~valid).sum())).detach()
