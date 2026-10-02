"""Historical DenseKP loading and inference operations shared by entry points.

Precision, device selection and evaluation policy belong to callers. The two
loaders intentionally retain their distinct initialization paths, including the
backbone checkpoint read performed by DenseKP construction.
"""
import torch

from .densekp_trainer import DenseKP


def load_checkpoint_state_dict(checkpoint, cfg):
    """Return (CPU model, load metadata), reading the payload before construction.

    Strict key checks and their exceptions are unchanged. The returned model
    retains its constructor mode; this function does not move it or call eval.
    Metadata field order matches the historical candidate evaluation report.
    """
    payload = torch.load(checkpoint, map_location='cpu')
    model = DenseKP(cfg=cfg)
    result = model.load_state_dict(payload['state_dict'], strict=True)
    details = dict(epoch=payload.get('epoch'), global_step=payload.get('global_step'),
                   state_dict_keys=len(payload['state_dict']), missing_keys=result.missing_keys,
                   unexpected_keys=result.unexpected_keys, strict=True)
    del payload
    return model, details


def load_lightning_checkpoint(checkpoint, cfg):
    """Load strictly through Lightning on CPU, preserving its constructor/hooks.

    Device transfer and model mode remain with the caller. This path must not be
    replaced by the manual state_dict path merely because their predictions match.
    """
    return DenseKP.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location='cpu', strict=True)


def predict_batch(model, images, device):
    """Forward normalized NumPy [B,3,256,256] images, returning the raw tensor.

    Do not crop the middle 192 columns here; DenseKP owns that operation. This
    function sets inference_mode but preserves the model's train/eval flag and
    the caller's precision/autocast settings. It neither casts nor validates the
    raw [B,138,3] result: entry points keep their distinct shape/nonfinite failure
    handling and diagnostic-file behavior. The third output remains uninterpreted.
    """
    with torch.inference_mode():
        return model({'img': torch.from_numpy(images).to(device)})['pred_keypoints']


def predict_item(model, item, device):
    """Historical smoke prediction: switch to eval and leave the model in eval.

    Return float32 NumPy [1,138,3]; reject malformed or nonfinite output with the
    original error. This deliberate mode side effect differs from predict_batch.
    """
    model.eval()
    value = predict_batch(model, item['img'][None], device)
    if value.shape != (1, 138, 3) or not torch.isfinite(value).all():
        raise ValueError('Expected finite 138x3 output')
    return value.float().cpu().numpy()
