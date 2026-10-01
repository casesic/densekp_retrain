"""Training annotation contract, independent of model output channel 2.

An explicit producer/config declaration is required: numeric values alone do
not establish confidence, visibility, correspondence or supervision semantics.
"""
import numpy as np
from .keypoint_supervision import SUPERVISION_POLICY

NUM_DENSE_KEYPOINTS = 138


def validate_annotations(data, value_semantics):
    if value_semantics != 'supervision_weight':
        raise ValueError('Declare DATASETS.TARGET_VALUE_SEMANTICS=supervision_weight '
                         'only after verifying the annotation producer. '
                         'DenseKP keypoint_value is not automatically a weight.')
    for name in ('imgname', 'center', 'scale', 'proj_verts'):
        if name not in data:
            raise ValueError(f'Missing annotation field: {name}')
    names, center, scale, points = (np.asarray(data[k]) for k in
                                    ('imgname', 'center', 'scale', 'proj_verts'))
    if names.ndim != 1 or names.size == 0:
        raise ValueError('imgname must be a nonempty [N] array')
    n = len(names)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError('imgname entries must be nonempty strings')
    if center.shape != (n, 2) or not np.isfinite(center).all():
        raise ValueError('center must be finite [N,2] in label-raster pixels')
    if scale.shape not in ((n,), (n, 2)) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('scale must be positive finite [N] or [N,2], in pixels / 200')
    if points.shape != (n, NUM_DENSE_KEYPOINTS, 3) or points.dtype.kind != 'f':
        raise ValueError('proj_verts must be floating [N,138,3] with verified DenseKP row order')
    weights = points[..., 2]
    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError('Declared supervision weights must be finite in [0,1]')
    if not np.isfinite(points[..., :2][weights > 0]).all():
        raise ValueError('Supervised xy must be finite; only zero-weight xy may be undefined')
    if 'supervision_policy' in data:
        policy = np.asarray(data['supervision_policy'])
        if policy.shape != () or policy.item() != SUPERVISION_POLICY:
            raise ValueError('Unknown supervision_policy')
        for key in ('in_view', 'visible'):
            mask = np.asarray(data.get(key))
            if mask.shape != (n, NUM_DENSE_KEYPOINTS) or mask.dtype != np.bool_:
                raise ValueError(f'{key} must be boolean [N,138] for {SUPERVISION_POLICY}')
        if np.any(data['visible'] & ~data['in_view']):
            raise ValueError('Visible points must be in_view')
        if not np.isfinite(points[..., :2][data['in_view']]).all():
            raise ValueError('In-view synthetic coordinates must be finite')
