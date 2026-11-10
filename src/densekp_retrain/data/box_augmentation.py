"""Bounded, deterministic ERP box variants; no image or renderer dependency."""
from __future__ import annotations

import random
import numpy as np

from densekp_retrain.data.boxes import change_box_for_image, crop_from_box
from densekp_retrain.imaging.regions import require_crop_in_region

POLICY_VERSION = 'bounded_native_box_v1'
DEFAULT_POLICY = {
    'version': POLICY_VERSION, 'native_probability': .50,
    'center_probability': .25, 'scale_probability': .15,
    'one_side_probability': .10, 'max_center_fraction': .03,
    'scale_range': [.9, 1.1], 'max_one_side_fraction': .10,
    'crop_policy': 'deployment', 'boundary_policy': 'erp_reject_seam',
    'interpolation_margin_px': 2,
}


def validate_box_policy(value):
    """Allow reviewed settings within the fixed envelope, never wider draws."""
    if not isinstance(value, dict) or set(value)!=set(DEFAULT_POLICY):
        raise ValueError('unsupported_box_augmentation_policy')
    if any(value[k]!=DEFAULT_POLICY[k] for k in
           ('version','crop_policy','boundary_policy','interpolation_margin_px')):
        raise ValueError('unsupported_box_augmentation_policy')
    probabilities=[value[k] for k in ('native_probability','center_probability','scale_probability','one_side_probability')]
    if (not np.isfinite(probabilities).all() or any(p<0 or p>1 for p in probabilities)
            or abs(sum(probabilities)-1)>1e-10):
        raise ValueError('invalid_box_augmentation_probabilities')
    low,high=value['scale_range']
    if (not np.isfinite([low,high,value['max_center_fraction'],value['max_one_side_fraction']]).all()
            or not .9<=low<=1<=high<=1.1
            or not 0<=value['max_center_fraction']<=.03
            or not 0<=value['max_one_side_fraction']<=.10):
        raise ValueError('box_augmentation_exceeds_reserved_envelope')
    return dict(value)


def policy_within_reservation(requested, reserved):
    requested=validate_box_policy(requested);reserved=validate_box_policy(reserved)
    return (requested['max_center_fraction']<=reserved['max_center_fraction']
            and requested['max_one_side_fraction']<=reserved['max_one_side_fraction']
            and requested['scale_range'][0]>=reserved['scale_range'][0]
            and requested['scale_range'][1]<=reserved['scale_range'][1])


def sample_box_variant(parent_xyxy, *, policy, seed, observation_key,
                       roi_xyxy, image_size_wh=(5760, 2880), variant_index=0):
    """Draw once, then reject unsupported boxes; never redraw to force support."""
    policy=validate_box_policy(policy)
    key = '|'.join(map(str, (POLICY_VERSION, seed, *observation_key, variant_index)))
    rng = random.Random(key)
    choice = rng.random()
    native=policy['native_probability'];center_p=policy['center_probability'];scale_p=policy['scale_probability']
    mode = ('native' if choice < native else 'center' if choice < native+center_p
            else 'scale' if choice < native+center_p+scale_p else 'one_side')
    parameters = dict(dx=0., dy=0., factor=1., edge=None, fraction=0.)
    if mode == 'center':
        amount=policy['max_center_fraction']
        parameters.update(dx=rng.uniform(-amount,amount), dy=rng.uniform(-amount,amount))
    elif mode == 'scale':
        parameters['factor'] = rng.uniform(*policy['scale_range'])
    elif mode == 'one_side':
        parameters.update(edge=rng.randrange(4), fraction=rng.uniform(0,policy['max_one_side_fraction']))
    box = change_box_for_image(parent_xyxy, image_size_wh=image_size_wh,
        boundary_policy='erp_reject_seam', min_side_px=2,
        shift_reference_expansion=1., **parameters)
    center, scale = crop_from_box(box, image_size_wh=image_size_wh,
        boundary_policy='erp_reject_seam', crop_policy='deployment', min_side_px=2)
    require_crop_in_region(center, scale, roi_xyxy, image_size_wh,
                           interpolation_margin_px=2)
    return dict(bbox_kind='visible_instance_tight' if mode == 'native' else 'synthetic_perturbation',
                parent_bbox_xyxy=np.asarray(parent_xyxy,float).tolist(), bbox_xyxy=box.tolist(),
                center=np.asarray(center,float).tolist(), scale=float(scale), mode=mode,
                parameters=parameters, box_seed=str(seed), observation_key=list(observation_key),
                variant_index=variant_index, crop_policy='deployment',
                support_checked=True, roi_xyxy=list(roi_xyxy))


def require_full_box_family_support(parent_xyxy, roi_xyxy, image_size_wh, policy):
    """Preflight every allowed draw, before training; no selective row dropping.

    A parent may occupy any subset of the projected geometry, but once its
    actual tight box is known the maximum augmented square extends at most
    0.63 of its maximum side beyond each parent edge (plus interpolation).
    """
    validate_box_policy(policy)
    box=np.asarray(parent_xyxy,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=box[:2]).any():
        raise ValueError('invalid_parent_bbox')
    side=float(max(box[2:]-box[:2]));pad=.63*side+2.001
    required=np.r_[box[:2]-pad,box[2:]+pad]
    roi=np.asarray(roi_xyxy,float)
    full=np.asarray(image_size_wh,float)
    if roi.shape!=(4,) or (required[:2]<roi[:2]).any() or (required[2:]>roi[2:]).any() or (required[:2]<0).any() or (required[2:]>full).any():
        raise ValueError('box_augmentation_family_outside_roi_support')
    return required.tolist()
