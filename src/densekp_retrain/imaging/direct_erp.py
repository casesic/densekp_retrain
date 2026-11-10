"""Direct ERP labels from the actual ERP range/ID buffers, without DS image remap.

Keep the calibrated physical domain, historical point sampling and visibility
rules. A local buffer must cover every labeled point's original 5x5 neighborhood.
"""
import numpy as np
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.visibility import erp_visibility
from .regions import require_point_neighborhoods


def direct_annotations(world,person_ids,pose,camera,valid_mask,depth,instance,*,coverage_region=None):
    if depth.shape!=valid_mask.shape or instance.shape!=valid_mask.shape or valid_mask.dtype!=bool:
        raise ValueError('raster_or_valid_mask_mismatch')
    world=np.asarray(world)
    if world.shape!=(len(person_ids),138,3) or not np.isfinite(world).all():raise ValueError('invalid_world_keypoints')
    xy,fish,distance,domain,in_view=point_in_view(world.reshape(-1,3),pose,camera,valid_mask)
    if coverage_region is not None:
        require_point_neighborhoods(xy,coverage_region,(valid_mask.shape[1],valid_mask.shape[0]),radius_px=2)
    _,surface_visible=erp_visibility(xy,distance,depth,instance,person_ids)
    visible=surface_visible&in_view;shape=(len(person_ids),138)
    return dict(person_ids=np.asarray(person_ids),xy=xy.reshape(*shape,2).astype(np.float32),fisheye_xy=fish.reshape(*shape,2).astype(np.float32),
        model_domain=domain.reshape(shape),in_view=in_view.reshape(shape),surface_visible=surface_visible.reshape(shape),visible=visible.reshape(shape),
        out_of_view=(~in_view).reshape(shape),occluded_in_view=(in_view&~surface_visible).reshape(shape),camera_to_world_erp=pose,world_densekp=world)
