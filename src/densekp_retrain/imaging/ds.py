"""Double Sphere scene post-processing, without Blender or experiment imports.

The original ERP range is used for surface visibility before any image remap.
RGB takes the saved preview -> fisheye -> normal ERP route. The explicit lens
footprint is applied in fisheye space and intersected again in ERP space.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image
from densekp_retrain.body.transforms import world_points
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.visibility import erp_visibility
from densekp_retrain.camera.double_sphere import (
    DS_20260320,
    CAMERA_EXTRINSIC,
    build_erp_to_fisheye_lut,
    build_fisheye_to_source_erp_lut,
    unproject_double_sphere,
)
from densekp_retrain.imaging.source import read_ds_source
from densekp_retrain.imaging.remap import remap_periodic_erp
from densekp_retrain.visualization.imaging import draw_overlay


@dataclass
class DSDomain:
    """Boolean ERP/lens masks; float32 ERP-to-fish maps in sensor pixels."""

    valid_mask: np.ndarray
    lens_mask: np.ndarray
    back_x: np.ndarray
    back_y: np.ndarray
    removed_erp_pixels: int


@dataclass
class DSProjection:
    """Flattened [person*138,2] pixels, metre ranges and boolean point groups."""

    xy: np.ndarray
    fish_xy: np.ndarray
    ranges: np.ndarray
    domain: np.ndarray
    in_view: np.ndarray
    source_visible: np.ndarray
    visible: np.ndarray
    ray_error: float


@dataclass
class DSImages:
    """Uint8 RGB on each raster; nearest-remapped rounded float32 ERP IDs."""

    fisheye_rgb: np.ndarray
    normal_erp_rgb: np.ndarray
    normal_erp_instance: np.ndarray


def read_world_keypoints(people, vertices, dense):
    """Read each person geometry and apply the shared body lift/yaw/world translation."""
    world = []
    for person in people:
        if "geometry" in person:
            geometry = Path(person["geometry"])
            body = np.load(geometry / "vertices.npy", mmap_mode="r")
            points = np.load(geometry / "densekp_3d.npy", mmap_mode="r")
        else:
            body, points = vertices, dense
        frame = person["frame"]
        world.append(world_points(points[frame], body[frame], person))
    world = np.stack(world)
    return world


def prepare_ds_domain(name, source, erp_mask, lens_mask):
    """Nearest-sample explicit lens mask; preserve ERP intersection and rounding."""
    height, width = source.ray_range_m.shape
    mask = cv2.resize(
        erp_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    lens_mask = np.asarray(lens_mask, dtype=bool)
    intrinsics = DS_20260320[name]
    if lens_mask.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"{name}: lens mask does not match DS image size")
    back_x, back_y = build_erp_to_fisheye_lut(intrinsics, name, width, height)
    lens_erp = cv2.remap(
        lens_mask.astype(np.uint8),
        back_x,
        back_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    removed_erp_pixels = int((mask & ~lens_erp).sum())
    # Labels, instances and RGB share the intersection with the measured footprint.
    mask &= lens_erp
    return DSDomain(mask, lens_mask, back_x, back_y, removed_erp_pixels)


def project_ds_scene(world, ids, pose, name, source, domain_result):
    """Use raw source ERP ray range for periodic 5x5 visibility and DS roundtrip preflight."""
    mask = domain_result.valid_mask
    depth, instance = source.ray_range_m, source.instance
    flat = world.reshape(-1, 3)
    xy, fish_xy, ranges, domain, in_view = point_in_view(flat, pose, name, mask)
    _, source_visible = erp_visibility(xy, ranges, depth, instance, ids)
    visible = source_visible & in_view
    q = (flat - pose[:3, 3]) @ pose[:3, :3]
    rays = q @ CAMERA_EXTRINSIC[name].T
    recovered, inverse_valid = unproject_double_sphere(fish_xy, DS_20260320[name])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    valid_roundtrip = domain & inverse_valid
    ray_error = float(
        np.max(np.linalg.norm(recovered[valid_roundtrip] - rays[valid_roundtrip], axis=1))
    )
    if ray_error > 1e-8:
        raise RuntimeError(f"{name}: DS projection roundtrip {ray_error}")
    return DSProjection(
        xy, fish_xy, ranges, domain, in_view, source_visible, visible, ray_error
    )


def remap_ds_images(name, source, domain_result):
    """Linear RGB and nearest rounded IDs; mask fish before restoring normal ERP."""
    rgb, instance = source.rgb, source.instance
    height, width = source.ray_range_m.shape
    lens_mask, mask = domain_result.lens_mask, domain_result.valid_mask
    back_x, back_y = domain_result.back_x, domain_result.back_y
    fish_x, fish_y = build_fisheye_to_source_erp_lut(DS_20260320[name], name, width, height)
    fish_rgb = remap_periodic_erp(rgb, fish_x, fish_y, cv2.INTER_LINEAR)
    fish_rgb[~lens_mask] = 0
    restored = cv2.remap(
        fish_rgb, back_x, back_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    fish_instance = remap_periodic_erp(
        np.rint(instance).astype(np.float32), fish_x, fish_y, cv2.INTER_NEAREST
    )
    fish_instance[~lens_mask] = 0
    final_instance = cv2.remap(
        fish_instance, back_x, back_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    final_instance[~mask] = 0
    restored[~mask] = 0
    return DSImages(fish_rgb, restored, final_instance)


def save_ds_images(camera_dir, images, domain_result):
    fish_rgb, restored = images.fisheye_rgb, images.normal_erp_rgb
    lens_mask, mask = domain_result.lens_mask, domain_result.valid_mask
    Image.fromarray(fish_rgb).save(camera_dir / "fisheye.png")
    Image.fromarray(lens_mask.astype(np.uint8) * 255).save(
        camera_dir / "reference_lens_mask.png"
    )
    Image.fromarray(restored).save(camera_dir / "normal_erp.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(camera_dir / "valid_mask.png")


def make_ds_annotations(people, ids, world, pose, projection):
    """Retain [person,138] domain/visibility groups; reliable occluded GT stays in_view."""
    xy, fish_xy = projection.xy, projection.fish_xy
    domain, in_view = projection.domain, projection.in_view
    source_visible, visible = projection.source_visible, projection.visible
    shape = (len(people), 138)
    data = dict(
        person_ids=np.array(ids),
        xy=xy.reshape(*shape, 2).astype(np.float32),
        fisheye_xy=fish_xy.reshape(*shape, 2).astype(np.float32),
        model_domain=domain.reshape(shape),
        in_view=in_view.reshape(shape),
        surface_visible=source_visible.reshape(shape),
        visible=visible.reshape(shape),
        out_of_view=(~in_view).reshape(shape),
        occluded_in_view=(in_view & ~source_visible).reshape(shape),
        camera_to_world_erp=pose,
        world_densekp=world,
    )
    return data


def save_ds_annotations(camera_dir, data, ids, images, domain_result):
    restored, final_instance = images.normal_erp_rgb, images.normal_erp_instance
    mask = domain_result.valid_mask
    np.savez(camera_dir / "annotations.npz", **data)
    draw_overlay(restored, data["xy"], data["visible"], data["in_view"], ids).save(
        camera_dir / "overlay.png"
    )
    boxes = []
    for index, person_id in enumerate(ids):
        selected = (np.rint(final_instance) == person_id) & mask
        Image.fromarray(selected.astype(np.uint8) * 255).save(
            camera_dir / f"person_{person_id}_mask.png"
        )
        yy, xx = np.where(selected)
        bbox = (
            [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            if len(xx)
            else None
        )
        boxes.append(
            dict(
                person_id=person_id,
                bbox_xyxy_exclusive=bbox,
                source="oracle rendered instance mask; not detector output",
            )
        )
    (camera_dir / "boxes.json").write_text(json.dumps(boxes, indent=2) + "\n")


def save_scene_pair(output, ids, world, states, camera_reports):
    paired = []
    for j, person_id in enumerate(ids):
        a, b = states["cam2"]["visible"][j], states["cam4"]["visible"][j]
        paired.append(
            dict(
                person_id=person_id,
                shared_visible=int((a & b).sum()),
                cam2_only_visible=int((a & ~b).sum()),
                cam4_only_visible=int((b & ~a).sum()),
                neither_visible=int((~a & ~b).sum()),
            )
        )
    np.savez(
        output / "pair_annotations.npz",
        person_ids=np.array(ids),
        world_densekp=world,
        shared_visible=states["cam2"]["visible"] & states["cam4"]["visible"],
        cam2_xy=states["cam2"]["xy"],
        cam4_xy=states["cam4"]["xy"],
    )
    panels = [Image.open(output / n / "overlay.png").convert("RGB") for n in ("cam2", "cam4")]
    sheet = Image.new("RGB", (panels[0].width, 2 * panels[0].height))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * panel.height))
    sheet.save(output / "pair_review.jpg")
    return dict(cameras=camera_reports, pairs=paired)


def process_scene(output, people, poses, masks, vertices, dense, *, lens_masks, lens_metadata):
    """Read -> prepare mapping/domain -> source-label preflight -> remap -> save.

    Preflight remains before raster remapping to preserve historical failures.
    Original range buffers never get replaced with remapped depth. Pixel and
    label saving order is retained, including partial outputs on an exception.
    """
    ids = [person["id"] for person in people]
    world = read_world_keypoints(people, vertices, dense)
    np.save(output / "shared_world_densekp.npy", world)
    camera_reports, states = [], {}
    for name in ("cam2", "cam4"):
        source = read_ds_source(output / "source", name)
        domain = prepare_ds_domain(name, source, masks[name], lens_masks[name])
        projection = project_ds_scene(world, ids, poses[name], name, source, domain)
        images = remap_ds_images(name, source, domain)
        camera_dir = output / name
        camera_dir.mkdir(exist_ok=True)
        save_ds_images(camera_dir, images, domain)
        data = make_ds_annotations(people, ids, world, poses[name], projection)
        save_ds_annotations(camera_dir, data, ids, images, domain)
        camera_reports.append(
            dict(
                camera=name,
                in_view=data["in_view"].sum(1).tolist(),
                visible=data["visible"].sum(1).tolist(),
                ray_roundtrip_max=projection.ray_error,
                lens_mask={
                    **lens_metadata[name],
                    "removed_erp_valid_pixels": domain.removed_erp_pixels,
                },
            )
        )
        states[name] = data
    return save_scene_pair(output, ids, world, states, camera_reports)
