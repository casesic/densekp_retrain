"""Historical Omni-Radtan post-processing with explicit image/label stages.

The sampling uses BORDER_CONSTANT without DS longitude padding. Visibility uses
nearest-remapped fisheye ray range (metres), a clipped 5x5 window and <0.04 m.
It intentionally keeps image-bounds 'inside' distinct from model-domain validity.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image
import hashlib
from densekp_retrain.camera import project_points_omni_radtan, unproject_pixels_omni_radtan
from densekp_retrain.io import linear_to_srgb
from densekp_retrain.imaging.source import read_source_erp
from densekp_retrain.visualization.imaging import draw_omni_overlay


@dataclass
class OmniMap:
    """Fish-shaped float32 source ERP pixel maps and explicit boolean model domain."""

    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray


@dataclass
class OmniBuffers:
    """Fish-shaped linear RGB, remapped metre range and raw float instance IDs."""

    linear_rgb: np.ndarray
    ray_range_m: np.ndarray
    instance: np.ndarray


@dataclass
class OmniRaster:
    """Uint8 display RGB and 0/255 object mask; object mask is not lens validity."""

    display: np.ndarray
    object_mask: np.ndarray


@dataclass
class OmniLabels:
    """[N,2] pixel projection, boolean groups and metre visibility residuals."""

    projected: np.ndarray
    details: dict
    finite: np.ndarray
    inside: np.ndarray
    visible: np.ndarray
    depth_error: np.ndarray


def scalar_reference(points, camera, k):
    inv = np.linalg.inv(camera)
    out = []
    for p in points:
        q = (inv @ np.r_[p, 1])[:3]
        q = q / np.linalg.norm(q)
        theta = np.arctan2(np.hypot(q[0], q[1]), q[2])
        phi = np.arctan2(q[1], q[0])
        m = np.sin(theta) / (np.cos(theta) + k.xi)
        r = k.f * (m + k.k1 * m**3 + k.k2 * m**5)
        out.append((k.cx + r * np.cos(phi), k.cy + r * np.sin(phi)))
    return np.asarray(out)


def build_omni_source_map(source, k):
    """Map CV rays to ERP pixel coordinates; preserve constant-border seam behavior."""
    h, w = source.ray_range_m.shape
    yy, xx = np.indices((k.height, k.width), dtype=np.float64)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    rays, domain = unproject_pixels_omni_radtan(pixels, k, return_valid=True)
    lon = np.arctan2(rays[:, 0], rays[:, 2])
    lat = np.arcsin(np.clip(-rays[:, 1], -1, 1))
    mapx = ((lon / (2 * np.pi) + 0.5) * w).reshape(k.height, k.width).astype(np.float32)
    mapy = ((0.5 - lat / np.pi) * h).reshape(k.height, k.width).astype(np.float32)
    valid = domain.reshape(k.height, k.width)
    mapx[~valid] = -1
    mapy[~valid] = -1
    return OmniMap(mapx, mapy, valid)


def remap_omni_source(source, mapping):
    rgb, depth, index = source.rgb, source.ray_range_m, source.instance
    mapx, mapy = mapping.x, mapping.y
    warped = cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_depth = cv2.remap(
        depth, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    warped_index = cv2.remap(
        index, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    return OmniBuffers(warped, warped_depth, warped_index)


def apply_omni_domain(buffers, mapping):
    """Zero invalid/nonfinite remapped range in place; keep historical RGB and ID handling."""
    warped, warped_depth, warped_index = (
        buffers.linear_rgb,
        buffers.ray_range_m,
        buffers.instance,
    )
    valid = mapping.valid
    warped_depth[(~valid) | (~np.isfinite(warped_depth))] = 0
    mask = (valid & (warped_index > 0.5)).astype(np.uint8) * 255
    display = (np.clip(linear_to_srgb(warped), 0, 1) * 255 + 0.5).astype(np.uint8)
    return OmniRaster(display, mask)


def save_omni_rasters(output, raster, buffers, mapping, points, person_ids, point_person_ids):
    """Keep partial raster outputs before ID-layout rejection, as the original writer did."""
    display, mask = raster.display, raster.object_mask
    warped_depth, warped_index = buffers.ray_range_m, buffers.instance
    valid = mapping.valid
    Image.fromarray(display).save(output / "rgb.png")
    Image.fromarray(mask).save(output / "mask.png")
    np.save(output / "depth.npy", warped_depth.astype(np.float32))
    if person_ids is not None:
        if point_person_ids is None or len(point_person_ids) != len(points):
            raise ValueError("point_person_ids must match points")
        for person_id in person_ids:
            person_mask = (valid & (np.rint(warped_index) == person_id)).astype(np.uint8) * 255
            Image.fromarray(person_mask).save(output / f"person_{person_id}_mask.png")


def make_omni_labels(points, camera, k, buffers, point_person_ids):
    """Use remapped fish range, optional rounded per-person IDs, and clipped 5x5 neighborhoods."""
    warped_depth, warped_index = buffers.ray_range_m, buffers.instance
    projected, details = project_points_omni_radtan(points, camera, k, True)
    finite = np.isfinite(projected).all(1)
    inside = (
        finite
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < k.width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < k.height)
    )
    camera_points = details["camera_points"]
    point_range = np.linalg.norm(camera_points, axis=1)
    visible = np.zeros(len(points), bool)
    depth_error = np.full(len(points), np.nan)
    for j, (u, v) in enumerate(projected):
        if not inside[j]:
            continue
        x = int(np.floor(u))
        y = int(np.floor(v))
        region = (
            slice(max(0, y - 2), min(k.height, y + 3)),
            slice(max(0, x - 2), min(k.width, x + 3)),
        )
        q = warped_depth[region]
        if point_person_ids is not None:
            q = q[(q > 0) & (np.rint(warped_index[region]) == point_person_ids[j])]
        else:
            q = q[q > 0]
        if q.size:
            depth_error[j] = np.min(np.abs(q - point_range[j]))
            visible[j] = depth_error[j] < 0.04
    return OmniLabels(projected, details, finite, inside, visible, depth_error)


def save_omni_labels(output, raster, labels):
    projected, inside, visible, depth_error = (
        labels.projected,
        labels.inside,
        labels.visible,
        labels.depth_error,
    )
    overlay = draw_omni_overlay(raster.display, projected, inside, visible)
    overlay.save(output / "densekp_overlay.png")
    annotation = projected.astype(np.float32)[None]
    np.save(output / "densekp_fisheye_2d.npy", annotation)
    np.save(output / "densekp_2d.npy", annotation)
    np.save(output / "densekp_valid.npy", inside[None])
    np.save(output / "densekp_visible.npy", visible[None])
    np.save(output / "densekp_depth_error.npy", depth_error.astype(np.float32)[None])


def make_omni_report(
    source, source_data, points, camera, k, labels, source_keypoints, source_fingerprint
):
    projected, details, finite = labels.projected, labels.details, labels.finite
    inside, visible, depth_error = labels.inside, labels.visible, labels.depth_error
    channels = source_data.channels
    reference = scalar_reference(points, camera, k)
    projection_error = np.linalg.norm(projected - reference, axis=1)
    back, back_valid = unproject_pixels_omni_radtan(projected, k, return_valid=True)
    truth = details["rays"]
    angular = np.arccos(np.clip(np.sum(back * truth, axis=1), -1, 1))
    good = back_valid & np.isfinite(angular)
    fingerprint_3d = (
        hashlib.blake2b(source_keypoints.tobytes(), digest_size=16).hexdigest()
        if source_keypoints is not None
        else None
    )
    fingerprint_3d_matches_source = (
        None if source_fingerprint is None else (fingerprint_3d == source_fingerprint)
    )
    report = {
        "source_equirectangular": str(source.resolve()),
        "source_channels": channels,
        "image_size": [k.width, k.height],
        "densekp_3d_fingerprint": {
            "algorithm": "blake2b-128",
            "value": fingerprint_3d,
            "matches_projection_source": fingerprint_3d_matches_source,
        },
        "projection_error_px": {
            "mean": float(projection_error.mean()),
            "max": float(projection_error.max()),
        },
        "roundtrip_angular_error_rad": {
            "mean": float(angular[good].mean()),
            "max": float(angular[good].max()),
        },
        "densekp_total": len(points),
        "finite_keypoints": int(finite.sum()),
        "valid_keypoints": int(inside.sum()),
        "visible_keypoints": int(visible.sum()),
        "pixel_bounds_finite": {
            "min": projected[finite].min(0).tolist(),
            "max": projected[finite].max(0).tolist(),
        },
        "depth_error_m": {
            "median": float(np.nanmedian(depth_error)),
            "p95": float(np.nanpercentile(depth_error, 95)),
        },
        "no_nan": bool(np.isfinite(projected).all()),
        "pass": bool(
            projection_error.max() < 1e-9
            and inside.any()
            and visible.any()
            and np.isfinite(projected).all()
        ),
    }
    return report


def warp_case(
    source,
    output,
    points,
    camera,
    k,
    source_keypoints=None,
    source_fingerprint=None,
    person_ids=None,
    point_person_ids=None,
):
    """Read source -> remap -> apply domain -> raster/labels -> preview/report.

    Raster writes precede label checks and report construction, preserving the
    existing failure/partial-output contract. This function never renders.
    """
    output.mkdir(parents=True, exist_ok=True)
    source_data = read_source_erp(source)
    mapping = build_omni_source_map(source_data, k)
    buffers = remap_omni_source(source_data, mapping)
    raster = apply_omni_domain(buffers, mapping)
    save_omni_rasters(output, raster, buffers, mapping, points, person_ids, point_person_ids)
    labels = make_omni_labels(points, camera, k, buffers, point_person_ids)
    save_omni_labels(output, raster, labels)
    report = make_omni_report(
        source, source_data, points, camera, k, labels, source_keypoints, source_fingerprint
    )
    (output / "case_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
