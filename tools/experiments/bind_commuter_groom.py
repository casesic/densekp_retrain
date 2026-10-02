"""Transfer official LH groom roots onto verified standard SMPL-X scalp semantics."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.clothing.binding import closest_surface
from densekp_retrain.clothing.shells import vertex_normals
from densekp_retrain.io.json import write_json

from densekp_retrain.clothing.binding import CONV  # historical export


from densekp_retrain.clothing.binding import frame  # historical export


from densekp_retrain.clothing.binding import triangle_frame  # historical export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    root = args.assets.resolve()
    start = time.monotonic()
    z = np.load(root / "groom_reference/reference.npz")
    reference = z["vertices"]
    faces = z["faces"]
    neutral = np.load(
        "/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/SMPLX_NEUTRAL.npz"
    )
    np.testing.assert_array_equal(faces, neutral["f"])
    head = z["weights"][:, list(z["group_names"]).index("head")] > 0.1
    body_error = float(np.abs(reference[~head] - neutral["v_template"][~head] @ CONV.T).max())
    if body_error > 2e-6:
        raise ValueError("Unverified reference vertex semantics")
    scalp = head & (
        (reference[:, 2] > 0.32)
        | (reference[:, 1] > -0.005)
        | ((np.abs(reference[:, 0]) > 0.055) & (reference[:, 2] > 0.23))
    )
    scalp_faces = faces[scalp[faces].all(1)]
    ref_normals = vertex_normals(reference, faces)
    audit = json.loads((root / "groom_import_audit_v2/report.json").read_text())
    report = json.loads((root / "report.json").read_text())
    records = []
    for character in report["characters"]:
        mp = Path(character["manifest"])
        manifest = json.loads(mp.read_text())
        name = (
            "vinefx_groom_18-1_short-bob-F"
            if manifest["gender"] == "female"
            else "vinefx_groom_01-1_curly-M"
        )
        source = next(c for c in audit["cases"] if Path(c["source"]).stem == name)
        data = np.load(source["export"])
        points = data["points"] * 0.01
        offsets = data["offsets"]
        roots = points[offsets[:-1]]
        tree = cKDTree(reference[scalp_faces].mean(1))
        nearest = []
        tri = []
        for i in range(0, len(roots), 8192):
            q, _, _, ids = closest_surface(roots[i : i + 8192], reference, scalp_faces, tree)
            nearest.append(q)
            tri.append(scalp_faces[ids])
        nearest = np.concatenate(nearest)
        tri = np.concatenate(tri)
        weights = trimesh.triangles.points_to_barycentric(reference[tri], nearest).astype(
            np.float32
        )
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum(1, keepdims=True)
        ref_normal = (ref_normals[tri] * weights[:, :, None]).sum(1)
        axis = np.tile([1.0, 0, 0], (len(roots), 1))
        axis[np.abs(ref_normal[:, 0]) > 0.9] = [0, 1, 0]
        sf = frame(ref_normal, axis)
        rest = np.load(Path(manifest["reference"]["geometry"]) / "vertices.npy")[0] @ CONV.T
        rest_normal = (vertex_normals(rest, faces)[tri] * weights[:, :, None]).sum(1)
        tf = frame(rest_normal, axis)
        rest_tri_frame = triangle_frame(rest, tri)
        point_curve = np.repeat(np.arange(len(roots)), np.diff(offsets))
        # Preserve curve shape relative to its first point, then move that root to scalp.
        # A common scalp affine preserves the coherent groom silhouette. Independent
        # per-root rest rotations magnified small LH/standard scalp-normal changes.
        affine = np.linalg.lstsq(
            np.c_[reference[scalp], np.ones(scalp.sum())], rest[scalp], rcond=None
        )[0]
        rest_relative = (points - roots[point_curve]) @ affine[:3]
        local = np.einsum("nji,nj->ni", tf[point_curve], rest_relative)
        folder = mp.parent / f"groom_binding_{args.version}"
        folder.mkdir()
        np.savez(
            folder / "binding.npz",
            body_faces=tri,
            weights=weights,
            curve_offsets=offsets,
            local_points=local.astype(np.float32),
            target_rest_frames=tf.astype(np.float32),
            source_root_distance=np.linalg.norm(roots - nearest, axis=1).astype(np.float32),
        )
        source_info = dict(
            source,
            units="source centimetres -> reference metres: 0.01",
            source_reference=str(root / "groom_reference/reference.npz"),
            nonhead_reference_vertex_max_m=body_error,
            root_end="first point",
            source_root_distance_quantiles_m=np.quantile(
                np.linalg.norm(roots - nearest, axis=1), [0.5, 0.95, 0.99, 1]
            ).tolist(),
            source_roots_over_3mm=int((np.linalg.norm(roots - nearest, axis=1) > 0.003).sum()),
            source_to_target="verified shared face semantics; exact root barycentrics, common rest scalp affine, posed triangle frame rotations",
            rest_scalp_affine=affine.tolist(),
            scalp_mask_definition="head weight >0.1 plus crown/back/temple anatomical region; excludes central lower face",
            root_clearance_m=0.0004,
            body_edited=False,
            split="integration_only",
        )
        write_json(folder / "binding_manifest.json", source_info)
        jobs = [manifest["reference"]] + [
            d for d in report["derivatives"] if d["character_id"] == character["character_id"]
        ]
        for job in jobs:
            geometry = Path(job["geometry"])
            body = np.load(geometry / "vertices.npy")[0] @ CONV.T
            anchor = (body[tri] * weights[:, :, None]).sum(1)
            rotation = np.einsum("nij,nkj->nik", triangle_frame(body, tri), rest_tri_frame)
            target_frame = np.einsum("nij,njk->nik", rotation, tf)
            target_root = anchor + target_frame[:, :, 2] * 0.0004
            transferred = target_root[point_curve] + np.einsum(
                "nij,nj->ni", target_frame[point_curve], local
            )
            out = geometry.parent / f"groom_{args.version}"
            out.mkdir()
            np.save(out / "points.npy", (transferred @ CONV).astype(np.float32))
            np.save(out / "radius.npy", (data["imported_radius"] * 0.01).astype(np.float32))
            np.save(out / "curve_offsets.npy", offsets)
            np.save(out / "target_roots.npy", (target_root @ CONV).astype(np.float32))
            info = dict(
                character_id=character["character_id"],
                body_geometry=str(geometry),
                binding=str(folder),
                source_abc=source["source"],
                source_objects=source["objects"],
                points=len(transferred),
                curves=len(roots),
                status="bound_candidate_collision_and_visual_pending",
                split="integration_only",
                units="metres; standard SMPL-X body coordinates",
                radius_scale_from_abc=0.01,
                root_clearance_max_error_m=float(
                    np.abs(np.linalg.norm(target_root - anchor, axis=1) - 0.0004).max()
                ),
            )
            write_json(out / "groom_manifest.json", info)
            job["groom_geometry"] = str(out)
            records.append(info)
        manifest["hair_status"] = "bound_candidate_collision_and_visual_pending"
        manifest["groom_binding"] = str(folder)
        write_json(mp, manifest)
    report["status"] = "hair_bound_candidates_clothing_needs_correction"
    write_json(root / "report.json", report)
    write_json(
        root / f"groom_binding_{args.version}.json",
        dict(seconds=time.monotonic() - start, cases=records, split="integration_only"),
    )


if __name__ == "__main__":
    main()
