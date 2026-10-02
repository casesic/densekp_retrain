#!/usr/bin/env python3
"""Build fitted jersey/shorts shells for each independent cycling body.

Garment regions are selected in the shaped rest pose; the shell follows the
saved posed body with a small normal offset. This is a procedural close-fitting
outfit, not cloth simulation or recovery of the source photograph's clothes.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from densekp_retrain.clothing.shells import vertex_normals  # historical export


from densekp_retrain.clothing.shells import outfit_regions  # historical export


from densekp_retrain.clothing.shells import shell  # historical export


def review(path, body, body_faces, clothing, clothing_faces, material_ids, materials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # y-up canonical geometry -> z-up display, with equal scale in all views.
    body = body[:, [0, 2, 1]] * [1, -1, 1]
    clothing = clothing[:, [0, 2, 1]] * [1, -1, 1]
    center = (body.max(0) + body.min(0)) / 2
    radius = np.ptp(body, axis=0).max() * 0.53
    fig = plt.figure(figsize=(12, 4))
    for column, azimuth in enumerate((-90, 0, 90, 180)):
        ax = fig.add_subplot(1, 4, column + 1, projection="3d")
        all_triangles = np.concatenate((body[body_faces], clothing[clothing_faces]))
        colors = np.concatenate(
            (
                np.tile([0.65, 0.44, 0.30], (len(body_faces), 1)),
                np.array([m["color"][:3] for m in materials])[material_ids],
            )
        )
        normals = np.cross(
            all_triangles[:, 1] - all_triangles[:, 0], all_triangles[:, 2] - all_triangles[:, 0]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([1.0, -1.0, 2.0]) / np.sqrt(6)
        colors *= (0.4 + 0.6 * np.maximum(normals @ light, 0))[:, None]
        ax.add_collection3d(Poly3DCollection(all_triangles, facecolors=colors, linewidths=0))
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
            title=f"{azimuth} degrees",
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=5, azim=azimuth)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-m", type=float, default=0.004)
    args = parser.parse_args()
    import torch
    from densekp_retrain.body.geometry import load_smplx_model

    torch.set_num_threads(2)
    conversion = json.loads((args.converted / "report.json").read_text())
    model = load_smplx_model(
        Path(conversion["model_dir"]), gender="neutral", num_betas=10, batch_size=1
    )
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, case in enumerate(conversion["cases"]):
        if not case["accepted_for_synthetic_pilot"]:
            raise ValueError("Clothing requires an accepted body conversion")
        sample = case["sample_id"]
        geometry = (args.converted / sample / "geometry").resolve()
        body = np.load(geometry / "vertices.npy")
        faces = np.load(geometry / "faces.npy")
        np.testing.assert_array_equal(faces, model.faces)
        if body.shape != (1, 10475, 3):
            raise ValueError("Expected a single independent SMPL-X pose")
        with np.load(geometry / "motion_params.npz") as z:
            betas = torch.from_numpy(z["betas"][None])
        with torch.no_grad():
            rest = model(betas=betas)
        selected, material_ids = outfit_regions(
            rest.vertices[0].numpy(), rest.joints[0].numpy(), faces
        )
        clothing, clothing_faces, body_ids, separation = shell(
            body[0], faces, selected, args.offset_m
        )
        jersey = [0.65, 0.035, 0.025, 1.0] if index % 2 == 0 else [0.025, 0.18, 0.65, 1.0]
        materials = [
            dict(name="cycling_jersey", color=jersey),
            dict(name="cycling_shorts", color=[0.022, 0.03, 0.045, 1.0]),
            dict(name="jersey_band", color=[0.85, 0.88, 0.9, 1.0]),
        ]
        out = args.output / sample
        out.mkdir()
        np.save(out / "clothing_vertices.npy", clothing[None])
        np.save(out / "clothing_faces.npy", clothing_faces)
        np.save(out / "face_material_ids.npy", material_ids)
        np.save(out / "body_vertex_ids.npy", body_ids)
        np.save(out / "reference_body_vertices.npy", body[:, body_ids])
        np.save(out / "body_face_ids.npy", np.flatnonzero(selected))
        (out / "materials.json").write_text(json.dumps(materials, indent=2) + "\n")
        review(
            out / "clothing_review.jpg",
            body[0],
            faces,
            clothing,
            clothing_faces,
            material_ids,
            materials,
        )
        report = dict(
            sample_id=sample,
            body_geometry=str(geometry),
            frames=1,
            sequence=False,
            method="shaped rest-pose jersey/shorts regions; posed-body normal shell with local fold clearance",
            offset_m=args.offset_m,
            vertices=len(clothing),
            faces=len(clothing_faces),
            max_vertex_offset_m=float(np.linalg.norm(clothing - body[0, body_ids], axis=1).max()),
            minimum_face_normal_separation_m=float(separation.min()),
            body_vertices_unchanged=True,
            densekp_source="original body, not clothing surface",
            placement="apply the BODY lift/yaw/translation; never lift clothing independently",
            limitation="procedural tight outfit, no cloth dynamics or exhaustive collision test",
        )
        np.testing.assert_array_equal(body, np.load(geometry / "vertices.npy"))
        (out / "clothing_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output / "report.json").write_text(json.dumps(dict(cases=reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
