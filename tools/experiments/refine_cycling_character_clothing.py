#!/usr/bin/env python3
"""Clip garment boundaries within body triangles instead of selecting whole faces."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.clothing.shells import vertex_normals
from densekp_retrain.io.json import write_json
from tools.experiments.prepare_cycling_characters import MATERIALS


from densekp_retrain.clothing.shells import clipped_shell  # historical export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets", type=Path)
    a = p.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    path = a.assets / "report.json"
    report = json.loads(path.read_text())
    backup = a.assets / "report_before_clothing_v2.json"
    if backup.exists():
        raise FileExistsError("Clothing revision already applied")
    write_json(backup, report)
    records = []
    for item in report["characters"]:
        mp = Path(item["manifest"])
        manifest = json.loads(mp.read_text())
        write_json(mp.with_name("manifest_before_clothing_v2.json"), manifest)
        model = load_smplx_model(
            Path(manifest["model_dir"]), gender=manifest["gender"], num_betas=10, batch_size=1
        )
        with torch.no_grad():
            shaped = model(betas=torch.tensor(manifest["betas"], dtype=torch.float32)[None])
        pelvis = shaped.joints[0, 0].numpy()
        rest = shaped.vertices[0].numpy() - pelvis
        joints = shaped.joints[0].numpy() - pelvis
        jobs = [manifest["reference"]] + [
            d for d in report["derivatives"] if d["character_id"] == item["character_id"]
        ]
        for job in jobs:
            geom = Path(job["geometry"])
            body = np.load(geom / "vertices.npy")[0]
            faces = np.load(geom / "faces.npy")
            arrays, meta = clipped_shell(rest, joints, body, faces)
            out = geom.parent / "clothing_v2"
            out.mkdir(exist_ok=False)
            for name, array in arrays.items():
                np.save(out / f"{name}.npy", array)
            np.save(
                out / "face_material_ids.npy", np.zeros(len(arrays["clothing_faces"]), np.int32)
            )
            write_json(out / "materials.json", MATERIALS)
            write_json(out / "garment_material.json", meta)
            write_json(
                out / "clothing_manifest.json",
                dict(
                    meta,
                    body_geometry=str(geom),
                    character_id=item["character_id"],
                    split="integration_only",
                    version=2,
                    gender=manifest["gender"],
                    betas=manifest["betas"],
                    derived_from=job["clothing_geometry"],
                    method="rest-space polygon clipping with barycentric posed-body correspondence; same 4mm outward shell",
                    limitation="tight shell, no cloth dynamics or exhaustive self-intersection guarantee",
                ),
            )
            job["clothing_geometry"] = str(out)
            records.append(dict(geometry=str(geom), clothing=str(out), **meta))
        manifest["clothing"]["geometry_version"] = 2
        manifest["clothing"][
            "revision_reason"
        ] = "smooth triangle-clipped hems/collar/sleeves; continuous rest-coordinate color bands"
        write_json(mp, manifest)
    write_json(path, report)
    write_json(
        a.assets / "clothing_revision_v2.json",
        dict(
            split="integration_only",
            seconds=time.monotonic() - start,
            body_geometry_changed=False,
            status="pending_review",
            cases=records,
        ),
    )


if __name__ == "__main__":
    main()
