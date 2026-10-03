"""Save the shared simple-clothing format from existing body arrays.

Body and rest vertices, and anatomical joints, use SMPL-X model-axis meters.
Role-specific style tables belong to their preparation recipes.
"""

import numpy as np

from densekp_retrain.clothing.shells import clipped_shell
from densekp_retrain.io.json import write_json


def prepare_simple_clothing(
    destination, geometry, rest, joints, style_id, character_id, *, style_table
):
    """Write a clipped shell and its materials; return ``str(destination)``.

    ``geometry`` and ``destination`` are Paths. Read the first body frame [V,3]
    and faces [F,3] before looking up the style. ``rest`` is [V,3]; ``joints``
    contains anatomical joints, not DenseKP semantic points. Each style entry
    is (sleeve, leg, style, top_rgb, pants_rgb, panel_rgb).

    Preserve the historical creation/write order, including partial artifacts
    on failure. The destination must not already exist. Material IDs remain
    all-zero int32 even though the materials file lists three palette entries.
    """
    body = np.load(geometry / "vertices.npy")[0]
    faces = np.load(geometry / "faces.npy")
    sleeve, leg, style, top, pants, band = style_table[style_id]
    profile = dict(sleeve=sleeve, leg=leg, collar=.045, style=style)
    arrays, meta = clipped_shell(rest, joints, body, faces, profile)
    destination.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    np.save(
        destination / "face_material_ids.npy",
        np.zeros(len(arrays["clothing_faces"]), np.int32),
    )
    materials = [
        dict(name=n, color=c + [1])
        for n, c in [("top", top), ("pants", pants), ("panel", band)]
    ]
    write_json(destination / "materials.json", materials)
    meta.update(
        profile=profile,
        fabric="plain",
        fabric_note="continuous palette/panels; no rib shader claimed",
    )
    write_json(destination / "garment_material.json", meta)
    write_json(
        destination / "clothing_manifest.json",
        dict(
            meta,
            body_geometry=str(geometry),
            character_id=character_id,
            split="integration_only",
            method="existing clipped 4mm fitted shell with anatomical sleeve/leg lengths",
            limitation="fitted simple clothes; not straight trousers, polo or physically simulated cloth",
            appearance_status="diagnostic_pending",
            legacy_clearance_gate_m=.0004,
            legacy_offset_gate_m=.01,
        ),
    )
    return str(destination)
