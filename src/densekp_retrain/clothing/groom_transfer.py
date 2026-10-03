"""Transfer a frozen groom binding onto existing posed body geometry.

Positions are meters in SMPL-X model axes unless transformed by CONV into
binding axes. This preserves the existing head shape, active curves and rest
collision corrections; it neither fits a groom nor prunes intersecting hair.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from densekp_retrain.clothing.binding import CONV, triangle_frame, closest_surface
from densekp_retrain.io.json import write_json


def transfer_groom(manifest, geometry, out):
    """Write the transferred arrays and diagnostic manifest; return str(out).

    geometry/out are Paths. body_faces[C,3] holds body vertex IDs per curve;
    weights[C,3] are barycentric weights and offsets[C+1] delimit local_points.
    Frame matrices store basis vectors in columns. Keep the binding's dtype.

    Preserve read/write and failure order: an empty active sample set fails
    after six arrays are saved. No cleanup, fallback or new validation is added.
    """
    source = Path(manifest['reference']['groom_geometry'])
    info = json.loads((source / 'groom_manifest.json').read_text())
    binding = np.load(Path(info['binding']) / 'binding.npz')
    tri, weights = binding['body_faces'], binding['weights']
    offsets = binding['curve_offsets']
    pc = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    rest = np.load(Path(manifest['reference']['geometry']) / 'vertices.npy')[0] @ CONV.T
    body = np.load(geometry / 'vertices.npy')[0]
    posed = body @ CONV.T
    # Per-curve rotation: posed frame times reference frame transpose.
    rotation = np.einsum(
        'nij,nkj->nik', triangle_frame(posed, tri), triangle_frame(rest, tri)
    )
    frames = np.einsum('nij,njk->nik', rotation, binding['target_rest_frames'])
    roots = (posed[tri] * weights[:, :, None]).sum(1) + frames[:, :, 2] * .0004
    base = (
        roots[pc] + np.einsum('nij,nj->ni', frames[pc], binding['local_points'])
    ) @ CONV
    correction = np.load(source / 'collision_correction.npy')
    correction = np.einsum('nij,nj->ni', rotation[pc], correction @ CONV.T) @ CONV
    points = (base + correction).astype(np.float32)
    out.mkdir(parents=True)
    for name, value in dict(
        points=points,
        target_roots=roots @ CONV,
        collision_correction=correction,
        radius=np.load(source / 'radius.npy'),
        curve_offsets=offsets,
        active_curve_mask=np.load(source / 'active_curve_mask.npy'),
    ).items():
        np.save(out / f'{name}.npy', value)
    faces = np.load(geometry / 'faces.npy')
    tree = cKDTree(body[faces].mean(1))
    active = np.load(source / 'active_curve_mask.npy')
    sample = np.arange(0, len(points), 50)
    sample = sample[active[pc[sample]]]
    signed = np.concatenate([
        closest_surface(points[sample[i:i + 8192]], body, faces, tree)[2]
        for i in range(0, len(sample), 8192)
    ])
    info.update(
        body_geometry=str(geometry),
        parent_groom=str(source),
        transfer='fixed reference binding and frozen rest corrections rotated by scalp triangle; active curves unchanged',
        sample_min_signed_m=float(signed.min()),
        sample_below_minus_1mm=int((signed < -.001).sum()),
        sample_count=len(sample),
        status='diagnostic_pending',
        split='integration_only',
    )
    write_json(out / 'groom_manifest.json', info)
    return str(out)
