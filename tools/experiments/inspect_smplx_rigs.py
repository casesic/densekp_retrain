"""Blender: compare standard and locked-head rotation conventions, not their betas."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[sys.argv.index('--')+1])
records = {}
for version, name in [('standard', 'smplx_model_20230302.blend'),
                      ('locked_head', 'smplx_model_lh_20230302.blend')]:
    path = root/'assets_external/smplx_blender_addon/data'/name
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n == 'SMPLX-mesh-neutral']
    arm = dst.objects[0].parent
    if arm is None or arm.type != 'ARMATURE':
        raise ValueError('Expected official neutral armature')
    records[version] = {b.name: dict(parent=b.parent.name if b.parent else None,
        matrix=np.asarray(b.matrix_local).tolist()) for b in arm.data.bones}
if records['standard'].keys() != records['locked_head'].keys():
    raise ValueError('Bone names differ')
errors = []
for name, a in records['standard'].items():
    b = records['locked_head'][name]
    if a['parent'] != b['parent']:
        raise ValueError('Bone hierarchy differs')
    delta = np.abs(np.array(a['matrix'])[:3,:3]-np.array(b['matrix'])[:3,:3]).max()
    errors.append(dict(bone=name, rotation_matrix_max_abs=float(delta)))
if max(r['rotation_matrix_max_abs'] for r in errors)>1e-6:
    raise ValueError('Pose rotation axes differ')
out.write_text(json.dumps(dict(status='rotation_conventions_match', rigs=records,
    errors=errors, limitation='No shape-space equivalence claimed; source 16 betas are never copied'),indent=2)+'\n')
