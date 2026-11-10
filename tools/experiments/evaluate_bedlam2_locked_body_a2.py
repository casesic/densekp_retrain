"""Evaluate one published BEDLAM2 locked-head body sequence for garment transfer.

The local official Blender model/rig and 16-beta joint regressor are kept intact.
This is a source-body correspondence cache, not the target M03 authority.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--joint-regressor', type=Path, required=True)
    ap.add_argument('--addon', type=Path, required=True)
    ap.add_argument('--first', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with np.load(args.source, allow_pickle=False) as src:
        poses = np.asarray(src['poses'])
        trans = np.asarray(src['trans'])
        betas = np.asarray(src['betas'])
    if not (poses.shape[1] == 165 and trans.shape == (len(poses), 3) and
            betas.shape == (16,) and 0 <= args.first < args.end <= len(poses)):
        raise ValueError('source_locked_head_contract_mismatch')
    tree = ast.parse(args.addon.read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'SMPLX_JOINT_NAMES' for t in n.targets))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.append(filename='SMPLX-mesh-neutral', directory=str(args.model) + '/Object/')
    obj = bpy.data.objects.get('SMPLX-mesh-neutral')
    if obj is None or len(obj.data.vertices) != 10475:
        raise ValueError('expected_locked_head_neutral_model_missing')
    arm = obj.parent
    if obj.animation_data: obj.animation_data_clear()
    if arm.animation_data: arm.animation_data_clear()
    keys = obj.data.shape_keys.key_blocks
    if obj.data.shape_keys.animation_data: obj.data.shape_keys.animation_data_clear()
    for key in keys:
        if key.name.startswith(('Shape', 'Pose', 'Exp')): key.value = 0
    for i, value in enumerate(betas):
        key = keys[f'Shape{i:03d}']
        key.slider_min = min(-5, float(value))
        key.slider_max = max(5, float(value))
        key.value = float(value)
    jr = json.loads(args.joint_regressor.read_text())
    joints = np.asarray(jr['betasJ_regr'])[..., :16] @ betas + np.asarray(jr['template_J'])
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for i, name in enumerate(names):
        bone = arm.data.edit_bones[name]
        bone.head = (0, 0, 0)
        bone.tail = (0, 0, .1)
        j = joints[i]
        bone.translate(Vector((float(j[0]), float(-j[2]), float(j[1]))))
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj
    for bone in arm.pose.bones:
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)
    conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    output = []
    faces = None
    for source_frame in range(args.first, args.end):
        rotations = []
        for name, rod in zip(names, poses[source_frame].reshape(55, 3)):
            angle = float(np.linalg.norm(rod))
            q = Quaternion(Vector(rod / angle), angle) if angle else Quaternion((1, 0, 0, 0))
            arm.pose.bones[name].rotation_quaternion = q
            rotations.append(np.asarray(q.to_matrix()))
        arm.pose.bones['pelvis'].location = Vector(trans[source_frame])
        weights = (np.asarray(rotations[1:]) - np.eye(3)).reshape(-1)
        for i, value in enumerate(weights):
            key = keys[f'Pose{i:03d}']
            key.slider_min = min(-5, float(value))
            key.slider_max = max(5, float(value))
            key.value = float(value)
        bpy.context.view_layer.update()
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = ev.to_mesh()
        coords = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get('co', coords)
        matrix = np.asarray(ev.matrix_world)
        world = coords.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        vertices = (world @ conv).astype(np.float32)
        current_faces = np.array([list(p.vertices) for p in mesh.polygons], np.int32)
        ev.to_mesh_clear()
        if faces is None: faces = current_faces
        elif not np.array_equal(faces, current_faces): raise ValueError('body_topology_changed')
        if not np.isfinite(vertices).all(): raise ValueError('nonfinite_body')
        output.append(vertices)
        if source_frame % 20 == 0: print('LOCKED_BODY_FRAME', source_frame, flush=True)
    np.save(out / 'vertices.npy', np.stack(output))
    np.save(out / 'faces.npy', faces)
    (out / 'manifest.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()), model=str(args.model.resolve()),
        first=args.first, end=args.end, frame_count=args.end - args.first,
        betas_source_16=betas.tolist(), source_trans_applied_once=True,
        output_axes='SMPL-X Y-up metres', role='source_body_for_garment_transfer_only',
        target_M03_unchanged=True), indent=2) + '\n')


if __name__ == '__main__': main()
