"""Evaluate the original B2 neutral locked-head body, without addon registration.

The shape, bone update and corrective formulas follow the already-local official
SMPL-X Blender addon. No source beta padding, target retargeting or garment binding.
"""
import argparse
import ast
import json
from pathlib import Path
import resource
import sys
import time

import bpy
import numpy as np
from mathutils import Vector, Quaternion
from mathutils.kdtree import KDTree

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.output.resolve()
    recipe=json.loads((out/'frozen_recipe.json').read_text());began=time.monotonic()
    geometry=out/'geometry';geometry.mkdir(exist_ok=False)
    source=dict(np.load(recipe['body_source'],allow_pickle=False))
    with np.load(recipe['clothing_source'],allow_pickle=False) as z:
        cloth=z['vertices_seq'][:91];cf=z['faces'];uv=z['vertices_uv'];uf=z['faces_uv']
    assert source['poses'].shape==(127,165) and source['betas'].shape==(16,) and len(cloth)==91
    assert np.isfinite(cloth).all() and np.isfinite(uv).all() and cf.min()>=0 and cf.max()<cloth.shape[1]
    assert uf.shape==cf.shape and uf.min()>=0 and uf.max()<len(uv)
    addon=ROOT/'assets_external/smplx_blender_addon/__init__.py'
    tree=ast.parse(addon.read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='SMPLX_JOINT_NAMES' for t in n.targets))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.append(filename='SMPLX-mesh-neutral',directory=recipe['model']+'/Object/')
    obj=bpy.data.objects.get('SMPLX-mesh-neutral')
    if obj is None:raise ValueError('Expected original neutral mesh was not appended')
    arm=obj.parent
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    assert obj.type=='MESH' and arm.type=='ARMATURE' and len(obj.data.vertices)==10475
    original_props={k:obj[k] for k in obj.keys() if isinstance(obj[k],(str,int,float))}
    if obj.animation_data:obj.animation_data_clear()
    if arm.animation_data:arm.animation_data_clear()
    keys=obj.data.shape_keys.key_blocks
    if obj.data.shape_keys.animation_data:obj.data.shape_keys.animation_data_clear()
    for k in keys:
        if k.name.startswith(('Shape','Pose','Exp')):k.value=0
    for i,beta in enumerate(source['betas']):
        key=keys[f'Shape{i:03d}'];key.slider_min=min(-5,float(beta));key.slider_max=max(5,float(beta));key.value=float(beta)
    np.testing.assert_allclose([keys[f'Shape{i:03d}'].value for i in range(16)],source['betas'],atol=1e-7,rtol=0)
    # Use exactly the supplied sixteen coefficients, not an invented target beta vector.
    jr=json.loads(Path(recipe['joint_regressor']).read_text())
    joints=np.asarray(jr['betasJ_regr'])[...,:16]@source['betas']+np.asarray(jr['template_J'])
    bpy.context.view_layer.objects.active=arm
    bpy.ops.object.mode_set(mode='EDIT')
    for i,name in enumerate(names):
        bone=arm.data.edit_bones[name];bone.head=(0,0,0);bone.tail=(0,0,.1)
        j=joints[i];bone.translate(Vector((float(j[0]),float(-j[2]),float(j[1]))))
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active=obj
    for bone in arm.pose.bones:
        bone.rotation_mode='QUATERNION';bone.rotation_quaternion=(1,0,0,0);bone.location=(0,0,0);bone.scale=(1,1,1)
    # Source NPZ uses SMPL-X Y-up; the bundled rig's rest basis supplies C once.
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    vertices=[];records=[];faces=None
    for frame in range(91):
        pose=source['poses'][frame].reshape(55,3);rotations=[]
        for name,rod in zip(names,pose):
            angle=float(np.linalg.norm(rod));q=Quaternion(Vector(rod/angle),angle) if angle else Quaternion((1,0,0,0))
            arm.pose.bones[name].rotation_quaternion=q;rotations.append(np.asarray(q.to_matrix()))
        arm.pose.bones['pelvis'].location=Vector(source['trans'][frame])
        weights=(np.asarray(rotations[1:])-np.eye(3)).reshape(-1)
        for i,w in enumerate(weights):
            key=keys[f'Pose{i:03d}'];key.slider_min=-5;key.slider_max=5;key.value=float(w)
        bpy.context.view_layer.update()
        ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
        v=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',v)
        M=np.asarray(ev.matrix_world);world=v.reshape(-1,3)@M[:3,:3].T+M[:3,3]
        v=(world@C).astype(np.float32)
        f=np.array([list(p.vertices) for p in mesh.polygons],np.int32)
        ev.to_mesh_clear()
        assert v.shape==(10475,3) and f.shape==(20908,3) and np.isfinite(v).all()
        if faces is None:faces=f
        else:np.testing.assert_array_equal(faces,f)
        record=dict(asset_frame=frame,body_frame=frame,cloth_frame=frame,simulation_frame=101+frame,
                    body_min=v.min(0).tolist(),body_max=v.max(0).tolist(),
                    cloth_min=cloth[frame].min(0).tolist(),cloth_max=cloth[frame].max(0).tolist())
        if frame in recipe['probe_frames']:
            kd=KDTree(len(v))
            for i,x in enumerate(v):kd.insert(Vector(x),i)
            kd.balance();dist=np.array([kd.find(Vector(x))[2] for x in cloth[frame]])
            record['cloth_to_body_vertex_distance_m']=dict(median=float(np.median(dist)),p95=float(np.quantile(dist,.95)),max=float(dist.max()),diagnostic_only_not_collision_test=True)
        vertices.append(v);records.append(record)
    vertices=np.array(vertices)
    R=np.load(ROOT/'assets/densekp_regressor_R.npy');assert R.shape==(138,10475)
    points=np.einsum('kv,tvc->tkc',R,vertices,optimize=True).astype(np.float32)
    assert np.isfinite(points).all()
    for name,value in [('vertices',vertices),('faces',faces),('densekp_3d_pending',points),
                       ('clothing_vertices',cloth.astype(np.float32)),('clothing_faces',cf),
                       ('clothing_vertices_uv',uv),('clothing_faces_uv',uf)]:
        np.save(geometry/(name+'.npy'),value)
    np.testing.assert_allclose(np.load(geometry/'densekp_3d_pending.npy'),
        np.einsum('kv,tvc->tkc',R,np.load(geometry/'vertices.npy'),optimize=True),atol=2e-6,rtol=0)
    standard_faces=np.load(ROOT/'outputs/baselines/p0_dressed_dataset/geometry/faces.npy')
    result=dict(status='native_geometry_evaluated_visual_review_pending',frames=91,
        source_betas=source['betas'].tolist(),source_betas_unchanged=True,original_model_properties=original_props,
        implementation_reference=str(addon),shape_keys_driven=16,unobserved_shape_expression_keys='remain model defaults zero',
        garment_rebinding=False,garment_interpolation=False,source_root_preserved=True,
        same_index_body_cloth=True,source_npz_preserved=True,root_applied_once=True,
        coordinate_matrix_SMPLX_Yup_to_Blender_Zup=C.tolist(),per_frame_grounding=False,
        faces_equal_to_existing_R_reference=bool(np.array_equal(faces,standard_faces)),
        densekp_label_ready='pending_locked_head_template_semantics_not_certified',
        R_numeric_readback=True,per_frame=records,
        seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        blender_version=bpy.app.version_string)
    (out/'geometry_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE_GEOMETRY_DONE',result['seconds'],flush=True)


if __name__=='__main__':main()
