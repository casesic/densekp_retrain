"""Bounded native body/cache studio preview; no garment regeneration."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix, Vector


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True)
    ap.add_argument('--stage',choices=['probe','full'],required=True);ap.add_argument('--gpu-uuid',required=True)
    ap.add_argument('--appearance-overlay',type=Path)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);asset=a.asset.resolve();began=time.monotonic()
    recipe=json.loads((asset/'frozen_recipe.json').read_text());g=asset/'geometry'
    overlay=None
    if a.appearance_overlay:
        asset=a.appearance_overlay.resolve()
        overlay=json.loads((asset/'appearance.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu_uuid
    v=np.load(g/'vertices.npy');cv=np.load(g/'clothing_vertices.npy')
    f=np.load(g/'faces.npy');cf=np.load(g/'clothing_faces.npy')
    uv=np.load(g/'clothing_vertices_uv.npy');uf=np.load(g/'clothing_faces_uv.npy')
    frames=recipe['probe_frames'] if a.stage=='probe' else list(range(91))
    out=asset/'renders';out.mkdir(exist_ok=True)
    if a.stage=='full':
        decision=json.loads((asset/'probe_decision.json').read_text());assert decision['proceed_full']
        prior=decision['probe_render_seconds']
    else:prior=0
    bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;s.render.engine='CYCLES'
    s.cycles.samples=64;s.cycles.use_denoising=False;s.cycles.seed=29;s.cycles.use_animated_seed=False;s.cycles.use_adaptive_sampling=False
    s.cycles.max_bounces=4;s.cycles.diffuse_bounces=1;s.cycles.glossy_bounces=2;s.cycles.transmission_bounces=2
    s.render.threads_mode='FIXED';s.render.threads=4;s.render.use_persistent_data=True
    pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='CUDA';pref.get_devices()
    for d in pref.devices:d.use=d.type=='CUDA'
    devices=[dict(name=d.name,id=d.id,type=d.type) for d in pref.devices if d.use]
    assert len(devices)==1;s.cycles.device='GPU'
    s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100
    s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.view_settings.view_transform='AgX'
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    world_v=v@C.T;world_cv=cv@C.T
    low=np.minimum(world_v.min((0,1)),world_cv.min((0,1)))
    high=np.maximum(world_v.max((0,1)),world_cv.max((0,1)))
    center=(low+high)/2;scale=float(max(high-low)*1.18)
    world=bpy.data.worlds.new('Studio');world.use_nodes=True;s.world=world
    world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.18,.18,.18,1)
    world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.5
    for name,relative,energy,size in [('Key',(-3,-4,4),500,4),('Fill',(3,-2,2),250,3),('Rim',(1,3,3),350,3)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.shape='DISK';light.size=size
        ob=bpy.data.objects.new(name,light);s.collection.objects.link(ob);ob.location=Vector(center)+Vector(relative)
        ob.rotation_euler=(Vector(center)-ob.location).to_track_quat('-Z','Y').to_euler()
    def mesh(name,verts,faces):
        data=bpy.data.meshes.new(name);data.from_pydata(verts.tolist(),[],faces.tolist())
        ob=bpy.data.objects.new(name,data);s.collection.objects.link(ob)
        for p in data.polygons:p.use_smooth=True
        return ob
    body=mesh('OriginalNeutralLockedHead',world_v[0],f)
    cloth=mesh('OriginalBEDLAMCache',world_cv[0],cf)
    mat=bpy.data.materials.new('NeutralMatteBody_NotOriginalSkinTexture');mat.use_nodes=True
    bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.45,.30,.23,1);bs.inputs['Roughness'].default_value=.58
    body.data.materials.append(mat)
    hair=[];head=None;hair_counts=[]
    if overlay:
        root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root),str(root/'src')]
        from tools.render.bedlam_walk_appearance import add_native_eyes
        from densekp_retrain.rendering.blender.groom import add_groom
        add_native_eyes(body.data,f,overlay)
        head=np.load(overlay['head_transforms']);display=np.eye(4);display[:3,:3]=C
        hair=add_groom(overlay['groom'],Matrix((display@head[0]).tolist()))
        hair_counts=[(len(o.data.curves),len(o.data.points)) for o in hair]
    layer=cloth.data.uv_layers.new(name='OriginalBEDLAM_UV');layer.data.foreach_set('uv',uv[uf].reshape(-1))
    mat=bpy.data.materials.new('OriginalBEDLAMTexture01');mat.use_nodes=True
    nodes=mat.node_tree.nodes;links=mat.node_tree.links;bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.58;bs.inputs['Metallic'].default_value=0
    di=nodes.new('ShaderNodeTexImage');di.image=bpy.data.images.load(recipe['diffuse']);di.image.colorspace_settings.name='sRGB';links.new(di.outputs['Color'],bs.inputs['Base Color'])
    ni=nodes.new('ShaderNodeTexImage');ni.image=bpy.data.images.load(recipe['normal']);ni.image.colorspace_settings.name='Non-Color'
    nm=nodes.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=1;links.new(ni.outputs['Color'],nm.inputs['Color']);links.new(nm.outputs['Normal'],bs.inputs['Normal']);cloth.data.materials.append(mat)
    cam=bpy.data.objects.new('FixedReviewCamera',bpy.data.cameras.new('FixedReviewCamera'));s.collection.objects.link(cam);s.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=scale
    matrices={}
    for label,offset in [('front',(0,-5,.1)),('side',(5,0,.1))]:
        cam.location=Vector(center)+Vector(offset);cam.rotation_euler=(Vector(center)-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
        matrices[label]=np.array(cam.matrix_world).tolist()
    settings=dict(samples=64,denoising=False,seed=29,animated_seed=False,adaptive=False,resolution_each=[512,512],
        view_transform='AgX',camera_center=center.tolist(),ortho_scale=scale,camera_matrices=matrices,
        model_to_display=C.tolist(),native_root_preserved=True,frame_tracking=False,per_frame_grounding=False,
        body_material='neutral matte preview; no official skin identity claimed',garment_uv_flip_v=False,
        normal_strength=1,roughness=.58,groom=None,shoes=None,gpu_uuid=a.gpu_uuid,devices=devices)
    if overlay:
        settings.update(groom=overlay['groom'],eyes=overlay['eye_mapping'],head_transform='display @ frame0_to_current_head; native root already included',hair_counts=[list(x) for x in hair_counts])
    settings_path=asset/'render_settings.json'
    if settings_path.exists():assert json.loads(settings_path.read_text())==settings
    else:settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    setup=time.monotonic()-began;records=[];reused=[]
    for i in frames:
        body.data.vertices.foreach_set('co',world_v[i].astype(np.float32).reshape(-1));body.data.update()
        cloth.data.vertices.foreach_set('co',world_cv[i].astype(np.float32).reshape(-1));cloth.data.update()
        if overlay:
            for obj,count in zip(hair,hair_counts):
                obj.matrix_world=Matrix((display@head[i]).tolist())
                assert (len(obj.data.curves),len(obj.data.points))==count
            for obj,expected in [(body,world_v[i]),(cloth,world_cv[i])]:
                check=np.empty(expected.size,np.float32);obj.data.vertices.foreach_get('co',check)
                np.testing.assert_array_equal(check.reshape(-1,3),expected.astype(np.float32))
        for label in matrices:
            path=out/f'frame_{i:03d}_{label}.png'
            if path.exists():
                assert a.stage=='full' and i in recipe['probe_frames'];reused.append(str(path));continue
            if time.monotonic()-began+prior>1800:raise RuntimeError('30 minute render budget exhausted')
            ts=time.monotonic();cam.matrix_world=Matrix(matrices[label]);s.render.filepath=str(path)
            bpy.ops.render.render(write_still=True)
            records.append(dict(asset_frame=i,source_frame=i,cloth_simulation_frame=101+i,time_tick=i,time_base=[1,30],view=label,seconds=time.monotonic()-ts))
            with (asset/(a.stage+'_timings.jsonl')).open('a') as handle:handle.write(json.dumps(records[-1])+'\n')
        if overlay and a.stage=='probe' and i==45:
            # One labelled static detail for eye/material inspection, not a dynamic acceptance substitute.
            labels=np.load(overlay['eye_mapping'])['component_labels'];eyes=world_v[i][labels>0].mean(0)
            cam.location=Vector(eyes)+Vector((0,-1.5,.03))
            cam.rotation_euler=(Vector(eyes)-cam.location).to_track_quat('-Z','Y').to_euler()
            cam.data.ortho_scale=.42;s.render.filepath=str(asset/'head_detail_frame045.png')
            ts=time.monotonic();bpy.ops.render.render(write_still=True)
            records.append(dict(asset_frame=i,view='static_head_detail',seconds=time.monotonic()-ts))
            cam.data.ortho_scale=scale
        print('NATIVE_RENDER_FRAME_DONE',i,flush=True)
    report=dict(stage=a.stage,records=records,reused_images=reused,setup_seconds=setup,elapsed_seconds=time.monotonic()-began,
        render_seconds=sum(r['seconds'] for r in records),gpu_uuid=a.gpu_uuid,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        status='rendered_user_dynamic_review_pending',integration_only=True)
    (asset/(a.stage+'_render_manifest.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_RENDER_DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
