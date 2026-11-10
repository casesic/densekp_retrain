"""Independent bounded studio consumer of the A2 BEDLAM garment candidate."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.rendering.blender.materials import apply_character_materials
from densekp_retrain.rendering.blender.groom import add_groom
BASE=ROOT/'outputs/experiments/continuous_cycling_asset_a2_20261009'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--stage',choices=['probe','full'],required=True);ap.add_argument('--gpu-uuid',required=True)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);began=time.monotonic();candidate=BASE/a.candidate
    asset=json.loads((BASE/'accepted/manifest.json').read_text());m=json.loads(Path(asset['character_manifest']).read_text());cm=json.loads((candidate/'manifest.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu_uuid
    if a.stage=='full':assert json.loads((candidate/'probe_decision.json').read_text())['proceed_full']
    out=candidate/(a.stage+'_renders');out.mkdir(exist_ok=False)
    ids=np.load(candidate/(a.stage+'_asset_frames.npy'));cv=np.load(candidate/(a.stage+'_clothing_vertices.npy'));v=np.load(Path(asset['geometry'])/'vertices.npy',mmap_mode='r');f=np.load(Path(asset['geometry'])/'faces.npy');H=np.load(Path(asset['geometry'])/'head_transforms.npy');cf=np.load(candidate/'clothing_faces.npy')
    bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=64;s.cycles.use_denoising=False;s.cycles.seed=29;s.cycles.use_animated_seed=False;s.cycles.use_adaptive_sampling=False
    s.cycles.max_bounces=4;s.cycles.diffuse_bounces=1;s.cycles.glossy_bounces=2;s.cycles.transmission_bounces=2
    s.render.threads_mode='FIXED';s.render.threads=4;s.render.use_persistent_data=True
    pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='CUDA';pref.get_devices()
    for d in pref.devices:d.use=d.type=='CUDA'
    devices=[dict(name=d.name,id=d.id,type=d.type) for d in pref.devices if d.use];assert len(devices)==1;s.cycles.device='GPU'
    s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Studio');world.use_nodes=True;s.world=world;world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.18,.18,.18,1);world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.5
    for name,loc,energy,size in [('Key',(-3,-4,4),500,4),('Fill',(3,-2,2),250,3),('Rim',(1,3,3),350,3)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.shape='DISK';light.size=size;ob=bpy.data.objects.new(name,light);s.collection.objects.link(ob);ob.location=loc;ob.rotation_euler=(-ob.location).to_track_quat('-Z','Y').to_euler()
    cam=bpy.data.objects.new('FixedReviewCamera',bpy.data.cameras.new('FixedReviewCamera'));s.collection.objects.link(cam);s.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=2.3
    C=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
    def mesh(name,vertices,faces):
        data=bpy.data.meshes.new(name);data.from_pydata(vertices.tolist(),[],faces.tolist());obj=bpy.data.objects.new(name,data);s.collection.objects.link(obj);obj.matrix_world=C
        for p in data.polygons:p.use_smooth=True
        return obj
    body=mesh('FrozenBody',v[ids[0]],f);apply_character_materials(body.data,f,m['appearance']);cloth=mesh('BEDLAMDerivedClothes',cv[0],cf)
    uv=np.load(cm['uv']['vertices']);uf=np.load(cm['uv']['faces']);layer=cloth.data.uv_layers.new(name='BEDLAM_UV');layer.data.foreach_set('uv',uv[uf].reshape(-1))
    material=bpy.data.materials.new('BEDLAMOriginalTexture01');material.use_nodes=True;nodes=material.node_tree.nodes;links=material.node_tree.links;bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.58;bs.inputs['Metallic'].default_value=0
    di=nodes.new('ShaderNodeTexImage');di.image=bpy.data.images.load(cm['material']['textures']['diffuse']);di.image.colorspace_settings.name='sRGB';links.new(di.outputs['Color'],bs.inputs['Base Color'])
    ni=nodes.new('ShaderNodeTexImage');ni.image=bpy.data.images.load(cm['material']['textures']['normal']);ni.image.colorspace_settings.name='Non-Color';nm=nodes.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=1;links.new(ni.outputs['Color'],nm.inputs['Color']);links.new(nm.outputs['Normal'],bs.inputs['Normal']);cloth.data.materials.append(material)
    hair=add_groom(asset['groom_reference'],C);counts=[(len(o.data.curves),len(o.data.points)) for o in hair];assert sum(x[0] for x in counts)==63652
    settings=dict(samples=64,denoising=False,seed=29,animated_seed=False,adaptive=False,views=[0,90],resolution_each=[512,512],AgX=True,display_matrix=np.array(C).tolist(),camera_target=[0,0,-.1],ortho_scale=2.3,material=cm['material'],body_parameters=asset['parameters'],groom_reference=asset['groom_reference'],head_transform=str(Path(asset['geometry'])/'head_transforms.npy'),gpu_uuid=a.gpu_uuid,devices=devices)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');setup=time.monotonic()-began;records=[]
    for k,i in enumerate(ids):
        if time.monotonic()-began>1800:raise RuntimeError('1800-second budget exhausted; preserve partial candidate')
        body.data.vertices.foreach_set('co',v[i].ravel());body.data.update();cloth.data.vertices.foreach_set('co',cv[k].ravel());cloth.data.update()
        for obj,count in zip(hair,counts):obj.matrix_world=C@Matrix(H[i].tolist());assert (len(obj.data.curves),len(obj.data.points))==count
        rb=np.empty(v[i].size,np.float32);body.data.vertices.foreach_get('co',rb);np.testing.assert_array_equal(rb.reshape(-1,3),v[i])
        for label,angle in [('front',0),('side',90)]:
            ts=time.monotonic();target=Vector((0,0,-.1));theta=math.radians(angle);cam.location=target+Vector((4*math.sin(theta),-4*math.cos(theta),.12));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
            s.render.filepath=str(out/f'frame_{i:03d}_{label}.png');bpy.ops.render.render(write_still=True);records.append(dict(asset_frame=int(i),source_frame=int(i)+30,view=label,seconds=time.monotonic()-ts,body_readback_exact=True,camera_from_model=np.array(cam.matrix_world.inverted()@C).tolist()))
            with (out/'timings.jsonl').open('a') as stream:stream.write(json.dumps(records[-1])+'\n')
        print('CANDIDATE_FRAME_DONE',int(i),flush=True)
    report=dict(status='rendered_visual_review_pending',integration_only=True,stage=a.stage,settings=settings,frames=ids.tolist(),records=records,setup_seconds=setup,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,body_unchanged=True,groom_unchanged=True,fixed_garment_topology=True,no_floor=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
