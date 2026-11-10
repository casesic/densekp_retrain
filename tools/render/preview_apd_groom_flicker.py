"""Controlled8/64 sample comparison; image formation inherited from dressed pilot."""
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
from mathutils import Matrix, Vector

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.rendering.blender.materials import apply_character_materials, add_clothing_materials
from densekp_retrain.rendering.blender.groom import add_groom
BASE=ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008'
OUT=ROOT/'outputs/experiments/apd_groom_flicker_20261009'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['calibration','short','full'],required=True)
    ap.add_argument('--render-folder',default='renders')
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);began=time.monotonic()
    recipe=json.loads((BASE/'frozen_recipe.json').read_text()); budget=dict(recipe['budget'])
    frozen=json.loads((OUT/'frozen_comparison.json').read_text())
    budget['samples']=8 if a.stage=='calibration' else 64
    if a.stage=='full':
        decision=json.loads((OUT/'short_decision.json').read_text())
        assert decision['proceed_full'] and decision['estimated_full_stage_seconds']<=1800
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==budget['gpu_uuid'], 'Explicit reserved UUID required'
    out=OUT/a.stage/a.render_folder;out.mkdir(parents=True,exist_ok=False)
    m=json.loads(Path(recipe['character_manifest']).read_text());bodyfile=Path(recipe['body_cache'])
    vertices=np.load(bodyfile,mmap_mode='r');faces=np.load(bodyfile.parent.parent/'faces.npy')
    cv=np.load(BASE/'full/clothing_vertices.npy');H=np.load(BASE/'full/head_transforms.npy');ids=np.load(BASE/'full/asset_frames.npy')
    cloth=Path(recipe['source_clothing']);cf=np.load(cloth/'clothing_faces.npy')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=budget['samples'];s.cycles.use_denoising=False
    s.cycles.seed=29;s.cycles.use_animated_seed=False;s.cycles.use_adaptive_sampling=False
    s.cycles.max_bounces=4;s.cycles.diffuse_bounces=1;s.cycles.glossy_bounces=2;s.cycles.transmission_bounces=2
    s.render.threads_mode='FIXED';s.render.threads=4;s.render.use_persistent_data=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='CUDA'
    selected=[dict(name=d.name,id=d.id,type=d.type) for d in prefs.devices if d.use]
    assert len(selected)==1, f'Expected exactly one visible GPU: {selected}'
    s.cycles.device='GPU'
    s.render.resolution_x,s.render.resolution_y=budget['resolution_each'];s.render.resolution_percentage=100
    s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Studio');world.use_nodes=True;s.world=world
    world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.18,.18,.18,1)
    world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.5
    for name,loc,energy,size in [('Key',(-3,-4,4),500,4),('Fill',(3,-2,2),250,3),('Rim',(1,3,3),350,3)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.shape='DISK';light.size=size
        ob=bpy.data.objects.new(name,light);s.collection.objects.link(ob);ob.location=loc
        ob.rotation_euler=(-ob.location).to_track_quat('-Z','Y').to_euler()
    cam=bpy.data.objects.new('FixedReviewCamera',bpy.data.cameras.new('FixedReviewCamera'));s.collection.objects.link(cam);s.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=2.3
    C=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
    def mesh(name,v,f):
        data=bpy.data.meshes.new(name);data.from_pydata(v.tolist(),[],f.tolist())
        obj=bpy.data.objects.new(name,data);s.collection.objects.link(obj);obj.matrix_world=C
        for poly in data.polygons:poly.use_smooth=True
        return obj
    body=mesh('FrozenBody',vertices[ids[0]],faces);apply_character_materials(body.data,faces,m['appearance'])
    clothing=mesh('BoundGarment',cv[0],cf);add_clothing_materials(clothing.data,cloth)
    hair=add_groom(recipe['source_groom'],C)
    counts=[dict(name=o.name,curves=len(o.data.curves),points=len(o.data.points)) for o in hair]
    original_hair=[np.array(o.data.position_data[0].vector) for o in hair]
    assert sum(x['curves'] for x in counts)==63652 and sum(x['points'] for x in counts)==1591300
    # Record scalar defaults as well as explicit settings; same executable/factory state as old run.
    def scalar_rna(obj):
        values={}
        for prop in obj.bl_rna.properties:
            name=prop.identifier
            if name=='rna_type' or prop.type in {'POINTER','COLLECTION'}:continue
            try:
                v=getattr(obj,name)
                if getattr(prop,'is_array',False):v=list(v)
                if isinstance(v,set):v=sorted(v)
                json.dumps(v);values[name]=v
            except (TypeError,ValueError,AttributeError):pass
        return values
    settings=dict(cycles=scalar_rna(s.cycles),render=scalar_rna(s.render),view=scalar_rna(s.view_settings),
        image=scalar_rna(s.render.image_settings),display=scalar_rna(s.display_settings),
        lights=[dict(name=o.name,location=list(o.location),rotation=list(o.rotation_euler),data=scalar_rna(o.data)) for o in s.objects if o.type=='LIGHT'],
        camera_data=scalar_rna(cam.data),world_color=[.18,.18,.18,1],world_strength=.5,
        materials=[m.name for m in bpy.data.materials],geometry_sources=dict(body=str(bodyfile),clothes=str(BASE/'full/clothing_vertices.npy'),head=str(BASE/'full/head_transforms.npy'),groom=recipe['source_groom']))
    (out/'settings_before_render.json').write_text(json.dumps(settings,indent=2)+'\n')
    setup=time.monotonic()-began;records=[]
    selected_frames=({frozen['calibration_asset_frame']} if a.stage=='calibration' else set(frozen['short_asset_frames']) if a.stage=='short' else set(range(76))-set(frozen['short_asset_frames']))
    reused=set(frozen['short_asset_frames']) if a.stage=='full' else set()
    for k,asset_frame in enumerate(ids.tolist()):
        if asset_frame not in selected_frames:continue
        if time.monotonic()-began>1800:raise RuntimeError('Frozen render time budget exceeded; partial outputs retained')
        frame_start=time.monotonic()
        body.data.vertices.foreach_set('co',vertices[asset_frame].ravel());body.data.update()
        clothing.data.vertices.foreach_set('co',cv[k].ravel());clothing.data.update()
        for obj,count,first in zip(hair,counts,original_hair):
            obj.matrix_world=C@Matrix(H[k].tolist())
            assert len(obj.data.points)==count['points'] and len(obj.data.curves)==count['curves']
            assert np.array_equal(np.array(obj.data.position_data[0].vector),first)
            actual=np.array(obj.matrix_world@Vector(first))
            desired=np.array(C)[:3,:3]@(H[k,:3,:3]@first+H[k,:3,3])
            assert np.max(np.abs(actual-desired))<2e-6
        # Read back Blender's body coordinates; no hidden mesh modifier or transform.
        read=np.empty(vertices[asset_frame].size,np.float32);body.data.vertices.foreach_get('co',read)
        assert np.array_equal(read.reshape(-1,3),vertices[asset_frame])
        for label,angle in [('front',0),('side',90)]:
            start=time.monotonic();target=Vector((0,0,-.10));theta=math.radians(angle)
            cam.location=target+Vector((4*math.sin(theta),-4*math.cos(theta),.12))
            cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()
            s.render.filepath=str(out/f'frame_{asset_frame:03d}_{label}.png');bpy.ops.render.render(write_still=True)
            row=dict(asset_frame=asset_frame,source_frame=asset_frame+30,view=label,seconds=time.monotonic()-start,samples=s.cycles.samples,
                camera_from_model=np.array(cam.matrix_world.inverted()@C).tolist(),ortho_scale_m=2.3,
                body_readback_exact=True,groom_constant_topology=True,head_matrix_applied_once=True)
            records.append(row)
            with (out/'timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(f'PILOT_FRAME_DONE {asset_frame} {time.monotonic()-frame_start:.3f}s',flush=True)
    result=dict(integration_only=True,stage=a.stage,gpu_uuid=budget['gpu_uuid'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        blender_version=bpy.app.version_string,devices=selected,setup_seconds=setup,elapsed_seconds=time.monotonic()-began,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rendered_views=len(records),reused_short_asset_frames=sorted(reused),samples=s.cycles.samples,denoising=s.cycles.use_denoising,seed=s.cycles.seed,use_animated_seed=s.cycles.use_animated_seed,
        records=records,groom_objects=counts,display_transform=np.array(C).tolist(),camera_fixed_target=[0,0,-.10],
        settings_snapshot=settings,no_floor=True,clothing_topology_fixed=True,body_modifiers=0,actual_dynamic_human_review='pending_user')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PILOT_FINISHED '+json.dumps({k:result[k] for k in ['stage','elapsed_seconds','rendered_views','gpu_uuid']}),flush=True)


if __name__=='__main__':main()
