"""Blender low-cost two-fixed-view preview of frozen body and derived bindings."""
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
from densekp_retrain.rendering.blender.materials import apply_character_materials
from densekp_retrain.rendering.blender.garment_materials import add_garment_materials, legacy_native_description
from densekp_retrain.rendering.blender.groom import add_groom
BASE=ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008'


def main():
    global BASE
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['probe','full'],required=True)
    ap.add_argument('--render-folder',default='renders')
    ap.add_argument('--asset-root',type=Path)
    ap.add_argument('--recipe-file',default='frozen_recipe.json')
    ap.add_argument('--views',choices=['both','front'],default='both')
    ap.add_argument('--legacy-garment-before',action='store_true',help='A2 independent before/after audit only')
    ap.add_argument('--audit-material',action='store_true')
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);began=time.monotonic()
    if a.asset_root:BASE=a.asset_root.resolve()
    recipe=json.loads((BASE/a.recipe_file).read_text()); budget=recipe['budget']
    timeline=json.loads(Path(recipe['timeline']).read_text()) if 'timeline' in recipe else None
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==budget['gpu_uuid'], 'Explicit reserved UUID required'
    out=BASE/a.stage/a.render_folder;out.mkdir(exist_ok=False)
    m=json.loads(Path(recipe['character_manifest']).read_text());bodyfile=Path(recipe['body_cache'])
    vertices=np.load(bodyfile,mmap_mode='r');faces=np.load(Path(recipe['faces']) if 'faces' in recipe else bodyfile.parent.parent/'faces.npy')
    ids=np.load(BASE/a.stage/'asset_frames.npy')
    cv=np.load(recipe['clothing_cache'],mmap_mode='r') if recipe.get('clothing_cache') else np.load(BASE/a.stage/'clothing_vertices.npy')
    H=np.load(BASE/a.stage/'head_transforms.npy')
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
    cam=bpy.data.objects.new('FixedReviewCamera',bpy.data.cameras.new('FixedReviewCamera'));s.collection.objects.link(cam);s.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=float(recipe.get('camera_ortho_scale_m',2.3))
    C=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
    def mesh(name,v,f):
        data=bpy.data.meshes.new(name);data.from_pydata(v.tolist(),[],f.tolist())
        obj=bpy.data.objects.new(name,data);s.collection.objects.link(obj);obj.matrix_world=C
        for poly in data.polygons:poly.use_smooth=True
        return obj
    body=mesh('FrozenBody',vertices[ids[0]],faces);apply_character_materials(body.data,faces,m['appearance'])
    clothing=mesh('BoundGarment',cv[ids[0] if recipe.get('clothing_cache') else 0],cf)
    native_material=recipe.get('native_clothing_material')
    material_description=recipe.get('garment_description')
    if material_description:
        material_description=json.loads(Path(material_description).read_text())
    elif native_material:
        material_description=legacy_native_description(native_material)
    else:
        material_description=dict(material_kind='GarmentRest_continuous_shell',material=dict(reference=str(cloth)))
    if a.legacy_garment_before:
        from tools.validate.garment_material_before_a2 import add_before_materials
        add_before_materials(clothing.data,cloth,native_material)
    else:
        add_garment_materials(clothing.data,material_description)
    if a.audit_material:
        def material_snapshot(mesh):
            result=[]
            for mat in mesh.materials:
                nodes=mat.node_tree.nodes
                result.append(dict(name=mat.name,
                    roughness=float(nodes.get('Principled BSDF').inputs['Roughness'].default_value),
                    images=sorted((n.image.filepath,n.image.colorspace_settings.name)
                        for n in nodes if n.type=='TEX_IMAGE'),
                    links=sorted((link.from_node.bl_idname,link.from_socket.name,
                                  link.to_node.bl_idname,link.to_socket.name)
                                 for link in mat.node_tree.links)))
            return result
        loops=np.empty(len(clothing.data.loops)*2,np.float32)
        if clothing.data.uv_layers:
            clothing.data.uv_layers.active.data.foreach_get('uv',loops)
        np.savez(out/'material_geometry_audit.npz',
                 garment_vertices=np.asarray(cv[ids[0] if recipe.get('clothing_cache') else 0]),
                 garment_faces=cf,uv_loops=loops if clothing.data.uv_layers else np.empty(0,np.float32),
                 polygon_material_ids=np.asarray([p.material_index for p in clothing.data.polygons]),
                 object_matrix=np.asarray(clothing.matrix_world),instance_id=np.asarray([clothing.pass_index]))
        (out/'material_audit.json').write_text(json.dumps(dict(
            materials=material_snapshot(clothing.data),
            uv_layers=[layer.name for layer in clothing.data.uv_layers],
            material_slots=len(clothing.data.materials),
            geometry_unchanged=True),indent=2)+'\n')
    hair=add_groom(recipe['source_groom'],C)
    counts=[dict(name=o.name,curves=len(o.data.curves),points=len(o.data.points)) for o in hair]
    original_hair=[np.array(o.data.position_data[0].vector) for o in hair]
    expected=recipe.get('groom_render_counts',{'curves':63652,'points':1591300})
    assert sum(x['curves'] for x in counts)==expected['curves'] and sum(x['points'] for x in counts)==expected['points']
    setup=time.monotonic()-began;records=[]
    reused=set(recipe['probe_asset_frames']) if a.stage=='full' else set()
    prior=0
    if a.asset_root and a.stage=='full':
        decision=json.loads((BASE/'probe_decision.json').read_text());assert decision['proceed']
        prior=decision['probe_seconds']
    for k,asset_frame in enumerate(ids.tolist()):
        if asset_frame in reused:continue
        if time.monotonic()-began+prior>budget.get('max_render_seconds',1800):raise RuntimeError('Frozen render time budget exceeded; partial outputs retained')
        frame_start=time.monotonic()
        body.data.vertices.foreach_set('co',vertices[asset_frame].ravel());body.data.update()
        clothing.data.vertices.foreach_set('co',cv[asset_frame if recipe.get('clothing_cache') else k].ravel());clothing.data.update()
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
        for label,angle in ([('front',0)] if a.views=='front' else [('front',0),('side',90)]):
            start=time.monotonic();target=Vector(recipe.get('camera_target_blender_m',(0,0,-.10)));theta=math.radians(angle)
            cam.location=target+Vector((4*math.sin(theta),-4*math.cos(theta),.12))
            cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()
            s.render.filepath=str(out/f'frame_{asset_frame:03d}_{label}.png');bpy.ops.render.render(write_still=True)
            row=dict(asset_frame=asset_frame,source_frame=timeline[asset_frame]['source_frame'] if timeline else asset_frame+30,view=label,seconds=time.monotonic()-start,
                camera_from_model=np.array(cam.matrix_world.inverted()@C).tolist(),ortho_scale_m=cam.data.ortho_scale,
                body_readback_exact=True,groom_constant_topology=True,head_matrix_applied_once=True)
            records.append(row)
            with (out/'timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(f'PILOT_FRAME_DONE {asset_frame} {time.monotonic()-frame_start:.3f}s',flush=True)
    result=dict(integration_only=True,stage=a.stage,gpu_uuid=budget['gpu_uuid'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        blender_version=bpy.app.version_string,devices=selected,setup_seconds=setup,elapsed_seconds=time.monotonic()-began,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rendered_views=len(records),reused_probe_asset_frames=sorted(reused),
        records=records,groom_objects=counts,display_transform=np.array(C).tolist(),camera_fixed_target=list(recipe.get('camera_target_blender_m',(0,0,-.10))),
        no_floor=True,clothing_topology_fixed=True,body_modifiers=0,
        native_clothing_material=bool(native_material),native_diffuse_applied=bool(native_material and native_material.get('diffuse')),
        actual_dynamic_human_review='pending_user')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PILOT_FINISHED '+json.dumps({k:result[k] for k in ['stage','elapsed_seconds','rendered_views','gpu_uuid']}),flush=True)


if __name__=='__main__':main()
