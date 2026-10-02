"""Blender: identical studio lighting, orthographic scale and clothing palette."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.rendering.blender.materials import apply_character_materials, add_clothing_materials
from densekp_retrain.rendering.blender.groom import add_groom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['CPU', 'CUDA'], default='CUDA')
    p.add_argument('--samples', type=int, default=48)
    p.add_argument('--heads-only', action='store_true', help='Review a material revision before full previews')
    p.add_argument('--record-projection', action='store_true', help='Save actual studio camera projections for loader diagnostics')
    p.add_argument('--character', help='Restrict a component revision review to one identity')
    p.add_argument('--only-views', nargs='+', help='Render only named views for a localized revision')
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--reference-heads-only', action='store_true', help='Avoid repeating full rest-pose outfit views')
    p.add_argument('--characters', nargs='+', help='Bounded multi-identity review batch')
    p.add_argument('--no-groom', action='store_true', help='Same-condition identity comparison without hairstyle cues')
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output.mkdir(parents=True, exist_ok=False); start=time.monotonic()
    report=json.loads((a.assets/'report.json').read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene; s.render.engine='CYCLES'; s.cycles.samples=a.samples
    s.cycles.use_denoising=False; s.cycles.seed=29
    s.render.threads_mode='FIXED'; s.render.threads=8
    if a.device=='CUDA':
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='CUDA'; prefs.get_devices()
        for device in prefs.devices: device.use=device.type=='CUDA'
        if not any(d.use for d in prefs.devices): raise RuntimeError('No CUDA device available')
        s.cycles.device='GPU'
    s.render.resolution_x=640; s.render.resolution_y=640; s.render.resolution_percentage=100
    s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGB'
    s.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Studio'); world.use_nodes=True; s.world=world
    world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.18,.18,.18,1)
    world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.5
    for name,location,energy,size in [('Key',(-3,-4,4),500,4),('Fill',(3,-2,2),250,3),('Rim',(1,3,3),350,3)]:
        light=bpy.data.lights.new(name,'AREA'); light.energy=energy; light.shape='DISK'; light.size=size
        obj=bpy.data.objects.new(name,light); s.collection.objects.link(obj); obj.location=location
        obj.rotation_euler=(-obj.location).to_track_quat('-Z','Y').to_euler()
    camera=bpy.data.objects.new('ReviewCamera',bpy.data.cameras.new('ReviewCamera'))
    s.collection.objects.link(camera); s.camera=camera; camera.data.type='ORTHO'
    conv=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
    jobs=[]
    for character in report['characters']:
        manifest=json.loads(Path(character['manifest']).read_text())
        jobs.append(dict(manifest['reference'], character_id=manifest['character_id'],
                         pose_source_id='reference', appearance=manifest['appearance']))
    if not a.heads_only and not a.reference_only:
        jobs.extend(report['derivatives'])
    records=[]
    for job in jobs:
        if a.character and job['character_id']!=a.character:continue
        if a.characters and job['character_id'] not in a.characters:continue
        folder=a.output/job['character_id']/job['pose_source_id']; folder.mkdir(parents=True)
        geometry=Path(job['geometry']); clothing=Path(job['clothing_geometry'])
        vertices=np.load(geometry/'vertices.npy')[0]; faces=np.load(geometry/'faces.npy')
        mesh=bpy.data.meshes.new('Body'); mesh.from_pydata(vertices.tolist(),[],faces.tolist())
        body=bpy.data.objects.new('Body',mesh); s.collection.objects.link(body); body.matrix_world=conv
        for poly in mesh.polygons: poly.use_smooth=True
        apply_character_materials(mesh,faces,job['appearance'])
        cv=np.load(clothing/'clothing_vertices.npy')[0]; cf=np.load(clothing/'clothing_faces.npy')
        cm=bpy.data.meshes.new('Clothes'); cm.from_pydata(cv.tolist(),[],cf.tolist())
        co=bpy.data.objects.new('Clothes',cm); s.collection.objects.link(co); co.matrix_world=conv
        for poly in cm.polygons: poly.use_smooth=True
        add_clothing_materials(cm,clothing)
        hair=add_groom(job['groom_geometry'],conv) if 'groom_geometry' in job and not a.no_groom else []
        hair_points=np.empty((0,3),np.float32)
        if hair:
            hair_folder=Path(job['groom_geometry']);hair_points=np.load(hair_folder/'points.npy')
            if (hair_folder/'active_curve_mask.npy').exists():
                active=np.load(hair_folder/'active_curve_mask.npy');offsets=np.load(hair_folder/'curve_offsets.npy')
                hair_points=hair_points[np.repeat(active,np.diff(offsets))]
        # Same metric scale for every person, without independently resizing bodies.
        views=[] if a.heads_only else [('front',0,2.25),('side',90,2.25),('back',180,2.25)]
        if job['pose_source_id']=='reference':
            views.extend([('head_front',0,.34),('head_threequarter',35,.34)])
            if not a.heads_only: views.append(('head_clay',35,.34))
        if hair and not a.heads_only:
            extra=[('head_side',90,.42),('head_back',180,.42)]
            if views[-1][0]=='head_clay':views=views[:-1]+extra+views[-1:]
            else:views.extend(extra)
        if a.only_views:views=[v for v in views if v[0] in a.only_views]
        if a.reference_heads_only and job['pose_source_id']=='reference':
            views=[v for v in views if v[0].startswith('head')]
        for name,angle,scale in views:
            if name.startswith('head'):
                with np.load(job['appearance']['eye_mapping']) as z:
                    labels=z['component_labels']
                eyes=vertices[labels!=0].mean(0)
                target=Vector((float(eyes[0]),float(-eyes[2]),float(eyes[1]+.025)))
            else:
                # All bodies already have pelvis at the origin.
                target=Vector((0,0,0))
            if name=='head_clay':
                mesh.materials.clear(); apply_character_materials(mesh,faces,job['appearance'],clay=True)
                for poly in mesh.polygons: poly.material_index=0
            theta=math.radians(angle)
            camera.location=target+Vector((4*math.sin(theta),-4*math.cos(theta),.12 if scale>1 else .015))
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
            camera.data.ortho_scale=scale
            s.render.filepath=str(folder/f'{name}.png'); bpy.ops.render.render(write_still=True)
            record=dict(character_id=job['character_id'],pose_source_id=job['pose_source_id'],
                        view=name,path=str(Path(s.render.filepath).resolve()),ortho_scale_m=scale,
                        material_version=job['appearance'].get('material_version'),
                        groom_geometry=job.get('groom_geometry'),clothing_geometry=job['clothing_geometry'])
            if a.record_projection and not name.startswith('head'):
                camera_from_body=np.asarray(camera.matrix_world.inverted()@body.matrix_world,dtype=np.float64)
                def project(points):
                    camera_points=points@camera_from_body[:3,:3].T+camera_from_body[:3,3]
                    return np.column_stack(((camera_points[:,0]/scale+.5)*640,
                                            (.5-camera_points[:,1]/scale)*640))
                dense=np.load(geometry/'densekp_3d.npy')[0]
                xy=project(dense);bounds=project(np.concatenate((vertices,cv,hair_points)))
                record.update(xy=xy.tolist(),bbox_xyxy=np.r_[bounds.min(0),bounds.max(0)].tolist(),
                              camera_from_body=camera_from_body.tolist(),resolution=[640,640],
                              visibility_status='not_computed_for_studio_crop_diagnostic')
            records.append(record)
        bpy.data.objects.remove(body,do_unlink=True); bpy.data.objects.remove(co,do_unlink=True)
        bpy.data.meshes.remove(mesh); bpy.data.meshes.remove(cm)
        for obj in hair:
            data=obj.data;bpy.data.objects.remove(obj,do_unlink=True);bpy.data.hair_curves.remove(data)
    (a.output/'review_manifest.json').write_text(json.dumps(dict(
        split='integration_only', seconds=time.monotonic()-start, samples=a.samples, device=a.device,
        lighting='fixed three area lights plus uniform world', fixed_clothing_palette=report.get('uniform_comparison',False),
        head_mode='per-job groom when specified; otherwise legacy albedo only', views=records),indent=2)+'\n')


if __name__=='__main__': main()
