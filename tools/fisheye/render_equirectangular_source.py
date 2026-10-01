"""Blender helper: render equirectangular source buffers for Omni-Radtan warping."""
import argparse, json, math, sys
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.render.character_materials import apply_character_materials, add_clothing_materials

def arguments():
    p=argparse.ArgumentParser(); q=sys.argv[sys.argv.index('--')+1:]
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--clothing-geometry',type=Path)
    p.add_argument('--cameras',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--frame',type=int,default=0)
    p.add_argument('--body-texture',type=Path); p.add_argument('--body-uv',type=Path); p.add_argument('--hdri',type=Path); p.add_argument('--width',type=int,default=2048); p.add_argument('--height',type=int,default=1024); p.add_argument('--samples',type=int,default=8)
    p.add_argument('--scene-blend',type=Path,help='Render people inside this scene, retaining its native World and lights')
    p.add_argument('--device',choices=['CPU','CUDA'],default='CPU')
    p.add_argument('--seed',type=int,help='Explicit Cycles seed; disables animated seed')
    p.add_argument('--people-config',type=Path,help='JSON with people: [{id, frame, location, yaw_deg}]')
    p.add_argument('--check-ground-only',action='store_true',help='Probe native terrain without importing people or rendering')
    return p.parse_args(q)

def main():
    a=arguments(); a.output_dir.mkdir(parents=True,exist_ok=True); vertices=np.load(a.geometry/'vertices.npy'); faces=np.load(a.geometry/'faces.npy'); f=a.frame
    if a.scene_blend:
        bpy.ops.wm.open_mainfile(filepath=str(a.scene_blend.resolve()))
        for obj in bpy.context.scene.objects: obj.pass_index=0
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene; s.render.resolution_x=a.width; s.render.resolution_y=a.height; s.render.resolution_percentage=100
    # Eevee cannot render panoramic cameras, so use Cycles with the same static geometry.
    s.render.engine='CYCLES'; s.cycles.samples=a.samples; s.cycles.use_denoising=False; s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGBA'; s.render.film_transparent=False
    if a.seed is not None:
        s.cycles.seed=a.seed; s.cycles.use_animated_seed=False
    s.view_layers[0].use_pass_z=True; s.view_layers[0].use_pass_object_index=True; s.view_settings.look=s.view_settings.look if a.scene_blend else 'None'
    if a.device=='CUDA':
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='CUDA'; prefs.get_devices()
        for device in prefs.devices: device.use=device.type=='CUDA'
        if not any(d.use for d in prefs.devices): raise RuntimeError('No CUDA Cycles device available')
        s.cycles.device='GPU'
    else: s.cycles.device='CPU'
    # Asset preview scenes disable compositing; annotations require depth/ID EXR output.
    s.render.use_compositing=True
    if hasattr(s,'compositing_node_group'):
        nt=bpy.data.node_groups.new('FisheyeCompositor','CompositorNodeTree'); s.compositing_node_group=nt
    else: s.use_nodes=True; nt=s.node_tree; nt.nodes.clear()
    rl=nt.nodes.new('CompositorNodeRLayers'); rl.scene=s; file_node=nt.nodes.new('CompositorNodeOutputFile')
    if hasattr(file_node,'directory'):
        file_node.directory=str(a.output_dir); file_node.file_name='source'; file_node.format.file_format='OPEN_EXR_MULTILAYER'; file_node.format.color_depth='32'
        file_node.file_output_items.new('RGBA','Combined'); file_node.file_output_items.new('FLOAT','Depth'); file_node.file_output_items.new('FLOAT','IndexOB')
    else:
        file_node.base_path=str(a.output_dir); file_node.format.file_format='OPEN_EXR_MULTILAYER'; file_node.format.color_depth='32'; file_node.file_slots.clear(); file_node.file_slots.new('Combined'); file_node.file_slots.new('Depth'); file_node.file_slots.new('IndexOB')
    index_output=next((socket for socket in rl.outputs if 'Index' in socket.name),None)
    if index_output is None: raise RuntimeError(f'object index render pass unavailable: {[x.name for x in rl.outputs]}')
    nt.links.new(rl.outputs['Image'],file_node.inputs['Combined']); nt.links.new(rl.outputs['Depth'],file_node.inputs['Depth']); nt.links.new(index_output,file_node.inputs['IndexOB'])
    if not a.scene_blend:
        world=bpy.data.worlds.new('FisheyeWorld'); s.world=world; world.use_nodes=True; nodes=world.node_tree.nodes; links=world.node_tree.links; nodes.clear(); wo=nodes.new('ShaderNodeOutputWorld'); bg=nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value=1
        if a.hdri:
            env=nodes.new('ShaderNodeTexEnvironment'); env.image=bpy.data.images.load(str(a.hdri.resolve())); env.image.colorspace_settings.name='Linear Rec.709'; links.new(env.outputs['Color'],bg.inputs['Color'])
        else: bg.inputs['Color'].default_value=(.08,.1,.14,1)
        links.new(bg.outputs['Background'],wo.inputs['Surface'])
    people=json.loads(a.people_config.read_text())['people'] if a.people_config else [{'id':1,'frame':f,'location':[0,0,0],'yaw_deg':0}]
    if not people or len({int(p['id']) for p in people})!=len(people): raise ValueError('people must have unique IDs')
    uv=np.load(a.body_uv) if a.body_texture and a.body_uv else None
    body_texture=bpy.data.images.load(str(a.body_texture.resolve())) if uv is not None else None
    cv=np.load(a.clothing_geometry/'clothing_vertices.npy') if a.clothing_geometry else None
    cf=np.load(a.clothing_geometry/'clothing_faces.npy') if a.clothing_geometry else None
    conv=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
    default_vertices,default_faces=vertices,faces
    # Query native ground before adding any person, so people cannot hit each other.
    ground_checks=[];ground_failed=False
    for person in people:
        if not person.get('ground_contact_check'): continue
        body_v=np.load(Path(person['geometry'])/'vertices.npy')[person['frame']]
        t=Matrix.Translation(person['location'])@Matrix.Rotation(math.radians(person['yaw_deg']),4,'Z')@(
            Matrix.Translation((0,0,-float(body_v[:,1].min())))@conv)
        matrix=np.asarray(t);world=body_v@matrix[:3,:3].T+matrix[:3,3]
        ids=np.flatnonzero(world[:,2]<world[:,2].min()+.02)[::5]
        distances=[];hits=[]
        for index in ids:
            point=world[index]
            hit,position,normal,face,obj,_=s.ray_cast(bpy.context.evaluated_depsgraph_get(),
                Vector((float(point[0]),float(point[1]),float(point[2]+.25))),Vector((0,0,-1)),distance=1.)
            if not hit: raise ValueError('Pedestrian support has no native ground below it')
            distances.append(float(point[2]-position.z));hits.append(obj.name)
        check=dict(person_id=person['id'],sampled_sole_vertices=len(ids),
            minimum_gap_m=min(distances),maximum_gap_m=max(distances),native_hit_objects=sorted(set(hits)),
            method='downward native-scene ray cast at lowest 2cm body vertices, before importing people',
            gate='support gap -10mm..30mm; upper foot clearance recorded separately')
        ground_checks.append(check)
        if min(distances)<-.01 or max(distances)>.03:
            ground_failed=True
            if a.check_ground_only:continue
            (a.output_dir/'ground_contact.json').write_text(json.dumps(dict(status='failed',cases=ground_checks),indent=2)+'\n')
            raise ValueError(f'Native ground contact fails: {check}')
    if ground_checks:
        (a.output_dir/'ground_contact.json').write_text(json.dumps(dict(status='failed' if ground_failed else 'passed',cases=ground_checks),indent=2)+'\n')
    if a.check_ground_only:return
    for person in people:
        person_geometry=Path(person['geometry']) if 'geometry' in person else a.geometry
        if 'geometry' in person:
            if a.clothing_geometry: raise ValueError('Shared motion clothing cannot be applied to independent bodies')
            vertices=np.load(person_geometry/'vertices.npy'); faces=np.load(person_geometry/'faces.npy')
        else:
            vertices,faces=default_vertices,default_faces
        if 'clothing_geometry' in person:
            clothing_path=Path(person['clothing_geometry'])
            clothing_info=json.loads((clothing_path/'clothing_manifest.json').read_text())
            if Path(clothing_info['body_geometry']).resolve()!=person_geometry.resolve():
                raise ValueError('Clothing belongs to a different body geometry')
            person_cv=np.load(clothing_path/'clothing_vertices.npy')
            person_cf=np.load(clothing_path/'clothing_faces.npy')
            face_materials=np.load(clothing_path/'face_material_ids.npy')
            materials=json.loads((clothing_path/'materials.json').read_text())
            barycentric=clothing_path/'body_barycentric_indices.npy'
            if barycentric.exists():
                indices=np.load(barycentric); weights=np.load(clothing_path/'body_barycentric_weights.npy')
                if indices.shape!=weights.shape or indices.ndim!=2 or indices.shape[1]!=3:
                    raise ValueError('Invalid garment barycentric correspondence')
                if indices.min()<0 or indices.max()>=vertices.shape[1] or not np.isfinite(weights).all() or (weights<0).any():
                    raise ValueError('Invalid garment barycentric indices/weights')
                np.testing.assert_allclose(weights.sum(1),1,atol=1e-6,rtol=0)
                current_reference=(vertices[:,indices]*weights[None,:,:,None]).sum(2)
            else:
                body_ids=np.load(clothing_path/'body_vertex_ids.npy')
                current_reference=vertices[:,body_ids]
            if person_cv.shape!=current_reference.shape:
                raise ValueError('Clothing frame count or body correspondence differs')
            reference=np.load(clothing_path/'reference_body_vertices.npy')
            distances=np.linalg.norm(person_cv-current_reference,axis=-1)
            if clothing_info.get('contract')=='commuter_surface_binding_v1':
                from tools.render.commuter_clothing_contract import validate_binding
                validate_binding(clothing_path,current_reference,person_cv,require_studio=True)
            elif not np.array_equal(reference,current_reference) or not np.isfinite(person_cv).all() or distances.max()>.010002:
                raise ValueError('Fitted clothing no longer matches this body pose')
        else:
            person_cv,person_cf=cv,cf
            face_materials=None
            materials=[dict(name='ClothingMaterial',color=[.04,.35,.5,1])]
        person_id=int(person['id']); frame=int(person['frame']); location=person['location']; yaw=float(person['yaw_deg'])
        if person_id<1 or frame<0 or frame>=len(vertices) or len(location)!=3: raise ValueError(f'invalid person {person}')
        transform=Matrix.Translation(location)@Matrix.Rotation(math.radians(yaw),4,'Z')@(Matrix.Translation((0,0,-float(vertices[frame,:,1].min())))@conv)
        mesh=bpy.data.meshes.new(f'SMPLX_{person_id}'); mesh.from_pydata(vertices[frame].tolist(),[],faces.tolist())
        body=bpy.data.objects.new(f'SMPLX_{person_id}',mesh); s.collection.objects.link(body); body.pass_index=person_id; body.matrix_world=transform
        # Persist Blender's actual object transform and world mesh for independent GT checks.
        matrix=np.asarray(body.matrix_world,dtype=np.float64)
        np.save(a.output_dir/f'person_{person_id}_world_vertices.npy',vertices[frame]@matrix[:3,:3].T+matrix[:3,3])
        (a.output_dir/f'person_{person_id}_geometry.json').write_text(json.dumps(dict(
            geometry=str(person_geometry.resolve()),frame=frame,object_to_world=matrix.tolist()),indent=2)+'\n')
        if 'appearance' in person:
            apply_character_materials(mesh, faces, person['appearance'])
            for polygon in mesh.polygons: polygon.use_smooth=True
        else:
            mat=bpy.data.materials.new(f'Body_{person_id}'); mat.use_nodes=True; bs=mat.node_tree.nodes.get('Principled BSDF'); bs.inputs['Roughness'].default_value=.58; bs.inputs['Base Color'].default_value=(.55,.25,.16,1); mesh.materials.append(mat)
        if uv is not None and 'appearance' not in person:
            layer=mesh.uv_layers.new(name='SMPLX_UV')
            for polygon,indices in zip(mesh.polygons,uv['faces_uv']):
                for loop,index in zip(polygon.loop_indices,indices): layer.data[loop].uv=uv['uv'][index]
            tex=mat.node_tree.nodes.new('ShaderNodeTexImage'); tex.image=body_texture; tex.image.colorspace_settings.name='sRGB'; mat.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
        if person_cv is not None:
            cm=bpy.data.meshes.new(f'Clothing_{person_id}'); cm.from_pydata(person_cv[frame].tolist(),[],person_cf.tolist())
            co=bpy.data.objects.new(f'Clothing_{person_id}',cm); s.collection.objects.link(co); co.pass_index=person_id if a.people_config else 2; co.matrix_world=transform
            if 'clothing_geometry' in person and (clothing_path/'garment_material.json').exists():
                add_clothing_materials(cm, clothing_path)
                for polygon in cm.polygons: polygon.use_smooth=True
            else:
                for material in materials:
                    cmat=bpy.data.materials.new(f"{material['name']}_{person_id}"); cmat.use_nodes=True; cb=cmat.node_tree.nodes.get('Principled BSDF'); cb.inputs['Base Color'].default_value=material['color']; cb.inputs['Roughness'].default_value=.75; cm.materials.append(cmat)
            if face_materials is not None and not ('clothing_geometry' in person and (clothing_path/'garment_material.json').exists()):
                if face_materials.shape!=(len(person_cf),) or face_materials.min()<0 or face_materials.max()>=len(materials):
                    raise ValueError('Invalid clothing face materials')
                for polygon,material_id in zip(cm.polygons,face_materials):
                    polygon.material_index=int(material_id)
                    polygon.use_smooth=True
            clothing_matrix=np.asarray(co.matrix_world,dtype=np.float64)
            np.save(a.output_dir/f'person_{person_id}_world_clothing.npy',person_cv[frame]@clothing_matrix[:3,:3].T+clothing_matrix[:3,3])
        if 'groom_geometry' in person:
            from tools.render.character_groom import add_groom
            folder=Path(person['groom_geometry'])
            info=json.loads((folder/'groom_manifest.json').read_text())
            if Path(info['body_geometry']).resolve()!=person_geometry.resolve():
                raise ValueError('Groom belongs to another body')
            objects=add_groom(folder,transform,person_id)
            rendered=[];counts=[]
            for obj in objects:
                points=np.empty((len(obj.data.points),3),np.float32)
                obj.data.position_data.foreach_get('vector',points.ravel())
                m=np.asarray(obj.matrix_world,dtype=np.float64)
                rendered.append(points@m[:3,:3].T+m[:3,3])
                counts.append(dict(name=obj.name,points=len(points),curves=len(obj.data.curves),
                                   pass_index=obj.pass_index,object_to_world=m.tolist()))
            np.save(a.output_dir/f'person_{person_id}_world_groom.npy',np.concatenate(rendered))
            (a.output_dir/f'person_{person_id}_groom.json').write_text(json.dumps(dict(
                geometry=str(folder.resolve()),objects=counts,depth_and_instance='native opaque Cycles curves'),indent=2)+'\n')
    if not a.scene_blend:
        ground_mesh=bpy.data.meshes.new('Ground'); ground_mesh.from_pydata([(-12,-12,-.01),(12,-12,-.01),(12,12,-.01),(-12,12,-.01)],[],[(0,1,2),(0,2,3)]); ground=bpy.data.objects.new('Ground',ground_mesh); s.collection.objects.link(ground); ground.pass_index=0
        gm=bpy.data.materials.new('GroundMaterial'); gm.use_nodes=True; gb=gm.node_tree.nodes.get('Principled BSDF'); gb.inputs['Base Color'].default_value=(.14,.16,.13,1); gb.inputs['Roughness'].default_value=.85; ground_mesh.materials.append(gm)
    cd=bpy.data.cameras.new('OmniSourceCamera'); cd.type='PANO'; pano=cd.cycles if hasattr(cd,'cycles') else cd
    pano.panorama_type='EQUIRECTANGULAR'; pano.longitude_min=-math.pi; pano.longitude_max=math.pi; pano.latitude_min=-math.pi/2; pano.latitude_max=math.pi/2
    cam=bpy.data.objects.new('OmniSourceCamera',cd); s.collection.objects.link(cam); s.camera=cam
    records=[]
    for item in json.loads(a.cameras.read_text())['cameras']:
        pose=np.asarray(item['camera_to_world'],float); r=pose[:3,:3]
        # Blender local axes: X right, Y up, -Z forward. CV: X right, Y down, Z forward.
        matrix=np.eye(4); matrix[:3,0]=r[:,0]; matrix[:3,1]=-r[:,1]; matrix[:3,2]=-r[:,2]; matrix[:3,3]=pose[:3,3]; cam.matrix_world=Matrix(matrix.tolist())
        if hasattr(file_node,'file_name'): file_node.file_name=item['id']
        else:
            for slot in file_node.file_slots: slot.path=item['id']
        s.render.filepath=str(a.output_dir/f"{item['id']}_preview.png"); bpy.ops.render.render(write_still=True)
        candidates=sorted(a.output_dir.glob(item['id']+'*.exr'))
        if len(candidates)!=1: raise RuntimeError(f'cannot locate compositor output for {item["id"]}: {candidates}')
        target=a.output_dir/f"{item['id']}.exr"
        if candidates[0]!=target: candidates[0].replace(target)
        records.append({'id':item['id'],'source':str(target.resolve()),'camera_to_world':item['camera_to_world']})
    (a.output_dir/'source_manifest.json').write_text(json.dumps({'projection':'equirectangular','resolution':[a.width,a.height],'frame':f,'samples':a.samples,'seed':s.cycles.seed,'animated_seed':s.cycles.use_animated_seed,'scene_blend':str(a.scene_blend.resolve()) if a.scene_blend else None,'lighting':'native_scene' if a.scene_blend else 'hdri_or_constant','device':a.device,'people':people,'sources':records},indent=2)+'\n')

if __name__=='__main__': main()
