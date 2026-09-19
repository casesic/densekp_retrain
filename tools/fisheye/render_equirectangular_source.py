"""Blender helper: render equirectangular source buffers for Omni-Radtan warping."""
import argparse, json, math, sys
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix

def arguments():
    p=argparse.ArgumentParser(); q=sys.argv[sys.argv.index('--')+1:]
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--clothing-geometry',type=Path)
    p.add_argument('--cameras',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--frame',type=int,default=0)
    p.add_argument('--body-texture',type=Path); p.add_argument('--body-uv',type=Path); p.add_argument('--hdri',type=Path); p.add_argument('--width',type=int,default=2048); p.add_argument('--height',type=int,default=1024); p.add_argument('--samples',type=int,default=8)
    return p.parse_args(q)

def main():
    a=arguments(); a.output_dir.mkdir(parents=True,exist_ok=True); vertices=np.load(a.geometry/'vertices.npy'); faces=np.load(a.geometry/'faces.npy'); f=a.frame
    bpy.ops.wm.read_factory_settings(use_empty=True); s=bpy.context.scene; s.render.engine='BLENDER_EEVEE'; s.render.resolution_x=a.width; s.render.resolution_y=a.height; s.render.resolution_percentage=100
    # Eevee cannot render panoramic cameras, so use Cycles with the same static geometry.
    s.render.engine='CYCLES'; s.cycles.samples=a.samples; s.cycles.use_denoising=False; s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGBA'; s.render.film_transparent=False
    s.view_layers[0].use_pass_z=True; s.view_layers[0].use_pass_object_index=True; s.view_settings.look='None'
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
    world=bpy.data.worlds.new('FisheyeWorld'); s.world=world; world.use_nodes=True; nodes=world.node_tree.nodes; links=world.node_tree.links; nodes.clear(); wo=nodes.new('ShaderNodeOutputWorld'); bg=nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value=1
    if a.hdri:
        env=nodes.new('ShaderNodeTexEnvironment'); env.image=bpy.data.images.load(str(a.hdri.resolve())); env.image.colorspace_settings.name='Linear Rec.709'; links.new(env.outputs['Color'],bg.inputs['Color'])
    else: bg.inputs['Color'].default_value=(.08,.1,.14,1)
    links.new(bg.outputs['Background'],wo.inputs['Surface'])
    mesh=bpy.data.meshes.new('SMPLX_Fisheye'); mesh.from_pydata(vertices[f].tolist(),[],faces.tolist()); body=bpy.data.objects.new('SMPLX_Fisheye',mesh); s.collection.objects.link(body); body.pass_index=1
    mat=bpy.data.materials.new('Body'); mat.use_nodes=True; bs=mat.node_tree.nodes.get('Principled BSDF'); bs.inputs['Roughness'].default_value=.58; bs.inputs['Base Color'].default_value=(.55,.25,.16,1); mesh.materials.append(mat)
    if a.body_texture and a.body_uv:
        uv=np.load(a.body_uv); layer=mesh.uv_layers.new(name='SMPLX_UV')
        for polygon,indices in zip(mesh.polygons,uv['faces_uv']):
            for loop,index in zip(polygon.loop_indices,indices): layer.data[loop].uv=uv['uv'][index]
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage'); tex.image=bpy.data.images.load(str(a.body_texture.resolve())); tex.image.colorspace_settings.name='sRGB'; mat.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
    conv=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1))); body.matrix_world=Matrix.Translation((0,0,-float(vertices[f,:,1].min())))@conv
    if a.clothing_geometry:
        cg=a.clothing_geometry; cv=np.load(cg/'clothing_vertices.npy'); cf=np.load(cg/'clothing_faces.npy'); cm=bpy.data.meshes.new('Clothing'); cm.from_pydata(cv[f].tolist(),[],cf.tolist()); co=bpy.data.objects.new('Clothing',cm); s.collection.objects.link(co); co.pass_index=2; co.matrix_world=body.matrix_world
        cmat=bpy.data.materials.new('ClothingMaterial'); cmat.use_nodes=True; cb=cmat.node_tree.nodes.get('Principled BSDF'); cb.inputs['Base Color'].default_value=(.04,.35,.5,1); cb.inputs['Roughness'].default_value=.6; cm.materials.append(cmat)
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
    (a.output_dir/'source_manifest.json').write_text(json.dumps({'projection':'equirectangular','resolution':[a.width,a.height],'frame':f,'sources':records},indent=2)+'\n')

if __name__=='__main__': main()
