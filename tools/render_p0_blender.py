"""Blender 后台单进程渲染 P0 RGB/Z/IndexOB，并验证投影。"""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import bpy, numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eye_background_render import procedural_eyes, save_linear, shadow_composite

def args_():
    p=argparse.ArgumentParser(); a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p.add_argument('--input-geometry',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--engine',choices=['BLENDER_EEVEE','CYCLES'],default='BLENDER_EEVEE'); p.add_argument('--device',default='OPTIX'); p.add_argument('--samples',type=int,default=8); p.add_argument('--resolution',type=int,default=512); p.add_argument('--appearance-baseline',action='store_true'); p.add_argument('--clothing-geometry',type=Path); p.add_argument('--clothing-texture-id','--clothing_texture_id'); p.add_argument('--clothing-texture-dir',type=Path); p.add_argument('--clothing-uv-vflip',action='store_true'); p.add_argument('--render-frame-count',type=int,default=16)
    p.add_argument('--body-texture',type=Path); p.add_argument('--body-uv',type=Path)
    p.add_argument('--environment-mode','--environment_mode',choices=['simple','hdri'],default='simple')
    p.add_argument('--hdri-path','--hdri_path',type=Path)
    p.add_argument('--hdri-strength','--hdri_strength',type=float,default=1.0)
    p.add_argument('--hdri-rotation-deg','--hdri_rotation_deg',type=float,default=0.0)
    p.add_argument('--hdri-visible-background','--hdri_visible_background',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--manual-lights','--manual_lights',choices=['studio','fill','none'],default='studio')
    p.add_argument('--eye-render-mode','--eye_render_mode',choices=['legacy','neutral','official','procedural'],default='legacy')
    p.add_argument('--eye-texture','--eye_texture',type=Path)
    p.add_argument('--eye-mapping',type=Path)
    p.add_argument('--ground-mode','--ground_mode',choices=['visible','shadow_catcher'],default='visible')
    p.add_argument('--camera-placement',type=Path,help='Optional deterministic per-frame camera matrices; does not alter intrinsics or local vertices')
    return p.parse_args(a)
def ml(m): return [[float(v) for v in row] for row in m]
def outputs(scene,dd,md=None):
    vl=scene.view_layers[0]; vl.use_pass_z=True; vl.use_pass_object_index=True
    if hasattr(scene,'compositing_node_group'):
        nt=bpy.data.node_groups.new(scene.name+'_Comp','CompositorNodeTree'); scene.compositing_node_group=nt
    else:
        scene.use_nodes=True; nt=scene.node_tree; nt.nodes.clear()
    rl=nt.nodes.new('CompositorNodeRLayers'); rl.scene=scene; dn=nt.nodes.new('CompositorNodeOutputFile')
    if hasattr(dn,'directory'):
        dn.directory=str(dd); dn.file_name='passes'; dn.format.file_format='OPEN_EXR_MULTILAYER'; dn.format.color_depth='32'; dn.file_output_items.new('FLOAT','Depth'); dn.file_output_items.new('FLOAT','Mask')
    else:
        dn.base_path=str(dd); dn.format.file_format='OPEN_EXR_MULTILAYER'; dn.format.color_depth='32'; dn.file_slots.clear(); dn.file_slots.new('Depth'); dn.file_slots.new('Mask')
    nt.links.new(rl.outputs['Depth'],dn.inputs['Depth']); nt.links.new(rl.outputs['Alpha'],dn.inputs['Mask']); return dn
def output_name(dn,name):
    if hasattr(dn,'file_name'): dn.file_name=name
    else: dn.file_slots[0].path=name
def rename_one(d,prefix,suffix,target):
    m=sorted(d.glob(prefix+'*'+suffix))
    if len(m)!=1: raise RuntimeError(f'无法唯一定位 {d}/{prefix}*{suffix}: {m}')
    m[0].replace(target)
def plane_test(out,w,h,lens,sensor):
    s=bpy.data.scenes.new('DepthSemantics'); s.render.engine='BLENDER_EEVEE'; s.render.resolution_x=w; s.render.resolution_y=h; s.render.resolution_percentage=100; s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGBA'; s.render.film_transparent=True
    cd=bpy.data.cameras.new('DepthCamera'); cd.lens=lens; cd.sensor_width=cd.sensor_height=sensor; cd.sensor_fit='HORIZONTAL'; c=bpy.data.objects.new('DepthCamera',cd); s.collection.objects.link(c); s.camera=c; c.matrix_world=Matrix.Identity(4)
    me=bpy.data.meshes.new('KnownPlane'); me.from_pydata([(-2,-2,-2),(2,-2,-2),(2,2,-2),(-2,2,-2)],[],[(0,1,2),(0,2,3)]); ob=bpy.data.objects.new('KnownPlane',me); ob.pass_index=1; s.collection.objects.link(ob)
    td=out/'depth_test_raw'; td.mkdir(exist_ok=True); dn=outputs(s,td); output_name(dn,'known_plane'); s.render.filepath=str(td/'known_plane_rgb.png'); s.frame_set(0); bpy.context.window.scene=s; bpy.ops.render.render(write_still=True)
    rename_one(td,'known_plane','.exr',out/'known_plane_passes.exr')
    cfg={'plane_camera_z_m':2.0,'sample_pixel':[3*w//4,h//2],'resolution':[w,h],'lens_mm':lens,'sensor_width_mm':sensor,'fx':lens/sensor*w,'depth_file':'known_plane_passes.exr'}; (out/'depth_test_config.json').write_text(json.dumps(cfg,indent=2)+'\n'); bpy.data.scenes.remove(s)
def main():
    a=args_(); gd=a.input_geometry.resolve(); out=a.output_dir.resolve()
    placement=json.loads(a.camera_placement.read_text()) if a.camera_placement else None
    if a.eye_render_mode=='procedural' and (not a.body_texture or not a.eye_mapping):
        raise RuntimeError('Procedural eyes require body texture, verified body UV and --eye-mapping')
    if a.ground_mode=='shadow_catcher' and a.environment_mode!='hdri':
        raise RuntimeError('Shadow composite requires HDRI setup')
    for n in ['rgb','depth','mask','masks/person','annotations','overlay','crosscheck','depth_raw','mask_raw']: (out/n).mkdir(parents=True,exist_ok=True)
    vs=np.load(gd/'vertices.npy'); fs=np.load(gd/'faces.npy'); dk=np.load(gd/'densekp_3d.npy'); geometry_fingerprint_before=hashlib.blake2b(vs.tobytes(),digest_size=16).hexdigest()
    gm=json.loads((gd/'geometry_manifest.json').read_text()); avatar_id=gm.get('avatar_id','smplx_neutral_zero_betas'); motion_id=gm.get('motion_id','procedural_p0_yaw_left_arm')
    dressed=a.clothing_geometry is not None; styled=a.appearance_baseline or dressed; clothing_manifest=None
    if dressed:
        cg=a.clothing_geometry.resolve(); cvs=np.load(cg/'clothing_vertices.npy'); cfs=np.load(cg/'clothing_faces.npy'); cuv=np.load(cg/'clothing_vertices_uv.npy'); cfuv=np.load(cg/'clothing_faces_uv.npy'); clothing_manifest=json.loads((cg/'clothing_manifest.json').read_text()); clothing_fingerprint_before=hashlib.blake2b(cvs.tobytes(),digest_size=16).hexdigest()
        if cvs.shape!=(len(vs),13027,3) or cfs.shape!=(26328,3): raise RuntimeError(f'clothing geometry shape 错误 {cvs.shape} {cfs.shape}')
    if vs.ndim!=3 or vs.shape[1:]!=(10475,3) or dk.shape!=(len(vs),138,3): raise RuntimeError(f'geometry shape 错误 {vs.shape} {dk.shape}')
    st=time.perf_counter(); bpy.ops.wm.read_factory_settings(use_empty=True); s=bpy.context.scene; s.name='DenseKP_P0'; s.render.engine=a.engine; devices=[]
    if placement: s['shadow_receiver_per_frame']=True
    if a.engine=='CYCLES':
        if a.device=='CPU': s.cycles.device='CPU'
        else:
            pr=bpy.context.preferences.addons['cycles'].preferences; pr.compute_device_type=a.device; pr.get_devices()
            for d in pr.devices: d.use=d.type==a.device; devices.append({'name':d.name,'type':d.type,'use':bool(d.use)})
            s.cycles.device='GPU'
        s.cycles.samples=a.samples
    if a.environment_mode=='hdri' and (not a.hdri_path or not a.hdri_path.is_file()):
        raise RuntimeError('HDRI mode requires an existing --hdri-path')
    w=h=a.resolution; s.render.resolution_x=w; s.render.resolution_y=h; s.render.resolution_percentage=100; s.render.pixel_aspect_x=s.render.pixel_aspect_y=1.; s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGBA'; s.render.film_transparent=True; s.world=bpy.data.worlds.new('P0World'); s.world.color=(.055,.075,.11)
    if styled:
        s.world.use_nodes=True; bg=s.world.node_tree.nodes.get('Background'); bg.inputs['Color'].default_value=(.035,.075,.14,1); bg.inputs['Strength'].default_value=.32
    hdri_colorspace=None
    if a.environment_mode=='hdri':
        nodes=s.world.node_tree.nodes; links=s.world.node_tree.links; nodes.clear()
        out_world=nodes.new('ShaderNodeOutputWorld'); env_bg=nodes.new('ShaderNodeBackground'); env_bg.inputs['Strength'].default_value=a.hdri_strength
        env=nodes.new('ShaderNodeTexEnvironment'); env.image=bpy.data.images.load(str(a.hdri_path.resolve())); hdri_colorspace=env.image.colorspace_settings.name
        coord=nodes.new('ShaderNodeTexCoord'); mapping=nodes.new('ShaderNodeMapping'); mapping.inputs['Rotation'].default_value[2]=math.radians(a.hdri_rotation_deg)
        links.new(coord.outputs['Generated'],mapping.inputs['Vector']); links.new(mapping.outputs['Vector'],env.inputs['Vector']); links.new(env.outputs['Color'],env_bg.inputs['Color'])
        if a.hdri_visible_background:
            links.new(env_bg.outputs['Background'],out_world.inputs['Surface'])
        else:
            simple_bg=nodes.new('ShaderNodeBackground'); simple_bg.inputs['Color'].default_value=(.035,.075,.14,1); simple_bg.inputs['Strength'].default_value=.32
            light_path=nodes.new('ShaderNodeLightPath'); mix=nodes.new('ShaderNodeMixShader')
            links.new(light_path.outputs['Is Camera Ray'],mix.inputs[0]); links.new(env_bg.outputs['Background'],mix.inputs[1]); links.new(simple_bg.outputs['Background'],mix.inputs[2]); links.new(mix.outputs[0],out_world.inputs['Surface'])
    me=bpy.data.meshes.new('SMPLX_FinalVertices'); me.from_pydata(vs[0].tolist(),[],fs.tolist()); ob=bpy.data.objects.new('SMPLX_FinalVertices',me); s.collection.objects.link(ob); ob.pass_index=1; mat=bpy.data.materials.new('P0SkinMaterial' if styled else 'P0BodyMaterial'); mat.diffuse_color=(.48,.20,.11,1) if styled else (.62,.32,.18,1)
    if styled:
        mat.use_nodes=True; bsdf=mat.node_tree.nodes.get('Principled BSDF'); bsdf.inputs['Base Color'].default_value=(.48,.20,.11,1); bsdf.inputs['Roughness'].default_value=.58
    me.materials.append(mat)
    if a.body_texture:
        if not a.body_uv: raise RuntimeError('body texture requires verified UV archive')
        bu=np.load(a.body_uv); uv=bu['uv']; uf=bu['faces_uv']; labels=bu['component_labels']
        if not np.array_equal(fs,bu['faces']) or uf.shape!=fs.shape or uf.min()<0 or uf.max()>=len(uv) or not np.isfinite(uv).all():
            raise RuntimeError('Body UV topology/index verification failed')
        if sorted(np.bincount(labels).tolist())!=[546,546,9383]: raise RuntimeError('Unexpected components')
        body_component=int(np.argmax(np.bincount(labels)))
        skin=bpy.data.materials.new('OfficialBodySkin'); skin.use_nodes=True
        sb=skin.node_tree.nodes.get('Principled BSDF'); sb.inputs['Metallic'].default_value=0.; sb.inputs['Roughness'].default_value=.58; sb.inputs['Alpha'].default_value=1.
        tex=skin.node_tree.nodes.new('ShaderNodeTexImage'); tex.image=bpy.data.images.load(str(a.body_texture.resolve())); tex.image.colorspace_settings.name='sRGB'
        skin.node_tree.links.new(tex.outputs['Color'],sb.inputs['Base Color']); me.materials.append(skin)
        if a.eye_render_mode=='official':
            if not a.eye_texture or not a.eye_texture.is_file(): raise RuntimeError('Official eye mode requires --eye-texture')
            eye_tex=skin.node_tree.nodes.new('ShaderNodeTexImage'); eye_tex.image=bpy.data.images.load(str(a.eye_texture.resolve())); eye_tex.image.colorspace_settings.name='sRGB'
            darken=skin.node_tree.nodes.new('ShaderNodeMixRGB'); darken.blend_type='MULTIPLY'; darken.inputs[0].default_value=1.; darken.inputs[2].default_value=(.2,.2,.2,1)
            overlay=skin.node_tree.nodes.new('ShaderNodeMixRGB')
            skin.node_tree.links.new(eye_tex.outputs['Color'],darken.inputs[1]); skin.node_tree.links.new(eye_tex.outputs['Alpha'],overlay.inputs[0]); skin.node_tree.links.new(tex.outputs['Color'],overlay.inputs[1]); skin.node_tree.links.new(darken.outputs['Color'],overlay.inputs[2]); skin.node_tree.links.new(overlay.outputs['Color'],sb.inputs['Base Color'])
        layer=me.uv_layers.new(name='Verified_SMPLX_UV')
        for polygon,indices in zip(me.polygons,uf):
            for loop,index in zip(polygon.loop_indices,indices): layer.data[loop].uv=uv[index]
            if np.all(labels[fs[polygon.index]]==body_component): polygon.material_index=1
        if a.eye_render_mode in ('neutral','official'):
            eye=bpy.data.materials.new('NeutralEyeFallback'); eye.use_nodes=True
            eb=eye.node_tree.nodes.get('Principled BSDF')
            eb.inputs['Base Color'].default_value=(.72,.74,.76,1); eb.inputs['Metallic'].default_value=0.; eb.inputs['Roughness'].default_value=.3
            if 'IOR Level' in eb.inputs: eb.inputs['IOR Level'].default_value=.25
            me.materials.append(eye)
            eye_material_index=len(me.materials)-1
            for polygon in me.polygons:
                if np.all(labels[fs[polygon.index]]!=body_component): polygon.material_index=eye_material_index
        if a.eye_render_mode=='procedural':
            procedural_eyes(me,fs,labels,body_component,a.eye_mapping)

    conv=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1))); ob.matrix_world=Matrix.Translation((0,0,-float(vs[0,:,1].min())))@conv
    cob=None; cme=None
    if dressed:
        cme=bpy.data.meshes.new('BEDLAM_Clothing'); cme.from_pydata(cvs[0].tolist(),[],cfs.tolist()); cob=bpy.data.objects.new('BEDLAM_Clothing',cme); s.collection.objects.link(cob); cob.pass_index=2; cob.matrix_world=ob.matrix_world
        uv_layer=cme.uv_layers.new(name='BEDLAM_UV')
        for polygon,uv_indices in zip(cme.polygons,cfuv):
            for loop_index,uv_index in zip(polygon.loop_indices,uv_indices):
                u,v=cuv[uv_index]; uv_layer.data[loop_index].uv=(float(u),float(1-v) if a.clothing_uv_vflip else float(v))
        cmat=bpy.data.materials.new('P0ClothingMaterial'); cmat.use_nodes=True; cbsdf=cmat.node_tree.nodes.get('Principled BSDF'); cbsdf.inputs['Roughness'].default_value=.58; cbsdf.inputs['Metallic'].default_value=0.; cme.materials.append(cmat)
        if a.clothing_texture_id:
            if not a.clothing_texture_dir: raise RuntimeError('指定 clothing texture id 时必须提供 texture dir')
            diffuse=(a.clothing_texture_dir/f'{a.clothing_texture_id}_diffuse_1001.png').resolve(); normal=(a.clothing_texture_dir/f'{a.clothing_texture_id}_normal_1001.png').resolve()
            if not diffuse.is_file() or not normal.is_file(): raise RuntimeError(f'clothing texture 缺失: {diffuse}, {normal}')
            nodes=cmat.node_tree.nodes; links=cmat.node_tree.links; di=nodes.new('ShaderNodeTexImage'); di.image=bpy.data.images.load(str(diffuse)); di.image.colorspace_settings.name='sRGB'; ni=nodes.new('ShaderNodeTexImage'); ni.image=bpy.data.images.load(str(normal)); ni.image.colorspace_settings.name='Non-Color'; nm=nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value=1.; links.new(di.outputs['Color'],cbsdf.inputs['Base Color']); links.new(ni.outputs['Color'],nm.inputs['Color']); links.new(nm.outputs['Normal'],cbsdf.inputs['Normal'])
        else: cbsdf.inputs['Base Color'].default_value=(.035,.16,.42,1)
    if ob.modifiers or ob.data.shape_keys or ob.parent: raise RuntimeError('禁止几何修改器/shape key/parent')
    cd=bpy.data.cameras.new('P0PerspectiveCamera'); cd.type='PERSP'; cd.lens=50.; cd.sensor_width=cd.sensor_height=36.; cd.sensor_fit='HORIZONTAL'; cd.shift_x=cd.shift_y=0.; cd.clip_start=.05; cd.clip_end=100.; cam=bpy.data.objects.new('P0PerspectiveCamera',cd); s.collection.objects.link(cam); s.camera=cam; cam.location=(0,-3.2,1.); cam.rotation_euler=(math.pi/2,0,0)
    light_specs=[]
    if a.manual_lights=='studio':
        light_specs=[('Key',600.,(-2,-2,4),4.),('Fill',300.,(2,-1,2),3.)]
        if styled: light_specs.append(('Rim',420.,(0,2.5,3),2.5))
    elif a.manual_lights=='fill':
        light_specs=[('Fill',90.,(2,-1,2),3.)]
    for n,e,loc,sz in light_specs:
        ld=bpy.data.lights.new(n,'AREA'); ld.energy=e; ld.shape='DISK'; ld.size=sz; lo=bpy.data.objects.new(n,ld); s.collection.objects.link(lo); lo.location=loc
    ground=None
    if styled:
        ge=bpy.data.meshes.new('P0Ground'); ge.from_pydata([(-10,-10,-.005),(10,-10,-.005),(10,10,-.005),(-10,10,-.005)],[],[(0,1,2),(0,2,3)]); ground=bpy.data.objects.new('P0Ground',ge); s.collection.objects.link(ground); ground.hide_render=True
        gmat=bpy.data.materials.new('P0GroundMaterial'); gmat.use_nodes=True; gbsdf=gmat.node_tree.nodes.get('Principled BSDF'); gbsdf.inputs['Base Color'].default_value=(.12,.16,.20,1); gbsdf.inputs['Roughness'].default_value=.82; ge.materials.append(gmat)
    dn=outputs(s,out/'depth_raw'); plane_test(out,w,h,50.,36.); bpy.context.window.scene=s
    frame_count=a.render_frame_count
    if not 1<=frame_count<=len(vs): raise RuntimeError(f'render frame count 无效: {frame_count}, geometry={len(vs)}')
    fx=fy=cd.lens/cd.sensor_width*w; cx=cy=w/2; pix=np.empty((frame_count,138,2),np.float32); camxyz=np.empty((frame_count,138,3),np.float32); diffs=[]; times=[]; rec=[]; maxdiff=0.
    api_pix=np.empty_like(pix); ray_checks=[]
    for f in range(frame_count):
        me.vertices.foreach_set('co',vs[f].reshape(-1)); me.update(); back=np.empty_like(vs[f]); me.vertices.foreach_get('co',back.reshape(-1)); d=float(np.max(np.abs(back-vs[f]))); maxdiff=max(maxdiff,d)
        if d!=0.: raise RuntimeError(f'frame {f} mesh source diff {d}')
        if dressed: cme.vertices.foreach_set('co',cvs[f].reshape(-1)); cme.update()
        if placement:
            assert placement['frames'][f]['frame_id']==f
            cam.matrix_world=Matrix(placement['frames'][f]['camera_matrix_world'])
            bpy.context.view_layer.update()
            back_clothing=np.empty_like(cvs[f]); cme.vertices.foreach_get('co',back_clothing.reshape(-1))
            assert np.array_equal(back_clothing,cvs[f])
        s.frame_set(f); s.render.filepath=str(out/'rgb'/f'{f:06d}.png'); output_name(dn,f'{f:06d}'); t=time.perf_counter(); bpy.ops.render.render(write_still=True)
        separate_ground_pass=dressed or a.ground_mode=='shadow_catcher'
        rename_one(out/'depth_raw',f'{f:06d}','.exr',(out/'mask_raw' if separate_ground_pass else out/'depth_raw')/f'{f:06d}.exr')
        if a.ground_mode=='shadow_catcher':
            (out/'appearance_raw').mkdir(exist_ok=True)
            save_linear(s,out/'appearance_raw'/f'{f:06d}_foreground.exr')
        if dressed:
            ground.hide_render=False; s.render.film_transparent=False; output_name(dn,f'{f:06d}'); bpy.ops.render.render(write_still=True); rename_one(out/'depth_raw',f'{f:06d}','.exr',out/'depth_raw'/f'{f:06d}.exr')
            if a.ground_mode=='shadow_catcher':
                shadow_info=shadow_composite(s,dn,ob,cob,ground,out,f)
            s.render.film_transparent=True; ground.hide_render=True
        elif a.ground_mode=='shadow_catcher':
            ground.hide_render=False; s.render.film_transparent=False; output_name(dn,f'{f:06d}'); bpy.ops.render.render(write_still=True)
            rename_one(out/'depth_raw',f'{f:06d}','.exr',out/'depth_raw'/f'{f:06d}.exr')
            shadow_info=shadow_composite(s,dn,ob,None,ground,out,f)
            s.render.film_transparent=True; ground.hide_render=True
        elif a.appearance_baseline:
            # 几何 pass 保持透明背景且不含地面；第二次仅覆盖 RGB，不改变深度/掩码监督。
            dn.mute=True; ground.hide_render=False; s.render.film_transparent=False; bpy.ops.render.render(write_still=True); s.render.film_transparent=True; ground.hide_render=True; dn.mute=False
        times.append(time.perf_counter()-t)
        ci=cam.matrix_world.inverted()
        for j,p in enumerate(dk[f]):
            world=ob.matrix_world@Vector(p.tolist()); q=ci@world; camxyz[f,j]=q; z=-float(q.z); an=np.array([fx*float(q.x)/z+cx,cy-fy*float(q.y)/z]); nd=world_to_camera_view(s,cam,world); api=np.array([float(nd.x)*w,(1-float(nd.y))*h]); pix[f,j]=an; diffs.append(float(np.linalg.norm(an-api)))
            api_pix[f,j]=api
        if placement:
            # Independent scene-ray samples at exact pixel centers, not GT rays.
            ground.hide_render=False; bpy.context.view_layer.update(); deps=bpy.context.evaluated_depsgraph_get()
            for py in range(8,h,16):
                for px in range(8,w,16):
                    direction=cam.matrix_world.to_3x3()@Vector(((px+.5-cx)/fx,(cy-py-.5)/fy,-1.))
                    direction.normalize()
                    hit,location,normal,index,obj,matrix=s.ray_cast(deps,cam.matrix_world.translation,direction,distance=cd.clip_end)
                    ray_checks.append({'frame':f,'x':px,'y':py,'hit':bool(hit),'depth':float(-(ci@location).z) if hit else None,'object':obj.name if hit else None})
            ground.hide_render=True
        row={'frame_id':f,'render_frame':f,'geometry_frame_id':f,'avatar_id':avatar_id,'motion_id':motion_id,'body_model':gm.get('model_dir'),'resolution_width':w,'resolution_height':h,'lens_mm':float(cd.lens),'sensor_width_mm':float(cd.sensor_width),'sensor_height_mm':float(cd.sensor_height),'sensor_fit':cd.sensor_fit,'pixel_aspect':1.,'shift_x':0.,'shift_y':0.,'near_clip':float(cd.clip_start),'far_clip':float(cd.clip_end),'camera_location_world':[float(x) for x in cam.location],'camera_rotation_world':[float(x) for x in cam.rotation_euler],'camera_matrix_world':ml(cam.matrix_world),'object_location_world':[float(x) for x in ob.location],'object_rotation_world':[float(x) for x in ob.rotation_euler],'object_scale_world':[float(x) for x in ob.scale],'object_matrix_world':ml(ob.matrix_world),'projection':'perspective_pinhole','intrinsics':{'fx':fx,'fy':fy,'cx':cx,'cy':cy},'rgb_path':f'rgb/{f:06d}.png','depth_path':f'depth/{f:06d}.npy','depth_raw_path':f'depth_raw/{f:06d}.exr','mask_path':f'masks/person/{f:06d}.png' if dressed else f'mask/{f:06d}.png','person_mask_path':f'masks/person/{f:06d}.png' if dressed else f'mask/{f:06d}.png','overlay_path':f'overlay/{f:06d}.png','densekp_3d_path':'geometry/densekp_3d.npy','densekp_3d_frame_index':f,'densekp_2d_path':'annotations/densekp_2d.npy','gender':gm.get('gender')}
        if dressed: row.update({'subject_id':clothing_manifest['subject_id'],'animation_id':clothing_manifest['animation_id'],'clothing_id':clothing_manifest['clothing_id'],'clothing_frame_index':clothing_manifest['clothing_frame_indices'][f],'local_motion_index':clothing_manifest['local_motion_indices'][f],'source_frame_30fps':clothing_manifest['source_frame_30fps'][f],'clothing_source_path':clothing_manifest['clothing_source'],'clothing_texture_id':a.clothing_texture_id,'clothing_uv_vflip':a.clothing_uv_vflip,'body_geometry_fingerprint':geometry_fingerprint_before})
        if a.body_texture: row.update({'body_texture_id':a.body_texture.stem,'body_diffuse_texture_path':str(a.body_texture.resolve()),'body_uv_path':str(a.body_uv.resolve()),'eye_texture_integrated':a.eye_render_mode=='official','eye_texture_path':str(a.eye_texture.resolve()) if a.eye_texture else None,'eye_status':{'legacy':'BLOCKED_UV_ATLAS_MISMATCH','neutral':'NEUTRAL_FALLBACK','official':'OFFICIAL_RGBA_BODY_UV_OVERLAY','procedural':'PROCEDURAL_NEUTRAL_EYE'}[a.eye_render_mode],'eye_render_mode':a.eye_render_mode})
        row['ground_mode']=a.ground_mode
        if placement:
            row['camera_stress_config']=placement['config']
            row['placement_solver']=placement['frames'][f]
        if a.ground_mode=='shadow_catcher': row['shadow_composite']=shadow_info
        row.update({'environment_mode':a.environment_mode,'hdri_path':str(a.hdri_path.resolve()) if a.hdri_path else None,'hdri_strength':a.hdri_strength if a.environment_mode=='hdri' else None,'hdri_rotation_deg':a.hdri_rotation_deg if a.environment_mode=='hdri' else None,'hdri_visible_background':a.hdri_visible_background if a.environment_mode=='hdri' else None,'manual_lights':a.manual_lights})
        rec.append(row)
    np.save(out/'annotations/densekp_2d.npy',pix); np.save(out/'annotations/densekp_camera_xyz.npy',camxyz); (out/'manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rec))
    if placement:
        np.save(out/'annotations/blender_projection_2d.npy',api_pix)
        (out/'ray_depth_checks.json').write_text(json.dumps(ray_checks)+'\n')
    geometry_fingerprint_after=hashlib.blake2b(vs.tobytes(),digest_size=16).hexdigest(); clothing_fingerprint_after=hashlib.blake2b(cvs.tobytes(),digest_size=16).hexdigest() if dressed else None; de=np.asarray(diffs); report={'frames':frame_count,'engine':a.engine,'cycles_devices':devices,'appearance_baseline':styled,'dressed':dressed,'clothing_texture_id':a.clothing_texture_id,'clothing_uv_vflip':a.clothing_uv_vflip,'eye_render_mode':a.eye_render_mode,'environment':{'mode':a.environment_mode,'hdri_path':str(a.hdri_path.resolve()) if a.hdri_path else None,'strength':a.hdri_strength if a.environment_mode=='hdri' else None,'rotation_deg':a.hdri_rotation_deg if a.environment_mode=='hdri' else None,'visible_background':a.hdri_visible_background if a.environment_mode=='hdri' else None,'image_colorspace':hdri_colorspace,'manual_lights':a.manual_lights,'ground_plane':bool(styled)},'color_management':{'display_device':s.display_settings.display_device,'view_transform':s.view_settings.view_transform,'look':s.view_settings.look,'exposure':float(s.view_settings.exposure),'gamma':float(s.view_settings.gamma)},'single_geometry_source':maxdiff==0. and geometry_fingerprint_before==geometry_fingerprint_after,'max_mesh_source_diff':maxdiff,'geometry_path':str(gd/'vertices.npy'),'geometry_fingerprint':{'algorithm':'blake2b-128','before':geometry_fingerprint_before,'after':geometry_fingerprint_after,'identical':geometry_fingerprint_before==geometry_fingerprint_after},'clothing_geometry_fingerprint':{'algorithm':'blake2b-128','before':clothing_fingerprint_before,'after':clothing_fingerprint_after,'identical':clothing_fingerprint_before==clothing_fingerprint_after} if dressed else None,'smpl_to_blender_matrix':ml(conv),'projection_error_px':{'mean':float(de.mean()),'median':float(np.median(de)),'p95':float(np.percentile(de,95)),'max':float(de.max())},'render_seconds':{'mean':float(np.mean(times)),'total':float(np.sum(times)),'values':times},'process_total_seconds':time.perf_counter()-st,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.}
    (out/'p0_render_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
