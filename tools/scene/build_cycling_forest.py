"""Adapt Poly Haven Pine Forest into a smooth asphalt cycling greenway.

Run with the original forest .blend open. Source assets and older scenes remain
untouched. The output references ../pine_forest/textures and local asphalt maps.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector



def main():
    ROOT=Path(__file__).resolve().parents[2]
    SOURCE=ROOT/'assets/scenes/photoreal/pine_forest'
    DEST=ROOT/'assets/scenes/photoreal/cycling_forest'
    OUT=ROOT/'outputs/scenes/cycling_forest'
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--width',type=int,default=1600)
    parser.add_argument('--skip-render',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    DEST.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'

    files={p.name:p for p in SOURCE.rglob('*') if p.is_file()}
    missing=[]
    for img in bpy.data.images:
        if img.source!='FILE' or img.packed_file:
            continue
        basename=img.filepath.replace('\\','/').split('/')[-1]
        path=files.get(basename)
        if path:
            img.filepath=str(path)
        else:
            missing.append({'name':img.name,'path':img.filepath})
    if missing:
        (OUT/'missing_images.json').write_text(json.dumps(missing,indent=2))
        raise RuntimeError(f'Missing {len(missing)} source textures')

    def center(y):
        return 15+10*math.sin(y/32)

    def derivative(y):
        return 10/32*math.cos(y/32)

    def point(y,lateral=0,z=0):
        tangent=derivative(y)
        normal=Vector((1,-tangent,0)).normalized()
        return Vector((center(y),y,z))+lateral*normal

    def smooth(t):
        t=max(0,min(1,t))
        return t*t*(3-2*t)

    # Remove the original narrow hiking paths and river from this adaptation only.
    terrain=bpy.data.objects['terrain_main']
    for m in list(terrain.modifiers):
        if m.name in ['path','river']:
            terrain.modifiers.remove(m)
    for name in ['path','river_bed','river_water','fog']:
        o=bpy.data.objects.get(name)
        if o:
            o.hide_render=True
            if o.name in bpy.context.view_layer.objects:
                o.hide_set(True)

    # Blend the original topography into a planar corridor. The asphalt is Z=0;
    # its shoulders join the ground at Z=-0.10, with no centerline undulation.
    for v in terrain.data.vertices:
        d=abs(v.co.x-center(v.co.y))
        mix=1-smooth((d-5)/10)
        v.co.z=v.co.z*(1-mix)-.10*mix
    for name in ['path','sides','river','river_rocks','tree_roots']:
        old=terrain.data.attributes.get(name)
        if old:
            terrain.data.attributes.remove(old)
        attr=terrain.data.attributes.new(name,'FLOAT','POINT')
        for v,a in zip(terrain.data.vertices,attr.data):
            distance=abs(v.co.x-center(v.co.y))
            if name=='path':a.value=1-smooth((distance-3.1)/1.0)
            elif name=='sides':a.value=max(0,1-abs(distance-4.8)/1.5)
            else:a.value=0
    terrain.data.update()
    # Keep the original high quality scatter, but remove geometric noise from the
    # support surface so it cannot push through the smooth asphalt.
    landscape=bpy.data.node_groups['NodeGroup']
    landscape.nodes['Value'].outputs[0].default_value=0

    def mathnode(group,operation,first=None,second=None):
        n=group.nodes.new('ShaderNodeMath')
        n.operation=operation
        if first is not None:
            if isinstance(first,(float,int)):n.inputs[0].default_value=first
            else:group.links.new(first,n.inputs[0])
        if second is not None:
            if isinstance(second,(float,int)):n.inputs[1].default_value=second
            else:group.links.new(second,n.inputs[1])
        return n.outputs[0]

    def clearance_field(group,margin):
        pos=group.nodes.new('GeometryNodeInputPosition')
        sep=group.nodes.new('ShaderNodeSeparateXYZ')
        group.links.new(pos.outputs['Position'],sep.inputs[0])
        phase=mathnode(group,'DIVIDE',sep.outputs['Y'],32)
        sine=mathnode(group,'SINE',phase)
        x=mathnode(group,'ADD',mathnode(group,'MULTIPLY',sine,10),15)
        distance=mathnode(group,'ABSOLUTE',mathnode(group,'SUBTRACT',sep.outputs['X'],x))
        return mathnode(group,'GREATER_THAN',distance,margin)

    scatter_groups=[]
    main_group=bpy.data.node_groups['terrain']
    for groupnode in main_group.nodes:
        if groupnode.bl_idname!='GeometryNodeGroup':continue
        group=groupnode.node_tree
        scatters=[n for n in group.nodes if n.bl_idname=='GeometryNodeDistributePointsOnFaces']
        if not scatters:continue
        label=' '.join(str(n.inputs['Collection'].default_value.name) for n in group.nodes
                       if n.bl_idname=='GeometryNodeCollectionInfo' and n.inputs['Collection'].default_value)
        margin=6 if any(w in label for w in ['trees','trunk','dead_tree']) else 4.0
        field=clearance_field(group,margin)
        for node in scatters:
            selection=node.inputs['Selection']
            if selection.is_linked:
                previous=selection.links[0].from_socket
                field=mathnode(group,'MULTIPLY',field,previous)
            group.links.new(field,selection)
        scatter_groups.append({'group':group.name,'asset_collections':label,'clearance_from_center_x':margin})

    # Clear hand-placed source objects conservatively by their world-space bounds.
    bpy.context.view_layer.update()
    hidden=[]
    hand=bpy.data.collections.get('hand_placed_assets')
    if hand:
        for o in hand.all_objects:
            if o.type!='MESH':continue
            corners=[o.matrix_world@Vector(p) for p in o.bound_box]
            xmin,xmax=min(v.x for v in corners),max(v.x for v in corners)
            ymin,ymax=min(v.y for v in corners),max(v.y for v in corners)
            yy=[ymin+(ymax-ymin)*i/20 for i in range(21)]
            if any(xmin-4<center(y)<xmax+4 for y in yy):
                o.hide_render=True
                hidden.append(o.name)

    roadcoll=bpy.data.collections.new('Cycling_Greenway_Road')
    scene.collection.children.link(roadcoll)

    def material(name,color,rough=.8):
        m=bpy.data.materials.new(name)
        m.diffuse_color=(*color,1)
        m.use_nodes=True
        bsdf=m.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value=(*color,1)
        bsdf.inputs['Roughness'].default_value=rough
        return m

    asphalt=material('Scanned asphalt — Poly Haven asphalt_02',(.12,.12,.12))
    n=asphalt.node_tree.nodes
    l=asphalt.node_tree.links
    shader=n.get('Principled BSDF')
    for suffix,socket in [('diff','Base Color'),('rough','Roughness'),('nor_gl',None)]:
        ext='png' if suffix=='nor_gl' else 'jpg'
        tex=n.new('ShaderNodeTexImage')
        tex.image=bpy.data.images.load(str(DEST/'textures'/f'asphalt_02_{suffix}_2k.{ext}'),check_existing=True)
        if suffix!='diff':tex.image.colorspace_settings.name='Non-Color'
        if socket:l.new(tex.outputs['Color'],shader.inputs[socket])
        else:
            normal=n.new('ShaderNodeNormalMap')
            normal.inputs['Strength'].default_value=.5
            l.new(tex.outputs['Color'],normal.inputs['Color'])
            l.new(normal.outputs['Normal'],shader.inputs['Normal'])
    edge=material('Warm white edge marking',(.64,.63,.52),.87)
    yellow=material('Muted yellow center marking',(.65,.39,.06),.85)
    gravel=bpy.data.materials['main_terrain'].copy()
    gravel.name='Forest aggregate shoulders'

    ys=[-96+i*.25 for i in range(769)]
    stations=[0.0]
    for i in range(1,len(ys)):
        stations.append(stations[-1]+(point(ys[i])-point(ys[i-1])).length)

    def strip(name,left,right,mat,zleft=0,zright=0,subset=None):
        values=ys if subset is None else subset
        vertices=[]
        for y in values:
            vertices.extend([point(y,left,zleft),point(y,right,zright)])
        faces=[(i*2,i*2+1,i*2+3,i*2+2) for i in range(len(values)-1)]
        data=bpy.data.meshes.new(name)
        data.from_pydata(vertices,[],faces)
        data.materials.append(mat)
        uv=data.uv_layers.new(name='Road_metre_UV')
        distance=0
        lengths=[0]
        for i in range(1,len(values)):
            distance+=(point(values[i])-point(values[i-1])).length
            lengths.append(distance)
        for polygon in data.polygons:
            for loop_index in polygon.loop_indices:
                vi=data.loops[loop_index].vertex_index
                uv.data[loop_index].uv=((left if vi%2==0 else right)/3,lengths[vi//2]/3)
        o=bpy.data.objects.new(name,data)
        roadcoll.objects.link(o)
        return o

    road=strip('Asphalt_5m_continuous',-2.5,2.5,asphalt)
    strip('Shoulder_left',-3.5,-2.5,gravel,-.10,-.008)
    strip('Shoulder_right',2.5,3.5,gravel,-.008,-.10)
    strip('White_edge_left',-2.33,-2.25,edge,.002,.002)
    strip('White_edge_right',2.25,2.33,edge,.002,.002)
    for start in range(-94,95,6):
        segment=[start+j*.2 for j in range(11)]
        strip('Dashed_center_line',-.045,.045,yellow,.003,.003,segment)

    # Editable path for later rider placement; no rider or bicycle is added.
    curve=bpy.data.curves.new('Cycling_centerline','CURVE')
    curve.dimensions='3D'
    spline=curve.splines.new('POLY')
    spline.points.add(len(ys)-1)
    for p,y in zip(spline.points,ys):p.co=(*point(y),1)
    path=bpy.data.objects.new('Cycling_centerline_DO_NOT_RENDER',curve)
    roadcoll.objects.link(path)
    path.hide_render=True
    path.display_type='WIRE'

    # Use the existing camera as the LOD reference for the original tree nodes.
    camera=scene.camera
    camera.location=point(-43,-.85,1.65)
    camera.rotation_euler=(point(-20,-.3,1.5)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='PERSP'
    camera.data.lens=25
    camera.data.clip_end=1000
    camera.data.dof.use_dof=False
    camera.name='Camera1'
    def addcamera(name,loc,target,pano=False):
        data=bpy.data.cameras.new(name)
        o=bpy.data.objects.new(name,data)
        scene.collection.objects.link(o)
        o.location=loc
        o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
        data.lens=27
        data.clip_end=1000
        if pano:
            data.type='PANO'
            data.panorama_type='EQUIRECTANGULAR'
        return o
    pano=addcamera('Camera_360_Greenway',point(-43,0,1.65),point(-42,0,1.65),True)
    overview=addcamera('Camera_Route_Overview',(80,-80,90),(15,0,0))
    reverse=addcamera('Camera_Reverse',point(25,.8,1.65),point(5,.3,1.5))

    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.adaptive_threshold=.04
    scene.cycles.max_bounces=7
    scene.cycles.transparent_max_bounces=12
    scene.render.threads_mode='FIXED'
    scene.render.threads=16
    scene.render.use_compositing=False
    scene.render.use_sequencer=False
    scene.render.use_border=False
    scene.render.use_crop_to_border=False
    scene.render.film_transparent=False
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.resolution_percentage=100
    scene.render.resolution_x=args.width
    scene.render.resolution_y=round(args.width*9/16)
    scene.render.filepath=str(OUT/'perspective.png')
    scene.camera=camera
    for obj in scene.objects:obj.pass_index=0
    scene['asset_credit']='Pine Forest by Poly Haven / Rob Tuytel and contributing artists, CC0; asphalt_02 by Poly Haven, CC0'
    scene['purpose']='Environment only; flat asphalt cycling greenway, 5m width, Z=0 road surface'
    for image in bpy.data.images:
        if image.source=='FILE' and image.filepath:
            image.filepath=bpy.path.relpath(image.filepath,start=str(DEST))

    with (OUT/'route_centerline.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['distance_m','x_m','y_m','z_m','heading_rad','width_m'])
        for y,d in zip(ys,stations):writer.writerow([d,center(y),y,0,math.atan2(1,derivative(y)),5])
    report={'source':'https://polyhaven.com/collections/pine_forest',
            'road_material':'https://polyhaven.com/a/asphalt_02',
            'length_m':stations[-1],'road_width_m':5,'road_surface_z_m':0,
            'grade_percent':0,'min_curve_radius_m':32**2/10,
            'people':0,'bicycles':0,'missing_textures':missing,
            'scatter_clearance':scatter_groups,'hidden_hand_placed_objects':hidden,
            'samples':args.samples,'camera_location':list(camera.location),
            'bounds_xy':[-103,-100,100,100]}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST/'cycling_forest.blend'),relative_remap=False)
    print('CYCLING_FOREST_SAVED',json.dumps(report),flush=True)
    if not args.skip_render:
        for cam,name,w,h in [(camera,'perspective.png',args.width,round(args.width*9/16)),
                             (pano,'panorama.png',args.width,args.width//2),
                             (overview,'overview.png',1200,900),
                             (reverse,'reverse.png',args.width,round(args.width*9/16))]:
            scene.camera=cam
            scene.render.resolution_x=w
            scene.render.resolution_y=h
            scene.render.filepath=str(OUT/name)
            print('RENDER_START',name,flush=True)
            bpy.ops.render.render(write_still=True)
            print('RENDER_DONE',name,flush=True)
    print('CYCLING_FOREST_COMPLETE',flush=True)


if __name__ == "__main__":
    main()
