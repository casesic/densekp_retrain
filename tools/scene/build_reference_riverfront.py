"""Blender background script: editable riverfront inspired by assets/example/00001.jpg.

blender -b --python tools/scene/build_reference_riverfront.py
No external textures or add-ons. Coordinates: meters, Z up, promenade along Y.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=24)
parser.add_argument('--width', type=int, default=1600)
parser.add_argument('--skip-render', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
OUT = ROOT / 'assets/scenes/riverfront'
PREVIEW = ROOT / 'outputs/scenes/riverfront'
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
random.seed(41)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'

def collection(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c

land = collection('01_Ground_and_river')
arch = collection('02_Railings_and_tower')
plants = collection('03_Trees_and_planting')
city = collection('04_Residential_skyline')
props = collection('05_Park_furniture')
cams = collection('06_Cameras_and_lighting')

def move(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj

def material(name, color, rough=.7, noise=0, scale=8):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    n = m.node_tree.nodes
    p = n.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = rough
    if noise:
        tex = n.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = scale
        tex.inputs['Detail'].default_value = 3
        geometry = n.new('ShaderNodeNewGeometry')
        m.node_tree.links.new(geometry.outputs['Position'], tex.inputs['Vector'])
        ramp = n.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].color = (*(v * (1-noise) for v in color), 1)
        ramp.color_ramp.elements[1].color = (*(min(1, v * (1+noise)) for v in color), 1)
        m.node_tree.links.new(tex.outputs['Fac'], ramp.inputs[0])
        m.node_tree.links.new(ramp.outputs[0], p.inputs['Base Color'])
        bump = n.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .22
        bump.inputs['Distance'].default_value = .035
        m.node_tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
        m.node_tree.links.new(bump.outputs[0], p.inputs['Normal'])
    return m

stone = [material('Limestone_%02d'%i, (.40+i*.017,.37+i*.016,.31+i*.015), noise=.24, scale=5) for i in range(6)]
grout = material('Paving joints', (.18,.18,.16))
curb = material('Grey granite edging', (.30,.32,.32), noise=.25)
grass = material('Dry olive park grass', (.29,.32,.10), noise=.65, scale=1.8)
bark = material('Tree bark', (.18,.12,.075), noise=.4, scale=12)
leaves = [material('Leaf_%d'%i, c, rough=.85) for i,c in enumerate([(.12,.22,.045),(.23,.32,.07),(.32,.35,.09),(.07,.17,.04)])]
brown = material('Warm brown painted railing', (.19,.085,.064), rough=.48, noise=.1)
support = material('Galvanized tree supports', (.36,.40,.39), rough=.4)
cream = material('Tower ivory plaster', (.69,.68,.60), noise=.10)
roof = material('Weathered grey dome', (.30,.31,.29), noise=.2)
glass = material('Desaturated blue window glass', (.11,.21,.25), rough=.25)
concrete = material('Residential facade', (.49,.48,.42), noise=.14)
trim = material('Pale facade grid', (.69,.68,.60))
orange = material('Lifebuoy orange', (.8,.055,.012), rough=.4)
white = material('Lifebuoy white bands', (.84,.82,.71))

def mesh(name, vertices, faces, mat, coll):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.materials.append(mat)
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    return obj

def boxes(name, specs, mat, coll):
    verts, faces = [], []
    for center, size in specs:
        x,y,z = center
        a,b,c = (v/2 for v in size)
        k = len(verts)
        verts.extend([(x-a,y-b,z-c),(x+a,y-b,z-c),(x+a,y+b,z-c),(x-a,y+b,z-c),
                      (x-a,y-b,z+c),(x+a,y-b,z+c),(x+a,y+b,z+c),(x-a,y+b,z+c)])
        faces.extend([tuple(k+i for i in f) for f in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]])
    return mesh(name, verts, faces, mat, coll)

def box(name, center, size, mat, coll):
    return boxes(name, [(center,size)], mat, coll)

def rod(name, start, end, radius, mat, coll, r2=None):
    d = Vector(end)-Vector(start)
    rotation = d.to_track_quat('Z','Y')
    count = 32 if radius > .5 else 8
    vertices = []
    for center, r in [(Vector(start),radius),(Vector(end),radius if r2 is None else r2)]:
        for i in range(count):
            a=2*math.pi*i/count
            vertices.append(center+rotation@Vector((r*math.cos(a),r*math.sin(a),0)))
    faces=[tuple(reversed(range(count))),tuple(range(count,2*count))]
    faces.extend((i,(i+1)%count,(i+1)%count+count,i+count) for i in range(count))
    return mesh(name,vertices,faces,mat,coll)

# A continuous level walking surface at Z=0, 5 m wide, 200 m long.
box('Promenade_ground', (0,25,-.13),(5,220,.24),grout,land)
tiles = [[] for _ in stone]
for row in range(245):
    y = -84+row*.9
    for col in range(7):
        x = -2.5+col*.8+(row%2)*.4
        lo,hi = max(-2.5,x),min(2.5,x+.8)
        if hi-lo > .03:
            tiles[random.randrange(len(tiles))].append((((lo+hi)/2,y+.45,-.035),(hi-lo-.012,.888,.07)))
for i, specs in enumerate(tiles):
    boxes('Promenade_limestone_%d'%i,specs,stone[i],land)
for x in [-2.62,2.62]:
    boxes('Granite_curb', [((x,y,0),(.24,.99,.13)) for y in range(-85,137)],curb,land)
box('River embankment',(-3,25,-1.3),(.6,225,2.6),curb,land)
water = material('River jade water',(.045,.13,.10),rough=.20)
wp = water.node_tree.nodes.get('Principled BSDF')
wp.inputs['Metallic'].default_value = .32
wn = water.node_tree.nodes.new('ShaderNodeTexNoise')
wn.inputs['Scale'].default_value = 2.8
wn.inputs['Detail'].default_value = 2
geometry = water.node_tree.nodes.new('ShaderNodeNewGeometry')
water.node_tree.links.new(geometry.outputs['Position'],wn.inputs['Vector'])
wb = water.node_tree.nodes.new('ShaderNodeBump')
wb.inputs['Strength'].default_value = .26
wb.inputs['Distance'].default_value = .09
water.node_tree.links.new(wn.outputs['Fac'],wb.inputs['Height'])
water.node_tree.links.new(wb.outputs[0],wp.inputs['Normal'])
box('River_water',(-28,25,-1.02),(49,330,.10),water,land)
box('Opposite stone embankment',(-53,25,-.3),(1.1,330,1.4),curb,land)
box('Opposite bank grass',(-68,25,.0),(29,330,.4),grass,land)
# Right bank slopes up gently to an elevated parallel promenade.
mesh('Sloping_grass_bank',[(2.75,-90,-.02),(2.75,145,-.02),(16,145,2.1),(16,-90,2.1)],[(0,1,2,3)],grass,land)
box('Upper park lawn',(25,27,1.9),(18,235,.4),grass,land)
box('Upper promenade_ground',(18,27,2.08),(3,235,.18),stone[2],land)
box('Brown retaining wall',(20,27,2.55),(.32,235,1.2),brown,arch)
box('City ground',(74,40,-.2),(90,300,.4),grass,land)

# Straight railing in world space; it curves naturally in the panorama.
for z,w,h in [(1.08,.24,.10),(.12,.075,.085)]:
    box('River railing continuous rail',(-2.52,25,z),(w,220,h),brown,arch)
boxes('Railing balusters',[((-2.52,-85+i*.22,.59),(.045,.045,.98)) for i in range(1001)],brown,arch)
boxes('Railing main posts',[((-2.52,y,.58),(.105,.105,1.16)) for y in range(-85,136,3)],brown,arch)
for y in [5,45,85,-35]:
    bpy.ops.mesh.primitive_torus_add(major_radius=.30,minor_radius=.061,major_segments=48,minor_segments=10,
                                   location=(-2.34,y,.62),rotation=(0,math.pi/2,0))
    o=bpy.context.object
    o.name='Orange rescue ring'
    o.data.materials.append(orange)
    o.data.materials.append(white)
    move(o,props)
    for a in range(4):
        t=a*math.pi/2
        box('Rescue ring reflective band',(-2.27,y+.30*math.cos(t),.62+.30*math.sin(t)),(.045,.105 if a%2 else .13,.13 if a%2 else .105),white,props)

# Small white cylindrical riverside tower, visible near the panorama seam.
tx,ty=-6,-9
rod('Tower round plaster shaft',(tx,ty,-1),(tx,ty,4.3),.95,cream,arch)
rod('Tower cornice',(tx,ty,4.15),(tx,ty,4.38),1.12,cream,arch)
bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,location=(tx,ty,4.34))
o=bpy.context.object
o.name='Tower curved cap'
o.scale=(1.10,1.10,.64)
o.data.materials.append(roof)
move(o,arch)
rod('Tower roof lantern',(tx,ty,4.72),(tx,ty,5.12),.24,cream,arch)
rod('Tower lantern cap',(tx,ty,5.12),(tx,ty,5.28),.37,roof,arch,r2=.05)
rod('Tower finial',(tx,ty,5.2),(tx,ty,5.95),.018,support,arch)
box('Tower tall window',(tx+.948,ty,2.85),(.035,.40,.92),glass,arch)
box('Tower window lintel',(tx+.97,ty,3.35),(.06,.5,.10),curb,arch)

def tree(x,y,z,height, canopy=1, stakes=False):
    rod('Tree trunk',(x,y,z),(x+.13,y,z+height*.72),height*.026,bark,plants,r2=height*.009)
    leafverts=[[] for _ in leaves]
    leaffaces=[[] for _ in leaves]
    for branch in range(7):
        a=branch*2.4+random.random()
        base=Vector((x+.1,y,z+height*(.39+branch*.043)))
        tip=Vector((x+math.cos(a)*height*.24,y+math.sin(a)*height*.24,z+height*(.76+random.random()*.23)))
        rod('Tree branch',base,tip,height*.009,bark,plants,r2=.012)
        for j in range(650 if canopy else 220):
            pt=tip+Vector((random.gauss(0,height*.11),random.gauss(0,height*.11),random.gauss(0,height*.065)))
            idx=random.randrange(len(leaves))
            v,f=leafverts[idx],leaffaces[idx]
            k=len(v)
            s=random.uniform(.07,.19)*(1 if stakes else 1.8)
            u=Vector((math.cos(a+j),math.sin(a+j),random.uniform(-.4,.4)))*s
            w=Vector((-math.sin(a+j),math.cos(a+j),random.uniform(-.4,.4)))*s*.45
            v.extend([pt-u,pt+w,pt+u,pt-w])
            f.append((k,k+1,k+2,k+3))
    for i in range(len(leaves)):
        mesh('Tree leaf canopy',leafverts[i],leaffaces[i],leaves[i],plants)
    if stakes:
        for a in [0,2.094,4.189]:
            rod('Tree tripod brace',(x+1.1*math.cos(a),y+1.1*math.sin(a),z),(x,y,z+2.6),.035,support,plants)

for y in range(-42,128,9):
    tree(7+random.uniform(-.6,.6),y,.67,random.uniform(6,8.5),canopy=0,stakes=True)
for y in range(-65,151,10):
    tree(25+random.uniform(-2,2),y,2.1,random.uniform(7,10))
for y in range(-115,180,7):
    tree(-59+random.uniform(-2,2),y,.2,random.uniform(9,13))
for y in range(-70,132,4):
    tree(-3.7,y,-.75,random.uniform(1.1,2),canopy=0)

def building(x,y,w,d,h):
    box('Apartment tower mass',(x,y,h/2),(w,d,h),concrete,city)
    windows=[]
    strips=[]
    for z in range(3,int(h)-1,3):
        for sign in [-1,1]:
            for j in range(int(w/2.5)):
                xx=x-w/2+1.25+j*2.5
                windows.append(((xx,y+sign*(d/2+.025),z),(1.55,.055,1.9)))
            for j in range(int(d/2.5)):
                yy=y-d/2+1.25+j*2.5
                windows.append(((x+sign*(w/2+.025),yy,z),(.055,1.55,1.9)))
        strips.extend([((x,y,z+1.2),(w+.25,d+.25,.14))])
    boxes('Apartment repetitive windows',windows,glass,city)
    boxes('Apartment floor ledges',strips,trim,city)
    ribs=[]
    for sign in [-1,1]:
        for j in range(int(w/2.5)+1):
            ribs.append(((x-w/2+j*2.5,y+sign*(d/2+.1),h/2),(.15,.22,h)))
        for j in range(int(d/2.5)+1):
            ribs.append(((x+sign*(w/2+.1),y-d/2+j*2.5,h/2),(.22,.15,h)))
    boxes('Apartment vertical mullions',ribs,trim,city)
    box('Tower rooftop pavilion',(x,y,h+1),(w*.5,d*.65,2),concrete,city)
    box('Tower rooftop floating cap',(x,y,h+2.1),(w*.8,d*.9,.3),trim,city)

for spec in [(43,-37,15,16,49),(48,-7,17,15,66),(44,20,16,18,73),(56,48,17,18,63),
             (42,71,14,17,54),(48,97,15,16,48),(53,123,15,17,44),(70,8,18,16,60),
             (73,78,16,16,71),(37,161,13,15,38),(34,206,12,15,40)]:
    building(*spec)

for y in [-30,2,32,62,92]:
    rod('Park lamp pole',(4.1,y,.25),(4.1,y,4.7),.048,brown,props)
    box('Park lamp head',(4.1,y,4.67),(.5,.23,.14),cream,props)
    box('Litter bin',(3.8,y+2,.58),(.48,.42,.94),brown,props)
    box('Litter bin opening',(3.55,y+2,.86),(.015,.28,.13),grout,props)
    for x in [4,4.2,4.4]:
        box('Bench timber slat',(x,y+5,.63),(.16,1.9,.09),brown,props)
    for yy in [y+4.3,y+5.7]:
        box('Bench legs',(4.2,yy,.4),(.45,.09,.45),support,props)

world=bpy.data.worlds.new('Soft overcast blue daylight')
scene.world=world
world.use_nodes=True
nodes=world.node_tree.nodes
bg=nodes.get('Background')
bg.inputs['Strength'].default_value=.12
sky=nodes.new('ShaderNodeTexSky')
sky_types=sky.bl_rna.properties['sky_type'].enum_items.keys()
sky.sky_type='NISHITA' if 'NISHITA' in sky_types else 'MULTIPLE_SCATTERING'
sky.sun_elevation=math.radians(35)
sky.sun_rotation=math.radians(135)
sky.sun_disc=False
sky.air_density=1.2
if hasattr(sky,'dust_density'):
    sky.dust_density=2
else:
    sky.aerosol_density=2
world.node_tree.links.new(sky.outputs[0],bg.inputs['Color'])
bpy.ops.object.light_add(type='SUN',location=(0,0,50))
sun=move(bpy.context.object,cams)
sun.name='Hazy afternoon sun'
sun.rotation_euler=(.5,-.6,-.6)
sun.data.energy=1.3
sun.data.angle=.35

def camera(name, location, target, pano=False):
    bpy.ops.object.camera_add(location=location)
    c=move(bpy.context.object,cams)
    c.name=name
    c.rotation_euler=(Vector(target)-c.location).to_track_quat('-Z','Y').to_euler()
    c.data.clip_end=1000
    if pano:
        c.data.type='PANO'
        c.data.panorama_type='EQUIRECTANGULAR'
    else:
        c.data.lens=23
    return c

pano=camera('Camera_Equirectangular_360',(0,0,1.65),(0,30,1.65),True)
persp=camera('Camera_Promenade',(1,-13,2.1),(-.5,30,3.5))
overview=camera('Camera_Overview',(-40,-55,39),(9,27,5))
scene.render.engine='CYCLES'
scene.cycles.device='CPU'
scene.cycles.samples=args.samples
scene.cycles.use_denoising=True
scene.cycles.max_bounces=5
scene.render.threads_mode='FIXED'
scene.render.threads=12
scene.render.image_settings.file_format='PNG'
scene.view_settings.view_transform='AgX'
scene.view_settings.look='AgX - Medium High Contrast'
scene.camera=pano
scene.render.resolution_x=args.width
scene.render.resolution_y=args.width//2
scene.render.resolution_percentage=100
scene.render.filepath=str(PREVIEW/'panorama.png')
scene['reference']='assets/example/00001.jpg; stitched fisheye equirectangular panorama'
scene['reconstruction_note']='Approximate metric reconstruction from one image; no calibrated geometry. Environment only.'
for obj in scene.objects:
    obj.pass_index=0
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_distance=65
            area.spaces.active.region_3d.view_location=(0,15,3)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'riverfront.blend'))
report={'blend':str(OUT/'riverfront.blend'),'objects':len(scene.objects),'units':'meters',
        'reference':scene['reference'],'camera_location':list(pano.location),
        'projection':'EQUIRECTANGULAR','environment_only':True,'seed':41}
(PREVIEW/'scene_report.json').write_text(json.dumps(report,indent=2)+'\n')
if not args.skip_render:
    for cam,filename,w,h in [(pano,'panorama.png',args.width,args.width//2),
                             (persp,'promenade.png',1200,800),(overview,'overview.png',1200,800)]:
        scene.camera=cam
        scene.render.resolution_x=w
        scene.render.resolution_y=h
        scene.render.filepath=str(PREVIEW/filename)
        bpy.ops.render.render(write_still=True)
print('RIVERFRONT_COMPLETE',json.dumps(report))
