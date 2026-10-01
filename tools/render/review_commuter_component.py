"""Small fixed-light component reviews, without regenerating body or hair."""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stages', nargs='+', default=['rest', 'bound', 'final'])
    p.add_argument('--count', type=int, default=2664)
    p.add_argument('--device', default='CPU')
    p.add_argument('--center', nargs=3, type=float)
    p.add_argument('--scale', type=float)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    z = np.load(args.input)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    if args.device == 'CUDA':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == 'CUDA'
        scene.cycles.device = 'GPU'
    scene.render.resolution_x = scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.3,.3,.3,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .5
    conv = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    for name, color in [('body', (.22,.22,.22,1)), ('cloth', (.65,.65,.65,1))]:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        mat.use_nodes = True
        mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
        mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .75
    for loc, energy, size in [((-2,-3,3),400,3),((2,-1,1),200,2),((0,2,3),300,2)]:
        light = bpy.data.lights.new('Area', 'AREA');light.energy=energy;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new('Area',light);scene.collection.objects.link(obj);obj.location=loc
        obj.rotation_euler=(-obj.location).to_track_quat('-Z','Y').to_euler()
    camera=bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'))
    scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='ORTHO'
    def mesh(name, vertices, faces, material):
        data=bpy.data.meshes.new(name);data.from_pydata((vertices@conv.T).tolist(),[],faces.tolist())
        obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);data.materials.append(bpy.data.materials[material])
        for poly in data.polygons:poly.use_smooth=True
        return obj
    body=mesh('Body',z['body'],z['body_faces'],'body')
    cf=z['cloth_faces'];cf=cf[(cf<args.count).all(1)]
    records=[]
    for stage in args.stages:
        v=z[stage];obj=mesh('Cloth',v,cf,'cloth')
        target=(v[np.unique(cf)].min(0)+v[np.unique(cf)].max(0))/2
        if args.center:target=np.asarray(args.center)
        target=Vector(target@conv.T)
        for name, offset in [('front',(0,-3,.15)),('side',(3,-.1,.1)),('back',(0,3,.15))]:
            camera.location=target+Vector(offset)
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
            camera.data.ortho_scale=1.45 if 'reference' in str(args.input) else .85
            if args.scale:camera.data.ortho_scale=args.scale
            scene.render.filepath=str(args.output/f'{stage}_{name}.png')
            bpy.ops.render.render(write_still=True)
            records.append(scene.render.filepath)
        bpy.data.objects.remove(obj,do_unlink=True)
    (args.output/'review.json').write_text(json.dumps({'seconds':time.monotonic()-start,'input':str(args.input),'images':records,'split':'integration_only'},indent=2)+'\n')


if __name__ == '__main__':main()
