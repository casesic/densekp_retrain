"""Blender Cloth pilot with moving body collision and boundary pinning; preserves failures."""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np


def keyed_mesh(name,path,faces):
    convert=np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
    values=path@convert.T
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(values[0].tolist(),[],faces.tolist())
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.shape_key_add(name='Basis')
    for i in range(1,len(values)):
        key=obj.shape_key_add(name=f'Initialization_{i}')
        key.data.foreach_set('co',values[i].ravel())
        key.value=0;key.keyframe_insert('value',frame=10+5*(i-1))
        key.value=1;key.keyframe_insert('value',frame=10+5*i)
        if i<len(values)-1:
            key.value=0;key.keyframe_insert('value',frame=10+5*(i+1))
    return obj


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);start=time.monotonic()
    z=np.load(args.input/'input.npz');bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=130;scene.render.fps=24
    scene.render.threads_mode='FIXED';scene.render.threads=8
    body=keyed_mesh('CollisionBody',z['body_path'],z['body_faces'])
    body.modifiers.new('BodyCollision','COLLISION');body.collision.thickness_outer=.001
    body.collision.thickness_inner=.001;body.collision.cloth_friction=5
    cloth=keyed_mesh('CommuterCloth',z['cloth_path'],z['cloth_faces'])
    pin=cloth.vertex_groups.new(name='BoundaryPins')
    for i,weight in enumerate(z['pin']):
        if weight:pin.add([i],float(weight),'REPLACE')
    modifier=cloth.modifiers.new('StaticClothPilot','CLOTH');settings=modifier.settings
    settings.quality=10;settings.mass=.15;settings.tension_stiffness=20;settings.compression_stiffness=20
    settings.shear_stiffness=5;settings.bending_stiffness=.5;settings.vertex_group_mass=pin.name
    settings.pin_stiffness=1;settings.air_damping=2
    collision=modifier.collision_settings;collision.use_collision=True;collision.collision_quality=6
    collision.distance_min=.001;collision.use_self_collision=True;collision.self_distance_min=.002
    modifier.point_cache.frame_start=1;modifier.point_cache.frame_end=130
    traces=[]
    for number in range(1,131):
        scene.frame_set(number);depsgraph=bpy.context.evaluated_depsgraph_get();evaluated=cloth.evaluated_get(depsgraph)
        mesh=evaluated.to_mesh();v=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',v.ravel());evaluated.to_mesh_clear()
        if not np.isfinite(v).all():raise ValueError('Nonfinite simulated cloth')
        if number%10==0:
            traces.append(dict(frame=number,seconds=time.monotonic()-start,bounds=[v.min(0).tolist(),v.max(0).tolist()]))
            print(json.dumps(traces[-1]),flush=True)
        if np.abs(v).max()>5:raise ValueError('Cloth simulation diverged beyond 5m safety bound')
    convert=np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
    np.save(args.input/'simulated_vertices.npy',(v@convert).astype(np.float32))
    (args.input/'simulation_report.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,frames=130,
        traces=traces,status='simulation_completed_validation_pending',split='integration_only',
        purpose='independent static garment fitting; interpolation is not source motion'),indent=2)+'\n')


if __name__=='__main__':main()
