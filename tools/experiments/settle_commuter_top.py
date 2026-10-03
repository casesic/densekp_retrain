"""Isolated, initially valid top on the preserved P2 initialization path.

Unlike the old whole-outfit solve, this excludes already-intersecting trousers,
uses the faired reference pattern, and checks the actual collider endpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.experiments.settle_commuter_cloth import keyed_mesh
from tools.experiments.repair_commuter_top import edges_of
from densekp_retrain.rendering.blender.intersections import crossings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--pattern',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    z=np.load(args.source/'input.npz');bodypath=z['body_path'];bf=z['body_faces']
    rest=np.load(args.pattern/'clothing_rest_vertices.npy');cf=np.load(args.pattern/'clothing_faces.npy')
    # The saved old path has the same body anchors and rotations; interpolate
    # rotations from body patch frames without changing any body coordinates.
    ids=np.load(args.pattern/'body_barycentric_indices.npy');w=np.load(args.pattern/'body_barycentric_weights.npy')
    offsets=np.load(args.pattern/'rest_offsets.npy')
    # Reconstruct the original LBS transforms, keeping all pose parameters fixed.
    # These are exported separately by the preparation step.
    transforms=np.load(args.pattern.parent/'P2_path_rotations.npy')
    anchors=(bodypath[:,ids]*w[None,:,:,None]).sum(2)
    clothpath=anchors+np.einsum('tnab,nb->tna',transforms,offsets)
    np.testing.assert_allclose(clothpath[0],rest,atol=2e-6,rtol=0)
    edges,counts=edges_of(cf);boundary=np.unique(edges[counts==1]);pin=np.zeros(len(rest),np.float32)
    selected=boundary[(rest[boundary,1]>.43)|(np.abs(rest[boundary,0])>.50)]
    pin[selected]=1
    np.savez_compressed(args.output/'input.npz',body_path=bodypath,cloth_path=clothpath,body_faces=bf,cloth_faces=cf,pin=pin)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=130;scene.render.fps=24
    scene.render.threads_mode='FIXED';scene.render.threads=8
    body=keyed_mesh('ImmutableBody',bodypath,bf);body.modifiers.new('BodyCollision','COLLISION')
    body.collision.thickness_outer=.001;body.collision.thickness_inner=.001;body.collision.cloth_friction=5
    cloth=keyed_mesh('IndependentTop',clothpath,cf)
    pins=cloth.vertex_groups.new(name='NeckAndCuffs')
    for i in selected:pins.add([int(i)],1,'REPLACE')
    modifier=cloth.modifiers.new('Cloth','CLOTH');settings=modifier.settings
    settings.quality=10;settings.mass=.15;settings.tension_stiffness=20;settings.compression_stiffness=20
    settings.shear_stiffness=5;settings.bending_stiffness=.5;settings.vertex_group_mass=pins.name
    settings.pin_stiffness=1;settings.air_damping=2
    collision=modifier.collision_settings;collision.use_collision=True;collision.collision_quality=6
    collision.distance_min=.001;collision.use_self_collision=True;collision.self_distance_min=.002
    modifier.point_cache.frame_start=1;modifier.point_cache.frame_end=130
    conv=np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
    traces=[]
    for frame in range(1,131):
        scene.frame_set(frame);graph=bpy.context.evaluated_depsgraph_get();evaluated=cloth.evaluated_get(graph)
        mesh=evaluated.to_mesh();v=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',v.ravel());evaluated.to_mesh_clear()
        if not np.isfinite(v).all() or np.abs(v).max()>5:raise ValueError('Solver divergence')
        if frame in [10,60,85,110,130]:
            world=v@conv
            actual=body.evaluated_get(graph).to_mesh();bv=np.array([v.co[:] for v in actual.vertices])@conv
            pairs,_,_=crossings(world,cf,bv,bf);own,_,_=crossings(world,cf)
            traces.append(dict(frame=frame,body_crossings=len(pairs),self_crossings=len(own)))
            np.save(args.output/f'frame_{frame:03d}.npy',world)
            body.evaluated_get(graph).to_mesh_clear()
    final=v@conv;bound=clothpath[-1];correction=final-bound
    np.savez_compressed(args.output/'evidence.npz',body=bodypath[-1],body_faces=bf,cloth_faces=cf,rest=rest,bound=bound,final=final)
    report=dict(split='integration_only',status='pending_independent_acceptance',source=str(args.source),pattern=str(args.pattern),
                body_changed=False,frames=130,pin_count=len(selected),traces=traces,
                max_correction_m=float(np.linalg.norm(correction,axis=1).max()),
                actual_collider_enabled=body.collision.use,
                collider_endpoint_max_m=float(np.abs(bv-bodypath[-1]).max()),seconds=time.monotonic()-start,
                changes_from_old_solver=['valid smooth top reference','no intersecting trousers/collar components','no global soft pins'],
                preserved_solver_parameters='same v6 quality, stiffness, collision distances and 130 frames')
    if report['max_correction_m']>=.06 or traces[-1]['body_crossings'] or traces[-1]['self_crossings']:
        report['status']='failed_geometry'
    (args.output/'simulation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
