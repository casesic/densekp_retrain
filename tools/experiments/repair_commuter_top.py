"""Component experiment: smooth the rest pattern, then elastic collision fitting.

The immutable body is never replaced by the voxel collision proxy. Exact body
triangle crossings remain a separate acceptance check. No dynamic cloth solve.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Matrix, Quaternion, Vector

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.rendering.blender.intersections import crossings


def edges_of(faces):
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    return np.unique(edges,axis=0,return_counts=True)


def average(vertices,edges):
    total=np.zeros_like(vertices);counts=np.zeros(len(vertices))
    for a,b in [(0,1),(1,0)]:
        np.add.at(total,edges[:,a],vertices[edges[:,b]])
        np.add.at(counts,edges[:,a],1)
    return total/np.maximum(counts[:,None],1)


def tree(vertices,faces):
    return BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True)


def qmultiply(a,b):
    return np.concatenate(((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True)),
                           a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)


def dual_quaternion_binding(vertices,skin,transforms):
    real=np.array([Matrix(t[:3,:3].tolist()).to_quaternion()[:] for t in transforms])
    dual=.5*qmultiply(np.column_stack((np.zeros(len(transforms)),transforms[:,:3,3])),real)
    reference=real[skin.argmax(1)]
    sign=np.where(reference@real.T<0,-1.,1.)
    qr=(skin*sign)@real;qd=(skin*sign)@dual
    length=np.linalg.norm(qr,axis=1,keepdims=True);qr/=length;qd/=length
    qd-=qr*(qr*qd).sum(1,keepdims=True)
    conjugate=qr*np.array([1,-1,-1,-1])
    translations=2*qmultiply(qd,conjugate)[:,1:]
    rotated=vertices+2*np.cross(qr[:,1:],np.cross(qr[:,1:],vertices)+qr[:,:1]*vertices)
    return rotated+translations,qr,translations


def project(vertices,bvh,clearance):
    output=vertices.copy()
    for index,point in enumerate(vertices):
        near,normal,_,distance=bvh.find_nearest(point.tolist())
        near,normal=np.asarray(near),np.asarray(normal)
        signed=np.dot(point-near,normal)
        if signed<clearance:
            output[index]=near+clearance*normal
    return output


def ray_degree(point,bvh):
    """Oriented crossings to infinity count overlapping interiors, unlike parity."""
    direction=np.array([.713,.237,.660]);direction/=np.linalg.norm(direction)
    position=np.asarray(point).copy();degree=0
    for _ in range(32):
        hit,normal,_,distance=bvh.ray_cast(position.tolist(),direction.tolist())
        if hit is None:return degree
        degree+=1 if np.dot(np.asarray(normal),direction)>0 else -1
        position=np.asarray(hit)+direction*2e-6
    raise RuntimeError('Ray crossing safety bound exceeded')


def exposed_collider(body,faces):
    bvh=tree(body,faces);tri=body[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    centers=tri.mean(1)+normal*.0002
    keep=np.array([ray_degree(p,bvh)==0 for p in centers])
    return body,faces[keep]


def collision_proxy(body,faces,voxel):
    mesh=bpy.data.meshes.new('Proxy');mesh.from_pydata(body.tolist(),[],faces.tolist())
    obj=bpy.data.objects.new('Proxy',mesh);bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active=obj;obj.select_set(True)
    mesh.remesh_voxel_size=voxel
    bpy.ops.object.voxel_remesh()
    v=np.empty((len(obj.data.vertices),3),np.float32);obj.data.vertices.foreach_get('co',v.ravel())
    obj.data.calc_loop_triangles()
    f=np.asarray([t.vertices[:] for t in obj.data.loop_triangles],np.int32)
    bpy.data.objects.remove(obj,do_unlink=True)
    return v,f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--character',default='cycling_F02_v2')
    p.add_argument('--pose',default='reference')
    p.add_argument('--component',choices=['top','trousers','folded_collar'],default='top')
    p.add_argument('--pattern',type=Path)
    p.add_argument('--exposed-collider',action='store_true')
    p.add_argument('--bridge-armpit',action='store_true')
    p.add_argument('--binding-inputs',type=Path)
    p.add_argument('--crotch-clearance',action='store_true')
    p.add_argument('--independent-pattern',action='store_true')
    p.add_argument('--dqs',action='store_true')
    p.add_argument('--lower-neck',action='store_true')
    p.add_argument('--use-fitted-pattern',action='store_true')
    p.add_argument('--smoothed-skin',type=Path)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.smoothed_skin and not args.binding_inputs:
        raise ValueError('Smoothed weights require the corresponding frozen joint transforms')
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=args.source.resolve()
    reference=source/'characters'/args.character/'reference'
    job=reference if args.pose=='reference' else source/'derivatives'/args.character/args.pose
    old=job/'commuter_clothing_v4'
    info=json.loads((old/'clothing_manifest.json').read_text())
    component=next(c for c in info['components'] if c['name']==args.component)
    first,count=component['vertex_start'],component['vertex_count'];sl=slice(first,first+count)
    faces=np.load(old/'clothing_faces.npy');faces=faces[((faces>=first)&(faces<first+count)).all(1)]-first
    edges,edgecounts=edges_of(faces);boundary=np.unique(edges[edgecounts==1])
    body=np.load(job/'geometry/vertices.npy')[0];bf=np.load(job/'geometry/faces.npy')
    restbody=np.load(reference/'geometry/vertices.npy')[0]
    oldrest=np.load(old/'clothing_rest_vertices.npy')[sl]
    index=np.load(old/'body_barycentric_indices.npy')[sl]
    weights=np.load(old/'body_barycentric_weights.npy')[sl]
    restanchor=(restbody[index]*weights[:,:,None]).sum(1)
    if args.pattern:
        rest=np.load(args.pattern/'clothing_rest_vertices.npy')
        if args.independent_pattern:
            if not args.binding_inputs:raise ValueError('Independent topology requires fresh binding inputs')
            faces=np.load(args.pattern/'clothing_faces.npy')
            edges,edgecounts=edges_of(faces);boundary=np.unique(edges[edgecounts==1])
            oldrest=rest.copy()
            if not (args.pattern/'repair.json').exists() and not args.use_fitted_pattern:
                restbvh=tree(restbody,bf)
                rest=project(rest,restbvh,.006)
                for _ in range(8):
                    delta=.3*(average(rest,edges)-rest);delta[boundary]=0
                    rest=project(rest+delta,restbvh,.006)
    else:
        rest=oldrest.copy();bvh=tree(restbody,bf)
        # Continuous fairing removes the discontinuous box/ellipse masks from v4.
        # Boundary rings retain the neckline, cuffs and hem dimensions.
        for _ in range(30):
            delta=.4*(average(rest,edges)-rest);delta[boundary]=0
            rest=project(rest+delta,bvh,.006)
        if args.crotch_clearance:
            # The v4 medial offsets push two distinct legs through each other.
            # Restore a narrow, smoothly blended crotch allowance, retaining the
            # independently widened straight legs outside this local strip.
            factor=np.clip((.035-np.abs(restanchor[:,0]))/.020,0,1)
            factor*=np.clip((-.10-restanchor[:,1])/.035,0,1)
            factor*=np.clip((restanchor[:,1]+.40)/.08,0,1)
            factor=factor*factor*(3-2*factor)
            tight=project(restanchor,bvh,.0008)
            rest=rest*(1-factor[:,None])+tight*factor[:,None]
    if args.lower_neck:
        neck=boundary[(rest[boundary,1]>.50)&(np.abs(rest[boundary,0])<.13)]
        if not len(neck):raise ValueError('No neck boundary found')
        rest[neck,1]-=.035
        bvh=tree(restbody,bf);rest=project(rest,bvh,.006)
        for _ in range(12):
            delta=.4*(average(rest,edges)-rest);delta[boundary]=0
            rest=project(rest+delta,bvh,.006)
    skin=np.load(old/'skin_weights.npy')[sl]
    rotations=np.load(old/'binding_rotations.npy')[sl]
    if args.binding_inputs:
        bvh=tree(restbody,bf)
        near=[];faceids=[]
        for point in rest:
            closest,_,faceid,_=bvh.find_nearest(point.tolist());near.append(closest[:]);faceids.append(faceid)
        index=bf[np.asarray(faceids)];tri=restbody[index].astype(float);near=np.asarray(near)
        a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];d=near-tri[:,0]
        aa=(a*a).sum(1);ab=(a*b).sum(1);bb=(b*b).sum(1);ad=(a*d).sum(1);bd=(b*d).sum(1)
        determinant=aa*bb-ab*ab
        v=(bb*ad-ab*bd)/determinant;w=(aa*bd-ab*ad)/determinant
        weights=np.clip(np.column_stack((1-v-w,v,w)),0,1).astype(np.float32);weights/=weights.sum(1,keepdims=True)
        sourcebinding=np.load(args.binding_inputs/f'{args.character}_{args.pose}.npz')
        skin=(sourcebinding['skin_weights'][index]*weights[:,:,None]).sum(1)
        rotations=np.einsum('nj,jab->nab',skin,sourcebinding['joint_rotations'])
        restanchor=(restbody[index]*weights[:,:,None]).sum(1)
    if args.smoothed_skin:
        replacement=np.load(args.smoothed_skin/'skin_weights.npy')
        if replacement.shape!=skin.shape:raise ValueError('Garment weight topology mismatch')
        np.testing.assert_allclose(replacement.sum(1),1,atol=1e-6)
        if replacement.min()<0:raise ValueError('Negative skinning weights')
        skin=replacement;rotations=np.einsum('nj,jab->nab',skin,sourcebinding['joint_rotations'])
    anchor=(body[index]*weights[:,:,None]).sum(1)
    offsets=rest-restanchor
    bound=anchor+np.einsum('nab,nb->na',rotations,offsets)
    lbs_bound=bound.copy();dqs_arrays={}
    if args.dqs:
        if not args.binding_inputs:raise ValueError('DQS requires full saved transforms')
        bound,quaternions,translations=dual_quaternion_binding(rest,skin,sourcebinding['transforms'])
        dqs_arrays=dict(binding_quaternions=quaternions,binding_translations=translations,lbs_bound=lbs_bound)
    final=bound.copy()
    bridge=np.zeros(len(final),bool)
    if args.bridge_armpit:
        pairs,_,_=crossings(body,bf)
        restcenters=restbody[bf].mean(1)
        arm=(np.abs(restcenters[:,0])>.12)&(restcenters[:,1]>.29)&(restcenters[:,1]<.49)
        pairs=pairs[arm[pairs].any(1)]
        affected=np.unique(bf[pairs])
        bridge=np.isin(index,affected).any(1)
        for _ in range(3):
            touched=bridge[edges].any(1);bridge[np.unique(edges[touched])]=True
        bridge[boundary]=False
        # Replace a folded interior by a continuous membrane spanned by its
        # unchanged boundary. This is a recorded garment-only pose corrective.
        for _ in range(300):
            target=average(final,edges)
            final[bridge]=.2*final[bridge]+.8*target[bridge]
    proxy,proxyfaces=exposed_collider(body,bf) if args.exposed_collider else collision_proxy(body,bf,.003)
    bvh=tree(proxy,proxyfaces)
    restlen=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    history=[]
    for iteration in range(0 if args.pose=='reference' else 60):
        # Edge strain is constrained jointly with contact, rather than independent
        # point pushes; small bending relaxation removes isolated corrective spikes.
        delta=final[edges[:,1]]-final[edges[:,0]]
        length=np.maximum(np.linalg.norm(delta,axis=1),1e-10)
        strain=(length-restlen)/length
        moves=.35*strain[:,None]*delta
        total=np.zeros_like(final);degree=np.zeros(len(final))
        np.add.at(total,edges[:,0],moves);np.add.at(total,edges[:,1],-moves)
        np.add.at(degree,edges.ravel(),1)
        final+=total/np.maximum(degree[:,None],1)
        smooth=.08*(average(final,edges)-final);smooth[boundary]=0
        final=project(final+smooth,bvh,.005)
        history.append(float(np.linalg.norm(final-bound,axis=1).max()))
        if history[-1]>=.06:break
    results={}
    arrays=dict(body=body,body_faces=bf,rest=rest,bound=bound,final=final,cloth_faces=faces)
    arrays.update(dqs_arrays)
    for name,vertices in [('rest',rest),('bound',bound),('final',final)]:
        collider=restbody if name=='rest' else body
        pairs,points,coplanar=crossings(vertices,faces,collider,bf)
        selfpairs,selfpoints,selfcoplanar=crossings(vertices,faces)
        results[name]=dict(body_crossings=len(pairs),self_crossings=len(selfpairs),coplanar_unresolved=None)
        arrays[name+'_body_pairs']=pairs;arrays[name+'_body_locations']=points
        arrays[name+'_self_pairs']=selfpairs;arrays[name+'_self_locations']=selfpoints
    arrays['bridge_mask']=bridge
    correction=final-bound
    if not all(np.isfinite(value).all() for value in [rest,bound,final,skin,rotations,correction]):
        raise ValueError('Nonfinite garment or binding data; cannot export a candidate')
    saved=dict(clothing_vertices=final[None].astype(np.float32),clothing_faces=faces,
               clothing_rest_vertices=rest.astype(np.float32),rest_offsets=offsets.astype(np.float32),
               body_barycentric_indices=index,body_barycentric_weights=weights,
               binding_rotations=rotations,reference_body_vertices=anchor[None],
               collision_correction=correction.astype(np.float32),
               skin_weights=skin)
    saved.update(dqs_arrays)
    for name,value in saved.items():np.save(args.output/f'{name}.npy',value)
    np.savez_compressed(args.output/'evidence.npz',**arrays)
    np.savez_compressed(args.output/'collision_proxy.npz',vertices=proxy,faces=proxyfaces)
    restored=saved['reference_body_vertices'][0]+np.einsum('nab,nb->na',rotations,saved['rest_offsets'])+saved['collision_correction']
    if args.dqs:
        restored=np.array([(Quaternion(q.tolist())@Vector(point.tolist()))[:]
                           for q,point in zip(quaternions,rest)])+translations+saved['collision_correction']
    report=dict(split='integration_only',status='candidate_requires_visual_and_contact_review',character_id=args.character,
        pose=args.pose,parent_clothing=str(old),body_geometry=str(job/'geometry'),body_changed=False,
        pattern_source=str(args.pattern) if args.pattern else str(old),component=args.component,
        config=dict(smoothed_skin=str(args.smoothed_skin) if args.smoothed_skin else None,
                    binding_inputs=str(args.binding_inputs) if args.binding_inputs else None,
                    neck_boundary_lowered_m=.035 if args.lower_neck else 0.,
                    skinning='dual_quaternion_whole_rest_garment' if args.dqs else 'posed_body_anchor_plus_LBS_offset',
                    dqs_vs_surface_LBS_max_m=float(np.linalg.norm(bound-lbs_bound,axis=1).max()),
                    independent_topology=args.independent_pattern,crotch_clearance_m=.0008 if args.crotch_clearance else None,
                    binding='recomputed_all_triangle_BVH_barycentric_and_LBS' if args.binding_inputs else 'inherited_v4_anchors_and_weights',
                    armpit_membrane_vertices=int(bridge.sum()),armpit_membrane_steps=300 if args.bridge_armpit else 0,
                    collider='exposed_face_center_ray_degree' if args.exposed_collider else 'voxel_proxy',
                    pattern_fairing_steps=0 if args.pattern else 30,pattern_fairing_factor=.4,reference_clearance_m=.006,
                    proxy_voxel_m=.003,proxy_clearance_m=.005,strain_factor=.35,bending_factor=.08,max_steps=60),
        independent_crossings=results,iterations=len(history),max_correction_m=max(history,default=0.),
        correction_quantiles_m=np.quantile(np.linalg.norm(correction,axis=1),[0,.5,.95,1]).tolist(),
        rest_ease_quantiles_m=np.quantile(np.linalg.norm(offsets,axis=1),[0,.5,.95,1]).tolist(),
        pattern_edit_quantiles_m=np.quantile(np.linalg.norm(rest-oldrest,axis=1),[0,.5,.95,1]).tolist(),
        binding_readback_max_m=float(np.abs(restored-saved['clothing_vertices'][0]).max()),
        limits=dict(correction_failure_m=.06,binding_readback_m=2e-6),seconds=time.monotonic()-start,
        limitations=['Collision proxies are solver aids; immutable-body triangle tests decide crossings.',
                     'Intersection tests do not detect every containment or coplanar overlap.'])
    if results['final']['body_crossings'] or results['final']['self_crossings'] or max(history,default=0.)>=.06:
        report['status']='failed_geometry'
    (args.output/'repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
