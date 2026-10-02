"""Bind locally available groom variants to new adult candidates, preserving bodies."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.io.json import write_json
from densekp_retrain.clothing.shells import vertex_normals
from densekp_retrain.clothing.binding import CONV, frame, triangle_frame
from densekp_retrain.clothing.binding import closest_surface
from tools.experiments.prepare_commuter_scalp import ramp
PILOT=ROOT/'outputs/experiments/cycling_commuter_pilot_20260930'
HAIR={'F01':'03-1_short-casual-MF','F03':'07-1_medium-layered-bob-MF','F04':'18-1_short-bob-F',
      'F05':'01-1_curly-M','M01':'03-1_short-casual-MF','M03':'07-1_medium-layered-bob-MF',
      'M04':'01-1_curly-M','M05':'03-1_short-casual-MF'}

def scalp(appearance,reference,out):
    x,y,z=reference['vertices'].T
    mask=np.maximum.reduce([ramp(z,.345,.36),ramp(np.abs(x),.052,.066)*ramp(z,.255,.28)*ramp(y,-.095,-.075),
                            ramp(y,-.005,.015)*ramp(z,.22,.27)])
    mask[z<.20]=0;mask*=reference['weights'][:,list(reference['group_names']).index('head')]>.1
    np.save(out/'scalp_mask.npy',mask.astype(np.float32))
    uv=np.load(appearance['body_uv']);corners=uv['uv'][uv['faces_uv']]
    forehead=(np.abs(x)<.026)&(z>.333)&(z<.351)&(y<-.065)
    coords=corners[forehead[uv['faces']]].copy();alignment=appearance['face_uv_alignment'];weight=np.ones(len(coords))
    for axis,key in [(0,'x_knots'),(1,'y_knots')]:
        a,b,c,d=alignment[key];weight*=np.clip((coords[:,axis]-a)/(b-a),0,1)*np.clip((d-coords[:,axis])/(d-c),0,1)
    coords+=weight[:,None]*(coords*np.array([alignment['u_scale']-1,0])+alignment['offset'])
    image=np.asarray(Image.open(appearance['body_texture']).convert('RGB'))/255
    pixels=image[np.clip(((1-coords[:,1])*image.shape[0]).astype(int),0,image.shape[0]-1),np.clip((coords[:,0]*image.shape[1]).astype(int),0,image.shape[1]-1)]
    rgb=np.median(pixels,axis=0);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    appearance['scalp_replacement']=dict(vertex_mask=str(out/'scalp_mask.npy'),color_linear=linear.tolist(),darkness_gate=False,
        source_skin=appearance['body_texture'],source='same-skin forehead median; local shader only, original preserved',limitation='flat fill under real groom')
    appearance['material_version']=6

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);p.add_argument('--characters',nargs='+',default=list(HAIR))
    p.add_argument('--version',default='v1');a=p.parse_args()
    start=time.monotonic();root=a.assets.resolve();report=json.loads((root/'report.json').read_text())
    ref=np.load(PILOT/'groom_reference/reference.npz');v=ref['vertices'];faces=ref['faces']
    head=ref['weights'][:,list(ref['group_names']).index('head')]>.1
    neutral=np.load('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/SMPLX_NEUTRAL.npz')
    np.testing.assert_array_equal(faces,neutral['f'])
    if np.abs(v[~head]-neutral['v_template'][~head]@CONV.T).max()>2e-6:raise ValueError('Reference semantics fail')
    mask=head&((v[:,2]>.32)|(v[:,1]>-.005)|((np.abs(v[:,0])>.055)&(v[:,2]>.23)))
    sf=faces[mask[faces].all(1)];tree=cKDTree(v[sf].mean(1));vn=vertex_normals(v,faces)
    cases=json.loads((PILOT/'groom_import_audit_v2/report.json').read_text())['cases']
    cases+=json.loads((root/'groom_import/report.json').read_text())['cases'];records=[];cache={}
    for char in report['characters']:
        mp=Path(char['manifest']);manifest=json.loads(mp.read_text());short=manifest['identity_family_id'].split('_')[-1]
        if short not in a.characters:continue
        source=next(s for s in cases if Path(s['source']).stem=='vinefx_groom_'+HAIR[short])
        if short in ('F02','M02'):raise ValueError('Existing groom must be reused unchanged')
        key=source['source']
        if key not in cache:
            data=np.load(source['export']);points=data['points']*.01;offsets=data['offsets'];radii=data['imported_radius']*.01
            roots=points[offsets[:-1]];near=[];tri=[]
            for i in range(0,len(roots),8192):
                q,_,_,ids=closest_surface(roots[i:i+8192],v,sf,tree);near.append(q);tri.append(sf[ids])
            near=np.concatenate(near);tri=np.concatenate(tri)
            weights=trimesh.triangles.points_to_barycentric(v[tri],near).astype(np.float32)
            weights=np.clip(weights,0,1);weights/=weights.sum(1,keepdims=True)
            normal=(vn[tri]*weights[:,:,None]).sum(1);axis=np.tile([1.,0,0],(len(roots),1));axis[np.abs(normal[:,0])>.9]=[0,1,0]
            source_frame=frame(normal,axis);pc=np.repeat(np.arange(len(roots)),np.diff(offsets))
            local=np.einsum('nji,nj->ni',source_frame[pc],points-roots[pc]).astype(np.float32)
            cache[key]=(tri,weights,axis,offsets,radii,pc,local,np.linalg.norm(roots-near,axis=1))
        tri,weights,axis,offsets,radii,pc,local,root_dist=cache[key]
        rest=np.load(Path(manifest['reference']['geometry'])/'vertices.npy')[0]@CONV.T
        tf=frame((vertex_normals(rest,faces)[tri]*weights[:,:,None]).sum(1),axis);rt=triangle_frame(rest,tri)
        binding=mp.parent/f'groom_binding_{a.version}';binding.mkdir()
        np.savez(binding/'binding.npz',body_faces=tri,weights=weights,curve_offsets=offsets,local_points=local,
                 target_rest_frames=tf,target_triangle_frames=rt,source_root_distance=root_dist)
        write_json(binding/'binding_manifest.json',dict(source=source['source'],reference=str(PILOT/'groom_reference/reference.npz'),
            method='verified face semantics, exact root barycentrics and local scalp frames',units='metres',source_unit_scale=.01,
            source_root_distance_max_m=float(root_dist.max()),license='https://bedlam2.is.tuebingen.mpg.de/license.html'))
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==manifest['character_id']]
        prepared=[];removed=set()
        for job in jobs:
            geometry=Path(job['geometry']);body=np.load(geometry/'vertices.npy')[0];b=body@CONV.T
            np.testing.assert_array_equal(np.load(geometry/'faces.npy'),faces)
            rotation=np.einsum('nij,nkj->nik',triangle_frame(b,tri),rt);target=np.einsum('nij,njk->nik',rotation,tf)
            roots=(b[tri]*weights[:,:,None]).sum(1)+target[:,:,2]*.0004
            original=((roots[pc]+np.einsum('nij,nj->ni',target[pc],local))@CONV).astype(np.float32)
            points=original.copy();vnbody=vertex_normals(body,faces);vt=cKDTree(body);ft=cKDTree(body[faces].mean(1))
            isroot=np.zeros(len(points),bool);isroot[offsets[:-1]]=True
            # Fixed bounded correction, not an escalating cloth solver.
            for _ in range(3):
                nearest=vt.query(points)[1];distance=((points-body[nearest])*vnbody[nearest]).sum(1)
                ids=np.flatnonzero((distance<.004)&~isroot)
                for i in range(0,len(ids),8192):
                    ix=ids[i:i+8192];_,normal,signed,_=closest_surface(points[ix],body,faces,ft)
                    points[ix]+=np.maximum(np.maximum(.0003,radii[ix])-signed,0)[:,None]*normal
            delta=points-original;sample=np.unique(np.r_[np.arange(0,len(points),20),offsets[:-1]])
            signed=np.concatenate([closest_surface(points[sample[i:i+8192]],body,faces,ft)[2] for i in range(0,len(sample),8192)])
            removed.update(np.unique(np.r_[pc[sample[signed<-.001]],pc[np.linalg.norm(delta,axis=1)>.03]]).tolist())
            folder=mp.parent/('reference' if job['pose_source_id']=='reference' else job['pose_source_id'])/f'groom_{a.version}';folder.mkdir(parents=True)
            for name,value in dict(points=points,radius=radii,curve_offsets=offsets,target_roots=roots@CONV,collision_correction=delta).items():np.save(folder/(name+'.npy'),value)
            info=dict(character_id=manifest['character_id'],body_geometry=str(geometry),binding=str(binding),source_abc=source['source'],source_objects=source['objects'],
                split='integration_only',status='candidate_visual_pending',units='metres; SMPL-X coordinates',
                correction_max_m=float(np.linalg.norm(delta,axis=1).max()),sample_min_signed_m=float(signed.min()),
                collision_method='nearest vertex filter, closest of 24 triangle centers; diagnostic sampling not exhaustive proof')
            prepared.append((job,folder,info,sample,signed,np.linalg.norm(delta,axis=1)))
        active=np.ones(len(offsets)-1,bool);active[list(removed)]=False
        if (~active).mean()>.01:raise ValueError(f'{short}: would prune more than 1%; choose another groom')
        for job,folder,info,sample,signed,magnitude in prepared:
            np.save(folder/'active_curve_mask.npy',active)
            info.update(removed_curves=len(removed),retained_sample_min_m=float(signed[active[pc[sample]]].min()),
                        retained_correction_max_m=float(magnitude[active[pc]].max()),retained_curves=int(active.sum()))
            write_json(folder/'groom_manifest.json',info);job['groom_geometry']=str(folder);records.append(info)
        scalp(manifest['appearance'],ref,mp.parent)
        for job in jobs:job['appearance']=manifest['appearance']
        manifest.update(groom_binding=str(binding),groom_source=source['source'],hair_status='bound_pruned_candidate_visual_pending')
        write_json(mp,manifest);write_json(root/'report.json',report);print(short,len(removed),flush=True)
        write_json(root/f'new_groom_bindings_{a.version}.json',dict(seconds=time.monotonic()-start,cases=records,split='integration_only'))

if __name__=='__main__':main()
