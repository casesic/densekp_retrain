#!/usr/bin/env python3
"""Clip garment boundaries within body triangles instead of selecting whole faces."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from tools.experiments.prepare_cycling_clothing import vertex_normals
from tools.experiments.prepare_cycling_characters import write_json, MATERIALS


def clipped_shell(rest, joints, body, faces, profile=None):
    waist=float(joints[0,1]+.10)
    hem=float((joints[4,1]+joints[5,1])*.5 + .35*((joints[1,1]+joints[2,1]-joints[4,1]-joints[5,1])*.5))
    collar=float(joints[12,1]+.045)
    left=float(joints[16,0]+.65*(joints[18,0]-joints[16,0]))
    right=float(joints[17,0]+.65*(joints[19,0]-joints[17,0]))
    if profile:
        # Anatomical interpolation changes garment coverage, not body geometry.
        sleeve=float(profile['sleeve'])
        left=float(joints[16,0]+sleeve*(joints[20,0]-joints[16,0]))
        right=float(joints[17,0]+sleeve*(joints[21,0]-joints[17,0]))
        hip=float(joints[1:3,1].mean()); ankle=float(joints[7:9,1].mean())
        hem=hip+float(profile['leg'])*(ankle-hip)
        collar=float(joints[12,1]+profile.get('collar',.045))
    planes=[(0,1,-right),(0,-1,left),(1,1,-hem),(1,-1,collar)]
    lookup={}; indices=[]; weights=[]; cloth_faces=[]
    for triangle in faces:
        rv=rest[triangle]
        if any(np.max(sign*rv[:,axis]+offset)<0 for axis,sign,offset in planes): continue
        polygon=list(np.eye(3))
        for axis,sign,offset in planes:
            if not polygon: break
            clipped=[]
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da=float(sign*(a@rv)[axis]+offset); db=float(sign*(b@rv)[axis]+offset)
                if da>=0: clipped.append(a)
                if (da>=0)!=(db>=0): clipped.append(a+(b-a)*(da/(da-db)))
            polygon=clipped
        ids=[]
        for bary in polygon:
            pairs=sorted((int(vertex),float(weight)) for vertex,weight in zip(triangle,bary) if weight>1e-9)
            key=tuple((v,round(w,9)) for v,w in pairs)
            if key not in lookup:
                lookup[key]=len(indices)
                indices.append([p[0] for p in pairs]+[pairs[0][0]]*(3-len(pairs)))
                weights.append([p[1] for p in pairs]+[0.]*(3-len(pairs)))
            ids.append(lookup[key])
        for i in range(1,len(ids)-1):
            if len({ids[0],ids[i],ids[i+1]})==3: cloth_faces.append([ids[0],ids[i],ids[i+1]])
    indices=np.asarray(indices,np.int32); weights=np.asarray(weights,np.float32)
    faces_out=np.asarray(cloth_faces,np.int32)
    weights/=weights.sum(1,keepdims=True)
    reference=(body[indices]*weights[:,:,None]).sum(1)
    rest_clothing=(rest[indices]*weights[:,:,None]).sum(1)
    original=reference[faces_out]
    normals=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0])
    lengths=np.linalg.norm(normals,axis=1)
    faces_out=faces_out[lengths>1e-12]; normals=normals[lengths>1e-12]/lengths[lengths>1e-12,None]
    body_normals=vertex_normals(body,faces)
    displacement=(body_normals[indices]*weights[:,:,None]).sum(1)
    displacement*=.004/np.maximum(np.linalg.norm(displacement,axis=1,keepdims=True),1e-12)
    for _ in range(100):
        separation=(displacement[faces_out].mean(1)*normals).sum(1)
        bad=np.flatnonzero(separation<.0004-1e-8)
        if not len(bad): break
        for i in bad:
            f=faces_out[i]; n=normals[i]
            displacement[f]+=max(.0004-displacement[f].mean(0)@n,0)*n
    else: raise ValueError('Clipped cloth cannot meet existing local clearance')
    clothing=(reference+displacement).astype(np.float32)
    separation=((clothing[faces_out].mean(1)-reference[faces_out].mean(1))*normals).sum(1)
    offset=float(np.linalg.norm(clothing-reference,axis=1).max())
    if separation.min()<.0004-2e-7 or offset>.01: raise ValueError('Existing shell gates failed')
    return dict(clothing_vertices=clothing[None],clothing_faces=faces_out,
                body_barycentric_indices=indices,body_barycentric_weights=weights,
                reference_body_vertices=reference[None],clothing_rest_vertices=rest_clothing), dict(
                    waist_y=waist,stripe_min_y=waist+.13,stripe_max_y=waist+.19,
                    hem_y=hem,collar_y=collar,sleeve_left_x=left,sleeve_right_x=right,
                    max_vertex_offset_m=offset,minimum_face_normal_separation_m=float(separation.min()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);a=p.parse_args()
    start=time.monotonic();torch.set_num_threads(2)
    path=a.assets/'report.json';report=json.loads(path.read_text())
    backup=a.assets/'report_before_clothing_v2.json'
    if backup.exists():raise FileExistsError('Clothing revision already applied')
    write_json(backup,report);records=[]
    for item in report['characters']:
        mp=Path(item['manifest']);manifest=json.loads(mp.read_text())
        write_json(mp.with_name('manifest_before_clothing_v2.json'),manifest)
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=10,batch_size=1)
        with torch.no_grad():shaped=model(betas=torch.tensor(manifest['betas'],dtype=torch.float32)[None])
        pelvis=shaped.joints[0,0].numpy();rest=shaped.vertices[0].numpy()-pelvis;joints=shaped.joints[0].numpy()-pelvis
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==item['character_id']]
        for job in jobs:
            geom=Path(job['geometry']);body=np.load(geom/'vertices.npy')[0];faces=np.load(geom/'faces.npy')
            arrays,meta=clipped_shell(rest,joints,body,faces)
            out=geom.parent/'clothing_v2';out.mkdir(exist_ok=False)
            for name,array in arrays.items():np.save(out/f'{name}.npy',array)
            np.save(out/'face_material_ids.npy',np.zeros(len(arrays['clothing_faces']),np.int32))
            write_json(out/'materials.json',MATERIALS);write_json(out/'garment_material.json',meta)
            write_json(out/'clothing_manifest.json',dict(meta,body_geometry=str(geom),character_id=item['character_id'],
                split='integration_only',version=2,gender=manifest['gender'],betas=manifest['betas'],
                derived_from=job['clothing_geometry'],method='rest-space polygon clipping with barycentric posed-body correspondence; same 4mm outward shell',
                limitation='tight shell, no cloth dynamics or exhaustive self-intersection guarantee'))
            job['clothing_geometry']=str(out);records.append(dict(geometry=str(geom),clothing=str(out),**meta))
        manifest['clothing']['geometry_version']=2
        manifest['clothing']['revision_reason']='smooth triangle-clipped hems/collar/sleeves; continuous rest-coordinate color bands'
        write_json(mp,manifest)
    write_json(path,report)
    write_json(a.assets/'clothing_revision_v2.json',dict(split='integration_only',seconds=time.monotonic()-start,
                body_geometry_changed=False,status='pending_review',cases=records))


if __name__=='__main__':main()
