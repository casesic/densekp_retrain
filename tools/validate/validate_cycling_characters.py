#!/usr/bin/env python3
"""Independent saved-parameter, identity, garment-boundary and DS checks."""
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
from densekp_retrain.body.parameters import PARAMETERS
from tools.experiments.prepare_cycling_characters import POSES
from densekp_retrain.io.json import write_json


def check_face_uv(config):
    u=np.linspace(.10,.50,801);v=np.linspace(.70,.97,541)
    x,y=np.meshgrid(u,v)
    def window(value,knots):
        return np.clip((value-knots[0])/(knots[1]-knots[0]),0,1)*np.clip((knots[3]-value)/(knots[3]-knots[2]),0,1)
    weight=window(x,config['x_knots'])*window(y,config['y_knots'])
    mapped_x=x+weight*((config['u_scale']-1)*x+config['offset'][0])
    mapped_y=y+weight*config['offset'][1]
    dx_dv,dx_du=np.gradient(mapped_x,v,u);dy_dv,dy_du=np.gradient(mapped_y,v,u)
    determinant=dx_du*dy_dv-dx_dv*dy_du
    minimum=float(determinant.min())
    if minimum<=0:raise ValueError('Face UV warp folds or reverses orientation')
    return minimum


def check_clothing(geometry, clothing):
    body=np.load(geometry/'vertices.npy')[0]
    cv=np.load(clothing/'clothing_vertices.npy')[0];cf=np.load(clothing/'clothing_faces.npy')
    indices=np.load(clothing/'body_barycentric_indices.npy');weights=np.load(clothing/'body_barycentric_weights.npy')
    if (weights<0).any():raise ValueError('Negative barycentric weights')
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-6,rtol=0)
    reference=(body[indices]*weights[:,:,None]).sum(1)
    np.testing.assert_array_equal(reference,np.load(clothing/'reference_body_vertices.npy')[0])
    offset=float(np.linalg.norm(cv-reference,axis=1).max())
    triangles=reference[cf]; normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    separation=((cv[cf].mean(1)-triangles.mean(1))*normals).sum(1)
    if separation.min()<.0004-2e-7 or offset>.01:raise ValueError('Existing clothing clearance/offset gate failed')
    edges=np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    if counts.max()>2:raise ValueError('Nonmanifold clothing edge')
    rest=np.load(clothing/'clothing_rest_vertices.npy');ep=rest[unique[counts==1]]
    material=json.loads((clothing/'garment_material.json').read_text())
    intentional=np.zeros(len(ep),bool)
    for axis,key in [(0,'sleeve_left_x'),(0,'sleeve_right_x'),(1,'hem_y'),(1,'collar_y')]:
        intentional|=(np.abs(ep[:,:,axis]-material[key])<2e-6).all(1)
    if not intentional.all():raise ValueError(f'Unexpected open garment boundary: {int((~intentional).sum())} edges')
    return dict(max_offset_m=offset,min_local_clearance_m=float(separation.min()),
                boundary_edges=len(ep),unexpected_open_edges=0,manifold_edges=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ds',type=Path)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.assets/'report.json').read_text())
    regressor=np.load(ROOT/'assets/densekp_regressor_R.npy');rows=[];uv_checks=[]
    for item in source['characters']:
        manifest=json.loads(Path(item['manifest']).read_text())
        uv_checks.append(dict(character_id=item['character_id'],minimum_sampled_uv_jacobian=check_face_uv(manifest['appearance']['face_uv_alignment'])))
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=manifest['num_betas'],batch_size=1)
        derivatives=[d for d in source['derivatives'] if d['character_id']==item['character_id']]
        if {d['pose_source_id'] for d in derivatives}!=set(POSES):raise ValueError('Missing independent pose')
        for d in derivatives:
            geom=Path(d['geometry']);params=np.load(geom/'motion_params.npz')
            np.testing.assert_array_equal(params['betas'],np.asarray(manifest['betas'],np.float32))
            if d['appearance']!=manifest['appearance']:raise ValueError('Identity appearance changed across poses')
            gm=json.loads((geom/'geometry_manifest.json').read_text())
            original=np.load(gm['source_parameters'])
            for key in PARAMETERS:
                if key not in ('betas','transl'):np.testing.assert_array_equal(params[key],original[key])
            tensors={k:torch.from_numpy(params[k]) for k in PARAMETERS};tensors['betas']=tensors['betas'][None]
            with torch.no_grad():restored=model(**tensors)
            body=np.load(geom/'vertices.npy');faces=np.load(geom/'faces.npy')
            np.testing.assert_array_equal(faces,model.faces)
            error=float(np.abs(restored.vertices.numpy()-body).max())
            op_error=float(np.abs(regressor@body[0]-np.load(geom/'densekp_3d.npy')[0]).max())
            if error>2e-5 or op_error>1e-6:raise ValueError('Existing mesh/operator thresholds failed')
            if np.abs(restored.joints[0,0].numpy()).max()>2e-6:raise ValueError('Pelvis not independently recentered')
            rows.append(dict(character_id=d['character_id'],pose_source_id=d['pose_source_id'],
                             export_max_error_m=error,operator_max_error_m=op_error,
                             clothing=check_clothing(geom,Path(d['clothing_geometry']))))
    scenes=[]
    if a.ds:
        from PIL import Image
        for name in ['cycling_forest','urban_street_assets']:
            directory=a.ds/name
            validation=json.loads((directory/'validation.json').read_text())
            if validation['status']!='passed':raise ValueError('DS validation not passed')
            people=json.loads((directory/'people.json').read_text())['people']
            for camera in ['cam2','cam4']:
                valid=np.asarray(Image.open(directory/camera/'valid_mask.png'))>0
                for b in json.loads((directory/camera/'boxes.json').read_text()):
                    mask=np.asarray(Image.open(directory/camera/f"person_{b['person_id']}_mask.png"))>0
                    if np.any(mask&~valid):raise ValueError('Oracle instance exceeds valid domain')
                    yy,xx=np.where(mask)
                    box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None
                    if b['bbox_xyxy_exclusive']!=box:raise ValueError('Oracle box does not bound saved mask')
            with np.load(directory/'diagnostic_loader.npz') as z:
                lookup={p['id']:p for p in people}
                ids=[lookup[int(i)] for i in z['person_id']]
                sidecar=dict(split='integration_only',rows=[dict(index=i,character_id=p['character_id'],
                    pose_source_id=p['pose_source_id'],camera=str(z['camera_id'][i]),source_image=str(z['imgname'][i])) for i,p in enumerate(ids)])
                write_json(directory/'diagnostic_loader_identity.json',sidecar)
            scenes.append(validation)
    write_json(a.output,dict(status='passed',split='integration_only',seconds=time.monotonic()-start,
         characters=4,derivatives=len(rows),cases=rows,ds_scenes=scenes,face_uv_checks=uv_checks,
         thresholds=dict(parameter_mesh_m=2e-5,operator_m=1e-6,world_mesh_m=5e-6,
                         clothing_world_m=5e-6,projection_crop_px=2e-4,
                         clothing_clearance_m=.0004,clothing_max_offset_m=.01,
                         clearance_solver_tolerance_m=1e-8,float32_clothing_readback_tolerance_m=2e-7),
         limitations=['garment topology/local outward clearance is not an exhaustive collision proof',
                      'visibility uses existing depth/instance approximation; no inference or training']))
    print(json.dumps(dict(status='passed',derivatives=len(rows),ds_scenes=len(scenes),seconds=time.monotonic()-start)))


if __name__=='__main__':main()
