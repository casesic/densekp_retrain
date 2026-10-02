"""Versioned simple-clothing candidates. Never edits a source body or pose."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
from scipy.optimize import lsq_linear

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import PARAMETERS, evaluate, zero_pose
from tools.experiments.prepare_cycling_characters import POSES
from densekp_retrain.io.json import write_json
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.body.measurements import measurement_rows, measure
from densekp_retrain.io.geometry import save_geometry
from densekp_retrain.clothing.shells import clipped_shell

PILOT=ROOT/'outputs/experiments/cycling_commuter_pilot_20260930'
OLD=ROOT/'outputs/experiments/cycling_four_characters_20260929'
# sleeve: shoulder -> wrist, leg: hip -> ankle. Styles intentionally simplified.
STYLES={
 'F01':(.34,.36,'sport',[.08,.24,.55],[.025,.04,.08],[.8,.85,.9]),
 'F02':(.93,.95,'daily',[.57,.45,.30],[.035,.055,.1],[.57,.45,.30]),
 'F03':(.38,.40,'daily',[.42,.09,.045],[.28,.20,.10],[.53,.16,.08]),
 'F04':(.92,.94,'sport',[.025,.30,.24],[.018,.025,.028],[.40,.65,.45]),
 'F05':(.70,.88,'daily',[.19,.25,.07],[.055,.055,.065],[.25,.32,.1]),
 'M01':(.32,.42,'sport',[.65,.19,.025],[.02,.04,.09],[.9,.75,.38]),
 'M02':(.42,.95,'daily',[.15,.24,.31],[.28,.29,.30],[.15,.24,.31]),
 'M03':(.93,.42,'daily',[.25,.40,.55],[.45,.32,.17],[.4,.55,.65]),
 'M04':(.92,.72,'sport',[.27,.035,.07],[.045,.045,.055],[.55,.20,.22]),
 'M05':(.90,.95,'daily',[.035,.17,.09],[.17,.095,.045],[.055,.22,.13]),
}

def fitted_shape(model, short, labels):
    base=np.zeros(30,np.float32)
    v,j,_=rest_body(model,base)
    rows=measurement_rows(v,j,labels,model.lbs_weights[:,15].numpy())
    original=measure(v,rows)
    # Bounded whole-model fit. These are design constraints, not population labels.
    targets={
      'F03':(1.60,1.04,1.12,1.05,1.06,.96),
      'F04':(1.76,1.04,.97,.98,.97,1.04),
      'F05':(1.66,1.02,1.15,1.035,1.045,.99),
      'M03':(1.70,.95,.95,.96,.94,.97),
      'M04':(1.81,1.055,1.015,1.035,1.05,.99),
      'M05':(1.75,1.025,1.15,1.06,1.07,.98)}
    target=original*np.array(targets[short]);target[0]=targets[short][0]
    jac=[]
    for i in range(30):
        beta=base.copy();beta[i]=.1
        jac.append((measure(rest_body(model,beta)[0],rows)-original)/.1)
    jac=np.array(jac).T;tol=np.array([.01,.008,.008,.002,.002,.004])
    fit=lsq_linear(np.vstack((jac/tol[:,None],np.eye(30)*.6)),
                   np.r_[(target-original)/tol,np.zeros(30)],bounds=(-1.5,1.5))
    beta=fit.x.astype(np.float32);final=rest_body(model,beta)[0]
    return beta,dict(names=[r[0] for r in rows],target_m=target.tolist(),achieved_m=measure(final,rows).tolist(),
                     bounds=[-1.5,1.5],meaning='engineering bound, not population standard deviations',
                     method='whole SMPL-X shape-space regularized fit, expression zero',visual_status='pending')

def clothing(destination,geometry,rest,joints,short,cid):
    body=np.load(geometry/'vertices.npy')[0];faces=np.load(geometry/'faces.npy')
    sleeve,leg,style,top,pants,band=STYLES[short]
    profile=dict(sleeve=sleeve,leg=leg,collar=.045,style=style)
    arrays,meta=clipped_shell(rest,joints,body,faces,profile)
    destination.mkdir(parents=True)
    for name,value in arrays.items():np.save(destination/f'{name}.npy',value)
    np.save(destination/'face_material_ids.npy',np.zeros(len(arrays['clothing_faces']),np.int32))
    materials=[dict(name=n,color=c+[1]) for n,c in [('top',top),('pants',pants),('panel',band)]]
    write_json(destination/'materials.json',materials)
    meta.update(profile=profile,fabric='plain',fabric_note='continuous palette/panels; no rib shader claimed')
    write_json(destination/'garment_material.json',meta)
    write_json(destination/'clothing_manifest.json',dict(meta,body_geometry=str(geometry),character_id=cid,
        split='integration_only',method='existing clipped 4mm fitted shell with anatomical sleeve/leg lengths',
        limitation='fitted simple clothes; not straight trousers, polo or physically simulated cloth',
        appearance_status='diagnostic_pending',legacy_clearance_gate_m=.0004,legacy_offset_gate_m=.01))
    return str(destination)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--characters',nargs='+',default=['F02','M02']);a=p.parse_args()
    start=time.monotonic();torch.set_num_threads(2);root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    report_path=root/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(
        created_utc=datetime.now(timezone.utc).isoformat(),split='integration_only',characters=[],derivatives=[],
        policy='20260930 simplified candidates; appearance and numerical acceptance separate',
        status='diagnostic_candidates_not_accepted')
    plan=json.loads((ROOT/'outputs/plans/characters_10plus1_20260930/candidates.json').read_text())
    planned={r['character_id'].split('_')[-1]:r for r in plan['adults']}
    regressor=np.load(ROOT/'assets/densekp_regressor_R.npy')
    labels=np.load(ROOT/'outputs/body_texture_audit/body_uv.npz')['component_labels']
    for short in a.characters:
        gender='female' if short[0]=='F' else 'male'
        existing=short[-1] in '12';pilot=short.endswith('02')
        version=3 if pilot else (2 if existing else 1);cid=f'cycling_{short}_v{version}'
        out=root/'characters'/cid;out.mkdir(parents=True,exist_ok=False)
        model=load_smplx_model(PILOT/'models',gender=gender,num_betas=30,batch_size=1).eval()
        parent_path=(PILOT if pilot else OLD)/'characters'/f'cycling_{short}_v{2 if pilot else 1}'/'manifest.json'
        parent=json.loads(parent_path.read_text()) if existing else None
        if parent:
            beta=np.asarray(parent['betas'],np.float32);beta=np.pad(beta,(0,30-len(beta)));fit={'method':'unchanged parent shape'}
        else:beta,fit=fitted_shape(model,short,labels)
        rest,joints,pelvis=rest_body(model,beta)
        base=json.loads((OLD/'characters'/f'cycling_{short[0]}01_v1/manifest.json').read_text())
        appearance=deepcopy(parent['appearance'] if parent else base['appearance'])
        if not parent:
            skin=planned[short]['skin'];texture=out/(skin['source_id']+'.png')
            with zipfile.ZipFile(skin['archive']) as z:texture.write_bytes(z.read(skin['member']))
            appearance['body_texture']=str(texture)
        coords=np.zeros_like(rest)
        for label in (1,2):
            ids=np.flatnonzero(labels==label);coords[ids]=rest[ids]-rest[ids].mean(0)
        np.savez(out/'eye_mapping.npz',coordinates=coords,faces=model.faces,component_labels=labels)
        appearance.update(character_id=cid,eye_mapping=str(out/'eye_mapping.npz'),hair_mode='3D groom candidate')
        manifest=dict(character_id=cid,identity_family_id=f'cycling_{short}',parent_asset_id=parent['character_id'] if parent else None,
            parent_manifest=str(parent_path) if parent else None,version=version,gender=gender,status='body_frozen_appearance_candidate',
            split='integration_only',model_dir=str(PILOT/'models'),model_file=str(PILOT/'models'/f'SMPLX_{gender.upper()}.npz'),
            num_betas=30,betas=beta.tolist(),shape_fit=fit,expression_policy='zero; full-file 300:310, shape 0:30',
            appearance=appearance,texture_source=planned[short]['skin'],model_license=base['model_license'],
            texture_license=base['texture_license'],proportions=dict(height_m=float(np.ptp(rest[:,1]))),
            outfit=dict(style=STYLES[short][2],simplification='simple fitted lengths; no complex collar/straight-pants claim'))
        np.save(out/'rest_joints.npy',joints)
        source_report=json.loads(((PILOT if pilot else OLD)/'report.json').read_text()) if parent else None
        jobs=[]
        # F02/M02 P1 first, P2 retained only as optional diagnostic; others one pose.
        selected_poses=POSES if pilot else [POSES[0]]
        for pose in ['reference']+selected_poses:
            folder=out/'reference' if pose=='reference' else root/'derivatives'/cid/pose
            if parent:
                oldjob=parent['reference'] if pose=='reference' else next(d for d in source_report['derivatives']
                    if d['character_id']==parent['character_id'] and d['pose_source_id']==pose)
                geom=Path(oldjob['geometry']);params=np.load(geom/'motion_params.npz')
                pars={k:params[k] for k in PARAMETERS};pars['betas']=beta
                v,j=evaluate(model,pars)
                if np.abs(v-np.load(geom/'vertices.npy')).max()>2e-5:raise ValueError('Parent model regression changed')
                if np.abs(regressor@v[0]-np.load(geom/'densekp_3d.npy')[0]).max()>1e-6:raise ValueError('Parent R failed')
                row=dict(geometry=str(geom),groom_geometry=oldjob['groom_geometry']) if pilot else dict(geometry=str(geom))
            else:
                params=zero_pose(beta,pelvis)
                if pose!='reference':
                    source=ROOT/'outputs/experiments/cycling_smplx_20260929'/pose/'canonical_pose.npz'
                    z=np.load(source);params={k:z[k] for k in PARAMETERS};params.update(betas=beta,transl=-pelvis[None])
                row=save_geometry(folder/'geometry',model,params,dict(manifest,pose_source_id=pose),regressor)
                geom=Path(row['geometry'])
            row['clothing_geometry']=clothing(folder/'simple_clothing_v1',geom,rest,joints,short,cid)
            row.update(character_id=cid,pose_source_id=pose,appearance=appearance,split='integration_only',
                       appearance_status='diagnostic_pending',pose_source_parameters=None if pose=='reference' else
                       str(ROOT/'outputs/experiments/cycling_smplx_20260929'/pose/'canonical_pose.npz'))
            if pose=='reference':manifest['reference']=row
            else:jobs.append(row)
        write_json(out/'manifest.json',manifest)
        report['characters'].append(dict(character_id=cid,manifest=str(out/'manifest.json')));report['derivatives'].extend(jobs)
        write_json(report_path,report);print(cid,flush=True)
    report.setdefault('compute_runs',[]).append(dict(seconds=time.monotonic()-start,characters=a.characters))
    write_json(report_path,report)

if __name__=='__main__':main()
