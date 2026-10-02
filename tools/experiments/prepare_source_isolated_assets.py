"""Reuse unchanged assets or regenerate body, simple clothes and bound hair per pose."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.io.geometry import save_geometry
from densekp_retrain.body.parameters import PARAMETERS
from densekp_retrain.io.json import write_json
from tools.experiments.audit_pedestrian_poses import parameters
from tools.experiments.prepare_adult_candidates import clothing
from tools.experiments.prepare_mixed_pedestrians import transfer_groom
from densekp_retrain.data.source_split import validate_split


def main():
    root=Path(sys.argv[1]).resolve();split=json.loads((root/'split_v2.json').read_text());validate_split(split)
    out=root/'assets';out.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.monotonic()
    prior=[]
    for name in ['cycling_ten_adults_candidates_20260930','mixed_riders_pedestrians_20260930']:
        prior.extend(json.loads((ROOT/'outputs/experiments'/name/'report.json').read_text())['derivatives'])
    R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    report=dict(characters=[{k:c[k] for k in ['character_id','manifest']} for c in split['characters']],
        derivatives=[],split='source_isolated_candidates',split_manifest=str(root/'split_v2.json'),
        quality_state='geometry_preparation_not_image_acceptance',compute_runs=[])
    if (out/'report.json').exists():report=json.loads((out/'report.json').read_text())
    foot=[]
    walk={'F01':50,'F03':30,'F05':50,'M01':60,'M04':40,'M05':60,'F02':20,'M02':40,'M03':20}
    for c in split['characters']:
        m=json.loads(Path(c['manifest']).read_text());cid=m['character_id'];short=cid.split('_')[1]
        side=c['partition'];src=split['source_groups'][side]
        model=load_smplx_model(Path(m['model_dir']),gender=m['gender'],num_betas=30,batch_size=1).eval()
        beta=np.array(m['betas'],np.float32);rest,joints,pelvis=rest_body(model,beta)
        source=np.load(src['source_file'])
        prefix='KIT314_bend_left01' if side=='train' else 'KIT425_walking_medium05'
        jobs=[(src['cycling'],'cycling',None),
              (f"{prefix}_f{src['standing_frame']:03d}",'standing',src['standing_frame']),
              (f'{prefix}_f{walk[short]:03d}','walking',walk[short])]
        for pose,action,frame in jobs:
            before=time.monotonic();old=next((d for d in prior if d['character_id']==cid and d['pose_source_id']==pose),None)
            completed=next((d for d in report['derivatives'] if d['character_id']==cid and d['pose_source_id']==pose),None)
            source_group=src['cycling'] if action=='cycling' else src['pedestrian']
            if completed:
                row=dict(completed)
            elif old:
                row=dict(old,reused_unchanged=True)
            else:
                folder=out/'derivatives'/cid/pose
                if action=='cycling':
                    source_params=ROOT/'outputs/experiments/cycling_smplx_20260929'/pose/'canonical_pose.npz'
                    z=np.load(source_params);params={k:z[k] for k in PARAMETERS}
                    params.update(betas=beta,transl=-pelvis[None])
                else:params=parameters(source['poses'][frame],beta,pelvis)
                row=save_geometry(folder/'geometry',model,params,dict(character_id=cid,pose_source_id=pose,
                    source_group=source_group,source_frame=frame,source_file=src['source_file'] if action!='cycling' else str(source_params),
                    source_betas=source['betas'].tolist() if action!='cycling' else None,
                    parent_manifest=c['manifest'],partition=side),R)
                geometry=Path(row['geometry'])
                row['clothing_geometry']=clothing(folder/'simple_clothing',geometry,rest,joints,short,cid)
                row['groom_geometry']=transfer_groom(m,geometry,folder/'groom')
                np.savez(folder/'canonical_pose.npz',**params)
                row.update(character_id=cid,pose_source_id=pose,appearance=m['appearance'],
                    pose_source_parameters=str(folder/'canonical_pose.npz'),reused_unchanged=False)
            row.update(action=action,source_frame=frame,source_group=source_group,
                original_pose_file=src['source_file'] if action!='cycling' else str(ROOT/'outputs/experiments/cycling_smplx_20260929'/pose/'canonical_pose.npz'),
                partition=side,identity_family_id=c['identity_family_id'],parent_manifest=c['manifest'],
                split='source_isolated_candidate_not_accepted',appearance_status='pending_new_scene_acceptance')
            saved=np.load(Path(row['geometry'])/'motion_params.npz')
            saved_beta=saved['betas'].reshape(-1)
            # Existing standard same-gender first-10 equivalence was independently
            # verified. This is not a locked-head donor shape conversion.
            np.testing.assert_allclose(np.pad(saved_beta,(0,len(beta)-len(saved_beta))),beta,atol=0,rtol=0)
            if action!='cycling':
                expected=parameters(source['poses'][frame],beta,pelvis)
                for key in PARAMETERS:np.testing.assert_array_equal(saved[key],expected[key])
                body=np.load(Path(row['geometry'])/'vertices.npy')[0];groups=model.lbs_weights.numpy().argmax(1)
                base=body[:,1].min();base_joint=int(groups[body[:,1].argmin()])
                if base_joint not in (7,8,10,11):raise ValueError('Lowest body point is not a foot')
                foot.append(dict(character_id=cid,pose_source_id=pose,base_joint=base_joint,
                    foot_height_m={s:float(body[np.isin(groups,ids),1].min()-base) for s,ids in [('left',[7,10]),('right',[8,11])]}))
            if not completed:
                report['derivatives'].append(row)
                report['compute_runs'].append(dict(character_id=cid,pose=pose,reused=bool(old),seconds=time.monotonic()-before))
            write_json(out/'report.json',report);print(cid,pose,'reused' if old else 'generated',flush=True)
    report['compute_seconds']=time.monotonic()-start;write_json(out/'report.json',report)
    write_json(out/'foot_geometry.json',dict(cases=foot,status='passed_local_foot_base; native_ground_not_yet_rendered'))

if __name__=='__main__':main()
