"""Transfer reviewed local walking poses onto frozen adult identities."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from tools.experiments.audit_pedestrian_poses import SOURCES, parameters
from tools.experiments.prepare_adult_candidates import clothing
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.io.geometry import save_geometry
from densekp_retrain.io.json import write_json
from densekp_retrain.clothing.binding import CONV, triangle_frame
from densekp_retrain.clothing.binding import closest_surface
from densekp_retrain.body.geometry import load_smplx_model

OLD = ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930'
JOBS = [('F01', 0), ('M01', 0), ('F02', 30), ('M02', 40), ('F01', 50), ('M01', 60)]


def transfer_groom(manifest, geometry, out):
    """Rotate the frozen rest correction with its bound scalp triangle; no new pruning."""
    source = Path(manifest['reference']['groom_geometry'])
    info = json.loads((source/'groom_manifest.json').read_text())
    binding = np.load(Path(info['binding'])/'binding.npz')
    tri, weights = binding['body_faces'], binding['weights']
    offsets = binding['curve_offsets']; pc = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    rest = np.load(Path(manifest['reference']['geometry'])/'vertices.npy')[0] @ CONV.T
    body = np.load(geometry/'vertices.npy')[0]; posed = body @ CONV.T
    rotation = np.einsum('nij,nkj->nik', triangle_frame(posed,tri), triangle_frame(rest,tri))
    frames = np.einsum('nij,njk->nik', rotation, binding['target_rest_frames'])
    roots = (posed[tri]*weights[:,:,None]).sum(1)+frames[:,:,2]*.0004
    base = (roots[pc]+np.einsum('nij,nj->ni',frames[pc],binding['local_points'])) @ CONV
    correction = np.load(source/'collision_correction.npy')
    correction = np.einsum('nij,nj->ni',rotation[pc],correction @ CONV.T) @ CONV
    points = (base+correction).astype(np.float32)
    out.mkdir(parents=True)
    for name, value in dict(points=points, target_roots=roots@CONV, collision_correction=correction,
                            radius=np.load(source/'radius.npy'), curve_offsets=offsets,
                            active_curve_mask=np.load(source/'active_curve_mask.npy')).items():
        np.save(out/f'{name}.npy',value)
    faces=np.load(geometry/'faces.npy');tree=cKDTree(body[faces].mean(1))
    active=np.load(source/'active_curve_mask.npy');sample=np.arange(0,len(points),50)
    sample=sample[active[pc[sample]]]
    signed=np.concatenate([closest_surface(points[sample[i:i+8192]],body,faces,tree)[2]
                           for i in range(0,len(sample),8192)])
    info.update(body_geometry=str(geometry), parent_groom=str(source),
                transfer='fixed reference binding and frozen rest corrections rotated by scalp triangle; active curves unchanged',
                sample_min_signed_m=float(signed.min()),sample_below_minus_1mm=int((signed<-.001).sum()),
                sample_count=len(sample),status='diagnostic_pending',split='integration_only')
    write_json(out/'groom_manifest.json',info)
    return str(out)


def main():
    root=Path(sys.argv[1]).resolve(); start=time.monotonic();torch.set_num_threads(2)
    old=json.loads((OLD/'report.json').read_text()); z=np.load(SOURCES[1]);R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    selected={short:next(c for c in old['characters'] if c['character_id'].split('_')[1]==short) for short,_ in JOBS}
    report=dict(characters=list(selected.values()),derivatives=[],split='integration_only',status='diagnostic_pending',
                source_group='AMASS_KIT_314_bend_left01',source_original_subject='KIT/314; target identity unrelated',
                parent_library=str(OLD),source_file=str(SOURCES[1]))
    records=[]
    for short,frame in JOBS:
        manifest=json.loads(Path(selected[short]['manifest']).read_text());cid=manifest['character_id']
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=30,batch_size=1).eval()
        beta=np.asarray(manifest['betas'],np.float32);rest,joints,pelvis=rest_body(model,beta)
        pose=f'KIT314_bend_left01_f{frame:03d}';folder=root/'derivatives'/cid/pose
        params=parameters(z['poses'][frame],beta,pelvis)
        row=save_geometry(folder/'geometry',model,params,dict(character_id=cid,pose_source_id=pose,
            source_file=str(SOURCES[1]),source_frame=frame,source_group=report['source_group'],
            source_transl=z['trans'][frame].tolist(),source_betas=z['betas'].tolist(),shape_source=selected[short]['manifest'],
            pose_policy='all 165 axis-angle values retained; target shape fixed; original translation not used'),R)
        geometry=Path(row['geometry'])
        row['clothing_geometry']=clothing(folder/'simple_clothing',geometry,rest,joints,short,cid)
        row['groom_geometry']=transfer_groom(manifest,geometry,folder/'groom')
        np.savez(folder/'canonical_pose.npz',**params)
        row.update(character_id=cid,pose_source_id=pose,source_frame=frame,source_group=report['source_group'],
            pose_source_parameters=str(folder/'canonical_pose.npz'),
            appearance=manifest['appearance'],action='standing' if frame==0 else 'walking',
            split='integration_only',parent_manifest=selected[short]['manifest'],ground_policy='lowest body sole at scene surface; no riding hover')
        report['derivatives'].append(row);write_json(root/'report.json',report)
        # Left/right foot support is measured on semantic ankle/toe joint skinning weights.
        body=np.load(geometry/'vertices.npy')[0];groups=model.lbs_weights.numpy().argmax(1)
        minimum=body[:,1].min();feet={side:float(body[np.isin(groups,ids),1].min()-minimum)
            for side,ids in [('left',[7,10]),('right',[8,11])]}
        records.append(dict(character_id=cid,pose_source_id=pose,foot_minimum_above_body_base_m=feet,
                            body_base_vertex=int(body[:,1].argmin()),body_base_joint=int(groups[body[:,1].argmin()])))
        print(cid,pose,feet,flush=True)
    # Reuse riding geometry unchanged for these four identities.
    report['derivatives'].extend(dict(d,action='cycling') for d in old['derivatives']
        if d['character_id'] in {c['character_id'] for c in selected.values()} and d['pose_source_id']=='coco_292226_person_1330527')
    report['compute_seconds']=time.monotonic()-start;write_json(root/'report.json',report)
    write_json(root/'foot_geometry.json',dict(cases=records,meaning='pose foot clearance before scene-ground lookup; minimum must belong to foot'))


if __name__=='__main__':main()
