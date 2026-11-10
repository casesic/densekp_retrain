"""Bounded local walking adapter: fixed target shape, source root, fixed bindings.

This adapter prepares generic asset manifests for the road_pass consumer. It
does not reinterpret the historical source split or alter character assets.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import sys
import time
import resource
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
BASE=ROOT/'outputs/experiments/pedestrian_seeded_pilot_a1_20261009'

def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    import torch
    from smplx.lbs import batch_rodrigues
    from densekp_retrain.body.geometry import load_smplx_model,compute_densekp_3d
    from densekp_retrain.body.parameters import rest_body,parameters_from_pose,evaluate
    from densekp_retrain.body.sequence_binding import bound_garment,head_transform
    torch.set_num_threads(2); began=time.monotonic()
    source=ROOT/'outputs/experiments/source_isolated_pilot_20260930/sources/us_2495_L_2200.npz'
    s=dict(np.load(source,allow_pickle=False)); assert float(s['mocap_frame_rate'])==30 and s['poses'].shape==(127,165)
    # Store real contiguous source frames once; episode selection uses stride=3.
    frames=np.arange(9,103); assert np.isfinite(s['poses']).all() and np.isfinite(s['trans']).all()
    R=np.load(ROOT/'assets/densekp_regressor_R.npy'); assets=[]
    for character in ['cycling_M01_v2','cycling_M04_v1']:
        char=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters'/character/'manifest.json'
        m=json.loads(char.read_text()); ref=Path(m['reference']['geometry']);cloth=Path(m['reference']['clothing_geometry']);hair=Path(m['reference']['groom_geometry'])
        out=BASE/'assets_attempt_2'/character;out.mkdir(parents=True,exist_ok=False)
        beta=np.array(m['betas'],np.float32);assert beta.shape==(30,)
        model=load_smplx_model(Path(m['model_dir']),gender=m['gender'],num_betas=30,batch_size=1).eval();model.requires_grad_(False)
        rest,jrest,pelvis=rest_body(model,beta);np.testing.assert_allclose(rest,np.load(ref/'vertices.npy')[0],atol=2e-6,rtol=0)
        ids=np.load(cloth/'body_barycentric_indices.npy');w=np.load(cloth/'body_barycentric_weights.npy');cf=np.load(cloth/'clothing_faces.npy')
        assert ids.min()>=0 and ids.max()<10475 and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-6)
        cr=bound_garment(rest,model.faces,ids,w,offset_m=.004)
        np.testing.assert_allclose(cr,np.load(cloth/'clothing_vertices.npy')[0],atol=2e-6,rtol=0)
        gi=json.loads((hair/'groom_manifest.json').read_text()); gb=np.load(Path(gi['binding'])/'binding.npz')
        active=np.load(hair/'active_curve_mask.npy');tri=gb['body_faces'][active];gw=gb['weights'][active]
        restroots=(rest[tri]*gw[...,None]).sum(1)
        vertices=[];joints=[];clothes=[];heads=[];parameters=[];records=[]
        for f in frames:
            p=parameters_from_pose(s['poses'][f],beta,pelvis)
            # Source-root relative to common frame9; scene consumer subtracts episode start once.
            p['transl']=(s['trans'][f]-s['trans'][9]-pelvis).astype(np.float32)[None]
            v,j=evaluate(model,p);v,j=v[0],j[0]
            cv=bound_garment(v,model.faces,ids,w,offset_m=.004)
            local=batch_rodrigues(torch.from_numpy(s['poses'][f,:66].astype(np.float32).reshape(22,3))).numpy()
            H=head_transform(local,model.parents,j[15],jrest[15])
            error=np.linalg.norm(restroots@H[:3,:3].T+H[:3,3]-(v[tri]*gw[...,None]).sum(1),axis=1)
            # Walking pose correctives are not identical to the APD fit. Preserve
            # the rigid head motion and fixed topology; report its actual error.
            # This diagnostic gate is explicitly different from A2's 1cm gate.
            assert np.isfinite(v).all() and error.max()<.02 and np.quantile(error,.95)<.005
            vertices.append(v);joints.append(j[:22]);clothes.append(cv);heads.append(H);parameters.append(p)
            records.append(dict(source_frame=int(f),lowest_model_y_m=float(v[:,1].min()),groom_anchor_max_m=float(error.max()),groom_anchor_p95_m=float(np.quantile(error,.95)),anchors_above_1cm=int((error>.01).sum())))
        vertices=np.array(vertices);joints=np.array(joints)
        for name,value in [('vertices',vertices),('joints',joints),('clothing_vertices',clothes),('head_transforms',heads),('source_frames',frames),('faces',model.faces),('densekp_3d',compute_densekp_3d(R,vertices))]:np.save(out/f'{name}.npy',value)
        np.savez(out/'parameters.npz',**{k:np.concatenate([p[k] for p in parameters]) for k in parameters[0] if k!='betas'},betas=beta,source_frame_ids=frames,source_root_relative=s['trans'][frames]-s['trans'][9])
        dynamics=[]
        for start in [9,12]:
            f=np.arange(start,start+91,3);root=s['trans'][f];steps=np.linalg.norm(np.diff(root[:,[0,2]],axis=0),axis=1)/.1
            relative=joints[f-9]-joints[f-9,:1]
            foot=np.linalg.norm(np.diff(relative[:,[7,8,10,11]],axis=0),axis=2)
            dynamics.append(dict(start_frame=start,end_frame=int(f[-1]),stride=3,span_seconds=3,
                root_speed_min_m_s=float(steps.min()),root_speed_max_m_s=float(steps.max()),root_speed_mean_m_s=float(steps.mean()),
                maximum_foot_motion_each_step_m=foot.max(1).tolist(),root_speeds_m_s=steps.tolist()))
            assert (steps>.1).all() and (foot.max(1)>.005).all(), 'Walking window contains stationary output interval'
        manifest=dict(schema='densekp_dynamic_asset_v1',asset_id=character+'_local_walk',character_manifest=str(char),
            character_family=character.rsplit('_v',1)[0],geometry=str(out),clothing_reference=str(cloth),groom_reference=str(hair),
            parameters=str(out/'parameters.npz'),source_ref=str(source),motion_source_id='KIT/425/walking_medium05',
            available_source_frames=[9,102],nominal_source_rate=[30,1],source_start_reference=9,
            original_100fps_mapping=dict(status='uncertain',nominal_index_multiplier=3,not_verified_original_timestamps=True),
            fixed_target_betas=beta.tolist(),source_betas_used=False,root_mode='source_root_relative_to_frame9',
            clothing_offset_m=.004,head_joint=15,available_combinations_verified=True,
            binding_reference_reproduced=True,dynamic_numeric_checked_frames=len(frames),human_dynamic_review='pending',
            walking_groom_diagnostic_gate=dict(max_m=.02,p95_m=.005,not_equivalent_to_A2_1cm=True),
            prior_split='historical_validation_source',integration_only=True,independent_validation=False,
            dynamics=dynamics,per_frame=records)
        write(out/'manifest.json',manifest)
        assets.append(dict(asset_id=manifest['asset_id'],manifest=str(out/'manifest.json'),character_family=manifest['character_family'],available_combinations_verified=True,allowed_start_frames=[9,12]))
        print(character, dynamics[0]['root_speed_min_m_s'],flush=True)
    write(BASE/'asset_pool.json',assets)
    write(BASE/'asset_preparation.json',dict(seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_threads=2,device='CPU',frames_per_character=94,characters=2,no_optimization=True))

if __name__=='__main__':main()
