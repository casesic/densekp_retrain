"""Read-only source/target validation, reusing project SMPL-X and R evaluators."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/experiments')]
from densekp_retrain.body.geometry import load_smplx_model,compute_densekp_3d
from prepare_cycling_female_asset_a2 import POSE_KEYS

def main():
    start=time.monotonic();torch.set_num_threads(2)
    out=ROOT/'outputs/experiments/mtb_f02_dressed_asset_a2_20261009'
    result=out/'readback.json';assert not result.exists()
    recipe=json.loads((out/'frozen_recipe.json').read_text())
    char=json.loads(Path(recipe['character_manifest']).read_text())
    p=dict(np.load(out/'parameters.npz'));src=dict(np.load(recipe['source_authoritative_parameters']))
    for key in src:
        if key not in ['betas','transl']:np.testing.assert_array_equal(p[key],src[key])
    np.testing.assert_array_equal(p['betas'],np.array(char['betas'],np.float32))
    np.testing.assert_array_equal(p['source_pts'],8670662+1001*np.arange(120))
    np.testing.assert_array_equal(p['source_decoded_frames'],np.arange(8662,8782))
    m=load_smplx_model(Path(char['model_dir']),gender='female',num_betas=30,batch_size=1).eval()
    m.requires_grad_(False)
    assert not m.use_pca and m.flat_hand_mean
    assert torch.count_nonzero(m.pose_mean)==0
    v=np.load(out/'full/vertices.npy');cloth=np.load(out/'full/clothing_vertices.npy');H=np.load(out/'full/head_transforms.npy')
    dense=np.load(out/'full/densekp_3d.npy');R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    errors=[]
    with torch.no_grad():
        for i in range(120):
            kw={k:torch.from_numpy(p[k][i:i+1]) for k in POSE_KEYS}
            q=m(**kw,betas=torch.from_numpy(p['betas'][None]),transl=torch.from_numpy(p['transl'][i:i+1]))
            errors.append(float(np.max(np.abs(q.vertices.numpy()[0]-v[i]))))
    assert max(errors)<2e-6
    rerr=float(np.max(np.abs(compute_densekp_3d(R,v)-dense)));assert rerr<2e-6
    for array in [v,cloth,H,dense]:assert np.isfinite(array).all()
    assert H.shape==(120,4,4)
    np.testing.assert_allclose(H[:,:3,:3]@H[:,:3,:3].transpose(0,2,1),np.broadcast_to(np.eye(3),(120,3,3)),atol=2e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.det(H[:,:3,:3]),1,atol=2e-6)
    # One diagnostic scene transform, not a produced road placement.
    S=np.array([[0,0,1,2],[1,0,0,3],[0,1,0,4],[0,0,0,1]],float)
    hair=np.load(Path(recipe['source_groom'])/'points.npy',mmap_mode='r')[::1024]
    composite=max(float(np.max(np.abs((hair@h[:3,:3].T+h[:3,3])@S[:3,:3].T+S[:3,3]-(hair@(S@h)[:3,:3].T+(S@h)[:3,3])))) for h in H)
    assert composite<1e-12
    diagnostic_world=compute_densekp_3d(R,v@S[:3,:3].T+S[:3,3])
    comm=float(np.max(np.abs(diagnostic_world-(dense@S[:3,:3].T+S[:3,3]))));assert comm<2e-6
    info=dict(integration_only=True,status='passed',frames=120,body_readback_max_m=max(errors),R_readback_max_m=rerr,
        rotation_and_time_arrays_exact=True,shape_fixed=True,model_gender='female',num_betas=30,
        hand_semantics=dict(use_pca=False,flat_hand_mean=True,pose_mean_all_zero=True,full_axis_angle_per_hand=45),
        expression='10 coefficients unchanged, full model expression basis indices300:310 per existing model_semantics.json',
        changed_fields=['female model/template','fixed original F02 30betas','constant transl=-F02 rest pelvis','gender/root_mode descriptors'],
        camera_axis_reapplied=False,source_cam_translation_reapplied=False,no_per_frame_grounding=True,
        body_bounds_model_m=[v.min((0,1)).tolist(),v.max((0,1)).tolist()],cloth_bounds_model_m=[cloth.min((0,1)).tolist(),cloth.max((0,1)).tolist()],
        body_minY_range_m=[float(v[:,:,1].min(1).min()),float(v[:,:,1].min(1).max())],
        scene_groom_composition_error_m=composite,scene_groom_check_stride=1024,R_scene_commutation_error_m=comm,
        original_all_groom_points_finite_check='validation.json',support_height='not selected; scene-owned fixed height, no M01 height inheritance',
        source_body_motion='human_body_motion_accepted',target_dressed_review='pending_user',elapsed_seconds=time.monotonic()-start)
    result.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))

if __name__=='__main__':main()
