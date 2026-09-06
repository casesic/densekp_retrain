#!/usr/bin/env python3
"""从 canonical real motion 生成 P0 单一几何制品。"""
import argparse,json,time
from pathlib import Path
import numpy as np, smplx, torch
def main():
    p=argparse.ArgumentParser(); p.add_argument('--motion',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--regressor',type=Path,default=Path('assets/densekp_regressor_R.npy')); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--avatar-id'); p.add_argument('--motion-id'); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); st=time.perf_counter()
    with np.load(a.motion,allow_pickle=False) as z: q={k:z[k] for k in z.files}
    ids=q['source_frame_ids']; t=len(ids); gender=str(q['gender'].item()); kwargs=dict(gender=gender,ext='npz',num_betas=len(q['betas']),use_pca=False,flat_hand_mean=True,batch_size=t)
    model=smplx.SMPLX(str(a.model_dir),**kwargs) if (a.model_dir/f'SMPLX_{gender.upper()}.npz').is_file() else smplx.create(str(a.model_dir),model_type='smplx',**kwargs)
    names=['global_orient','body_pose','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','transl']; inp={n:torch.from_numpy(np.asarray(q[n],np.float32)) for n in names}; inp['betas']=torch.from_numpy(np.repeat(q['betas'][None],t,axis=0).astype(np.float32))
    with torch.no_grad(): vertices=model(return_verts=True,**inp).vertices.cpu().numpy().astype(np.float32)
    faces=np.asarray(model.faces,np.int32); r=np.load(a.regressor)
    if vertices.shape!=(t,10475,3) or r.shape!=(138,10475) or not np.isfinite(vertices).all(): raise RuntimeError(f'geometry invalid {vertices.shape} R={r.shape}')
    dense=np.einsum('kv,tvc->tkc',r,vertices,optimize=True).astype(np.float32)
    np.save(a.output_dir/'vertices.npy',vertices); np.save(a.output_dir/'faces.npy',faces); np.save(a.output_dir/'densekp_3d.npy',dense)
    # canonical 参数原样进入 geometry 目录，保留 source frame mapping 与未使用 translation。
    np.savez(a.output_dir/'motion_params.npz',**q)
    m={'artifact':'p0_real_smplx_geometry','training_data':False,'source_motion':str(a.motion.resolve()),'avatar_id':a.avatar_id or f'smplx_{gender}_source_betas','motion_id':a.motion_id or a.motion.stem,'gender':gender,'frames':t,'source_total_frames':int(q['source_total_frames']),'source_frame_ids':ids.tolist(),'source_translation_used':bool(q['source_translation_used'].item()),'vertices_shape':list(vertices.shape),'faces_shape':list(faces.shape),'densekp_3d_shape':list(dense.shape),'dtype':'float32','model_dir':str(a.model_dir.resolve()),'regressor':str(a.regressor.resolve()),'generation_seconds':time.perf_counter()-st,'max_vertex_change_test_first_to_last_m':float(np.max(np.abs(vertices[-1]-vertices[0])))}
    (a.output_dir/'geometry_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
