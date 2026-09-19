#!/usr/bin/env python3
"""从匹配的 BEDLAM 6 fps clothing NPZ 提取衣物几何。"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def main():
    p=argparse.ArgumentParser(); p.add_argument('--body-geometry',type=Path,required=True); p.add_argument('--clothing-npz',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--all-frames',action='store_true'); a=p.parse_args()
    clothing_ids=np.arange(39,dtype=np.int64) if a.all_frames else np.asarray([0,2,5,7,10,12,15,17,20,22,25,27,30,32,35,38],np.int64); motion_ids=clothing_ids*5; source_ids=101+motion_ids
    with np.load(a.clothing_npz,allow_pickle=True) as z:
        vertices=np.asarray(z['vertices_seq'][clothing_ids],np.float32); faces=np.asarray(z['faces'],np.int32); vertices_uv=np.asarray(z['vertices_uv']); faces_uv=np.asarray(z['faces_uv']); metadata=dict(z['metadata'].item())
    body=np.load(a.body_geometry/'vertices.npy')
    if vertices.shape!=(len(clothing_ids),13027,3) or faces.shape!=(26328,3): raise RuntimeError(f'clothing shape 异常: {vertices.shape}, {faces.shape}')
    if metadata!={'fps':6,'source_fps':30,'start_frame':101,'end_frame':292}: raise RuntimeError(f'metadata 与已审计映射不符: {metadata}')
    if not np.array_equal(np.load(a.body_geometry/'motion_params.npz')['source_frame_ids'],motion_ids): raise RuntimeError('body/clothing frame mapping 不一致')
    before=hashlib.blake2b(body.tobytes(),digest_size=16).hexdigest(); nearest=[]
    for b,c in zip(body,vertices): nearest.append(cKDTree(b).query(c,k=1,workers=-1)[0])
    nearest=np.concatenate(nearest); after=hashlib.blake2b(body.tobytes(),digest_size=16).hexdigest()
    a.output_dir.mkdir(parents=True,exist_ok=True); np.save(a.output_dir/'clothing_vertices.npy',vertices); np.save(a.output_dir/'clothing_faces.npy',faces); np.save(a.output_dir/'clothing_vertices_uv.npy',vertices_uv); np.save(a.output_dir/'clothing_faces_uv.npy',faces_uv)
    rows=['render_frame,clothing_frame_index,local_motion_index,source_frame_30fps']+[f'{i},{c},{m},{s}' for i,(c,m,s) in enumerate(zip(clothing_ids,motion_ids,source_ids))]; (a.output_dir/'dressed_frame_mapping.csv').write_text('\n'.join(rows)+'\n')
    report={'subject_id':'rp_aaron_posed_009','animation_id':'1095','clothing_id':'rp_aaron_posed_009/1095','clothing_source':str(a.clothing_npz.resolve()),'clothing_frame_indices':clothing_ids.tolist(),'local_motion_indices':motion_ids.tolist(),'source_frame_30fps':source_ids.tolist(),'metadata':metadata,'vertices_shape':list(vertices.shape),'faces_shape':list(faces.shape),'vertices_uv_shape':list(vertices_uv.shape),'faces_uv_shape':list(faces_uv.shape),'body_geometry_fingerprint':{'algorithm':'blake2b-128','before':before,'after':after,'identical':before==after},'alignment_nearest_body_m':{'median':float(np.median(nearest)),'p95':float(np.percentile(nearest,95)),'p99':float(np.percentile(nearest,99)),'max':float(nearest.max())}}
    (a.output_dir/'clothing_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
