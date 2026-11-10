#!/usr/bin/env python3
"""Read-only geometric checks of the frozen-body dressed pilot; write new diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.clothing.binding import closest_surface


def main():
    began=time.perf_counter();b=ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008';out=b/'diagnostics';out.mkdir(exist_ok=False)
    recipe=json.loads((b/'frozen_recipe.json').read_text());body=np.load(recipe['body_cache'],mmap_mode='r');cfolder=Path(recipe['source_clothing']);gfolder=Path(recipe['source_groom'])
    faces=np.load(Path(recipe['body_cache']).parent.parent/'faces.npy');cf=np.load(cfolder/'clothing_faces.npy')
    ids=np.load(cfolder/'body_barycentric_indices.npy');w=np.load(cfolder/'body_barycentric_weights.npy');c=np.load(b/'full/clothing_vertices.npy');H=np.load(b/'full/head_transforms.npy')
    assert c.shape==(76,2612,3) and H.shape==(76,4,4) and np.isfinite(c).all() and np.isfinite(H).all()
    active=np.load(gfolder/'active_curve_mask.npy');off=np.load(gfolder/'curve_offsets.npy');points=np.load(gfolder/'points.npy',mmap_mode='r')
    # Fixed membership, three points per selected curve; neither resample nor prune by pose.
    curves=np.flatnonzero(active)[::64];sample_ids=np.concatenate([off[curves],off[curves]+12,off[curves+1]-1]);sample=points[sample_ids]
    np.save(out/'groom_diagnostic_point_ids.npy',sample_ids)
    base=(body[:,ids]*w[None,...,None]).sum(2);disp=c-base
    distances=np.linalg.norm(disp,axis=-1);assert abs(distances-.004).max()<1e-6
    records=[]
    for n in range(76):
        t=base[n,cf];nn=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);ct=c[n,cf];cn=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]);area=np.linalg.norm(cn,axis=-1)/2
        bad=(cn*nn).sum(1)<0
        hair=sample@H[n,:3,:3].T+H[n,:3,3];near,_,signed,_=closest_surface(hair,body[n],faces)
        records.append(dict(asset_frame=n,source_frame=n+30,flipped_face_ids=np.flatnonzero(bad).tolist(),
            flipped_area_fraction=float(area[bad].sum()/area.sum()),max_flipped_face_area_mm2=float(area[bad].max()*1e6) if bad.any() else 0.,
            groom_sample_count=len(sample),groom_sample_signed_min_m=float(signed.min()),groom_sample_below_minus1mm=int((signed<-.001).sum()),
            groom_sample_below_minus5mm=int((signed<-.005).sum()),groom_sample_nearest_distance_max_m=float(np.linalg.norm(hair-near,axis=1).max())))
    p=np.load(recipe['authoritative_body']);np.testing.assert_array_equal(np.load(b/'full/asset_frames.npy'),np.arange(76))
    probe=np.load(b/'probe/asset_frames.npy');assert np.array_equal(c[probe],np.load(b/'probe/clothing_vertices.npy'));assert np.array_equal(H[probe],np.load(b/'probe/head_transforms.npy'))
    assert np.array_equal(p['source_frame_ids'],np.arange(30,106)) and (np.diff(p['timestamps'])>0).all()
    for k in ['body_pose','global_orient','transl','betas','left_hand_pose','right_hand_pose','expression','jaw_pose','leye_pose','reye_pose']:assert np.isfinite(p[k]).all(),k
    hair_first=sample[0]@H[:,:3,:3].transpose(0,2,1)+H[:,:3,3]
    result=dict(integration_only=True,all76_geometry_finite=True,probe_full_outputs_exactly_equal=True,
        max_clothing_offset_error_m=float(abs(distances-.004).max()),
        max_adjacent_clothing_step_m=float(np.linalg.norm(np.diff(c,axis=0),axis=2).max()),
        max_adjacent_offset_change_m=float(np.linalg.norm(np.diff(disp,axis=0),axis=2).max()),
        flipped_face_count_max=max(len(r['flipped_face_ids']) for r in records),
        flipped_area_fraction_max=max(r['flipped_area_fraction'] for r in records),
        max_flipped_face_area_mm2=max(r['max_flipped_face_area_mm2'] for r in records),
        groom_sample_signed_min_m=min(r['groom_sample_signed_min_m'] for r in records),
        groom_sample_below_minus1mm_max=max(r['groom_sample_below_minus1mm'] for r in records),
        groom_sample_below_minus5mm_max=max(r['groom_sample_below_minus5mm'] for r in records),
        groom_diagnostic_points=len(sample),groom_first_point_axis_ranges_m=np.ptp(hair_first,axis=0).tolist(),
        limits='local signed nearest-of24-triangles sample is not inside/outside proof or exhaustive collision certificate; static/numeric checks do not certify perceived continuity',
        elapsed_seconds=time.perf_counter()-began,human_dynamic_review='pending_user')
    (out/'per_frame.json').write_text(json.dumps(records,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
