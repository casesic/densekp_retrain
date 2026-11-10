"""Read-only asset-side checker; no model/Blender/scene/Dataset imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import time
import numpy as np


def read(p):return json.loads(Path(p).read_text())


def check(item):
    started=time.monotonic();n=item['frame_count'];timeline=read(item['timeline'])
    assert len(timeline)==n and [t['asset_frame'] for t in timeline]==list(range(n))
    ticks=np.array([t['source_time_tick'] for t in timeline],np.int64)
    assert (np.diff(ticks)>0).all()
    assert all(t['time_base']==item['time_base'] for t in timeline)
    assert np.array_equal([t['episode_time_tick'] for t in timeline],ticks-ticks[0])
    with np.load(item['parameters'],allow_pickle=False) as p:
        np.testing.assert_array_equal(p['betas'],np.array(item['model']['betas'],dtype=p['betas'].dtype))
        if 'timestamps' in p:
            np.testing.assert_array_equal(p['timestamps'],ticks)
            np.testing.assert_array_equal(p['source_frame_ids'],[t['source_frame'] for t in timeline])
            assert np.all(p['transl']==p['transl'][0])
        elif 'source_pts' in p:
            np.testing.assert_array_equal(p['source_pts'],ticks)
            np.testing.assert_array_equal(p['source_decoded_frames'],[t['source_frame'] for t in timeline])
            np.testing.assert_array_equal(p['time_base'],item['time_base'])
            assert np.all(p['transl']==p['transl'][0])
    assert Path(item['model']['file']).is_file()
    g=item['geometry'];v=np.load(g['vertices'],mmap_mode='r');faces=np.load(g['faces']);kp=np.load(g['densekp'])
    R=np.load(item['regressor']);assert v.shape==(n,10475,3) and kp.shape==(n,138,3) and R.shape==(138,10475)
    assert np.isfinite(v).all() and np.isfinite(kp).all()
    assert faces.shape==(20908,3) and faces.min()>=0 and faces.max()<10475
    error=float(np.max(np.abs(np.einsum('kv,tvc->tkc',R,v,optimize=True)-kp)));assert error<2e-6
    garment=item['garment'];cv=np.load(garment['vertices'],mmap_mode='r');cf=np.load(garment['faces'])
    assert cv.shape[0]==n and cv.shape[-1]==3 and np.isfinite(cv).all()
    assert cf.min()>=0 and cf.max()<cv.shape[1]
    if garment['representation']=='simple_fixed_barycentric_normal_offset':
        ix=np.load(garment['binding_indices']);w=np.load(garment['binding_weights'])
        assert ix.shape==w.shape==(cv.shape[1],3) and ix.min()>=0 and ix.max()<10475
        assert np.isfinite(w).all() and np.allclose(w.sum(1),1,atol=1e-6)
        for name in ['materials','material_reference','rest_coordinates']:assert Path(garment[name]).exists()
    elif garment['representation']=='native_simulation_cache':
        uv=np.load(garment['uv']);uf=np.load(garment['face_corner_uv'])
        assert uf.shape==cf.shape and uf.min()>=0 and uf.max()<len(uv) and np.isfinite(uv).all()
        for name in ['original_npz','diffuse','normal']:assert Path(garment[name]).is_file()
    else:raise ValueError('Unknown garment representation')
    # Fixed arbitrary diagnostic placement only; not a road, camera, projection or scene observation.
    M=np.array([[1,0,0,2],[0,0,-1,-1],[0,1,0,.5],[0,0,0,1]],float)
    world=v@M[:3,:3].T+M[:3,3]
    world_error=float(np.max(np.abs(np.einsum('kv,tvc->tkc',R,world,optimize=True)-(kp@M[:3,:3].T+M[:3,3]))));assert world_error<2e-6
    hair=item['groom'];hair_record=None
    if hair:
        points=np.load(hair['points'],mmap_mode='r');radius=np.load(hair['radius']);offsets=np.load(hair['curve_offsets'])
        H=np.load(hair['head_transforms']);assert H.shape==(n,4,4) and np.isfinite(H).all()
        assert np.isfinite(points).all() and np.isfinite(radius).all() and len(radius)==len(points)
        assert offsets[0]==0 and offsets[-1]==len(points) and (np.diff(offsets)>0).all()
        active=np.load(hair['active_mask']) if hair['active_mask'] else np.ones(len(offsets)-1,bool)
        assert len(active)==len(offsets)-1
        np.testing.assert_allclose(H[:,:3,:3].transpose(0,2,1)@H[:,:3,:3],np.broadcast_to(np.eye(3),(n,3,3)),atol=2e-6)
        sample=points[::1024].astype(float);max_composition=0
        for h in H:
            composite=M@h
            direct=sample@composite[:3,:3].T+composite[:3,3]
            staged=(sample@h[:3,:3].T+h[:3,3])@M[:3,:3].T+M[:3,3]
            max_composition=max(max_composition,float(np.abs(direct-staged).max()))
        assert max_composition<2e-6
        info=read(Path(hair['reference'])/'groom_manifest.json');assert Path(info['source_abc']).is_file()
        hair_record=dict(source_points=len(points),active_curves=int(active.sum()),active_points=int(np.diff(offsets)[active].sum()),
                         full_source_finite=True,composition_sample_stride=1024,max_composition_error_m=max_composition)
    assert item['integration_only'] and not item['public_consumer_tested'] and not item['A1_schema_effective']
    assert Path(item['preview_html']).is_file() and Path(item['preview_mp4']).is_file()
    return dict(asset_id=item['asset_id'],frames=n,source_tick_interval_min=int(np.diff(ticks).min()),source_tick_interval_max=int(np.diff(ticks).max()),
        body_finite=True,cloth_finite=True,R_max_error_m=error,scene_R_commutation_error_m=world_error,groom=hair_record,
        densekp_status=item['densekp_status'],human_acceptance=item['human_acceptance'],seconds=time.monotonic()-started,
        scope='asset-side source/cache validation and fixed diagnostic transform only; no scene consumption')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'Do not overwrite a prior check'
    started=time.monotonic();catalog=read(a.index);results=[];ids=[]
    for entry in catalog['assets']:
        item=read(entry['manifest']);assert item['asset_id']==entry['asset_id'];ids.append(item['asset_id']);results.append(check(item))
    assert len(set(ids))==len(ids)==catalog['combination_count']
    result=dict(status='passed_asset_side_only',integration_only=True,source_files_written=False,scene_consumption=False,
                elapsed_seconds=time.monotonic()-started,assets=results)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':result['status'],'seconds':result['elapsed_seconds'],'assets':len(results)}))


if __name__=='__main__':main()
