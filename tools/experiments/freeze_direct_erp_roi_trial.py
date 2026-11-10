"""Freeze one existing scene state and bounded same-pixel A/B/C comparisons."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.data.boxes import change_box_for_image,crop_from_box
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.imaging.output_modes import prepare_source_workspace

OUT=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009'
OLD=ROOT/'outputs/experiments/pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/road_pass_00'
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    cv2.setNumThreads(1);ep=json.loads((OLD/'recipe.json').read_text());rig=json.loads((OLD/'rig.json').read_text())
    p=ep['people'][0];a=json.loads(Path(p['asset_manifest']).read_text());g=Path(a['geometry']);ix=p['asset_frames'][0];M=np.array(p['model_to_world']);H=np.load(g/'head_transforms.npy')[ix]
    body=np.load(g/'vertices.npy',mmap_mode='r')[ix];cloth=np.load(g/'clothing_vertices.npy',mmap_mode='r')[ix]
    hairdir=Path(a['groom_reference']);active=np.load(hairdir/'active_curve_mask.npy');offsets=np.load(hairdir/'curve_offsets.npy');hp=np.load(hairdir/'points.npy',mmap_mode='r')[np.repeat(active,np.diff(offsets))]
    world=dict(body=body@M[:3,:3].T+M[:3,3],clothing=cloth@M[:3,:3].T+M[:3,3],groom=hp@(M@H)[:3,:3].T+(M@H)[:3,3])
    cameras={};tiny=None
    for c in rig['cameras']:
        name=c['id'];pose=np.array(c['camera_to_world']);box=json.loads((OLD/f'frame_000/{name}/boxes.json').read_text())[0]['bbox_xyxy_exclusive'];crops=[]
        for label,params in [('reference',{}),('center_minus_5pct',dict(dx=-.05)),('center_plus_5pct',dict(dx=.05)),('scale_1p1',dict(factor=1.1))]:
            b=change_box_for_image(box,image_size_wh=(5760,2880),boundary_policy='erp_reject_seam',min_side_px=2,shift_reference_expansion=1.0,**params)
            center,scale=crop_from_box(b,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
            bounds=np.r_[center-scale*100,center+scale*100]
            crops.append(dict(name=label,box_xyxy=b.tolist(),center=center.tolist(),scale=float(scale),full_square_xyxy=bounds.tolist(),crop_policy='deployment',parameters=params))
        bounds={}
        for key,points in world.items():
            xy,_=project_erp(points,pose,5760,2880)
            assert np.isfinite(xy).all() and xy[:,0].ptp()<2880,'Seam-crossing geometry rejected'
            bounds[key]=np.r_[xy.min(0),xy.max(0)].tolist()
        union=np.array(list(bounds.values())+[r['full_square_xyxy'] for r in crops]);lo=np.floor(union[:,:2].min(0)-32).astype(int);hi=np.ceil(union[:,2:].max(0)+32).astype(int)
        if (lo<0).any() or (hi>[5760,2880]).any():raise ValueError('ROI crosses full-image boundary; no silent clipping')
        roi=np.r_[lo,hi].tolist()
        cameras[name]=dict(camera_to_world=c['camera_to_world'],physical_lens_to_world=c['physical_lens_to_world'],
            old_box=box,diagnostic_crops=crops,geometry_bounds_full_erp=bounds,roi_xyxy=roi,
            margin_final_erp_px=32,required_visibility_radius_px=2,crop_interpolation_margin_px=2,
            valid_mask_ref=str(OLD/f'frame_000/{name}/valid_mask.png'),lens_mask_ref=str(ROOT/f'outputs/acceptance/ds_output_modes_20261008/attempt_4/full/{name}/reference_lens_mask.png'))
        if name=='cam2':
            mask=np.asarray(Image.open(OLD/f'frame_000/{name}/person_1_mask.png'))>0
            dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5);yy,xx=np.unravel_index(dist.argmax(),dist.shape)
            tiny=dict(camera=name,region_xyxy=[int(xx-16),int(yy-16),int(xx+16),int(yy+16)],samples=64,expected_shape_hw=[32,32],method='raw range and index comparison at exact full-image offset against full render',range_abs_tolerance_m=.0001,range_matching_fraction_required=.99,index_matching_fraction_required=1.0)
    settings=json.loads((OUT/'scene_settings.json').read_text())
    deps=[Path(p['asset_manifest']),OLD/'recipe.json',OLD/'rig.json',Path(ep['scene']),ROOT/'assets/densekp_regressor_R.npy',Path(a['parameters'])]
    for folder in [g,Path(a['clothing_reference']),Path(a['groom_reference'])]:deps.extend(p for p in folder.iterdir() if p.is_file())
    manifest=dict(schema='direct_erp_roi_comparison_v1',integration_only=True,old_episode=str(OLD),frame_index=0,source_frame=12,asset_frame=ix,recipe=ep,rig=rig,
        model_to_world=M.tolist(),head_transform=H.tolist(),full_image_size_wh=[5760,2880],calibration_target='20260320-riding-dataset3',synthetic_baseline_m=.18,
        source_render=dict(samples=64,seed=1009,animated_seed=False,adaptive_sampling=False,denoising=False,device='CUDA',gpu_uuid=ep['render']['gpu_uuid']),
        scene_settings=settings['settings'],blender_version=settings['blender_version'],background=dict(resolution_wh=[1440,720],samples=8,resize='OpenCV INTER_LINEAR display RGB; outside protected high-quality ROI only'),
        cameras=cameras,tiny_preflight=tiny,composition='hard insert at integer ROI offset; boundary >=32px outside geometry/crop union; no blending inside protected crops',
        coverage_contract='valid_mask is physical lens domain; high_quality_region is independent render support; unknown depth/ID outside ROI; only contained crops accepted',
        crop_selection='four fixed diagnostic boxes per camera per route + one native production box per camera per route =30 actual Dataset reads',
        thresholds=dict(geometric_xy_max_abs_px=.0003,world_max_abs_m=.000002,in_view_mismatch_allowed=0,actual_affine_image_and_point_exact=True,
            RGB_bitwise_equality_required=False,RGB_metrics=['MAE','RMSE','p95_abs','max_abs','nonzero_pixel_fraction'],
            instance_metrics=['xor_pixels','IoU','tight_box_delta','raw_ROI_offset_match'],
            visible_rule='same original periodic5x5 range/ID, strict4cm; each route own actual buffers; list every changed semantic ID',
            common_GT='intersection of each view in_view and inside_full_crop by semantic ID; no visibility exclusion',
            visual_quality_pass='human review; RGB statistics describe differences, not a trained model metric'),
        budget=dict(max_render_calls=10,render_seconds=1200,full_dual_calls=2,background_calls=2,roi_calls=2,tiny_preflight_calls=1,optional_roi_correction_calls_max=2,model_forwards=0,optimizer_updates=0),
        code=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True),
        input_inventory={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in deps})
    assert not (OUT/'frozen_comparison.json').exists();write(OUT/'frozen_comparison.json',manifest)
    for kind in ['full','background','tiny','roi']:
        base=OUT/'buffers'/kind;base.mkdir(parents=True);prepare_source_workspace(base,'compact')
    for stage in ['full','tiny','region']:
        stub=OUT/f'{stage}_assembly';stub.mkdir();source=prepare_source_workspace(stub,'compact')
        write(OUT/f'{stage}_job.json',dict(recipe=str(OLD/'recipe.json'),rig=str(OLD/'rig.json'),frames=[0],source_dirs=[str(source)],remaining_budget_seconds=1200))
    print(json.dumps({c:dict(roi=m['roi_xyxy'],geometry=m['geometry_bounds_full_erp']) for c,m in cameras.items()},indent=2))

if __name__=='__main__':main()
