"""Three-route trial consumers and fixed-crop comparisons; never loads a model."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
import cv2
import numpy as np
from PIL import Image,ImageDraw
from densekp_retrain.imaging.source import read_ds_source
from densekp_retrain.imaging.ds import (prepare_ds_domain,project_ds_scene,remap_ds_images,save_ds_images,
    make_ds_annotations,save_ds_annotations,DSImages)
from densekp_retrain.imaging.direct_erp import direct_annotations
from densekp_retrain.imaging.regions import (apply_physical_mask,embed_region,region_mask,require_crop_in_region,require_point_neighborhoods)
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.data.boxes import crop_from_box,validate_box_context
from densekp.core.keypoint_supervision import SUPERVISION_POLICY,make_targets,supervision_for_crop
from densekp_retrain.data.packing import pack_derivative_rows
from densekp_retrain.data.loading import load_box_dataset
OUT=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009'

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def pixels(p):return np.asarray(Image.open(p).convert('RGB'))
def mask(p):return np.asarray(Image.open(p))>0
def source_dir(kind):return OUT/'buffers'/kind/'_temporary_ds_source/source'
def stats(a,b,selection=None):
    d=np.abs(a.astype(np.float32)-b.astype(np.float32))
    if selection is not None:d=d[selection]
    return dict(MAE=float(d.mean()),RMSE=float(np.sqrt((d*d).mean())),p95_abs=float(np.quantile(d,.95)),max_abs=float(d.max()),nonzero_pixel_fraction=float((d>0).any(-1).mean()),pixels=int(d.size//3))

def save_direct(folder,rgb,instance,data,valid):
    folder.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(folder/'normal_erp.png');Image.fromarray(valid.astype(np.uint8)*255).save(folder/'valid_mask.png')
    # Reuse annotation/mask/tight-box writer, but with the direct route's actual IDs.
    save_ds_annotations(folder,data,data['person_ids'].tolist(),DSImages(None,rgb,instance),SimpleNamespace(valid_mask=valid),output_mode='compact')

def full():
    began=time.monotonic();f=read(OUT/'frozen_comparison.json');old=Path(f['old_episode']);world=np.load(old/'world_densekp.npy')[0];ids=[p['person_id'] for p in f['recipe']['people']]
    erp_masks=np.load(ROOT/'assets_external/real_erp_valid_masks.npz');times=[];checks=[]
    for route in ['A','B']:(OUT/route).mkdir(exist_ok=False)
    for c,info in f['cameras'].items():
        t=time.monotonic();source=read_ds_source(source_dir('full'),c);read_seconds=time.monotonic()-t
        assert source.rgb.shape==(2880,5760,3)
        pose=np.array(info['camera_to_world']);t=time.monotonic()
        domain=prepare_ds_domain(c,source,erp_masks[f'valid_masks_{c}'],mask(Path(info['lens_mask_ref'])),output_size=(5760,2880))
        domain_s=time.monotonic()-t;np.testing.assert_array_equal(domain.valid_mask,mask(Path(info['valid_mask_ref'])))
        t=time.monotonic();projection=project_ds_scene(world,ids,pose,c,source,domain);adata=make_ds_annotations(f['recipe']['people'],ids,world,pose,projection);labels_a=time.monotonic()-t
        t=time.monotonic();images=remap_ds_images(c,source,domain);remap_s=time.monotonic()-t
        dest=OUT/'A'/c;dest.mkdir();t=time.monotonic();save_ds_images(dest,images,domain,output_mode='compact');save_ds_annotations(dest,adata,ids,images,domain,output_mode='compact');save_a=time.monotonic()-t
        olda=dict(np.load(old/f'frame_000/{c}/annotations.npz'));reproduce=dict(rgb=stats(images.normal_erp_rgb,pixels(old/f'frame_000/{c}/normal_erp.png')),
            xy_max_abs=float(np.abs(adata['xy']-olda['xy']).max()),in_view_mismatch=int((adata['in_view']!=olda['in_view']).sum()),visible_mismatch=int((adata['visible']!=olda['visible']).sum()),person_mask_xor=int((mask(dest/'person_1_mask.png')!=mask(old/f'frame_000/{c}/person_1_mask.png')).sum()))
        del images
        # B: independently read already-calibrated physical mask. No DS LUT/remap needed.
        t=time.monotonic();valid=mask(Path(info['valid_mask_ref']));valid_s=time.monotonic()-t
        t=time.monotonic();bdata=direct_annotations(world,ids,pose,c,valid,source.ray_range_m,source.instance);labels_b=time.monotonic()-t
        t=time.monotonic();brgb=apply_physical_mask(source.rgb,valid);rgb_b=time.monotonic()-t
        direct_ids=source.instance.copy();direct_ids[~valid]=0
        t=time.monotonic();save_direct(OUT/'B'/c,brgb,direct_ids,bdata,valid);save_b=time.monotonic()-t
        xy,_=project_erp(world.reshape(-1,3),pose,5760,2880);xy_error=float(np.abs(xy-bdata['xy'].reshape(-1,2)).max())
        # Verify the actual Blender camera matrix: right/up/back -> right/down/forward.
        actual=np.array(read(OUT/f'full_{c}_render.json')['camera_matrix_blender']);actual[:3,1:3]*=-1
        actual_xy,_=project_erp(world.reshape(-1,3),actual,5760,2880)
        camera_error=float(np.abs(actual_xy-xy).max())
        passed=xy_error<=f['thresholds']['geometric_xy_max_abs_px'] and camera_error<=f['thresholds']['geometric_xy_max_abs_px'] and np.array_equal(adata['xy'],bdata['xy']) and np.array_equal(adata['in_view'],bdata['in_view'])
        checks.append(dict(camera=c,passed=bool(passed),float32_xy_rounding_max_px=xy_error,blender_matrix_xy_max_px=camera_error,
            A_B_xy_exact=bool(np.array_equal(adata['xy'],bdata['xy'])),A_B_in_view_exact=bool(np.array_equal(adata['in_view'],bdata['in_view'])),
            A_B_visible_differences=np.flatnonzero(adata['visible'][0]!=bdata['visible'][0]).tolist(),old_probe_reproduction=reproduce))
        times.append(dict(camera=c,common_exr_and_display_rgb_read=read_seconds,A_domain_LUT_and_mask=domain_s,A_labels=labels_a,A_RGB_and_ID_remap=remap_s,A_encode_labels_masks=save_a,
            B_explicit_physical_mask_read=valid_s,B_labels=labels_b,B_apply_mask=rgb_b,B_encode_labels_masks=save_b))
    gate=dict(passed=all(x['passed'] for x in checks),checks=checks,geometry_tolerance_px=f['thresholds']['geometric_xy_max_abs_px'],
        pixel_grid='full ERP unchanged; projection continuous raster coordinates; no half-pixel correction added to legacy contract; Cycles border grid tested separately',cam4_extra_flip=False)
    write(OUT/'B_geometry_gate.json',gate);write(OUT/'AB_processing.json',dict(times=times,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    for route in ['A','B']:
        write(OUT/route/'manifest.json',dict(route=route,frame_index=0,episode_ref=str(old/'recipe.json'),frozen_comparison=str(OUT/'frozen_comparison.json'),final_erp_resolution=[5760,2880],
            image_chain='AgX display ERP -> DS2880 -> normal ERP' if route=='A' else 'AgX display ERP + calibrated physical valid mask; no DS image resampling',
            labels='own actual instance raster; raw original ERP range visibility',independent_production=True,shared_full_render_cost_must_be_charged=True))
    assert gate['passed'];print(json.dumps(gate,indent=2))

def tiny():
    f=read(OUT/'frozen_comparison.json');p=f['tiny_preflight'];c=p['camera'];r=p['region_xyxy'];x0,y0,x1,y1=r
    local=read_ds_source(source_dir('tiny'),c);full=read_ds_source(source_dir('full'),c)
    shape_ok=list(local.ray_range_m.shape)==p['expected_shape_hw'] and local.rgb.shape[:2]==local.ray_range_m.shape
    result=dict(passed=False,expected_shape_hw=p['expected_shape_hw'],actual_shape_hw=list(local.ray_range_m.shape),roi_xyxy=r)
    if shape_ok:
        d=np.abs(local.ray_range_m-full.ray_range_m[y0:y1,x0:x1]);equal=(np.rint(local.instance)==np.rint(full.instance[y0:y1,x0:x1]));frac=float((d<=p['range_abs_tolerance_m']).mean());idxfrac=float(equal.mean())
        result.update(range_max_abs_m=float(d.max()),range_matching_fraction=frac,index_matching_fraction=idxfrac,RGB=stats(local.rgb,full.rgb[y0:y1,x0:x1]),
            passed=frac>=p['range_matching_fraction_required'] and idxfrac>=p['index_matching_fraction_required'])
    write(OUT/'tiny_gate.json',result);print(json.dumps(result,indent=2));assert result['passed'],'Border pixel alignment failed; do not advance ROI'

def region():
    began=time.monotonic();f=read(OUT/'frozen_comparison.json');assert read(OUT/'B_geometry_gate.json')['passed'] and read(OUT/'tiny_gate.json')['passed']
    old=Path(f['old_episode']);world=np.load(old/'world_densekp.npy')[0];ids=[p['person_id'] for p in f['recipe']['people']];(OUT/'C').mkdir(exist_ok=False);timings=[]
    for c,info in f['cameras'].items():
        t=time.monotonic();local=read_ds_source(source_dir('roi'),c);low=pixels(source_dir('background')/f'{c}_preview.png');read_s=time.monotonic()-t
        roi=info['roi_xyxy'];x0,y0,x1,y1=roi;assert local.rgb.shape==(y1-y0,x1-x0,3) and low.shape==(720,1440,3)
        t=time.monotonic();depth=embed_region(local.ray_range_m,roi,(5760,2880),unknown_value=np.nan);instance=embed_region(local.instance,roi,(5760,2880),unknown_value=np.nan);coverage=region_mask(roi,(5760,2880));buffers_s=time.monotonic()-t
        t=time.monotonic();valid=mask(Path(info['valid_mask_ref']));mask_s=time.monotonic()-t
        t=time.monotonic();data=direct_annotations(world,ids,np.array(info['camera_to_world']),c,valid,depth,instance,coverage_region=roi);label_s=time.monotonic()-t
        t=time.monotonic();rgb=cv2.resize(low,(5760,2880),interpolation=cv2.INTER_LINEAR);rgb[y0:y1,x0:x1]=local.rgb;rgb=apply_physical_mask(rgb,valid);compose_s=time.monotonic()-t
        t=time.monotonic();dest=OUT/'C'/c;save_direct(dest,rgb,instance,data,valid);Image.fromarray(coverage.astype(np.uint8)*255).save(dest/'high_quality_region.png');encode_s=time.monotonic()-t
        write(dest/'render_coverage.json',dict(full_image_size_wh=[5760,2880],high_quality_region_xyxy=roi,label_coverage_region_xyxy=roi,
            region_offset_xy=[x0,y0],local_shape_hw=list(local.ray_range_m.shape),physical_valid_mask='valid_mask.png',
            outside_region_RGB='independent 1440x720 8sample whole-scene background upsample',outside_region_depth_and_ID='unknown; never low-resolution GT',
            person_mask_zero_outside_region='unknown, not certified background',supported_person_ids=ids,
            all_body_clothing_groom_bounds_covered=True,visibility_neighborhood_radius_px=2,
            crop_policy='require_crop_in_region including2px interpolation guard before packing every new crop',full_B_buffer_dependency=False))
        timings.append(dict(camera=c,ROI_EXR_and_background_read=read_s,embed_unknown_buffers=buffers_s,physical_mask_read=mask_s,labels=label_s,compose=compose_s,encode=encode_s))
    write(OUT/'C'/'manifest.json',dict(route='C',frame_index=0,frozen_comparison=str(OUT/'frozen_comparison.json'),episode_ref=str(old/'recipe.json'),final_erp_resolution=[5760,2880],
        independent_production=True,full_B_buffer_dependency=False,labels='only actual 64sample ROI high-resolution range/ID; each138 point5x5 neighborhood checked',
        crop_scope='finite high-quality region; producer rejects crops outside support',timings=timings,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('C independently consumed ROI range/ID and low-resolution background; no full buffer read')

def pack_and_compare():
    began=time.monotonic();f=read(OUT/'frozen_comparison.json');comparisons={};readbacks=[];route_times=[]
    for route in ['A','B','C']:
        route_start=time.monotonic()
        folder=OUT/route;rows=[];index=[]
        for c,info in f['cameras'].items():
            cd=folder/c;ann=dict(np.load(cd/'annotations.npz'));valid=mask(cd/'valid_mask.png');pm=mask(cd/'person_1_mask.png');native=read(cd/'boxes.json')[0]
            boxes=[dict(name='production_native',box_xyxy=native['bbox_xyxy_exclusive'],bbox_kind='visible_instance_tight')]+[dict(x,bbox_kind='fixed_reference') for x in info['diagnostic_crops']]
            for b in boxes:
                xyxy=b['box_xyxy'];source='route-own rendered instance mask' if b['name']=='production_native' else 'frozen same-pixel comparison box from prior A native mask; diagnostic only'
                validate_box_context(xyxy,bbox_kind=b['bbox_kind'],bbox_source=source,person_id=1,mask_person_id=1,annotation_person_id=int(ann['person_ids'][0]),image_size_wh=(5760,2880),manifest_size_wh=(5760,2880),valid_mask=valid,person_mask=pm,boundary_policy='erp_reject_seam',min_side_px=2)
                center,scale=crop_from_box(xyxy,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                if route=='C':require_crop_in_region(center,scale,info['roi_xyxy'],(5760,2880))
                targets=make_targets(ann['xy'][0],ann['in_view'][0],center,scale);assert np.isfinite(targets).all() and (targets[:,2]>0).any()
                rows.append(dict(imgname=str(cd/'normal_erp.png'),center=center,scale=scale,proj_verts=targets,in_view=ann['in_view'][0],visible=ann['visible'][0],person_id=1,camera_id=c))
                index.append(dict(packed_row_id=len(rows)-1,camera=c,person_id=1,name=b['name'],bbox_kind=b['bbox_kind'],bbox_source=source,given_bbox_xyxy=xyxy,crop_policy='deployment',frame_index=0,source_frame_index=12,asset_frame_index=3,time_tick=0,time_base=[1,10],episode_ref=f['old_episode'],label_ref=str(cd/'annotations.npz')))
        np.savez(folder/'packed.npz',**pack_derivative_rows(rows),supervision_policy=SUPERVISION_POLICY,split='integration_only_same_frame_comparison');write(folder/'index.json',dict(rows=index))
        dataset=load_box_dataset(folder/'packed.npz');(folder/'actual_inputs').mkdir();pack_seconds=time.monotonic()-route_start;readback_start=time.monotonic()
        for i,row in enumerate(index):
            item=dataset[i];image=cv2.imread(item['imgname']);warped=cv2.warpAffine(image,item['_trans'],(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            np.testing.assert_array_equal(warped,item['img_disp']);xy=((np.c_[item['proj_verts'][:,:2],np.ones(138)]@item['_trans'].T).astype(np.float32)/np.array([256,256])-.5).astype(np.float32)
            np.testing.assert_array_equal(xy,item['proj_verts_cropped'][:,:2]);sup=supervision_for_crop(dataset.data['in_view'][i],xy);np.testing.assert_array_equal(sup,item['proj_verts_cropped'][:,2]>0)
            target=cv2.warpAffine(mask(folder/row['camera']/'person_1_mask.png').astype(np.uint8),item['_trans'],(256,256),flags=cv2.INTER_NEAREST);assert target.any()
            Image.fromarray(warped[:,:,::-1]).save(folder/'actual_inputs'/f'{row["camera"]}_{row["name"]}.png')
            np.savez(folder/'actual_inputs'/f'geometry_{i:02d}.npz',affine=item['_trans'],crop_xy=item['proj_verts_cropped'],supervised=sup)
            readbacks.append(dict(route=route,**row,image_affine_exact=True,point_affine_exact=True,supervised=int(sup.sum()),occluded_supervised=int((sup&~dataset.data['visible'][i]).sum())))
        route_times.append(dict(route=route,metadata_masks_pack_and_Dataset_init_seconds=pack_seconds,ten_Dataset_reads_affine_checks_and_crop_exports_seconds=time.monotonic()-readback_start))
    assert len(readbacks)==30
    for c,info in f['cameras'].items():
        image={r:pixels(OUT/r/c/'normal_erp.png') for r in ['A','B','C']};pm={r:mask(OUT/r/c/'person_1_mask.png') for r in image};ann={r:dict(np.load(OUT/r/c/'annotations.npz')) for r in image};valid=mask(OUT/'A'/c/'valid_mask.png')
        roi=info['roi_xyxy'];x0,y0,x1,y1=roi;protected=np.zeros_like(valid);protected[y0+32:y1-32,x0+32:x1-32]=True
        results={}
        for r1,r2 in [('A','B'),('B','C')]:
            xor=pm[r1]!=pm[r2];inter=(pm[r1]&pm[r2]).sum();union=(pm[r1]|pm[r2]).sum()
            native1=np.array(read(OUT/r1/c/'boxes.json')[0]['bbox_xyxy_exclusive']);native2=np.array(read(OUT/r2/c/'boxes.json')[0]['bbox_xyxy_exclusive'])
            changes=np.flatnonzero(ann[r1]['visible'][0]!=ann[r2]['visible'][0])
            case=dict(RGB_full_valid=stats(image[r1],image[r2],valid),RGB_high_quality_protected=stats(image[r1],image[r2],protected&valid),
                RGB_person_union=stats(image[r1],image[r2],pm[r1]|pm[r2]),instance_mask_xor_pixels=int(xor.sum()),instance_IoU=float(inter/union),tight_box_delta=(native2-native1).tolist(),
                xy_max_abs_px=float(np.abs(ann[r1]['xy']-ann[r2]['xy']).max()),in_view_mismatch=int((ann[r1]['in_view']!=ann[r2]['in_view']).sum()),
                visibility_changes=[dict(semantic_id=int(i),before=bool(ann[r1]['visible'][0,i]),after=bool(ann[r2]['visible'][0,i]),in_view=bool(ann[r1]['in_view'][0,i]),reason='actual raw range/ID neighborhood difference; see visibility_difference_details.json') for i in changes],fixed_crop_RGB={})
            for b in info['diagnostic_crops']:
                n=b['name'];case['fixed_crop_RGB'][n]=stats(pixels(OUT/r1/'actual_inputs'/f'{c}_{n}.png'),pixels(OUT/r2/'actual_inputs'/f'{c}_{n}.png'))
            results[r1+'_'+r2]=case
        comparisons[c]=results
        # Same-pixel panels: both body (no rescaling) and groom bounds (no rescaling).
        review=OUT/'review';review.mkdir(exist_ok=True)
        for label,bounds in [('body',info['geometry_bounds_full_erp']['body']),('head',info['geometry_bounds_full_erp']['groom'])]:
            x,y=np.floor(np.array(bounds[:2])-12).astype(int);xx,yy=np.ceil(np.array(bounds[2:])+12).astype(int);w,h=xx-x,yy-y
            sheet=Image.new('RGB',(3*w,h+28));draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(['A','B','C']):sheet.paste(Image.fromarray(image[r][y:yy,x:xx]),(i*w,28));draw.text((i*w+4,5),r,fill='white')
            sheet.save(review/f'{c}_{label}_same_pixels.png')
        sheet=Image.new('RGB',(1440,3*720));draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(['A','B','C']):
            sheet.paste(Image.fromarray(image[r]).resize((1440,720)),(0,i*720));draw.text((10,i*720+8),r,fill='red')
            if r=='C':draw.rectangle((x0/4,y0/4+i*720,x1/4,y1/4+i*720),outline='yellow',width=2)
        sheet.save(review/f'{c}_full_ERP_overview.jpg')
        for b in info['diagnostic_crops']:
            n=b['name'];sheet=Image.new('RGB',(3*256,280));draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(['A','B','C']):sheet.paste(Image.open(OUT/r/'actual_inputs'/f'{c}_{n}.png'),(i*256,24));draw.text((i*256+4,4),r,fill='white')
            sheet.save(review/f'{c}_{n}_actual_inputs.png')
    # Paired support uses each view's own GT and crop. Visibility never gates this set.
    pairs=[]
    for route in ['A','B','C']:
        idx=read(OUT/route/'index.json')['rows'];npz=np.load(OUT/route/'packed.npz')
        for label in ['production_native']+[x['name'] for x in f['cameras']['cam2']['diagnostic_crops']]:
            i=next(r['packed_row_id'] for r in idx if r['name']==label and r['camera']=='cam2');j=next(r['packed_row_id'] for r in idx if r['name']==label and r['camera']=='cam4')
            common=(npz['proj_verts'][i,:,2]>0)&(npz['proj_verts'][j,:,2]>0)
            pairs.append(dict(route=route,name=label,common_semantic_ids=np.flatnonzero(common).tolist(),common_count=int(common.sum()),no_common=not bool(common.any())))
    write(OUT/'comparisons.json',comparisons);write(OUT/'loader_readback.json',dict(status='passed',actual_Dataset_reads=30,checks=readbacks,paired_support=pairs,route_times=route_times,elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_forwards=0))
    print('30 actual Dataset crops passed; comparison metrics saved')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['AB','tiny','C','compare']);a=ap.parse_args();cv2.setNumThreads(1)
    {'AB':full,'tiny':tiny,'C':region,'compare':pack_and_compare}[a.stage]()

if __name__=='__main__':main()
