#!/usr/bin/env python3
"""Run independent Omni-Radtan center and dual-camera overlap validation."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from densekp_retrain.camera import OmniRadtanIntrinsics, pose, project_points_omni_radtan, unproject_pixels_omni_radtan
from densekp_retrain.io import linear_to_srgb, read_combined_depth_index
from densekp_retrain.rendering.process import run_logged
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline

def scalar_reference(points, camera, k):
    inv=np.linalg.inv(camera); out=[]
    for p in points:
        q=(inv@np.r_[p,1])[:3]; q=q/np.linalg.norm(q); theta=np.arctan2(np.hypot(q[0],q[1]),q[2]); phi=np.arctan2(q[1],q[0]); m=np.sin(theta)/(np.cos(theta)+k.xi); r=k.f*(m+k.k1*m**3+k.k2*m**5); out.append((k.cx+r*np.cos(phi),k.cy+r*np.sin(phi)))
    return np.asarray(out)

def warp_case(source, output, points, camera, k, source_keypoints=None, source_fingerprint=None):
    output.mkdir(parents=True,exist_ok=True); rgb,depth,index,channels=read_combined_depth_index(source); h,w=depth.shape
    yy,xx=np.indices((k.height,k.width),dtype=np.float64); pixels=np.column_stack((xx.ravel(),yy.ravel())); rays,domain=unproject_pixels_omni_radtan(pixels,k,return_valid=True)
    lon=np.arctan2(rays[:,0],rays[:,2]); lat=np.arcsin(np.clip(-rays[:,1],-1,1)); mapx=((lon/(2*np.pi)+.5)*w).reshape(k.height,k.width).astype(np.float32); mapy=((.5-lat/np.pi)*h).reshape(k.height,k.width).astype(np.float32)
    valid=domain.reshape(k.height,k.width); mapx[~valid]=-1; mapy[~valid]=-1
    warped=cv2.remap(rgb,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT); warped_depth=cv2.remap(depth,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT); warped_index=cv2.remap(index,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    warped_depth[(~valid)|(~np.isfinite(warped_depth))]=0; mask=(valid&(warped_index>.5)).astype(np.uint8)*255
    display=(np.clip(linear_to_srgb(warped),0,1)*255+.5).astype(np.uint8); Image.fromarray(display).save(output/'rgb.png'); Image.fromarray(mask).save(output/'mask.png'); np.save(output/'depth.npy',warped_depth.astype(np.float32))
    projected,details=project_points_omni_radtan(points,camera,k,True); finite=np.isfinite(projected).all(1); inside=finite&(projected[:,0]>=0)&(projected[:,0]<k.width)&(projected[:,1]>=0)&(projected[:,1]<k.height)
    camera_points=details['camera_points']; point_range=np.linalg.norm(camera_points,axis=1); visible=np.zeros(len(points),bool); depth_error=np.full(len(points),np.nan)
    for j,(u,v) in enumerate(projected):
        if not inside[j]: continue
        x=int(np.floor(u)); y=int(np.floor(v)); q=warped_depth[max(0,y-2):min(k.height,y+3),max(0,x-2):min(k.width,x+3)]; q=q[q>0]
        if q.size: depth_error[j]=np.min(np.abs(q-point_range[j])); visible[j]=depth_error[j]<.04
    overlay=Image.fromarray(display); draw=ImageDraw.Draw(overlay)
    for j,(uv,ok) in enumerate(zip(projected,inside)):
        if ok:
            u,v=uv; color=(20,245,70) if visible[j] else (255,60,40); draw.ellipse((u-8,v-8,u+8,v+8),fill=color)
            if j%20==0: draw.text((u+9,v+4),str(j),fill=color)
    overlay.save(output/'densekp_overlay.png'); annotation=projected.astype(np.float32)[None]
    np.save(output/'densekp_fisheye_2d.npy',annotation); np.save(output/'densekp_2d.npy',annotation); np.save(output/'densekp_valid.npy',inside[None]); np.save(output/'densekp_visible.npy',visible[None]); np.save(output/'densekp_depth_error.npy',depth_error.astype(np.float32)[None])
    reference=scalar_reference(points,camera,k); projection_error=np.linalg.norm(projected-reference,axis=1); back,back_valid=unproject_pixels_omni_radtan(projected,k,return_valid=True); truth=details['rays']; angular=np.arccos(np.clip(np.sum(back*truth,axis=1),-1,1)); good=back_valid&np.isfinite(angular)
    fingerprint_3d=hashlib.blake2b(source_keypoints.tobytes(),digest_size=16).hexdigest() if source_keypoints is not None else None
    fingerprint_3d_matches_source=None if source_fingerprint is None else (fingerprint_3d==source_fingerprint)
    report={'source_equirectangular':str(source.resolve()),'source_channels':channels,'image_size':[k.width,k.height],'densekp_3d_fingerprint':{'algorithm':'blake2b-128','value':fingerprint_3d,'matches_projection_source':fingerprint_3d_matches_source},'projection_error_px':{'mean':float(projection_error.mean()),'max':float(projection_error.max())},'roundtrip_angular_error_rad':{'mean':float(angular[good].mean()),'max':float(angular[good].max())},'densekp_total':len(points),'finite_keypoints':int(finite.sum()),'valid_keypoints':int(inside.sum()),'visible_keypoints':int(visible.sum()),'pixel_bounds_finite':{'min':projected[finite].min(0).tolist(),'max':projected[finite].max(0).tolist()},'depth_error_m':{'median':float(np.nanmedian(depth_error)),'p95':float(np.nanpercentile(depth_error,95))},'no_nan':bool(np.isfinite(projected).all()),'pass':bool(projection_error.max()<1e-9 and inside.any() and visible.any() and np.isfinite(projected).all())}
    (output/'case_report.json').write_text(json.dumps(report,indent=2)+'\n'); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/geometry')); p.add_argument('--clothing-geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/clothing_geometry')); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,default=Path('outputs/baselines/fisheye_camera_p0')); p.add_argument('--frame',type=int,default=0); p.add_argument('--pano-width',type=int,default=2048); p.add_argument('--pano-height',type=int,default=1024); p.add_argument('--skip-render',action='store_true'); p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path); a=p.parse_args(); root=ROOT; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    vertices_all=np.load(a.geometry/'vertices.npy'); dense_all=np.load(a.geometry/'densekp_3d.npy'); vertices=vertices_all[a.frame]; dense=dense_all[a.frame]
    source_fingerprint=hashlib.blake2b(dense[None].tobytes(),digest_size=16).hexdigest()
    np.save(out/'densekp_3d_source.npy',dense[None])
    if hashlib.blake2b(np.load(out/'densekp_3d_source.npy').tobytes(),digest_size=16).hexdigest()!=source_fingerprint: raise RuntimeError('densekp_3d_source.npy fingerprint mismatch after save')
    lift=-float(vertices[:,1].min()); world=np.column_stack((dense[:,0],-dense[:,2],dense[:,1]+lift)); vw=np.column_stack((vertices[:,0],-vertices[:,2],vertices[:,1]+lift)); target=(vw.min(0)+vw.max(0))/2; origin=target-np.array([0,3.2,0])
    cams=[{'id':'fisheye_center','camera_to_world':pose(target,origin,0).tolist()},{'id':'cam_left','camera_to_world':pose(target,origin,75).tolist()},{'id':'cam_right','camera_to_world':pose(target,origin,-75).tolist()}]; camera_file=out/'camera_config.json'; intr=OmniRadtanIntrinsics(); camera_file.write_text(json.dumps({'camera_model':'Omni-Radtan','intrinsics':intr.as_dict(),'coordinate_system':'CV +X right, +Y down, +Z forward; camera_to_world','cameras':cams},indent=2)+'\n')
    source=out/'equirectangular_source'; source.mkdir(exist_ok=True)
    if not a.skip_render:
        baseline,_=load_appearance_baseline(a.appearance_manifest); cmd=[str(find_blender_executable(a.blender_bin)),'-b','--python',str(root/'tools/fisheye/render_equirectangular_source.py'),'--','--geometry',str(a.geometry.resolve()),'--clothing-geometry',str(a.clothing_geometry.resolve()),'--cameras',str(camera_file.resolve()),'--output-dir',str(source.resolve()),'--frame',str(a.frame),'--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],'--hdri',baseline['hdri_path'],'--width',str(a.pano_width),'--height',str(a.pano_height)]
        run_logged(cmd,cwd=root,log_path=out/'render.log')
        if not (source/'source_manifest.json').is_file(): raise RuntimeError(f'Blender source render failed; inspect {out/"render.log"}')
    reports={}
    reports['center']=warp_case(source/'fisheye_center.exr',out/'center',world,np.asarray(cams[0]['camera_to_world']),intr,dense[None],source_fingerprint)
    for i,name in enumerate(['cam_left','cam_right'],1): reports[name]=warp_case(source/f'{name}.exr',out/'seam'/name,world,np.asarray(cams[i]['camera_to_world']),intr,dense[None],source_fingerprint)
    fingerprint_consistent=all(x['densekp_3d_fingerprint']['matches_projection_source'] and x['densekp_3d_fingerprint']['value']==source_fingerprint for x in reports.values())
    fingerprint_consistent&=hashlib.blake2b(np.load(out/'densekp_3d_source.npy').tobytes(),digest_size=16).hexdigest()==source_fingerprint
    report={'baseline_commit':'1ad425978882ac96fbdaacec6fa15d1e21ec2a1b','geometry':str(a.geometry.resolve()),'geometry_frame':a.frame,'densekp_3d_shape':[1,len(world),3],'densekp_3d_fingerprint':{'algorithm':'blake2b-128','value':source_fingerprint,'consistent_across_cases':fingerprint_consistent},'intrinsics':intr.as_dict(),'case_center_pass':reports['center']['pass'],'case_seam_pass':reports['cam_left']['pass'] and reports['cam_right']['pass'],'densekp_3d_fingerprint_consistent':fingerprint_consistent,'fisheye_camera_pass':all(x['pass'] for x in reports.values()) and fingerprint_consistent,'cases':reports,'elapsed_seconds':time.perf_counter()-started}
    (out/'regression_report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={'phase':'FISHEYE_CAMERA_P0','camera_model':'Omni-Radtan','intrinsics':intr.as_dict(),'baseline_commit':report['baseline_commit'],'geometry':report['geometry'],'geometry_frame':a.frame,'densekp_3d_fingerprint':report['densekp_3d_fingerprint'],'annotation_format':{'densekp_fisheye_2d':'[T,138,2] float32, top-left-origin pixels','densekp_valid':'[T,138] bool, projected inside image','densekp_visible':'[T,138] bool, depth-visibility within 4 cm','densekp_depth_error':'[T,138] float32, |surface_depth - keypoint_range| in meters'},'outputs':{'center':{'rgb':'center/rgb.png','depth':'center/depth.npy','mask':'center/mask.png','densekp_2d':'center/densekp_fisheye_2d.npy','densekp_valid':'center/densekp_valid.npy','densekp_visible':'center/densekp_visible.npy','densekp_3d_source':'densekp_3d_source.npy','overlay':'center/densekp_overlay.png','case_report':'center/case_report.json'},'seam':{'cam_left':{'rgb':'seam/cam_left/rgb.png','depth':'seam/cam_left/depth.npy','mask':'seam/cam_left/mask.png','densekp_2d':'seam/cam_left/densekp_fisheye_2d.npy','densekp_valid':'seam/cam_left/densekp_valid.npy','densekp_visible':'seam/cam_left/densekp_visible.npy','overlay':'seam/cam_left/densekp_overlay.png','case_report':'seam/cam_left/case_report.json'},'cam_right':{'rgb':'seam/cam_right/rgb.png','depth':'seam/cam_right/depth.npy','mask':'seam/cam_right/mask.png','densekp_2d':'seam/cam_right/densekp_fisheye_2d.npy','densekp_valid':'seam/cam_right/densekp_valid.npy','densekp_visible':'seam/cam_right/densekp_visible.npy','overlay':'seam/cam_right/densekp_overlay.png','case_report':'seam/cam_right/case_report.json'}},'equirectangular_source':'equirectangular_source/source_manifest.json','camera_config':'camera_config.json','regression_report':'regression_report.json'},'metrics':{'finite_keypoints':{n:int(r['finite_keypoints']) for n,r in reports.items()},'valid_keypoints':{n:int(r['valid_keypoints']) for n,r in reports.items()},'visible_keypoints':{n:int(r['visible_keypoints']) for n,r in reports.items()},'max_projection_error_px':{n:float(r['projection_error_px']['max']) for n,r in reports.items()},'max_roundtrip_angular_error_rad':{n:float(r['roundtrip_angular_error_rad']['max']) for n,r in reports.items()}},'status':{'FISHEYE_CAMERA_P0':'PASS' if report['fisheye_camera_pass'] else 'FAIL','CENTER_CASE':'PASS' if report['case_center_pass'] else 'FAIL','SEAM_CASE':'PASS' if report['case_seam_pass'] else 'FAIL'}}
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
