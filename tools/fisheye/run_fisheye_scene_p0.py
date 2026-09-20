#!/usr/bin/env python3
"""Phase 18: place the SMPL-X subject into a static Blender scene and render the
Phase 17 yaw90 dual-fisheye pair through the unchanged Omni-Radtan warp pipeline.

Scene assets are not shipped with this repository; the driver validates the
scene spec, reports SCENE_ASSET_MISSING with the asset requirements instead of
faking a scene, and still runs every asset-independent regression gate.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from fisheye.omni_radtan import OmniRadtanIntrinsics
from scene.scene_loader import SceneAssetSpec, SceneAssetMissing
from fisheye.run_fisheye_camera_p0 import warp_case, pose
from tools.common.runtime import find_blender_executable, load_appearance_baseline

def fingerprint(array): return hashlib.blake2b(np.asarray(array).tobytes(),digest_size=16).hexdigest()

def gates(cam_dir, k):
    depth=np.load(cam_dir/'depth.npy'); mask=np.asarray(Image.open(cam_dir/'mask.png')); mask=mask[...,0] if mask.ndim==3 else mask
    valid=np.load(cam_dir/'densekp_valid.npy')[0]; pix=np.load(cam_dir/'densekp2d.npy')[0]
    person=mask>=128; person_pixels=int(person.sum())
    on_person=0; checked=0
    for (u,v),ok in zip(pix,valid):
        if not ok: continue
        x,y=int(round(float(u))),int(round(float(v)))
        if 0<=x<mask.shape[1] and 0<=y<mask.shape[0]: checked+=1; on_person+=int(person[y,x])
    # Phase 17 frozen side-view baseline (no scene, yaw75) measures 0.891/0.906
    # on-person for silhouette-boundary joints (eyes/ears/head/toes); 0.85 is the calibrated gate.
    mask_pass=bool(person_pixels>1000 and checked>0 and on_person/checked>=.85)
    body_depth=depth[person]; body_depth=body_depth[body_depth>0]
    outside=depth[~person]; scene_depth=outside[outside>0]
    depth_pass=bool(depth.shape==(k.height,k.width) and body_depth.size>1000 and np.isfinite(body_depth).all() and scene_depth.size>depth.size*.05 and float(np.median(scene_depth))>float(np.median(body_depth))+.5)
    return {'mask':{'person_pixels':person_pixels,'valid_keypoints_on_person':on_person,'valid_keypoints_checked':checked,'mask_pass':mask_pass},'depth':{'person_depth_median_m':float(np.median(body_depth)) if body_depth.size else None,'scene_depth_median_m':float(np.median(scene_depth)) if scene_depth.size else None,'scene_pixels_fraction':float(scene_depth.size/depth.size),'depth_pass':depth_pass}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene-spec',type=Path,default=Path('configs/fisheye_scene_p0.json')); p.add_argument('--geometry',type=Path,default=Path('outputs/temporal_case_aaron_1095/geometry')); p.add_argument('--clothing-geometry',type=Path,default=Path('outputs/temporal_case_aaron_1095/clothing_geometry')); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,default=Path('outputs/fisheye_scene_p0')); p.add_argument('--frame',type=int,default=0); p.add_argument('--pano-width',type=int,default=2048); p.add_argument('--pano-height',type=int,default=1024); p.add_argument('--skip-render',action='store_true'); p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path); a=p.parse_args()
    root=ROOT; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    vertices_all=np.load(a.geometry/'vertices.npy'); dense_all=np.load(a.geometry/'densekp_3d.npy'); vertices=vertices_all[a.frame]; dense=dense_all[a.frame]
    body_fp=fingerprint(vertices); clothing_fp=fingerprint(np.load(a.clothing_geometry/'clothing_vertices.npy')); dense_fp=fingerprint(dense[None])
    np.save(out/'densekp_3d_source.npy',dense[None])
    phase17_fp=json.loads((root/'outputs/fisheye_camera_p0/regression_report.json').read_text())['densekp_3d_fingerprint']['value']
    try: spec=SceneAssetSpec.from_file(root/a.scene_spec).validate(root); scene_error=None
    except SceneAssetMissing as e: spec=None; scene_error=str(e)
    world=np.column_stack((dense[:,0],-dense[:,2],dense[:,1]+lift_offset(vertices)))
    if spec is not None and (spec.body_yaw_deg or any(spec.body_location)):
        yaw=np.deg2rad(spec.body_yaw_deg); rot=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]]); world=world@rot.T+np.asarray(spec.body_location)
    vw=np.column_stack((vertices[:,0],-vertices[:,2],vertices[:,1]+lift_offset(vertices)))
    if spec is not None and (spec.body_yaw_deg or any(spec.body_location)):
        yaw=np.deg2rad(spec.body_yaw_deg); rot=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]]); vw=vw@rot.T+np.asarray(spec.body_location)
    target=(vw.min(0)+vw.max(0))/2; origin=target-np.array([0,3.2,0])
    intr=OmniRadtanIntrinsics(); cams=[{'id':'cam_left','camera_to_world':pose(target,origin,90).tolist()},{'id':'cam_right','camera_to_world':pose(target,origin,-90).tolist()}]
    camera_file=out/'camera_config.json'; camera_file.write_text(json.dumps({'camera_model':'Omni-Radtan','intrinsics':intr.as_dict(),'coordinate_system':'CV +X right, +Y down, +Z forward; camera_to_world','seam_placement':'phase17 yaw90','cameras':cams},indent=2)+'\n')
    source=out/'equirectangular_source'; scene_rendered=False
    if spec is not None and not a.skip_render:
        baseline,_=load_appearance_baseline(a.appearance_manifest); cmd=[str(find_blender_executable(a.blender_bin)),'-b','--python',str(root/'tools/fisheye/render_equirectangular_scene_source.py'),'--','--geometry',str(a.geometry.resolve()),'--clothing-geometry',str(a.clothing_geometry.resolve()),'--cameras',str(camera_file.resolve()),'--output-dir',str(source.resolve()),'--frame',str(a.frame),'--scene-blend',str((root/spec.blend_path).resolve()),'--scene-collection',spec.collection or '','--hide-objects',','.join(spec.hide_objects),'--ground-mode',spec.ground_mode,'--body-location',*[str(x) for x in spec.body_location],'--body-yaw-deg',str(spec.body_yaw_deg),'--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],'--hdri',baseline['hdri_path'],'--width',str(a.pano_width),'--height',str(a.pano_height)]
        with (out/'render.log').open('w') as log: subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (source/'source_manifest.json').is_file(): raise RuntimeError(f'Blender scene render failed; inspect {out/"render.log"}')
        scene_rendered=True
    body_fp_after=fingerprint(np.load(a.geometry/'vertices.npy')[a.frame]); clothing_fp_after=fingerprint(np.load(a.clothing_geometry/'clothing_vertices.npy'))
    test_env=os.environ.copy(); test_env['PYTHONPATH']=str(root)
    projection=subprocess.run([sys.executable,'-m','unittest','tools.fisheye.test_omni_radtan'],cwd=root,env=test_env,capture_output=True,text=True); projection_pass=projection.returncode==0
    cases={}
    if scene_rendered or (spec is not None and a.skip_render and (source/'source_manifest.json').is_file()):
        for name,cam in (('cam_left',cams[0]),('cam_right',cams[1])):
            report=warp_case(source/f'{name}.exr',out/name,world,np.asarray(cam['camera_to_world']),intr,dense[None],dense_fp)
            np.save(out/name/'densekp2d.npy',np.load(out/name/'densekp_fisheye_2d.npy'))
            cases[name]={**report,**gates(out/name,intr)}
    dense_same=dense_fp==phase17_fp
    def yes(v): return 'YES' if v else ('NOT_RUN' if v is None else 'NO')
    all_cases=cases.get('cam_left') and cases.get('cam_right')
    report={'phase':'FISHEYE_SCENE_P0','baseline_commit':'1ad425978882ac96fbdaacec6fa15d1e21ec2a1b','geometry':str(a.geometry.resolve()),'geometry_frame':a.frame,'scene_spec':spec.describe() if spec else None,'scene_error':scene_error,'scene_rendered':scene_rendered,'intrinsics':intr.as_dict(),'seam_placement':'phase17 yaw90 (cam_left +90, cam_right -90)','densekp_3d_fingerprint':{'algorithm':'blake2b-128','value':dense_fp,'phase17_value':phase17_fp},'cases':cases,'status':{'BODY_GEOMETRY_UNCHANGED':yes(body_fp==body_fp_after),'CLOTHING_GEOMETRY_UNCHANGED':yes(clothing_fp==clothing_fp_after),'DENSEKP_3D_FINGERPRINT':'SAME' if dense_same else 'DIFFERENT','PROJECTION_REGRESSION':'PASS' if projection_pass else 'FAIL','DEPTH_PASS':yes(all_cases and all(c['depth']['depth_pass'] for c in cases.values())),'MASK_PASS':yes(all_cases and all(c['mask']['mask_pass'] for c in cases.values())),'SCENE_ASSET':'LOADED' if spec else 'MISSING'},'projection_regression_output':projection.stderr.strip().splitlines()[-2:],'elapsed_seconds':time.perf_counter()-started}
    gates_ok=report['status']['BODY_GEOMETRY_UNCHANGED']=='YES' and report['status']['CLOTHING_GEOMETRY_UNCHANGED']=='YES' and dense_same and projection_pass
    report['PHASE18_STATUS']='PASS' if (gates_ok and spec and all_cases and report['status']['DEPTH_PASS']=='YES' and report['status']['MASK_PASS']=='YES') else ('SCENE_ASSET_MISSING' if spec is None else 'FAIL')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'PHASE18_STATUS':report['PHASE18_STATUS'],'status':report['status'],'scene_error':scene_error},indent=2))

def lift_offset(vertices): return -float(vertices[:,1].min())

if __name__=='__main__': main()
