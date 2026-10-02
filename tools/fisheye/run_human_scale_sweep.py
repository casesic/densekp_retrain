#!/usr/bin/env python3
"""Phase 17.2: human scale sweep for the frozen Phase 17 fisheye pipeline.

`scale` parameterizes the camera-human distance: the camera sits at
3.2 * scale meters from the body bounding-box center (scale 1.0 = the frozen
Phase 17 placement). Body geometry, DenseKP 3D, intrinsics and the yaw
placement rule (center 0, seam +/-90) are untouched; only the camera origin
changes. Uses the unmodified Phase 17 renderer and warp pipeline.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from densekp_retrain.camera import OmniRadtanIntrinsics
from densekp_retrain.camera import pose
from densekp_retrain.imaging.omni import warp_case
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline
from densekp_retrain.rendering.process import run_logged

SCALES=(1.0,0.8,0.7,0.6,0.5); BASELINE_DISTANCE=3.2; DIAMETER=3000

def fingerprint(array): return hashlib.blake2b(np.asarray(array).tobytes(),digest_size=16).hexdigest()

def human_metrics(case_dir):
    pix=np.load(case_dir/'densekp2d.npy')[0]; report=json.loads((case_dir/'case_report.json').read_text())
    finite=np.isfinite(pix).all(1)
    height=float(pix[finite,1].max()-pix[finite,1].min()) if finite.any() else 0.
    return {'human_pixel_height':height,'human_height_fraction_of_diameter':height/DIAMETER,'visible_keypoints':report['visible_keypoints'],'in_frame_keypoints':report['valid_keypoints'],'projection_error_px_max':report['projection_error_px']['max']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/geometry')); p.add_argument('--clothing-geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/clothing_geometry')); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,default=Path('outputs/experiments/human_scale_sweep')); p.add_argument('--frame',type=int,default=0); p.add_argument('--pano-width',type=int,default=2048); p.add_argument('--pano-height',type=int,default=1024); p.add_argument('--scales',type=float,nargs='+',default=list(SCALES)); p.add_argument('--skip-render',action='store_true'); p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path); a=p.parse_args()
    root=ROOT; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    vertices_all=np.load(a.geometry/'vertices.npy'); dense_all=np.load(a.geometry/'densekp_3d.npy'); vertices=vertices_all[a.frame]; dense=dense_all[a.frame]
    body_fp=fingerprint(vertices); dense_fp=fingerprint(dense[None]); np.save(out/'densekp_3d_source.npy',dense[None])
    phase17_fp=json.loads((root/'outputs/baselines/fisheye_camera_p0/regression_report.json').read_text())['densekp_3d_fingerprint']['value']
    lift=-float(vertices[:,1].min()); world=np.column_stack((dense[:,0],-dense[:,2],dense[:,1]+lift)); vw=np.column_stack((vertices[:,0],-vertices[:,2],vertices[:,1]+lift)); target=(vw.min(0)+vw.max(0))/2
    intr=OmniRadtanIntrinsics(); cams=[]; id_map={}
    for scale in a.scales:
        tag=f's{scale:g}'; origin=target-np.array([0,BASELINE_DISTANCE*scale,0])
        for name,yaw in (('center',0),('cam_left',90),('cam_right',-90)):
            cid=f'{name}_{tag}'; id_map[cid]=(scale,name); cams.append({'id':cid,'camera_to_world':pose(target,origin,yaw).tolist()})
    camera_file=out/'camera_config.json'; camera_file.write_text(json.dumps({'camera_model':'Omni-Radtan','intrinsics':intr.as_dict(),'coordinate_system':'CV +X right, +Y down, +Z forward; camera_to_world','placement':'camera at 3.2*scale m from body bbox center; center yaw=0, seam yaw=+/-90','cameras':cams},indent=2)+'\n')
    source=out/'equirectangular_source'; source.mkdir(exist_ok=True)
    if not a.skip_render:
        baseline,_=load_appearance_baseline(a.appearance_manifest); cmd=[str(find_blender_executable(a.blender_bin)),'-b','--python',str(root/'tools/fisheye/render_equirectangular_source.py'),'--','--geometry',str(a.geometry.resolve()),'--clothing-geometry',str(a.clothing_geometry.resolve()),'--cameras',str(camera_file.resolve()),'--output-dir',str(source.resolve()),'--frame',str(a.frame),'--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],'--hdri',baseline['hdri_path'],'--width',str(a.pano_width),'--height',str(a.pano_height)]
        run_logged(cmd,cwd=root,log_path=out/'render.log')
        if not (source/'source_manifest.json').is_file(): raise RuntimeError(f'Blender source render failed; inspect {out/"render.log"}')
    body_fp_after=fingerprint(np.load(a.geometry/'vertices.npy')[a.frame])
    rows=[]; poses={c['id']:np.asarray(c['camera_to_world']) for c in cams}
    for scale in a.scales:
        tag=f's{scale:g}'; case={'scale':scale,'camera_distance_m':BASELINE_DISTANCE*scale,'cameras':{}}
        for name in ('center','cam_left','cam_right'):
            cid=f'{name}_{tag}'
            warp_case(source/f'{cid}.exr',out/f'scale_{scale:g}'/name,world,poses[cid],intr,dense[None],dense_fp)
            np.save(out/f'scale_{scale:g}'/name/'densekp2d.npy',np.load(out/f'scale_{scale:g}'/name/'densekp_fisheye_2d.npy'))
            case['cameras'][name]=human_metrics(out/f'scale_{scale:g}'/name)
        rows.append(case); (out/f'scale_{scale:g}'/'case_report.json').write_text(json.dumps(case,indent=2)+'\n')
    test_env=os.environ.copy(); test_env['PYTHONPATH']=str(root)
    projection=subprocess.run([sys.executable,'-m','unittest','tests.test_omni_radtan'],cwd=root,env=test_env,capture_output=True,text=True); projection_pass=projection.returncode==0
    projection_ok=projection_pass and all(c['cameras'][n]['projection_error_px_max']<1e-9 for c in rows for n in c['cameras'])
    def seam_ok(c): l,r=c['cameras']['cam_left'],c['cameras']['cam_right']; return 50<=l['visible_keypoints']<=100 and 50<=r['visible_keypoints']<=100
    def center_gap(c):
        f=c['cameras']['center']['human_height_fraction_of_diameter']
        return 0. if .40<=f<=.50 else min(abs(f-.40),abs(f-.50))
    seam_eligible=[c for c in rows if seam_ok(c)]
    pool=seam_eligible or rows; best=min(pool,key=lambda c:(center_gap(c),-c['scale']))
    report={'phase':'HUMAN_SCALE_SWEEP','baseline_commit':'1ad425978882ac96fbdaacec6fa15d1e21ec2a1b','geometry':str(a.geometry.resolve()),'geometry_frame':a.frame,'placement_rule':f'camera at {BASELINE_DISTANCE}*scale m from body bbox center; center yaw=0, seam yaw=+/-90','intrinsics':intr.as_dict(),'densekp_3d_fingerprint':{'algorithm':'blake2b-128','value':dense_fp,'phase17_value':phase17_fp},'scales':rows,'recommended_scale':best['scale'],'recommended':best,'center_target_fraction':[0.40,0.50],'seam_visible_target':[50,100],'status':{'BODY_GEOMETRY_UNCHANGED':'YES' if body_fp==body_fp_after else 'NO','DENSEKP_3D_FINGERPRINT':'SAME' if dense_fp==phase17_fp else 'DIFFERENT','PROJECTION_REGRESSION':'PASS' if projection_ok else 'FAIL'},'projection_regression_output':projection.stderr.strip().splitlines()[-2:],'elapsed_seconds':time.perf_counter()-started}
    report['PHASE17_2_STATUS']='PASS' if (report['status']['BODY_GEOMETRY_UNCHANGED']=='YES' and report['status']['DENSEKP_3D_FINGERPRINT']=='SAME' and report['status']['PROJECTION_REGRESSION']=='PASS') else 'FAIL'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'PHASE17_2_STATUS':report['PHASE17_2_STATUS'],'status':report['status'],'recommended_scale':report['recommended_scale'],'recommended_center_fraction':best['cameras']['center']['human_height_fraction_of_diameter'],'recommended_seam_visible':[best['cameras']['cam_left']['visible_keypoints'],best['cameras']['cam_right']['visible_keypoints']]},indent=2))

if __name__=='__main__': main()
