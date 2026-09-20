#!/usr/bin/env python3
"""Phase 17.1: seam yaw sweep stress for the dual-fisheye overlap scenario.

Reuses the Phase 17 Omni-Radtan module, warp pipeline and Blender equirectangular
source renderer without modification; only the camera yaw placement varies.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from densekp_retrain.camera import OmniRadtanIntrinsics
from densekp_retrain.camera import pose
from fisheye.run_fisheye_camera_p0 import warp_case
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline
from densekp_retrain.rendering.process import run_logged

YAWS=(75,90,100,110)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/geometry')); p.add_argument('--clothing-geometry',type=Path,default=Path('outputs/baselines/temporal_case_aaron_1095/clothing_geometry')); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,default=Path('outputs/baselines/fisheye_camera_p0/seam_sweep')); p.add_argument('--frame',type=int,default=0); p.add_argument('--pano-width',type=int,default=2048); p.add_argument('--pano-height',type=int,default=1024); p.add_argument('--yaws',type=int,nargs='+',default=list(YAWS)); p.add_argument('--skip-render',action='store_true'); p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path); a=p.parse_args()
    root=ROOT; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    vertices_all=np.load(a.geometry/'vertices.npy'); dense_all=np.load(a.geometry/'densekp_3d.npy'); vertices=vertices_all[a.frame]; dense=dense_all[a.frame]
    import hashlib
    source_fingerprint=hashlib.blake2b(dense[None].tobytes(),digest_size=16).hexdigest()
    np.save(out/'densekp_3d_source.npy',dense[None])
    if hashlib.blake2b(np.load(out/'densekp_3d_source.npy').tobytes(),digest_size=16).hexdigest()!=source_fingerprint: raise RuntimeError('densekp_3d_source.npy fingerprint mismatch after save')
    lift=-float(vertices[:,1].min()); world=np.column_stack((dense[:,0],-dense[:,2],dense[:,1]+lift)); vw=np.column_stack((vertices[:,0],-vertices[:,2],vertices[:,1]+lift)); target=(vw.min(0)+vw.max(0))/2; origin=target-np.array([0,3.2,0])
    cams=[]
    for yaw in a.yaws:
        for name,sign in (('cam_left',1),('cam_right',-1)): cams.append({'id':f'{name}_yaw{yaw}','camera_to_world':pose(target,origin,sign*yaw).tolist()})
    camera_file=out/'camera_config.json'; intr=OmniRadtanIntrinsics(); camera_file.write_text(json.dumps({'camera_model':'Omni-Radtan','intrinsics':intr.as_dict(),'coordinate_system':'CV +X right, +Y down, +Z forward; camera_to_world','cameras':cams},indent=2)+'\n')
    source=out/'equirectangular_source'; source.mkdir(exist_ok=True)
    if not a.skip_render:
        baseline,_=load_appearance_baseline(a.appearance_manifest); cmd=[str(find_blender_executable(a.blender_bin)),'-b','--python',str(root/'tools/fisheye/render_equirectangular_source.py'),'--','--geometry',str(a.geometry.resolve()),'--clothing-geometry',str(a.clothing_geometry.resolve()),'--cameras',str(camera_file.resolve()),'--output-dir',str(source.resolve()),'--frame',str(a.frame),'--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],'--hdri',baseline['hdri_path'],'--width',str(a.pano_width),'--height',str(a.pano_height)]
        run_logged(cmd,cwd=root,log_path=out/'render.log')
        if not (source/'source_manifest.json').is_file(): raise RuntimeError(f'Blender source render failed; inspect {out/"render.log"}')
    poses={c['id']:np.asarray(c['camera_to_world']) for c in cams}
    stats={}
    for yaw in a.yaws:
        case={}
        for name in ('cam_left','cam_right'):
            report=warp_case(source/f'{name}_yaw{yaw}.exr',out/f'yaw{yaw}'/name,world,poses[f'{name}_yaw{yaw}'],intr,dense[None],source_fingerprint)
            case[name]=report
        lv={int(x) for x in np.flatnonzero(np.load(out/f'yaw{yaw}/cam_left/densekp_visible.npy')[0])}; rv={int(x) for x in np.flatnonzero(np.load(out/f'yaw{yaw}/cam_right/densekp_visible.npy')[0])}
        inter=len(lv&rv); union=len(lv|rv); stats[yaw]={'cam_left':case['cam_left'],'cam_right':case['cam_right'],'overlap':{'left_visible_set':sorted(lv),'right_visible_set':sorted(rv),'intersection_count':inter,'union_count':union,'intersection_over_union':inter/union if union else 0.,'visible_difference':len(lv.symmetric_difference(rv))}}
        (out/f'yaw{yaw}'/'case_report.json').write_text(json.dumps(stats[yaw],indent=2)+'\n')
    def row(yaw):
        l,r=stats[yaw]['cam_left'],stats[yaw]['cam_right']; s=stats[yaw]['overlap']
        return {'seam_yaw_deg':yaw,'cam_left':{'in_frame_keypoints':l['valid_keypoints'],'visible_keypoints':l['visible_keypoints'],'finite_keypoints':l['finite_keypoints'],'max_projection_error_px':l['projection_error_px']['max']},'cam_right':{'in_frame_keypoints':r['valid_keypoints'],'visible_keypoints':r['visible_keypoints'],'finite_keypoints':r['finite_keypoints'],'max_projection_error_px':r['projection_error_px']['max']},'left_visible_set':s['left_visible_set'],'right_visible_set':s['right_visible_set'],'intersection_count':s['intersection_count'],'union_count':s['union_count'],'intersection_over_union':s['intersection_over_union'],'visible_difference':s['visible_difference']}
    rows=[row(y) for y in a.yaws]
    def score(r):
        l,v=r['cam_left']['visible_keypoints'],r['cam_right']['visible_keypoints']; lf,rf=r['cam_left']['in_frame_keypoints'],r['cam_right']['in_frame_keypoints']
        different=int(l!=v)*100; in_range=(40<=l<=90)*50+(40<=v<=90)*50; vanished=(lf<30 or rf<30)*200; iou=r['intersection_over_union']; return different+in_range-vanished-100*iou
    eligible=[r for r in rows if r['cam_left']['visible_keypoints']!=r['cam_right']['visible_keypoints'] and 40<=r['cam_left']['visible_keypoints']<=90 and 40<=r['cam_right']['visible_keypoints']<=90 and r['cam_left']['in_frame_keypoints']>=30 and r['cam_right']['in_frame_keypoints']>=30]
    best=max(eligible or rows,key=score); recommended=f"yaw{best['seam_yaw_deg']}"
    projection_ok=all(r['cam_left']['max_projection_error_px']<1e-9 and r['cam_right']['max_projection_error_px']<1e-9 for r in rows)
    fingerprint_ok=all(y['cam_left']['densekp_3d_fingerprint']['matches_projection_source'] and y['cam_left']['densekp_3d_fingerprint']['value']==source_fingerprint and y['cam_right']['densekp_3d_fingerprint']['value']==source_fingerprint for y in stats.values()) and hashlib.blake2b(np.load(out/'densekp_3d_source.npy').tobytes(),digest_size=16).hexdigest()==source_fingerprint
    subset_different=any(r['left_visible_set']!=r['right_visible_set'] and r['visible_difference']>=10 for r in rows)
    report={'baseline_commit':'1ad425978882ac96fbdaacec6fa15d1e21ec2a1b','geometry':str(a.geometry.resolve()),'geometry_frame':a.frame,'intrinsics':intr.as_dict(),'densekp_3d_fingerprint':{'algorithm':'blake2b-128','value':source_fingerprint,'consistent_across_cases':fingerprint_ok},'rows':rows,'recommended_seam_yaw':recommended,'recommended_row':best,'projection_regression_pass':projection_ok,'densekp_3d_fingerprint_pass':fingerprint_ok,'subset_different_pass':subset_different,'phase17_1_pass':bool(projection_ok and fingerprint_ok and subset_different),'elapsed_seconds':time.perf_counter()-started}
    (out/'seam_sweep_report.json').write_text(json.dumps(report,indent=2)+'\n')
    tile=384; sheet=Image.new('RGB',(tile*2+16,len(rows)*(tile+26)+8),(24,24,28)); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        y=8+i*(tile+26); draw.text((8,y-2),f"yaw{r['seam_yaw_deg']}  left_visible={r['cam_left']['visible_keypoints']} right_visible={r['cam_right']['visible_keypoints']} intersection={r['intersection_count']} IoU={r['intersection_over_union']:.2f}"+('  <= RECOMMENDED' if f"yaw{r['seam_yaw_deg']}"==recommended else ''),fill=(255,220,80) if f"yaw{r['seam_yaw_deg']}"==recommended else (230,230,230))
        for j,name in enumerate(('cam_left','cam_right')):
            im=Image.open(out/f"yaw{r['seam_yaw_deg']}"f'/{name}/densekp_overlay.png').resize((tile,tile)); sheet.paste(im,(8+j*(tile+4),y+16))
    sheet.save(out/'contact_sheet.png'); print(json.dumps({k:report[k] for k in ('recommended_seam_yaw','projection_regression_pass','densekp_3d_fingerprint_pass','subset_different_pass','phase17_1_pass')},indent=2)); return report

if __name__=='__main__': main()
