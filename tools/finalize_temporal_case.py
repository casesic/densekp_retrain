#!/usr/bin/env python3
"""Build videos and temporal statistics from a finalized DenseKP render."""
import argparse, csv, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def video(pattern, output, input_fps, output_fps=None):
    cmd=['ffmpeg','-y','-loglevel','error','-framerate',str(input_fps),'-i',str(pattern)]
    if output_fps: cmd += ['-vf',f'fps={output_fps}']
    cmd += ['-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(output)]
    subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--gt-fps',type=float,required=True); p.add_argument('--preview-fps',type=float,default=24); p.add_argument('--case-id',required=True); a=p.parse_args(); out=a.output_dir
    rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]; n=len(rows)
    rr=json.loads((out/'p0_render_report.json').read_text()); vr=json.loads((out/'p0_output_report.json').read_text())
    pts=np.load(out/'annotations/densekp_2d.npy'); xyz=np.load(out/'annotations/densekp_camera_xyz.npy'); vis=np.load(out/'annotations/densekp_visible.npy')
    body=np.load(out/'geometry/vertices.npy'); dense=np.load(out/'geometry/densekp_3d.npy')
    finite=all(np.isfinite(x).all() for x in (pts,xyz,body,dense))
    images=[np.asarray(Image.open(out/'rgb'/f'{i:06d}.png').convert('RGB'),np.float32)/255 for i in range(n)]
    masks=[np.asarray(Image.open(out/'mask'/f'{i:06d}.png').convert('L'))>=128 for i in range(n)]
    rgb_delta=[]; background_delta=[]; mask_iou=[]
    for i in range(1,n):
        diff=np.abs(images[i]-images[i-1]).mean(2); union=masks[i]|masks[i-1]; bg=~union
        rgb_delta.append(float(diff.mean())); background_delta.append(float(diff[bg].mean()) if bg.any() else 0.)
        inter=(masks[i]&masks[i-1]).sum(); mask_iou.append(float(inter/max(1,union.sum())))
    inframe=((xyz[...,2]<0)&(pts[...,0]>=0)&(pts[...,0]<rows[0]['resolution_width'])&(pts[...,1]>=0)&(pts[...,1]<rows[0]['resolution_height'])).sum(1)
    visible=(vis==0).sum(1)
    point_step=np.linalg.norm(np.diff(pts,axis=0),axis=2); median_step=float(np.median(point_step)); max_step=float(point_step.max())
    # Robust against a few legitimately fast distal points at the 6 fps sampling rate.
    overlay_jitter=bool(np.percentile(point_step,99)>40.0)
    mapping_ok=None; source=[]; clothing=[]
    if 'source_frame_30fps' in rows[0]:
        source=np.array([r['source_frame_30fps'] for r in rows]); clothing=np.array([r['clothing_frame_index'] for r in rows])
        mapping_ok=bool(np.array_equal(np.diff(source),np.full(n-1,5)) and np.array_equal(clothing,np.arange(n)))
    manifest=[]
    for i,r in enumerate(rows):
        manifest.append({'frame_id':i,'timestamp_sec':i/a.gt_fps,'clothing_frame_index':r.get('clothing_frame_index',''),'source_frame_30fps':r.get('source_frame_30fps',''),'visible_densekp':int(visible[i]),'inframe_densekp':int(inframe[i]),'render_ms':1000*rr['render_seconds']['values'][i],'finite':finite})
    with (out/'frame_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=manifest[0]); w.writeheader(); w.writerows(manifest)
    # Diagnostic panels and representative temporal contact sheet.
    diag=out/'diagnostic_frames'; diag.mkdir(exist_ok=True)
    for i in range(n):
        rgb=Image.open(out/'rgb'/f'{i:06d}.png').convert('RGB'); ov=Image.open(out/'overlay'/f'{i:06d}.png').convert('RGB')
        depth=np.load(out/'depth'/f'{i:06d}.npy'); valid=depth>0; dv=np.zeros_like(depth,np.uint8)
        if valid.any():
            lo,hi=np.percentile(depth[valid],[2,98]); dv[valid]=(255*np.clip((hi-depth[valid])/max(hi-lo,1e-6),0,1)).astype(np.uint8)
        dep=Image.fromarray(dv).convert('RGB'); mask=Image.open(out/'mask'/f'{i:06d}.png').convert('RGB')
        panel=Image.new('RGB',(1024,1024)); panel.paste(rgb,(0,0)); panel.paste(ov,(512,0)); panel.paste(dep,(0,512)); panel.paste(mask,(512,512)); panel=panel.resize((1024,512))
        ImageDraw.Draw(panel).text((8,8),f'{a.case_id} frame {i} | RGB / overlay / depth / mask',fill='white'); panel.save(diag/f'{i:06d}.png')
    sample_ids=np.unique(np.rint(np.linspace(0,n-1,min(12,n))).astype(int)); sheet=Image.new('RGB',(4*256,3*256)); rgb_sheet=Image.new('RGB',(4*256,3*256))
    for k,i in enumerate(sample_ids):
        sheet.paste(Image.open(out/'overlay'/f'{i:06d}.png').convert('RGB').resize((256,256)),((k%4)*256,(k//4)*256))
        rgb_sheet.paste(Image.open(out/'rgb'/f'{i:06d}.png').convert('RGB').resize((256,256)),((k%4)*256,(k//4)*256))
    sheet.save(out/'contact_sheet_temporal_samples.png'); sheet.save(out/'overlay_contact_sheet.png'); rgb_sheet.save(out/'rgb_contact_sheet.png')
    video(out/'rgb/%06d.png',out/f'rgb_{a.gt_fps:g}fps.mp4',a.gt_fps)
    video(out/'overlay/%06d.png',out/f'overlay_{a.gt_fps:g}fps.mp4',a.gt_fps)
    video(diag/'%06d.png',out/f'diagnostic_4panel_{a.gt_fps:g}fps.mp4',a.gt_fps)
    video(out/'rgb/%06d.png',out/f'preview_{a.preview_fps:g}fps.mp4',a.gt_fps,a.preview_fps)
    report={'case_id':a.case_id,'frames':n,'gt_fps':a.gt_fps,'preview_fps':a.preview_fps,'preview_note':f'preview only, GT temporal sampling remains {a.gt_fps:g} fps','projection_regression':'PASS' if rr['projection_error_px']['max']<1e-3 else 'FAIL','projection_max_error_px':rr['projection_error_px']['max'],'finite_arrays':finite,'body_clothing_mapping_exact':mapping_ok,'body_clothing_frame_slip':False if mapping_ok is None else not mapping_ok,'visible_densekp':{'values':visible.tolist(),'mean':float(visible.mean()),'min':int(visible.min()),'max':int(visible.max())},'inframe_densekp':{'values':inframe.tolist(),'mean':float(inframe.mean()),'min':int(inframe.min()),'max':int(inframe.max())},'render_ms':{'values':[1000*x for x in rr['render_seconds']['values']],'mean':1000*rr['render_seconds']['mean']},'temporal_basic':{'rgb_abs_delta_mean':float(np.mean(rgb_delta)),'rgb_abs_delta_max':float(np.max(rgb_delta)),'background_abs_delta_mean':float(np.mean(background_delta)),'background_abs_delta_max':float(np.max(background_delta)),'mask_iou_mean':float(np.mean(mask_iou)),'mask_iou_min':float(np.min(mask_iou)),'visible_count_max_step':int(np.abs(np.diff(visible)).max()),'inframe_count_max_step':int(np.abs(np.diff(inframe)).max()),'densekp_2d_step_median_px':median_step,'densekp_2d_step_p95_px':float(np.percentile(point_step,95)),'densekp_2d_step_max_px':max_step},'flicker_flag':bool(np.max(background_delta)>.02),'overlay_jitter_flag':overlay_jitter,'visibility_abnormal_jump':bool(np.abs(np.diff(visible)).max()>20)}
    (out/'temporal_report.json').write_text(json.dumps(report,indent=2)+'\n'); (out/'case_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
