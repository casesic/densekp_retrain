#!/usr/bin/env python3
"""把 Blender EXR Z 转 NPY，计算 visibility，并生成 overlay/contact sheet。"""
import argparse,json,math
from pathlib import Path
import OpenEXR,numpy as np
from PIL import Image,ImageDraw
COL={0:(30,240,70),1:(255,65,45),2:(160,160,160),3:(180,50,220)}; N={0:'VISIBLE',1:'SELF_OCCLUDED',2:'OUT_OF_FRAME',3:'BEHIND_CAMERA'}
def passes(p):
    f=OpenEXR.File(str(p)); found={}
    for part in f.parts:
        for name,channel in part.channels.items(): found[name.split('.')[0]]=np.asarray(channel.pixels,np.float32)
    if 'Depth' not in found or 'Mask' not in found: raise RuntimeError(f'EXR 缺少 Depth/Mask: {p}, {list(found)}')
    return found['Depth'],found['Mask']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=Path('outputs/baselines/p0_perspective')); a=p.parse_args(); out=a.output_dir
    rec=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]; pts=np.load(out/'annotations/densekp_2d.npy'); xyz=np.load(out/'annotations/densekp_camera_xyz.npy')
    cfg=json.loads((out/'depth_test_config.json').read_text()); pd,_=passes(out/cfg['depth_file']); h,w=pd.shape; sx,sy=cfg['sample_pixel']; center=float(pd[h//2,w//2]); off=float(pd[sy,sx]); rx=((sx+.5)-w/2)/cfg['fx']; ez=cfg['plane_camera_z_m']; er=ez*math.sqrt(1+rx*rx); ze=abs(off-ez); re=abs(off-er); sem='EUCLIDEAN_RAY_DISTANCE' if re<ze else 'CAMERA_SPACE_Z'; dt=dict(cfg,center_depth=center,off_axis_depth=off,expected_camera_z=ez,expected_euclidean_range=er,camera_z_error=ze,euclidean_range_error=re,semantics=sem); (out/'depth_semantics.json').write_text(json.dumps(dt,indent=2)+'\n')
    frames=len(rec)
    if pts.shape!=(frames,138,2) or xyz.shape!=(frames,138,3): raise RuntimeError(f'annotation shape mismatch: {pts.shape}, {xyz.shape}, frames={frames}')
    separate_mask=(out/'mask_raw'/'000000.exr').is_file(); dressed=bool(rec[0].get('clothing_id')); errs=np.full((frames,138),np.nan,np.float32); state=np.ones((frames,138),np.uint8); pdense=np.linalg.norm(xyz,axis=2) if sem=='EUCLIDEAN_RAY_DISTANCE' else -xyz[...,2]; ds=[]
    for f in range(frames):
        d,_=passes(out/'depth_raw'/f'{f:06d}.exr'); _,m=passes((out/'mask_raw' if separate_mask else out/'depth_raw')/f'{f:06d}.exr'); d[~np.isfinite(d)]=0; np.save(out/'depth'/f'{f:06d}.npy',d); mask_image=Image.fromarray((m>=.5).astype(np.uint8)*255); mask_image.save(out/'mask'/f'{f:06d}.png');
        if dressed: mask_image.save(out/'masks/person'/f'{f:06d}.png')
        ds.append(d)
        for j,(u,v) in enumerate(pts[f]):
            if xyz[f,j,2]>=0: state[f,j]=3
            elif not(0<=u<d.shape[1] and 0<=v<d.shape[0]): state[f,j]=2
            else:
                x,y=int(np.floor(u)),int(np.floor(v)); q=d[max(0,y-1):min(d.shape[0],y+2),max(0,x-1):min(d.shape[1],x+2)]; q=q[q>0]
                if q.size: errs[f,j]=float(np.min(np.abs(q-pdense[f,j])))
    valid=errs[np.isfinite(errs)]; es={f'p{x}':float(np.percentile(valid,x)) for x in [50,75,90,95,99]}; tol=max(.002,2*float(np.median(pdense[state==1]))/rec[0]['intrinsics']['fx']); vis=state.copy(); vis[(state==1)&np.isfinite(errs)&(errs<=tol)]=0
    np.save(out/'annotations/densekp_depth.npy',pdense.astype(np.float32)); np.save(out/'annotations/densekp_valid.npy',np.ones_like(vis,dtype=np.uint8)); np.save(out/'annotations/densekp_visible.npy',vis); np.save(out/'annotations/densekp_depth_error.npy',errs)
    thumbs=[]
    for f in range(frames):
        im=Image.open(out/'rgb'/f'{f:06d}.png').convert('RGB'); mask=np.asarray(Image.open(out/'mask'/f'{f:06d}.png')); mask=mask[...,0] if mask.ndim==3 else mask
        if ds[f].shape!=(im.height,im.width) or mask.shape!=ds[f].shape or not set(np.unique(mask)).issubset({0,255}): raise RuntimeError(f'frame {f} RGB/depth/mask 无效')
        ov=im.copy(); dr=ImageDraw.Draw(ov)
        for j,((u,v),s) in enumerate(zip(pts[f],vis[f])):
            c=COL[int(s)]; r=3 if s==0 else 2; dr.ellipse((u-r,v-r,u+r,v+r),fill=c)
            if j%20==0: dr.text((u+3,v+3),str(j),fill=c)
        dr.text((8,8),f'frame={f} green=visible red=occluded',fill=(255,255,255)); ov.save(out/'overlay'/f'{f:06d}.png'); thumbs.append(ov.resize((256,256)))
    cols=min(8,frames); rows=math.ceil(frames/cols); sh=Image.new('RGB',(cols*256,rows*256)); [sh.paste(t,((i%cols)*256,(i//cols)*256)) for i,t in enumerate(thumbs)]; sh.save(out/'overlay/contact_sheet.png'); sh.save(out/'contact_sheet.png')
    labels=dict(N); labels[1]='OCCLUDED' if dressed else labels[1]; report={'frames':frames,'rgb_complete':True,'depth_complete':True,'mask_complete':True,'overlay_complete':True,'depth_semantics':dt,'depth_error_m':es,'visibility_tolerance_m':tol,'visibility_counts':{labels[x]:int(np.count_nonzero(vis==x)) for x in labels},'visibility_per_frame':[{labels[x]:int(np.count_nonzero(vis[f]==x)) for x in labels} for f in range(frames)]}; (out/'p0_output_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
