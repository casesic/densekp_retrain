"""Bounded six same-buffer comparisons plus production packing/ROI preview.

Full canvases exist ONLY here, never in the production consumer or Dataset ROI
path. Successful owned references and the six retained render buffers are
cleaned after exact comparisons; exceptions preserve the failure scene.
"""
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image,ImageDraw
from densekp.core.datasets.image_layout import LAYOUT_FIELDS
from densekp.core.datasets import dataset_keypoints as dataset_module
from densekp_retrain.data.loading import load_box_dataset
from densekp_retrain.data.sequence_packing import pack_rendered_sequence
from densekp_retrain.data.sequence_index import validate_sequence_index
from densekp_retrain.imaging.output_modes import prepare_source_workspace,complete_output
from densekp_retrain.imaging.roi_consumer import write


def timed_item(dataset,row):
    read=dataset_module.read_image;timing={}
    def timed_read(*args,**kwargs):
        t=time.monotonic();value=read(*args,**kwargs);timing['decode_seconds']=time.monotonic()-t;return value
    with patch.object(dataset_module,'read_image',timed_read):
        t=time.monotonic();item=dataset[row];timing['total_seconds']=time.monotonic()-t
    timing['crop_and_metadata_seconds']=timing['total_seconds']-timing['decode_seconds']
    return item,timing


def finish_roi_only(root):
    root=Path(root);start=time.monotonic()
    pack,ep,index,coverage=pack_rendered_sequence(root,output_name='pack');pack_seconds=time.monotonic()-start
    assert index['packed']['row_count']==22
    dataset=load_box_dataset(pack/'packed.npz');checks=[]
    verification=root/'same_buffer_reference';source=prepare_source_workspace(verification,'compact')
    original=dict(np.load(pack/'packed.npz',allow_pickle=False));reference_bytes_peak=0;extra_start=time.monotonic()
    keys=['img_disp','img','proj_verts','proj_verts_cropped','visible','visibility_known','_trans','img_size','_scale','box_size','box_center']
    for k in [0,5,10]:
        for c in ['cam2','cam4']:
            oid=f'f{k}_{c}_p1';row=next(b['packed_row_id'] for b in index['boxes'] if b['observation_id']==oid)
            cd=root/f'frame_{k:03d}'/c;meta=json.loads((cd/'raster_storage.json').read_text());x0,y0,x1,y1=meta['roi_xyxy']
            # Independent reference from the retained SAME rendered display RGB,
            # not the new stored image or any old C full canvas.
            raw=np.asarray(Image.open(root/f'frame_{k:03d}/_temporary_ds_source/source/roi/{c}_preview.png').convert('RGB')).copy()
            physical=np.asarray(Image.open(meta['physical_mask_ref']))>0
            raw[~physical[y0:y1,x0:x1]]=0
            full=np.zeros((2880,5760,3),np.uint8);full[y0:y1,x0:x1]=raw
            image=source/f'f{k}_{c}.png';Image.fromarray(full).save(image);del full
            arrays={key:(value if value.ndim==0 else value[row:row+1]) for key,value in original.items() if key not in LAYOUT_FIELDS}
            arrays['imgname']=np.array([str(image)]);npz=source/f'f{k}_{c}.npz';np.savez(npz,**arrays)
            ref=load_box_dataset(npz)
            roi_item,roi_time=timed_item(dataset,row);full_item,full_time=timed_item(ref,0)
            differences={}
            for key in keys:
                a=np.asarray(roi_item[key]);b=np.asarray(full_item[key]);different=a!=b
                differences[key]=dict(different_values=int(different.sum()),max_abs=float(np.abs(a.astype(float)-b.astype(float)).max()))
            check=dict(frame_index=k,camera=c,row=row,ROI_read=roi_time,full_reference_read=full_time,differences=differences,
                supervised=int((roi_item['proj_verts_cropped'][:,2]>0).sum()),
                occluded_supervised=int(((roi_item['proj_verts_cropped'][:,2]>0)&~roi_item['visible']).sum()))
            checks.append(check);write(pack/'same_buffer_comparison.json',dict(status='in_progress',checks=checks))
            assert all(v['different_values']==0 for v in differences.values()),check
            Image.fromarray(roi_item['img_disp'][:,:,::-1]).save(pack/f'frame_{k:03d}_{c}_crop.png')
            reference_bytes_peak=max(reference_bytes_peak,sum(p.stat().st_size for p in source.iterdir() if p.is_file()))
    # One old sample, read before and after the core changes. No history repack.
    before=json.loads((root/'legacy_before.json').read_text());legacy=load_box_dataset(before['path']);old,old_time=timed_item(legacy,before['row'])
    with np.load(root/'legacy_before.npz',allow_pickle=False) as baseline:
        for key in old:np.testing.assert_array_equal(old[key],baseline[key],err_msg=key)
    verification_seconds=time.monotonic()-extra_start
    write(pack/'same_buffer_comparison.json',dict(status='passed_exact',checks=checks,ROI_Dataset_reads=6,full_reference_Dataset_reads=6,
        legacy_full_reads_this_stage=1,legacy_full_reads_before_code=1,legacy_exact=True,legacy_after_timing=old_time,
        same_buffer_verification_seconds=verification_seconds,temporary_reference_bytes_peak=reference_bytes_peak))
    # All necessary comparisons succeeded. No training index points here.
    t=time.monotonic();complete_output(verification,'compact',source,checks=dict(status='passed',exact_comparisons=6))
    for k in [0,5,10]:
        out=root/f'frame_{k:03d}';complete_output(out,'compact',out/'_temporary_ds_source/source',checks=dict(status='passed',same_buffer_Dataset_comparisons=2))
    cleanup_seconds=time.monotonic()-t;t=time.monotonic()
    import av
    from fractions import Fraction
    video=av.open(str(pack/'dual_roi.mp4'),'w');stream=video.add_stream('libx264',rate=10)
    stream.width=1024;stream.height=544;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
    for k in range(11):
        canvas=Image.new('RGB',(1024,544),(32,32,32));draw=ImageDraw.Draw(canvas)
        for i,c in enumerate(['cam2','cam4']):
            with Image.open(root/f'frame_{k:03d}'/c/'roi.png') as image:
                preview=image.convert('RGB');preview.thumbnail((512,512));canvas.paste(preview,(i*512+(512-preview.width)//2,32+(512-preview.height)//2))
            draw.text((i*512+8,8),f'{c} tick={k}/10 s source={12+3*k}/30 s ROI only',fill='white')
        fr=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');fr.pts=k;fr.time_base=Fraction(1,10)
        for packet in stream.encode(fr):video.mux(packet)
    for packet in stream.encode():video.mux(packet)
    video.close()
    with av.open(str(pack/'dual_roi.mp4')) as video:pts=[float(fr.pts*fr.time_base) for fr in video.decode(video=0)]
    np.testing.assert_allclose(pts,np.arange(11)/10,atol=1e-8,rtol=0)
    video_seconds=time.monotonic()-t;t=time.monotonic()
    validation=validate_sequence_index(ep,index,base_dir=pack,check_image_headers=True,check_packed=True)
    assert not list(root.rglob('_temporary_ds_source')) and not list(root.glob('frame_*/cam*/normal_erp.png'))
    write(pack/'readback.json',dict(status='passed',ROI_Dataset_reads=6,full_reference_Dataset_reads=6,legacy_reads_before_and_after=2,
        pack_seconds=pack_seconds,verification_seconds=verification_seconds,cleanup_seconds=cleanup_seconds,
        video_seconds=video_seconds,final_metadata_seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-start,
        video=dict(frames=11,pts_seconds=pts,time_base=[1,10],first_last_span_seconds=1.,last_frame_hold_seconds=.1,
            padding='gray presentation placeholder; not scene background'),validation=validation,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_forwards=0))
