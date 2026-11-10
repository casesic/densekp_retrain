"""Synchronous compact consumption and bounded sequence Dataset/video delivery."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
import cv2
import numpy as np
from PIL import Image
from densekp_retrain.imaging.roi_consumer import consume_roi_frame,write


def finish(root):
    from densekp_retrain.data.sequence_packing import pack_rendered_sequence
    from densekp_retrain.data.loading import load_box_dataset
    from densekp.core.keypoint_supervision import supervision_for_crop
    from densekp_retrain.data.sequence_index import validate_sequence_index
    start=time.monotonic();pack,ep,index,coverage=pack_rendered_sequence(root,output_name='pack');pack_seconds=time.monotonic()-start
    assert index['packed']['row_count']==22,'Unexpected missing/rejected rows; inspect sequence sidecar'
    dataset=load_box_dataset(pack/'packed.npz');checks=[];t=time.monotonic()
    for k in [0,5,10]:
        for c in ['cam2','cam4']:
            oid=f'f{k}_{c}_p1';row=next(b['packed_row_id'] for b in index['boxes'] if b['observation_id']==oid)
            item=dataset[row];image=cv2.imread(item['imgname']);warped=cv2.warpAffine(image,item['_trans'],(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            np.testing.assert_array_equal(warped,item['img_disp'])
            xy=((np.c_[item['proj_verts'][:,:2],np.ones(138)]@item['_trans'].T).astype(np.float32)/np.array([256,256])-.5).astype(np.float32)
            np.testing.assert_array_equal(xy,item['proj_verts_cropped'][:,:2])
            sup=supervision_for_crop(dataset.data['in_view'][row],xy);np.testing.assert_array_equal(sup,item['proj_verts_cropped'][:,2]>0)
            Image.fromarray(warped[:,:,::-1]).save(pack/f'frame_{k:03d}_{c}_crop.png')
            checks.append(dict(row=row,frame_index=k,camera=c,image_affine_exact=True,point_affine_exact=True,
                supervised=int(sup.sum()),occluded_supervised=int((sup&~dataset.data['visible'][row]).sum())))
    readback_seconds=time.monotonic()-t;t=time.monotonic()
    import av
    from fractions import Fraction
    video=av.open(str(pack/'dual_view.mp4'),'w');stream=video.add_stream('libx264',rate=10)
    stream.width=1440;stream.height=360;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
    for k in range(11):
        canvas=Image.new('RGB',(1440,360))
        for i,c in enumerate(['cam2','cam4']):
            with Image.open(root/f'frame_{k:03d}'/c/'normal_erp.png') as image:canvas.paste(image.convert('RGB').resize((720,360)),(720*i,0))
        fr=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');fr.pts=k;fr.time_base=Fraction(1,10)
        for packet in stream.encode(fr):video.mux(packet)
    for packet in stream.encode():video.mux(packet)
    video.close()
    with av.open(str(pack/'dual_view.mp4')) as video:pts=[float(fr.pts*fr.time_base) for fr in video.decode(video=0)]
    np.testing.assert_allclose(pts,np.arange(11)/10,atol=1e-8,rtol=0)
    validation=validate_sequence_index(ep,index,base_dir=pack,check_image_headers=True,check_packed=True)
    assert not list(root.rglob('_temporary_ds_source'))
    write(pack/'readback.json',dict(status='passed',actual_Dataset_reads=6,checks=checks,
        pack_seconds=pack_seconds,readback_seconds=readback_seconds,video_and_final_metadata_seconds=time.monotonic()-t,
        elapsed_seconds=time.monotonic()-start,video=dict(frames=11,time_base=[1,10],source_frames=list(range(12,43,3)),
            pts_seconds=pts,first_last_span_seconds=1.0,last_frame_hold_seconds=.1),validation=validation,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_forwards=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['frame','finish','serve']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--frame',type=int)
    a=ap.parse_args();cv2.setNumThreads(1)
    if a.stage=='serve':
        from densekp_retrain.runtime.frame_consumer import serve_frames
        raise SystemExit(serve_frames(a.root,consume_roi_frame))
    if a.stage=='frame':consume_roi_frame(a.root,a.frame)
    elif json.loads((a.root/'run_config.json').read_text()).get('consumer_mode')=='persistent':
        from tools.experiments.validate_persistent_roi import finish_short
        finish_short(a.root)
    elif json.loads((a.root/'run_config.json').read_text()).get('storage_mode')=='roi_only':
        from tools.experiments.validate_roi_only import finish_roi_only
        finish_roi_only(a.root)
    else:finish(a.root)

if __name__=='__main__':main()
