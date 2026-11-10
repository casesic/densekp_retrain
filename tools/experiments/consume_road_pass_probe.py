"""Final-raster DS consumer + per-episode packing + bounded real Dataset checks."""
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
from densekp_retrain.imaging.ds import prepare_ds_domain,project_ds_scene,remap_ds_images,save_ds_images,make_ds_annotations,save_ds_annotations,save_scene_pair
from densekp_retrain.imaging.source import read_ds_source
from densekp_retrain.imaging.output_modes import preserve_source_recipe,complete_output
from densekp_retrain.data.sequence_packing import pack_rendered_sequence
from densekp_retrain.data.loading import load_box_dataset
from densekp.core.keypoint_supervision import supervision_for_crop
from densekp_retrain.visualization.imaging import draw_overlay

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--pack-name',default='probe_pack');a=ap.parse_args()
    cv2.setNumThreads(1);began=time.monotonic();job=json.loads(a.job.read_text());ep_path=Path(job['recipe']);ep=json.loads(ep_path.read_text());directory=ep_path.parent
    people=ep['people'];ids=[p['person_id'] for p in people];world=np.load(directory/'world_densekp.npy')
    poses={c['id']:np.array(c['camera_to_world']) for c in json.loads(Path(job['rig']).read_text())['cameras']}
    masks=np.load(ROOT/'assets_external/real_erp_valid_masks.npz');domains={};counts=[]
    for k,source_folder in zip(job['frames'],job['source_dirs']):
        out=directory/f'frame_{k:03d}';source_folder=Path(source_folder);start=time.monotonic();states={};camera_reports=[]
        np.save(out/'shared_world_densekp.npy',world[k])
        for c in ['cam2','cam4']:
            source=read_ds_source(source_folder,c)
            assert source.rgb.shape==(2880,5760,3) and source.instance.shape==(2880,5760)
            if c not in domains:
                lens=np.asarray(Image.open(ROOT/f'outputs/acceptance/ds_output_modes_20261008/attempt_4/full/{c}/reference_lens_mask.png'))>0
                domains[c]=prepare_ds_domain(c,source,masks[f'valid_masks_{c}'],lens,output_size=(5760,2880))
            domain=domains[c];projection=project_ds_scene(world[k],ids,poses[c],c,source,domain);images=remap_ds_images(c,source,domain)
            cd=out/c;cd.mkdir(exist_ok=False);save_ds_images(cd,images,domain,output_mode='compact')
            data=make_ds_annotations(people,ids,world[k],poses[c],projection);save_ds_annotations(cd,data,ids,images,domain,output_mode='compact');states[c]=data
            camera_reports.append(dict(camera=c,in_view=data['in_view'].sum(1).tolist(),visible=data['visible'].sum(1).tolist(),ray_roundtrip_max=projection.ray_error))
            if k==job['frames'][0]:draw_overlay(images.normal_erp_rgb,data['xy'],data['visible'],data['in_view'],ids).resize((1440,720)).save(out/f'{c}_overlay.jpg')
            del source,images
        report=save_scene_pair(out,ids,world[k],states,camera_reports,output_mode='compact')
        report.update(final_erp_resolution=[5760,2880],source_erp_resolution=[5760,2880],fisheye_resolution=[2880,2880],output_mode='compact',frame_index=k,time_tick=k,time_base=[1,10],postprocess_seconds=time.monotonic()-start)
        write(out/'report.json',report);preserve_source_recipe(out,source_folder,extra=dict(episode_recipe=str(ep_path),source_frame_map={str(p['person_id']):p['source_frames'][k] for p in people}))
        counts.append(dict(frame_index=k,seconds=time.monotonic()-start));print('DS_FRAME_DONE',k,counts[-1]['seconds'],flush=True)
    pack,episode,index,coverage=pack_rendered_sequence(directory,output_name=a.pack_name)
    dataset=load_box_dataset(pack/'packed.npz');checks=[]
    chosen=np.unique(np.linspace(0,len(dataset)-1,min(6,len(dataset)),dtype=int))
    for i in chosen:
        item=dataset[int(i)];source=cv2.imread(item['imgname']);warped=cv2.warpAffine(source,item['_trans'],(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        np.testing.assert_array_equal(warped,item['img_disp'])
        pixels=(np.c_[item['proj_verts'][:,:2],np.ones(138)]@item['_trans'].T).astype(np.float32)
        xy=(pixels/np.array([256,256])-.5).astype(np.float32);np.testing.assert_array_equal(xy,item['proj_verts_cropped'][:,:2])
        supervised=supervision_for_crop(dataset.data['in_view'][i],xy);np.testing.assert_array_equal(supervised,item['proj_verts_cropped'][:,2]>0)
        Image.fromarray(warped[:,:,::-1]).save(pack/f'crop_{i:03d}.png')
        checks.append(dict(row=int(i),points=138,image_affine_exact=True,point_affine_exact=True,supervised=int(supervised.sum()),occluded_supervised=int((supervised&~dataset.data['visible'][i]).sum()),affine=item['_trans'].tolist()))
    # Only after Dataset + sidecar readback, clean this job's owned sources.
    for k,source_folder in zip(job['frames'],job['source_dirs']):
        complete_output(directory/f'frame_{k:03d}','compact',Path(source_folder),checks=dict(status='passed',packed_rows=len(dataset),readback_rows=len(checks)))
    # A probe video is explicitly partial: actual timestamps only, never stretched to3s.
    import av
    from fractions import Fraction
    video=av.open(str(pack/'dual_view_probe.mp4'),'w');stream=video.add_stream('libx264',rate=10);stream.width=1440;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
    for k in job['frames']:
        panels=[Image.open(directory/f'frame_{k:03d}'/c/'normal_erp.png').convert('RGB').resize((720,360)) for c in ['cam2','cam4']]
        canvas=Image.new('RGB',(1440,720));canvas.paste(panels[0],(0,180));canvas.paste(panels[1],(720,180));fr=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');fr.pts=k;fr.time_base=Fraction(1,10)
        for packet in stream.encode(fr):video.mux(packet)
    for packet in stream.encode():video.mux(packet)
    video.close()
    write(pack/'readback.json',dict(status='passed',actual_readbacks=len(checks),checks=checks,postprocess=counts,
        elapsed_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        video=dict(frames=job['frames'],time_base=[1,10],last_frame_hold_seconds=.1,first_last_span_seconds=(job['frames'][-1]-job['frames'][0])/10,partial_probe=len(job['frames'])<31),
        source_cleanup='owned source only; after actual readback',no_model_constructed=True))
    print('PACK_READBACK_DONE',len(dataset),flush=True)

if __name__=='__main__':main()
