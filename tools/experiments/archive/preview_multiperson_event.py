"""Three label/ID evidence overlays from final ROI data, no renderer or model."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();start=time.monotonic()
    report=json.loads((root/'pack/multiperson_acceptance.json').read_text());recipe=json.loads((root/'recipe.json').read_text());outputs=[]
    cameras=['cam2','cam4'];colors={1:'#50ffff',2:'#ff80ff',3:'#aaff88'}
    for k in report['readback_frame_indices']:
        canvas=Image.new('RGB',(1024,570),(32,32,32));draw=ImageDraw.Draw(canvas)
        anns={c:dict(np.load(root/f'frame_{k:03d}'/c/'annotations.npz')) for c in cameras};target=recipe['target_person_id'];target_index=list(anns['cam2']['person_ids']).index(target)
        common=anns['cam2']['in_view'][target_index]&anns['cam4']['in_view'][target_index]
        for col,c in enumerate(cameras):
            folder=root/f'frame_{k:03d}'/c;meta=json.loads((folder/'raster_storage.json').read_text());offset=np.array(meta['offset_xy'])
            with Image.open(folder/'roi.png') as src:
                image=src.convert('RGB');image.thumbnail((512,512));width,height=image.size
            ratio=np.array([width/meta['stored_size_wh'][0],height/meta['stored_size_wh'][1]])
            origin=np.array([col*512+(512-width)//2,30+(512-height)//2]);canvas.paste(image,tuple(origin))
            for box in json.loads((folder/'boxes.json').read_text()):
                if box['bbox_xyxy_exclusive'] is None:continue
                pid=box['person_id'];b=(np.array(box['bbox_xyxy_exclusive']).reshape(2,2)-offset)*ratio+origin
                draw.rectangle(tuple(b.ravel()),outline=colors[pid],width=2);draw.text(tuple(b[0]+[2,2]),f'person {pid}',fill=colors[pid])
            evidence=(anns[c]['occlusion_evidence_code'][target_index]==2)&common
            for xy in (anns[c]['xy'][target_index][evidence]-offset)*ratio+origin:
                x,y=xy;draw.ellipse((x-2,y-2,x+2,y+2),fill='#ffff00')
            row=recipe['selected_timeline'][k]
            draw.text((col*512+6,5),f'{c} frame={k} source={row["source_frame"]} tick={row["episode_time_tick"]}',fill='white')
            draw.text((col*512+6,544),f'person {target}: neighbor evidence {int(evidence.sum())}/{int(common.sum())} common GT',fill='#ffff00')
        draw.text((6,558),'Grey = unrendered layout. Yellow = conservative neighbor depth/ID evidence; no predictions.',fill='white')
        destination=root/'pack'/f'event_evidence_frame_{k:03d}.png'
        if destination.exists():raise FileExistsError(destination)
        canvas.save(destination);outputs.append(str(destination))
    (root/'pack/event_overlay_record.json').write_text(json.dumps(dict(outputs=outputs,seconds=time.monotonic()-start,additional_Dataset_reads=0,model_forwards=0),indent=2)+'\n')


if __name__=='__main__':main()
