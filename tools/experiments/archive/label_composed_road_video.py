#!/usr/bin/env python3
"""Add component and original-time captions to an existing ROI-only preview.

This is video-only encoding. It never renders a camera or reads Dataset items.
The input MP4 is retained and output is append-only.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from densekp_retrain.data.dynamic_scene import frame_record


def label_video(run):
    run=Path(run).resolve()
    recipe=json.loads((run/'recipe.json').read_text())
    composition=recipe['component_provenance']['selection']
    source=run/'pack/dual_roi.mp4'
    target=run/'pack/dual_roi_components.mp4'
    if target.exists():raise FileExistsError(target)
    base=Fraction(*recipe['time_base'])
    ids=' | '.join(f'{label}={composition[key]}' for label,key in
                 [('body','body_id'),('motion','motion_id'),('cloth','clothing_id'),('hair','hair_id')])
    expected=[]
    with av.open(str(source)) as video, av.open(str(target),'w') as output:
        stream=output.add_stream('libx264',rate=10)
        stream.width=1024;stream.height=576;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
        stream.time_base=base;stream.codec_context.time_base=base
        for k,frame in enumerate(video.decode(video=0)):
            row=frame_record(recipe,k)
            tick=row['episode_time_tick']
            if frame.pts is None or Fraction(frame.pts)*frame.time_base!=tick*base:
                raise ValueError('source_video_episode_time_conflict')
            person=row['person_records'][0]
            canvas=Image.new('RGB',(1024,576),(24,24,24))
            canvas.paste(Image.fromarray(frame.to_ndarray(format='rgb24')),(0,0))
            draw=ImageDraw.Draw(canvas)
            draw.text((8,548),f'{recipe["requested_view_class"]} {recipe["coverage_goal"]} | {ids}',fill='white')
            draw.text((8,562),f'episode={tick}/10s | source frame={person["source_frame"]} tick={person["source_time_tick"]} base={person["source_time_base"]} | grey=unrendered',fill='yellow')
            rendered=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
            rendered.pts=tick;rendered.time_base=base
            for packet in stream.encode(rendered):output.mux(packet)
            expected.append(tick*base)
        for packet in stream.encode():output.mux(packet)
    with av.open(str(target)) as video:
        observed=[Fraction(frame.pts)*frame.time_base for frame in video.decode(video=0)]
    if observed!=expected or len(observed)!=recipe['frame_count']:
        raise ValueError('labeled_video_pts_or_count_conflict')
    record=dict(status='passed',source_video=str(source),labeled_video=str(target),frames=len(observed),
                episode_pts_seconds=[float(x) for x in observed],last_frame_display_hold_seconds=float(base),
                component_ids=composition,model_forwards=0,blender_calls=0,dataset_reads=0)
    (run/'pack/labeled_video.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(label_video(parser.parse_args().run)))
