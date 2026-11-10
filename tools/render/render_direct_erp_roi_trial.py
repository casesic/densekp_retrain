"""Bounded A/B full-buffer and C independent region rendering on existing scene assembly."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.render import render_road_pass_sources as assembly
from densekp_retrain.rendering.blender.assembly import render_cameras

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def scene_settings(s):
    return dict(view_transform=s.view_settings.view_transform,look=s.view_settings.look,exposure=s.view_settings.exposure,gamma=s.view_settings.gamma,
        display_device=s.display_settings.display_device,view_use_curve_mapping=s.view_settings.use_curve_mapping,
        world=s.world.name if s.world else None,
        bounces={k:getattr(s.cycles,k) for k in ['max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces','volume_bounces','transparent_max_bounces']},
        png_color_mode=s.render.image_settings.color_mode,png_color_depth=s.render.image_settings.color_depth,png_compression=s.render.image_settings.compression,
        film_transparent=s.render.film_transparent,filter_type=s.cycles.pixel_filter_type,filter_width=s.cycles.filter_width)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['inspect','full','region','tiny'],required=True);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.root.resolve();start=time.monotonic()
    if args.stage=='inspect':
        scene=json.loads((ROOT/'configs/fisheye_scene_urban_street_assets.json').read_text());bpy.ops.wm.open_mainfile(filepath=str(ROOT/scene['blend_path']))
        # Apply precisely the same non-geometric settings as the old renderer.
        from densekp_retrain.rendering.blender.assembly import configure_render
        configure_render(bpy.context.scene,5760,2880,64)
        write(out/'scene_settings.json',dict(settings=scene_settings(bpy.context.scene),blender_version=bpy.app.version_string,elapsed_seconds=time.monotonic()-start,render_calls=0));return
    frozen=json.loads((out/'frozen_comparison.json').read_text());ledger=out/'render_calls.jsonl';entries=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    used=max(sum(e.get('seconds',0) for e in entries if e['event']=='completed'),
             sum(json.loads(p.read_text())['elapsed_seconds'] for p in out.glob('*_render_summary.json')))
    used+=json.loads((out/'scene_settings.json').read_text())['elapsed_seconds']
    job=out/f'{args.stage}_job.json'
    def render(s,cam,node,cameras_path,folder):
        assert scene_settings(s)==frozen['scene_settings'], 'Color/light/bounce settings changed'
        setup=time.monotonic()-start
        write(out/f'{args.stage}_setup.json',dict(seconds=setup,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,settings=scene_settings(s)))
        cameras=json.loads(cameras_path.read_text())['cameras'];records=[]
        tasks=[]
        if args.stage=='full':tasks=[('full',c,None,5760,2880,64) for c in cameras]
        elif args.stage=='tiny':tasks=[('tiny',cameras[0],frozen['tiny_preflight']['region_xyxy'],5760,2880,64)]
        else:
            gate=json.loads((out/'B_geometry_gate.json').read_text());assert gate['passed']
            assert json.loads((out/'tiny_gate.json').read_text())['passed']
            tasks=[('background',c,None,1440,720,8) for c in cameras]+[('roi',c,frozen['cameras'][c['id']]['roi_xyxy'],5760,2880,64) for c in cameras]
        for kind,camera,roi,w,h,samples in tasks:
            old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
            calls=sum(e['event']=='started' for e in old);elapsed=used+(time.monotonic()-start)
            if calls>=frozen['budget']['max_render_calls'] or elapsed>=frozen['budget']['render_seconds']:
                raise RuntimeError('Frozen total render budget exhausted')
            dest=out/'buffers'/kind/'_temporary_ds_source/source';dest.mkdir(parents=True,exist_ok=True)
            # Owned directories/markers are created by the external driver, never cleaned here.
            assert (dest.parent/'.densekp_ds_temporary_source.json').is_file()
            s.render.resolution_x=w;s.render.resolution_y=h;s.render.resolution_percentage=100;s.cycles.samples=samples
            s.render.use_border=roi is not None;s.render.use_crop_to_border=roi is not None
            if roi:
                x0,y0,x1,y1=roi
                # Integers are top-left/exclusive; Blender borders are bottom-left.
                s.render.border_min_x=(x0+.01)/w;s.render.border_max_x=(x1+.01)/w
                s.render.border_min_y=(h-y1+.01)/h;s.render.border_max_y=(h-y0+.01)/h
            if hasattr(node,'directory'):node.directory=str(dest)
            else:node.base_path=str(dest)
            camera_file=out/f'{kind}_{camera["id"]}_camera.json';write(camera_file,dict(cameras=[camera]))
            began=time.monotonic()
            with ledger.open('a') as f:f.write(json.dumps(dict(event='started',call=calls+1,stage=kind,camera=camera['id'],remaining_seconds=frozen['budget']['render_seconds']-elapsed,roi_xyxy=roi,full_grid=[w,h],samples=samples))+'\n')
            record=render_cameras(s,cam,node,camera_file,dest)[0]
            seconds=time.monotonic()-began
            with ledger.open('a') as f:f.write(json.dumps(dict(event='completed',call=calls+1,stage=kind,camera=camera['id'],seconds=seconds,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
            record.update(kind=kind,roi_xyxy=roi,full_grid=[w,h],samples=samples,seconds=seconds,crop_to_border=bool(roi),camera_matrix_blender=np.asarray(cam.matrix_world).tolist(),border_normalized=[s.render.border_min_x,s.render.border_min_y,s.render.border_max_x,s.render.border_max_y])
            write(out/f'{kind}_{camera["id"]}_render.json',record);records.append(record)
        write(out/f'{args.stage}_render_summary.json',dict(setup_seconds=setup,elapsed_seconds=time.monotonic()-start,records=records,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        return records
    sys.argv=[sys.argv[0],'--','--job',str(job)]
    assembly.main(render_cameras_fn=render)

if __name__=='__main__':main()
