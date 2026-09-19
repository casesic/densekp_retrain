"""Assemble renderer-native camera stress annotations and validate all samples."""
import json
import shutil
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/perspective_camera_stress'
NAMES=['baseline','near_70','near_90','near_110','edge_left','edge_right','trunc_left_25','trunc_bottom_25']

def exr_depth(path):
    f=OpenEXR.File(str(path)); found={}
    for part in f.parts:
        for name,channel in part.channels.items():found[name.split('.')[0]]=np.asarray(channel.pixels,np.float32)
    return found['Depth']

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def sheet(images,rows,cols,path,tw=128,th=128):
    canvas=Image.new('RGB',(cols*tw,rows*th),(20,20,20))
    for i,image in enumerate(images):canvas.paste(image.resize((tw,th),Image.Resampling.LANCZOS),((i%cols)*tw,(i//cols)*th))
    canvas.save(path)

def main():
    for folder in ['rgb','depth','masks','overlay','annotations','manifests']:(OUT/folder).mkdir(exist_ok=True)
    all_rows=[];stats={};projection_max=0.;ray_errors=[]
    body_hashes=set();clothing_hashes=set();by_name={}
    geometry=ROOT/'outputs/p0_dressed_eye_background_fixed/geometry'
    densekp3d=np.load(geometry/'densekp_3d.npy')
    for config_id,name in enumerate(NAMES):
        src=OUT/'_renders'/name
        rows=[json.loads(x) for x in (src/'manifest.jsonl').read_text().splitlines()]
        report=json.loads((src/'p0_render_report.json').read_text());output_report=json.loads((src/'p0_output_report.json').read_text())
        xy=np.load(src/'annotations/densekp_2d.npy');xyz=np.load(src/'annotations/densekp_camera_xyz.npy')
        blender_xy=np.load(src/'annotations/blender_projection_2d.npy')
        state=np.load(src/'annotations/densekp_visible.npy')
        valid=np.isfinite(xy).all(2)&np.isfinite(xyz).all(2)&(xyz[:,:,2]<0)
        in_frame=valid&(xy[:,:,0]>=0)&(xy[:,:,0]<512)&(xy[:,:,1]>=0)&(xy[:,:,1]<512)
        visible=(state==0)&in_frame
        depth=-xyz[:,:,2]
        assert valid.all() and np.array_equal(in_frame,state!=2) and not np.any(visible&~in_frame)
        annotation_dir=OUT/'annotations'/name;annotation_dir.mkdir(exist_ok=True)
        for filename,array in [('densekp_3d.npy',densekp3d),('densekp_2d.npy',xy),('densekp_depth.npy',depth),
                               ('densekp_valid.npy',valid.astype(np.uint8)),('densekp_in_frame.npy',in_frame.astype(np.uint8)),
                               ('densekp_visible.npy',visible.astype(np.uint8)),('densekp_visibility_state.npy',state)]:np.save(annotation_dir/filename,array)
        body_hashes.add(report['geometry_fingerprint']['before']);clothing_hashes.add(report['clothing_geometry_fingerprint']['before'])
        projection_max=max(projection_max,report['projection_error_px']['max'])
        config_rows=[]
        for frame,row in enumerate(rows):
            p=row['placement_solver'];bbox=p['full_body_bbox']
            sample=f'{name}_{frame:06d}'
            rgb=OUT/'rgb'/name/f'{frame:06d}.png';dep=OUT/'depth'/name/f'{frame:06d}.npy';mask=OUT/'masks'/name/f'{frame:06d}.png';overlay=OUT/'overlay'/name/f'{frame:06d}.png'
            copy(src/'rgb'/f'{frame:06d}.png',rgb);copy(src/'depth'/f'{frame:06d}.npy',dep);copy(src/'mask'/f'{frame:06d}.png',mask)
            image=Image.open(rgb).convert('RGB');draw=ImageDraw.Draw(image)
            for point,is_in,is_vis in zip(xy[frame],in_frame[frame],visible[frame]):
                if not is_in:continue
                u,v=point;color=(30,240,70) if is_vis else (255,65,45);draw.ellipse((u-2,v-2,u+2,v+2),fill=color)
            draw.rectangle((4,4,260,44),fill=(0,0,0));draw.text((8,7),f'{name} frame={frame}',fill='white')
            draw.text((8,23),f'visible={visible[frame].sum()} occluded={(in_frame[frame]&~visible[frame]).sum()} out={138-in_frame[frame].sum()}',fill='white')
            overlay.parent.mkdir(parents=True,exist_ok=True);image.save(overlay)
            record={'sample_id':sample,'config_id':f'C{config_id}','config_name':name,'source_motion_frame':row['source_frame_30fps'],
                    'camera_intrinsics':row['intrinsics'],'camera_extrinsics':np.linalg.inv(row['camera_matrix_world']).tolist(),
                    'body_world_transform':row['object_matrix_world'],'full_body_bbox':bbox,'in_frame_body_bbox':p['in_frame_body_bbox'],
                    **{k:p[k] for k in ['bbox_height_fraction','bbox_width_fraction','bbox_visible_fraction_x','bbox_visible_fraction_y',
                       'left_truncation_fraction','right_truncation_fraction','top_truncation_fraction','bottom_truncation_fraction',
                       'camera_near_clip','min_body_camera_depth','min_clothing_camera_depth']},
                    'densekp_valid_count':int(valid[frame].sum()),'densekp_in_frame_count':int(in_frame[frame].sum()),
                    'densekp_visible_count':int(visible[frame].sum()),'rgb_path':str(rgb.relative_to(OUT)),
                    'depth_path':str(dep.relative_to(OUT)),'mask_path':str(mask.relative_to(OUT)),'overlay_path':str(overlay.relative_to(OUT)),
                    'projection_mean_error_px':report['projection_error_px']['mean'],'projection_p95_error_px':report['projection_error_px']['p95'],
                    'projection_max_error_px':report['projection_error_px']['max']}
            config_rows.append(record);all_rows.append(record)
        (OUT/'manifests'/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in config_rows))
        by_name[name]=config_rows
        # Compare compositor depth against independent scene rays at stable 3x3 regions.
        checks=json.loads((src/'ray_depth_checks.json').read_text())
        depths={frame:exr_depth(src/'depth_raw'/f'{frame:06d}.exr') for frame in range(16)}
        for q in checks:
            if not q['hit']:continue
            z=depths[q['frame']];y,x=q['y'],q['x'];patch=z[y-1:y+2,x-1:x+2]
            patch=patch[np.isfinite(patch)]
            if len(patch)==9 and np.ptp(patch)<.002:ray_errors.append(abs(float(np.median(patch))-q['depth']))
        stats[name]={'samples':16,'mean_body_bbox_height_ratio':float(np.mean([x['bbox_height_fraction'] for x in config_rows])),
                     'mean_densekp_in_frame_count':float(in_frame.sum(1).mean()),'min_densekp_in_frame_count':int(in_frame.sum(1).min()),
                     'max_densekp_in_frame_count':int(in_frame.sum(1).max()),'mean_densekp_visible_count':float(visible.sum(1).mean()),
                     'mean_out_of_frame_densekp_count':float((138-in_frame.sum(1)).mean()),
                     'projection_error_px':report['projection_error_px'],'rgb_complete':output_report['rgb_complete'],
                     'depth_complete':output_report['depth_complete'],'mask_complete':output_report['mask_complete']}
    assert len(body_hashes)==len(clothing_hashes)==1
    (OUT/'manifests/all_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_rows))
    # Actual target checks over every frame.
    acceptance={
      'baseline':all(x['bbox_visible_fraction_x']==1 and x['bbox_visible_fraction_y']==1 for x in by_name['baseline']),
      'near_70':all(abs(x['bbox_height_fraction']-.70)<=.03 and x['bbox_visible_fraction_y']==1 for x in by_name['near_70']),
      'near_90':all(abs(x['bbox_height_fraction']-.90)<=.03 and x['bbox_visible_fraction_y']==1 for x in by_name['near_90']),
      'near_110':all(1.05<=x['bbox_height_fraction']<=1.15 and x['top_truncation_fraction']>0 and x['bottom_truncation_fraction']>0 for x in by_name['near_110']),
      'edge_left':all(.02<=x['full_body_bbox'][0]/512<=.05 and x['bbox_visible_fraction_x']>=.95 for x in by_name['edge_left']),
      'edge_right':all(.02<=(512-x['full_body_bbox'][2])/512<=.05 and x['bbox_visible_fraction_x']>=.95 for x in by_name['edge_right']),
      'trunc_left_25':all(.20<=x['left_truncation_fraction']<=.30 for x in by_name['trunc_left_25']),
      'trunc_bottom_25':all(.20<=x['bottom_truncation_fraction']<=.30 for x in by_name['trunc_bottom_25'])}
    safety=all(x['min_body_camera_depth']>x['camera_near_clip']+.1 and x['min_clothing_camera_depth']>x['camera_near_clip']+.1 for x in all_rows)
    statistics={'configs':stats,'total_samples':len(all_rows),'acceptance':acceptance,'body_local_geometry_hashes':list(body_hashes),
                'clothing_local_geometry_hashes':list(clothing_hashes),'projection_max_error_px':projection_max,
                'independent_scene_ray_depth_check':{'samples':len(ray_errors),'mean_error_m':float(np.mean(ray_errors)),
                  'p95_error_m':float(np.percentile(ray_errors,95)),'max_error_m':float(np.max(ray_errors))},
                'near_clip_safety_pass':safety,'pass':all(acceptance.values()) and safety and projection_max<.5 and len(all_rows)==128}
    (OUT/'stress_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    frames=[[Image.open(OUT/'overlay'/name/f'{f:06d}.png').convert('RGB') for name in NAMES] for f in range(16)]
    sheet([x for row in frames for x in row],16,8,OUT/'contact_sheet_by_config.png')
    sheet([frames[f][c] for c in range(8) for f in range(16)],8,16,OUT/'contact_sheet_by_frame.png')
    print(json.dumps(statistics,indent=2))
    if not statistics['pass']:raise SystemExit(1)

if __name__=='__main__':main()
