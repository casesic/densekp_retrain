"""Small integration-only DS layouts; numerical pass never implies appearance pass."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.prepare_cycling_characters import POSES
from densekp_retrain.io.json import write_json
from tools.experiments.render_cycling_ds import layout
from tools.fisheye.run_ds_scene_pair_pilot import SCENES
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.imaging.output_modes import (prepare_source_workspace,
    preserve_source_recipe, complete_output, record_failure)
from densekp_retrain.camera.double_sphere import DS_20260320
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.body.transforms import world_points
from tools.validate.validate_ds_scene_pair_pilot import validate
from densekp_retrain.camera.double_sphere import build_erp_to_fisheye_lut
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg

def crops(out):
    folder=out/'actual_inputs';folder.mkdir();panels=[]
    dataset = DatasetKeypoints(
        audit_cfg("supervision_weight"),
        "adult-diagnostic",
        is_train=False,
        use_augmentation=False,
        annotation_path=str(out / "diagnostic_loader.npz"),
        image_root=str(out),
    )
    z = np.load(out / 'diagnostic_loader.npz')
    for i in range(len(z['person_id'])):
        item = dataset[i]
        rgb = cv2.cvtColor(item['img_disp'], cv2.COLOR_BGR2RGB)
        name = f"{z['camera_id'][i]}_person_{z['person_id'][i]}"
        Image.fromarray(rgb).save(folder / (name + '_square.png'))
        Image.fromarray(rgb[:, 32:-32]).save(folder / (name + '_center.png'))
        np.save(folder / (name + '_input.npy'), item['img'][:, :, 32:-32])
        panel = Image.new('RGB', (448, 280), (30, 30, 30))
        panel.paste(Image.fromarray(rgb), (0, 0))
        panel.paste(Image.fromarray(rgb[:, 32:-32]), (256, 0))
        ImageDraw.Draw(panel).text((5, 260), name, fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(896,280*((len(panels)+1)//2)),(30,30,30))
    for i,panel in enumerate(panels):sheet.paste(panel,(i%2*448,i//2*280))
    sheet.save(out/'actual_inputs.jpg')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--characters',nargs='+',required=True,help='2 to 4 versioned asset IDs')
    p.add_argument('--pose-indices',nargs='+',type=int,help='0=P1, 1=P2; one per character')
    p.add_argument('--pose-source-ids',nargs='+',help='Explicit saved pose IDs, e.g. reviewed pedestrian frames')
    p.add_argument('--pedestrian-layout',action='store_true',help='Place pedestrian soles on native ground at z=0; keep riding height')
    p.add_argument('--control-distance',type=float,help='Optional distance of complete-control slot along scene road')
    p.add_argument('--layout-config',type=Path,help='Explicit source-isolated placement variation; omitted preserves old pilots')
    p.add_argument('--reuse-layout',type=Path,help='Use saved target placements/cameras for an occlusion control')
    p.add_argument('--reuse-rendered-source',type=Path,help='Completed source directory from the exact reused layout; postprocess only into a new output')
    p.add_argument('--layout-slots',nargs='+',type=int,help='0/1 paired edge, 2 rear overlap, 3 full control')
    p.add_argument('--scene',choices=list(SCENES),default='cycling_forest')
    p.add_argument('--samples',type=int,default=24);p.add_argument('--width',type=int,default=5760)
    p.add_argument('--final-width',type=int,default=5760);p.add_argument('--final-height',type=int,default=2880)
    p.add_argument('--output-mode',choices=['compact','full'],default='compact')
    p.add_argument('--temporary-root',type=Path,help='Fresh compact-owned temporary directory')
    p.add_argument('--device',choices=['CPU','CUDA'],default='CUDA');a=p.parse_args();start=time.monotonic()
    if not 2<=len(a.characters)<=4:raise ValueError('Bounded layout requires 2..4 people')
    assets=json.loads((a.assets/'report.json').read_text());selected=[]
    indices=a.pose_indices or [0]*len(a.characters)
    if len(indices)!=len(a.characters):raise ValueError('Pose count mismatch')
    pose_ids=a.pose_source_ids or [POSES[index] for index in indices]
    if len(pose_ids)!=len(a.characters):raise ValueError('Pose ID count mismatch')
    for cid,pose_id in zip(a.characters,pose_ids):
        selected.append(next(d for d in assets['derivatives'] if d['character_id']==cid and d['pose_source_id']==pose_id))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source_dir=prepare_source_workspace(out,a.output_mode,a.temporary_root)
    failure_stage={'value':'preflight'}
    original_hook=sys.excepthook
    def preserve_failure(error_type,error,traceback):
        record_failure(out,a.output_mode,source_dir,failure_stage['value'],error)
        original_hook(error_type,error,traceback)
    sys.excepthook=preserve_failure
    old=ROOT/'outputs/experiments/cycling_four_characters_20260929/ds_v1/cycling_forest'
    lenses={n:np.array(Image.open(old/n/'reference_lens_mask.png'))>0 for n in ('cam2','cam4')}
    old_report=json.loads((old/'report.json').read_text())
    metadata={row['camera']:dict(row['lens_mask'],reused_explicit_mask=str(old/row['camera']/'reference_lens_mask.png')) for row in old_report['cameras']}
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as z:masks={n:z[f'valid_masks_{n}'] for n in lenses}
    layout_masks={}
    for n in lenses:
        x,y=build_erp_to_fisheye_lut(DS_20260320[n],n,a.width,a.width//2)
        layout_masks[n]=cv2.resize(masks[n].astype(np.uint8),(a.width,a.width//2),interpolation=cv2.INTER_NEAREST).astype(bool)
        layout_masks[n]&=cv2.remap(lenses[n].astype(np.uint8),x,y,cv2.INTER_NEAREST).astype(bool)
    config,origin,yaw_deg=SCENES[a.scene]
    variation=json.loads(a.layout_config.read_text()) if a.layout_config else {}
    if variation:
        angle=np.deg2rad(yaw_deg)
        road=np.array([-np.sin(angle),np.cos(angle),0.])
        origin=(np.asarray(origin)+road*variation.get('road_offset_m',0.)).tolist()
    poses,physical,yaw=rig_poses(origin,yaw_deg,.18)
    slots=a.layout_slots or list(range(len(selected)))
    if len(slots)!=len(selected) or len(set(slots))!=len(slots) or any(s not in range(4) for s in slots):
        raise ValueError('Invalid bounded layout slots')
    if a.reuse_layout:
        saved=json.loads((a.reuse_layout/'people.json').read_text())['people']
        people=[dict(next(p for p in saved if p['character_id']==d['character_id'] and p['pose_source_id']==d['pose_source_id'])) for d in selected]
        original_rig=json.loads((a.reuse_layout/'rig.json').read_text())
        poses={c['id']:np.array(c['camera_to_world']) for c in original_rig['cameras']}
        physical={c['id']:np.array(c['physical_ds_camera_to_world']) for c in original_rig['cameras']}
    else:
        base_layout=layout([Path(d['geometry']) for d in selected[:2]],np.array(origin),yaw_deg,yaw,poses,layout_masks)
        people=[base_layout[s] for s in slots]
    for person,row in zip(people,selected):
        person.update({k:row[k] for k in ('geometry','clothing_geometry','groom_geometry','character_id','pose_source_id','appearance')})
        person.update(split='integration_only',quality_state='diagnostic_render',sample_id=f"{row['character_id']}__{row['pose_source_id']}")
        person['action']=row.get('action','cycling')
        for field in ('partition','source_group','identity_family_id'):
            if field in row:person[field]=row[field]
        if variation and not a.reuse_layout:
            relative=np.asarray(person['location'])-origin
            if person['category']=='edge_partial':
                relative[:2]*=variation.get('edge_distance_scale',1.)
            elif person['category']=='rear_overlap':
                relative[:2]*=variation.get('rear_distance_scale',1.)
            person['location']=(np.asarray(origin)+relative).tolist()
            person['yaw_deg']+=variation.get('person_yaw_delta_deg',0.)
        if a.control_distance is not None and person['category']=='complete_control' and not a.reuse_layout:
            person['location']=(np.asarray(origin)+yaw@np.array([0.,a.control_distance,person['location'][2]-origin[2]])).tolist()
        if a.pedestrian_layout and person['action']!='cycling' and not a.reuse_layout:
            ground_offset=variation.get('pedestrian_ground_offsets_m',{}).get(str(person['id']),0.)
            person['location'][2]=float(origin[2])+float(ground_offset)
            if ground_offset:person['native_ground_placement_offset_m']=float(ground_offset)
            person['ground_contact_check']=True
        v=np.load(Path(row['geometry'])/'vertices.npy')[0];r=np.load(ROOT/'assets/densekp_regressor_R.npy')
        if np.abs(r@v-np.load(Path(row['geometry'])/'densekp_3d.npy')[0]).max()>1e-6:raise ValueError('R gate failed before render')
    rig=dict(calibration_group='20260320',baseline_m=.18,baseline_status='synthetic, not measured',split='integration_only',
        intrinsics={n:asdict(DS_20260320[n]) for n in poses},cameras=[dict(id=n,camera_to_world=poses[n].tolist(),physical_ds_camera_to_world=physical[n].tolist()) for n in poses])
    write_json(out/'rig.json',rig);write_json(out/'people.json',dict(people=people,split='integration_only',quality_state='diagnostic_render'))
    bounds=[]
    for person in people:
        body=np.load(Path(person['geometry'])/'vertices.npy')[person['frame']]
        points=world_points(body,body,person)
        if person.get('clothing_geometry'):
            cloth=np.load(Path(person['clothing_geometry'])/'clothing_vertices.npy')[person['frame']]
            points=np.concatenate((points,world_points(cloth,body,person)))
        bounds.append((points.min(0),points.max(0)))
    spacing_failures=[]
    for i,(low,high) in enumerate(bounds):
        for j,(other_low,other_high) in enumerate(bounds[i+1:],start=i+1):
            if not ((high+.05<other_low)|(other_high+.05<low)).any():
                spacing_failures.append([people[i]['id'],people[j]['id']])
    write_json(out/'spacing_preflight.json',dict(status='failed' if spacing_failures else 'passed',
        failed_person_pairs=spacing_failures,gate='Existing body+clothing AABBs separated with 5cm margin; actual rendered geometry still validated afterward'))
    if spacing_failures:raise ValueError('Body AABB overlap or insufficient clearance before render')
    scene=json.loads((ROOT/config).read_text())
    command=['/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender','-b','-t','8','--disable-autoexec','--python-exit-code','1',
        '--python',str(ROOT/'tools/fisheye/render_equirectangular_source.py'),'--','--geometry',selected[0]['geometry'],
        '--people-config',str(out/'people.json'),'--cameras',str(out/'rig.json'),'--scene-blend',str(ROOT/scene['blend_path']),
        '--output-dir',str(source_dir),'--width',str(a.width),'--height',str(a.width//2),'--samples',str(a.samples),'--seed','29','--device',a.device]
    write_json(out/'render_command.json',command)
    if a.reuse_rendered_source:
        source=a.reuse_rendered_source.resolve()
        if not a.reuse_layout or source!=a.reuse_layout.resolve()/'source':
            raise ValueError('Source reuse requires its exact parent layout')
        saved_people=json.loads((a.reuse_layout/'people.json').read_text())['people']
        if people!=saved_people:raise ValueError('Cannot reuse render after changing people')
        if rig['cameras']!=json.loads((a.reuse_layout/'rig.json').read_text())['cameras']:
            raise ValueError('Cannot reuse render after changing cameras')
        for name in ['source_manifest.json','cam2.exr','cam4.exr']:
            if not (source/name).is_file():raise ValueError('Source render is incomplete')
        if a.output_mode == 'compact':
            for path in source.iterdir():
                target=source_dir/path.name
                shutil.copytree(path,target) if path.is_dir() else shutil.copy2(path,target)
        else:
            source_dir.rmdir();source_dir.symlink_to(source,target_is_directory=True)
        write_json(out/'render_reuse.json',dict(source=str(source),render_executed=False,
            reason='Complete existing render; interrupted downstream processing resumed in a new directory'))
    else:
        failure_stage['value']='render'
        with (out/'render.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    failure_stage['value']='postprocess'
    result=process_scene(out,people,poses,masks,None,None,lens_masks=lenses,lens_metadata=metadata,
        output_size=(a.final_width,a.final_height),output_mode=a.output_mode,source_dir=source_dir)
    result.update(scene=a.scene,people=people,split='integration_only',quality_state='diagnostic_render',
        appearance_acceptance='pending actual ERP/square/network-center review',seconds=time.monotonic()-start,
        source_resolution=[a.width,a.width//2],physical_fisheye_resolution=[2880,2880],
        final_erp_resolution=[a.final_width,a.final_height],output_mode=a.output_mode)
    if variation:result['layout_variation']=variation
    write_json(out/'report.json',result);failure_stage['value']='validation'
    checked=validate(out,source_dir=source_dir,output_mode=a.output_mode)
    hair_checks=[]
    for person in people:
        folder=Path(person['groom_geometry']);info=json.loads((folder/'groom_manifest.json').read_text())
        points=np.load(folder/'points.npy');offsets=np.load(folder/'curve_offsets.npy')
        active=np.load(folder/'active_curve_mask.npy') if (folder/'active_curve_mask.npy').exists() else np.ones(len(offsets)-1,bool)
        expected=world_points(points[np.repeat(active,np.diff(offsets))],np.load(Path(person['geometry'])/'vertices.npy')[0],person)
        actual=np.load(source_dir/f"person_{person['id']}_world_groom.npy")
        if actual.shape!=expected.shape:raise ValueError('Rendered groom point count mismatch')
        error=float(np.abs(expected-actual).max())
        if error>5e-6:raise ValueError('Groom world-transform gate failed')
        meta=json.loads((source_dir/f"person_{person['id']}_groom.json").read_text())
        if any(o['pass_index']!=person['id'] for o in meta['objects']):raise ValueError('Groom instance ID mismatch')
        hair_checks.append(dict(character_id=person['character_id'],world_max_m=error,actual_control_points=len(actual),instance_id=person['id']))
    if a.output_mode == 'full':crops(out)
    write_json(out/'groom_validation.json',dict(status='world_transform_and_instance_assignment_passed',cases=hair_checks,
        limitation='actual curve control points and opaque render assignment; final image contacts require separate review'))
    preserve_source_recipe(out,source_dir,render_command=command,
        source_buffers_retained=a.output_mode=='full',extra=dict(
        output_mode=a.output_mode,final_erp_resolution=[a.final_width,a.final_height],
        ds_resolution=[2880,2880],seed=29,samples=a.samples))
    failure_stage['value']='cleanup'
    complete_output(out,a.output_mode,source_dir,checks=checked)
    sys.excepthook=original_hook
    print(json.dumps(checked),flush=True)

if __name__=='__main__':main()
