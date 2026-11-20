"""Bounded world-placement repair using native terrain probes, never pose edits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.fisheye.run_ds_scene_pair_pilot import SCENES


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--layouts',nargs='+',required=True);a=p.parse_args();root=a.root.resolve()
    recipe=json.loads((root/'layout_recipe.json').read_text());records=[]
    for name in a.layouts:
        source=root/'layouts'/name;job=next(j for j in recipe['layouts'] if j['id']==name)
        original=json.loads((source/'people.json').read_text());people=deepcopy(original)
        folder=root/'ground_repairs'/name;folder.mkdir(parents=True,exist_ok=False)
        (folder/'rig.json').write_bytes((source/'rig.json').read_bytes())
        config,origin,yaw_deg=SCENES[job['scene']]
        theta=np.deg2rad(yaw_deg);origin=np.array(origin)+np.array([-np.sin(theta),np.cos(theta),0])*job['variation']['road_offset_m']
        blend=ROOT/json.loads((ROOT/config).read_text())['blend_path'];passed=False;probes=[]
        for iteration in range(3):
            write(folder/'people.json',people)
            write(folder/f'people_probe_{iteration}.json',people)
            output=folder/f'probe_{iteration}'
            command=['/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender','-b','-t','2','--disable-autoexec','--python-exit-code','1',
                '--python',str(ROOT/'tools/fisheye/render_equirectangular_source.py'),'--','--geometry',people['people'][0]['geometry'],
                '--people-config',str(folder/'people.json'),'--cameras',str(folder/'rig.json'),'--scene-blend',str(blend),
                '--output-dir',str(output),'--check-ground-only','--device','CPU']
            with (folder/f'probe_{iteration}.log').open('w') as log:done=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:break
            result=json.loads((output/'ground_contact.json').read_text());probes.append(result)
            if result['status']=='passed':passed=True;break
            for case in result['cases']:
                lo,hi=case['minimum_gap_m'],case['maximum_gap_m']
                if lo>=-.01 and hi<=.03:continue
                person=next(v for v in people['people'] if v['id']==case['person_id'])
                if hi-lo<=.04:
                    correction=.01-(lo+hi)/2
                    if abs(correction)>.15:continue
                    person['location'][2]+=correction
                elif person['category']=='rear_overlap' and iteration==0:
                    rider=next((v for v in people['people'] if v['action']=='cycling'),None)
                    if rider is not None:
                        # The sloped shoulder may not support a flat-foot walking
                        # pose. Exchange complete world placements with a rider;
                        # preserve both poses and report the changed layout roles.
                        for key in ['location','yaw_deg','category']:
                            person[key],rider[key]=deepcopy(rider[key]),deepcopy(person[key])
                        person['location'][2]=float(origin[2])
                        rider['location'][2]=float(origin[2])+.25
        deltas=[]
        for before,after in zip(original['people'],people['people']):
            deltas.append(dict(person_id=before['id'],before=before['location'],after=after['location'],
                world_translation_delta_m=(np.array(after['location'])-before['location']).tolist(),
                before_category=before['category'],after_category=after['category'],
                before_yaw_deg=before['yaw_deg'],after_yaw_deg=after['yaw_deg']))
        result=dict(id=name+'_ground_v2',parent_layout=name,partition=job['partition'],scene=job['scene'],
            status='ground_probe_passed_render_pending' if passed else 'not_applicable_native_ground',
            reuse_layout=str(folder),characters=job['characters'],pose_source_ids=job['pose_source_ids'],
            deltas=deltas,probes=probes,policy='Only whole-body world placement changed; geometry, pose, shape, R, attachments and numerical gates unchanged')
        write(folder/'repair.json',result);records.append(result);print(name,result['status'],flush=True)

if __name__=='__main__':main()
