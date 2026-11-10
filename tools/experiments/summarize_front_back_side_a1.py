"""Small SVG/JSON review plots for a frozen five-person, three-rig scene."""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.data.viewpoint import classify_view, view_angle

COLORS={'rig_back':'#4fc3f7','rig_front':'#ffcf67','rig_side':'#f693bf'}
PEOPLE=['#d9f2ff','#e3b75f','#73daae','#cc91fa','#f58e82']


def read(path):
    return json.loads(Path(path).read_text())


def summarize(root):
    root=Path(root).resolve()
    scene=read(root/'production/recipe.json')
    mats=np.load(root/'prepared_back/scene_matrices.npy')
    rigs={rid:read(root/'production/rigs'/rid/'rig.json') for rid in COLORS}
    people=[]
    for person in scene['people']:
        asset=read(person['asset_manifest'])
        with np.load(asset['parameters']) as z:
            orient=z['global_orient'][person['asset_frames']]
        people.append(orient)
    angles=[]
    for k in range(31):
        row=[]
        for j in range(5):
            forward=mats[k,j,:3,:3] @ cv2.Rodrigues(people[j][k])[0] @ [0.,0.,1.]
            per_rig={}
            for rid,rig in rigs.items():
                per_camera={}
                for camera in rig['cameras']:
                    center=np.asarray(camera['camera_to_world'])[:3,3]
                    angle=view_angle(mats[k,j,:3,3],forward,center)
                    per_camera[camera['id']]=dict(angle_deg=angle,class_name=classify_view(angle))
                per_rig[rid]=per_camera
            row.append(dict(person_id=j+1,rigs=per_rig))
        angles.append(dict(frame_index=k,time_tick=k,time_base=[1,10],people=row))
    (root/'view_angles_by_person.json').write_text(json.dumps(dict(schema='five_person_three_rig_angles_v1',
        definition='horizontal body forward from source global_orient composed once with scene matrix to each camera center',
        frames=angles),indent=2,allow_nan=False)+'\n')

    # An explanatory top-down schematic in world metres; no RGB background.
    X=lambda x:100+(x+4.5)*80
    Y=lambda y:960-(y+1.0)*80
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1050" viewBox="0 0 900 1050">',
       '<rect width="900" height="1050" fill="#171d25"/>',
       f'<rect x="{X(-2.5):.1f}" y="60" width="{X(2.5)-X(-2.5):.1f}" height="900" fill="#37414c"/>',
       f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="60" y2="960" stroke="#e0e7ef" stroke-dasharray="12 12" opacity=".5"/>',
       '<text x="35" y="36" fill="#f0f4f8" font-size="20">City five riders: fixed trajectories and three static paired rigs</text>',
       '<text x="35" y="1020" fill="#c1ccd9" font-size="14">Schematic world XY (metres); paired camera centers marked separately. No stitched seam is implied.</text>']
    for j in range(5):
        coords=' '.join(f'{X(mats[k,j,0,3]):.1f},{Y(mats[k,j,1,3]):.1f}' for k in range(31))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{PEOPLE[j]}" stroke-width="4"/>')
        lines.append(f'<circle cx="{X(mats[0,j,0,3]):.1f}" cy="{Y(mats[0,j,1,3]):.1f}" r="5" fill="{PEOPLE[j]}"/>')
        lines.append(f'<text x="{X(mats[30,j,0,3])+7:.1f}" y="{Y(mats[30,j,1,3])-8:.1f}" fill="{PEOPLE[j]}" font-size="16">p{j+1}</text>')
    for rid,rig in rigs.items():
        for camera in rig['cameras']:
            center=np.asarray(camera['camera_to_world'])[:3,3]
            x,y=X(center[0]),Y(center[1])
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLORS[rid]}" stroke="#101820" stroke-width="2"/>')
            lines.append(f'<text x="{x+10:.1f}" y="{y-9:.1f}" fill="{COLORS[rid]}" font-size="14">{rid}/{camera["id"]}</text>')
    lines.append('</svg>')
    (root/'topdown_rigs.svg').write_text('\n'.join(lines)+'\n')

    # Person 1 angle curves; the exact angular bins remain the source of truth.
    px=lambda k:75+k*32
    py=lambda angle:455-angle*2.2
    plot=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="550" viewBox="0 0 1100 550">',
          '<rect width="1100" height="550" fill="#171d25"/>',
          '<text x="28" y="26" fill="#eef4fa" font-size="19">Person 1 viewing angle, body forward to each camera center</text>']
    for low,high,color,name in ((0,45,'#233d39','front'),(80,100,'#3d3543','side'),(135,180,'#2a3d54','back')):
        plot.append(f'<rect x="75" y="{py(high):.1f}" width="960" height="{py(low)-py(high):.1f}" fill="{color}"/>')
        plot.append(f'<text x="1040" y="{py((low+high)/2):.1f}" fill="#d9e3ec" font-size="12">{name}</text>')
    for angle in (0,45,80,100,135,180):
        plot.append(f'<line x1="75" x2="1035" y1="{py(angle):.1f}" y2="{py(angle):.1f}" stroke="#6b7785" opacity=".5"/>')
        plot.append(f'<text x="38" y="{py(angle)+4:.1f}" fill="#d5e0eb" font-size="12">{angle}°</text>')
    for rid,color in COLORS.items():
        for camera,dash in (('cam2',''),('cam4','stroke-dasharray="7 5"')):
            coords=' '.join(f'{px(k):.1f},{py(angles[k]["people"][0]["rigs"][rid][camera]["angle_deg"]):.1f}' for k in range(31))
            plot.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>')
    plot.append('<text x="75" y="505" fill="#d5e0eb" font-size="14">0s</text><text x="1000" y="505" fill="#d5e0eb" font-size="14">3s</text>')
    plot.append('<text x="75" y="535" fill="#4fc3f7" font-size="14">back</text><text x="170" y="535" fill="#ffcf67" font-size="14">front</text><text x="265" y="535" fill="#f693bf" font-size="14">side</text><text x="355" y="535" fill="#d5e0eb" font-size="14">solid cam2 · dashed cam4</text></svg>')
    (root/'target_view_angles.svg').write_text('\n'.join(plot)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root)
