"""Build honest native-pixel crops and before/after comparison artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(); p.add_argument('--after',type=Path,required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]
before=root/'outputs/p0_dressed_hdri_eye_fixed'; eye=root/'outputs/eye_fix_v2'; bg=root/'outputs/background_fix'
bg.mkdir(exist_ok=True)
im0=Image.open(before/'rgb/000000.png').convert('RGB'); im1=Image.open(a.after/'rgb/000000.png').convert('RGB')
def pair(left,right,path,labels=('Before','After')):
    sheet=Image.new('RGB',(left.width+right.width,max(left.height,right.height)+28),(22,22,22))
    sheet.paste(left,(0,28));sheet.paste(right,(left.width,28))
    d=ImageDraw.Draw(sheet);d.text((8,8),labels[0],fill='white');d.text((left.width+8,8),labels[1],fill='white');sheet.save(path)
box=(231,129,279,167)
for image,name in [(im0,'eye_before.png'),(im1,'eye_after.png')]:
    image.crop(box).resize((576,456),Image.Resampling.NEAREST).save(eye/name)
pair(Image.open(eye/'eye_before.png'),Image.open(eye/'eye_after.png'),eye/'eye_compare.png',('Before: duplicated albedo eyes + neutral eyeballs','After: albedo correction + procedural iris/pupil'))
im0.save(bg/'fullbody_before.png');im1.save(bg/'fullbody_after.png')
pair(im0,im1,bg/'background_compare.png',('Before: visible ground','After: HDRI + differential shadow'))
shutil.copyfile(a.after/'appearance_raw/000000_shadow.png',bg/'shadow_debug.png')
r=json.loads((a.after/'manifest.jsonl').read_text().splitlines()[0])
v=np.load(a.after/'geometry/vertices.npy')[0]
mapping=np.load(eye/'procedural_eye_coordinates.npz'); coords=mapping['coordinates']
labels=np.load(root/'outputs/body_texture_audit/body_uv.npz')['component_labels']
m=np.linalg.inv(r['camera_matrix_world'])@np.asarray(r['object_matrix_world'])
q=np.c_[v,np.ones(len(v))]@m.T;k=r['intrinsics']
xy=np.c_[k['fx']*q[:,0]/-q[:,2]+k['cx'],k['cy']-k['fy']*q[:,1]/-q[:,2]]
debug=im1.crop((225,118,285,185)).resize((720,804),Image.Resampling.NEAREST);d=ImageDraw.Draw(debug)
for c,color in [(1,'lime'),(2,'cyan')]:
    ids=np.where(labels==c)[0]; front=ids[coords[ids,2]>0]
    radial=np.linalg.norm(coords[front,:2],axis=1)
    pole=front[radial.argmin()]; center=xy[ids].mean(0); target=xy[pole]
    local=lambda p:((p[0]-225)*12,(p[1]-118)*12)
    x,y=local(center);d.ellipse((x-7,y-7,x+7,y+7),outline=color,width=2)
    d.line((local(center),local(target)),fill=color,width=2)
    x,y=local(target);d.line((x-9,y,x+9,y),fill=color,width=2);d.line((x,y-9,x,y+9),fill=color,width=2)
d.text((12,12),'Circle: mesh center. Cross: native +Z iris/pupil pole.',fill='white',stroke_fill='black',stroke_width=1)
debug.save(eye/'eye_debug.png')
diagnostic=eye/'preview_v2/rgb/000000.png'
if diagnostic.exists():
    Image.open(diagnostic).crop((225*4,118*4,285*4,185*4)).resize((720,804),Image.Resampling.NEAREST).save(eye/'diagnostic_face.png')
