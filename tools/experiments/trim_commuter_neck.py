"""Trim the excess neck tube after checking actual faired shoulder height."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.clothing.binding import clip_surface

p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
p.add_argument('--height',type=float,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
v=np.load(a.source/'clothing_rest_vertices.npy');f=np.load(a.source/'clothing_faces.npy')
shoulder_max=float(v[np.abs(v[:,0])>.10,1].max())
if a.height<shoulder_max+.003:raise ValueError('Neck plane would remove shoulder caps')
idx,w,faces=clip_surface(v,f,[(1,-1,a.height)]);vertices=(v[idx]*w[:,:,None]).sum(1)
np.save(a.output/'clothing_rest_vertices.npy',vertices);np.save(a.output/'clothing_faces.npy',faces)
(a.output/'pattern.json').write_text(json.dumps(dict(split='integration_only',parent=str(a.source),
    operation='remove neck tube above checked plane, without moving vertices or shoulder caps',
    plane_y_m=a.height,shoulder_max_y_m=shoulder_max,vertices=len(vertices),faces=len(faces)),indent=2)+'\n')
