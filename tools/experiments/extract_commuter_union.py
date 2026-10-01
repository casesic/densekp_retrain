"""Extract a 3 mm outward envelope; no anatomical or pose edits."""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt,gaussian_filter
from skimage.measure import marching_cubes

p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
p.add_argument('--sigma',type=float,default=.5);p.add_argument('--output',type=Path)
a=p.parse_args();start=time.monotonic();out=a.output or a.folder
if a.output:
    out.mkdir(parents=True,exist_ok=False)
    info=json.loads((a.folder/'volume.json').read_text());info['volume_path']=str(a.folder/'volume.npz')
    (out/'volume.json').write_text(json.dumps(info,indent=2)+'\n')
z=np.load(a.folder/'volume.npz');inside=z['occupancy'];step=float(z['step'])
signed=(distance_transform_edt(~inside)-distance_transform_edt(inside))*step
signed=gaussian_filter(signed,a.sigma)
vertices,faces,_,_=marching_cubes(signed,.003,spacing=(step,)*3,gradient_direction='ascent')
vertices+=z['origin'];np.savez_compressed(out/'closed_envelope.npz',vertices=vertices,faces=faces)
(out/'extraction.json').write_text(json.dumps(dict(outward_isovalue_m=.003,gaussian_sigma_voxels=a.sigma,
    seconds=time.monotonic()-start,vertices=len(vertices),faces=len(faces)),indent=2)+'\n')
