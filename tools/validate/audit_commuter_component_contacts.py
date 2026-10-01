"""Check proposed collar against the corresponding revised shirt, independently."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.validate.commuter_intersections import crossings

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);rows=[]
for pose in ['P1','P2']:
    collar=a.root/'M02_collar_binding'/pose;shirt=a.root/'M02_top_trimmed'/pose
    cv=np.load(collar/'clothing_vertices.npy')[0];cf=np.load(collar/'clothing_faces.npy')
    tv=np.load(shirt/'clothing_vertices.npy')[0];tf=np.load(shirt/'clothing_faces.npy')
    pairs,locations,_=crossings(cv,cf,tv,tf)
    np.savez_compressed(collar/'shirt_contact.npz',pairs=pairs,locations=locations)
    rows.append(dict(pose=pose,collar=str(collar),shirt=str(shirt),proper_crossing_pairs=len(pairs),
                     status='failed_contact' if len(pairs) else 'crossing_test_only_not_visual_acceptance'))
(a.root/'collar_shirt_contacts.json').write_text(json.dumps(dict(split='integration_only',cases=rows),indent=2)+'\n')
print(json.dumps(rows),flush=True)
