"""Numerical rest-to-pose initialization for static cloth, not a reconstructed motion."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from tools.experiments.prepare_cycling_characters import PARAMETERS,evaluate,write_json
from tools.experiments.build_commuter_clothing import pose_rotations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets',type=Path);parser.add_argument('--character',default='cycling_F02_v2')
    parser.add_argument('--pose',default='coco_306800_person_225160')
    parser.add_argument('--version',default='v6');args=parser.parse_args()
    root=args.assets.resolve();mp=root/'characters'/args.character/'manifest.json'
    manifest=json.loads(mp.read_text());model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=manifest['num_betas'],batch_size=1)
    torch.set_num_threads(2);joints=np.load(mp.parent/'rest_joints.npy')
    geom=root/'derivatives'/args.character/args.pose/'geometry';clothing=geom.parent/'commuter_clothing_v4'
    out=geom.parent/f'cloth_simulation_{args.version}';out.mkdir()
    params=np.load(geom/'motion_params.npz');rest_params=np.load(Path(manifest['reference']['geometry'])/'motion_params.npz')
    idx=np.load(clothing/'body_barycentric_indices.npy');weights=np.load(clothing/'body_barycentric_weights.npy')
    offsets=np.load(clothing/'rest_offsets.npy');skin=np.load(clothing/'skin_weights.npy')
    cloth_faces=np.load(clothing/'clothing_faces.npy');rest=np.load(clothing/'clothing_rest_vertices.npy')
    info=json.loads((clothing/'clothing_manifest.json').read_text())
    edges=np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1]);pin=np.full(len(rest),.15,np.float32)
    for component in info['components']:
        ids=boundary[(boundary>=component['vertex_start'])&(boundary<component['vertex_start']+component['vertex_count'])]
        if component['name']=='top':pin[ids[(rest[ids,1]>.4)|(np.abs(rest[ids,0])>.4)]]=1
        elif component['name']=='trousers':pin[ids[rest[ids,1]>.10]]=1
        else:pin[component['vertex_start']:component['vertex_start']+component['vertex_count']]=1
    body_path=[];cloth_path=[]
    for t in np.linspace(0,1,21):
        p={key:(1-t)*rest_params[key]+t*params[key] for key in PARAMETERS}
        body,_=evaluate(model,p);body=body[0];body_path.append(body)
        ref=(body[idx]*weights[:,:,None]).sum(1)
        rotations=np.einsum('nj,jab->nab',skin,pose_rotations(model,p,joints))
        cloth_path.append(ref+np.einsum('nab,nb->na',rotations,offsets))
    np.testing.assert_allclose(body_path[-1],np.load(geom/'vertices.npy')[0],atol=2e-5,rtol=0)
    np.savez(out/'input.npz',body_path=np.asarray(body_path),cloth_path=np.asarray(cloth_path),
             body_faces=model.faces,cloth_faces=cloth_faces,pin=pin)
    write_json(out/'input_manifest.json',dict(source_clothing=str(clothing),body_geometry=str(geom),
               path_samples=21,purpose='static cloth solver initialization only; not real motion',split='integration_only'))


if __name__=='__main__':main()
