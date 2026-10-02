"""Common outfit, illumination and no-groom comparison inputs; never new identities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.clothing.shells import clipped_shell
from densekp_retrain.io.json import write_json
from tools.experiments.prepare_cycling_characters import MATERIALS
from tools.experiments.bind_adult_grooms import scalp,PILOT

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);a=p.parse_args();torch.set_num_threads(2)
    root=a.assets.resolve();out=root/'uniform_inputs';out.mkdir(exist_ok=False)
    report=json.loads((root/'report.json').read_text());rows=[]
    for row in sorted(report['characters'],key=lambda r:r['character_id']):
        manifest=deepcopy(json.loads(Path(row['manifest']).read_text()));folder=out/manifest['character_id'];folder.mkdir()
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=30,batch_size=1)
        rest,joints,_=rest_body(model,np.array(manifest['betas'],np.float32))
        geom=Path(manifest['reference']['geometry']);body=np.load(geom/'vertices.npy')[0]
        arrays,meta=clipped_shell(rest,joints,body,model.faces)
        clothing=folder/'clothing';clothing.mkdir()
        for name,value in arrays.items():np.save(clothing/(name+'.npy'),value)
        np.save(clothing/'face_material_ids.npy',np.zeros(len(arrays['clothing_faces']),np.int32))
        write_json(clothing/'materials.json',MATERIALS);write_json(clothing/'garment_material.json',meta)
        manifest['reference']['clothing_geometry']=str(clothing)
        manifest['reference'].pop('groom_geometry',None)
        scalp(manifest['appearance'],np.load(PILOT/'groom_reference/reference.npz'),folder)
        # Keep original skin but remove its painted scalp uniformly for face review.
        manifest['comparison_parent']=row['manifest'];manifest['status']='comparison_only_not_another_identity'
        write_json(folder/'manifest.json',manifest);rows.append(dict(character_id=manifest['character_id'],manifest=str(folder/'manifest.json')))
    write_json(out/'report.json',dict(characters=rows,derivatives=[],uniform_comparison=True,split='integration_only'))

if __name__=='__main__':main()
