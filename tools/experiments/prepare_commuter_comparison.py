"""Same sports outfit and studio setup for old/new identity shape comparison only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.experiments.refine_cycling_character_clothing import clipped_shell
from tools.experiments.prepare_cycling_characters import MATERIALS,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets',type=Path)
    args=parser.parse_args()
    output=args.assets/'uniform_comparison';output.mkdir(exist_ok=False)
    old=ROOT/'outputs/experiments/cycling_four_characters_20260929'
    characters=[]
    for short,version,root in [('F01',1,old),('F02',2,args.assets),('M01',1,old),('M02',2,args.assets)]:
        cid=f'cycling_{short}_v{version}'
        source=root/'characters'/cid/'manifest.json'
        manifest=json.loads(source.read_text())
        if version==2:
            geom=Path(manifest['reference']['geometry'])
            body=np.load(geom/'vertices.npy')[0];faces=np.load(geom/'faces.npy')
            joints=np.load(source.parent/'rest_joints.npy')
            arrays,meta=clipped_shell(body,joints,body,faces)
            clothes=output/cid/'uniform_sports';clothes.mkdir(parents=True)
            for key,value in arrays.items():np.save(clothes/f'{key}.npy',value)
            np.save(clothes/'face_material_ids.npy',np.zeros(len(arrays['clothing_faces']),np.int32))
            write_json(clothes/'materials.json',MATERIALS)
            write_json(clothes/'garment_material.json',meta)
            manifest['reference']['clothing_geometry']=str(clothes.resolve())
        manifest['usage']='uniform identity comparison only; not the commuter garment'
        manifest['source_manifest']=str(source.resolve())
        destination=output/cid/'manifest.json';destination.parent.mkdir(exist_ok=True)
        write_json(destination,manifest)
        characters.append(dict(character_id=cid,manifest=str(destination.resolve())))
    write_json(output/'report.json',dict(characters=characters,derivatives=[],split='integration_only'))


if __name__=='__main__':main()
