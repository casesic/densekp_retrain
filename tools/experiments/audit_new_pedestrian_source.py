"""Review a complete new SMPL-X motion on unchanged target identities."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.audit_pedestrian_poses import parameters, mesh_panel
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.body.parameters import evaluate
from densekp_retrain.io.json import write_json
from densekp_retrain.body.geometry import load_smplx_model


def main():
    root=Path(sys.argv[1]).resolve(); out=root/'pose_audit';out.mkdir(exist_ok=False)
    source=json.loads((root/'sources/acquisition.json').read_text())
    z=np.load(source['local_file']);torch.set_num_threads(2);start=time.monotonic()
    if str(z['surface_model_type'])!='smplx' or z['poses'].shape[1]!=165:
        raise ValueError('Not complete SMPL-X axis-angle')
    frames=np.unique(np.r_[np.arange(0,len(z['poses']),10),len(z['poses'])-1])
    records=[]
    for short in ['F02','M02']:
        path=ROOT/f'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_{short}_v3/manifest.json'
        m=json.loads(path.read_text());model=load_smplx_model(Path(m['model_dir']),gender=m['gender'],num_betas=30,batch_size=1).eval()
        beta=np.array(m['betas'],np.float32);_,_,pelvis=rest_body(model,beta)
        sheet=Image.new('RGB',(1200,280*((len(frames)+3)//4)),'white')
        vs=[];js=[];checks=[];groups=model.lbs_weights.numpy().argmax(1)
        for i,frame in enumerate(frames):
            params=parameters(z['poses'][frame],beta,pelvis);v,j=evaluate(model,params)
            vs.append(v[0]);js.append(j[0]);base=v[0,:,1].min()
            checks.append(dict(frame=int(frame),base_joint=int(groups[v[0,:,1].argmin()]),
                foot_height_m={side:float(v[0,np.isin(groups,ids),1].min()-base)
                   for side,ids in [('left',[7,10]),('right',[8,11])]}))
            sheet.paste(mesh_panel(v[0],model.faces,f'{short} frame {frame}'),(i%4*300,i//4*280))
        sheet.save(out/f'{short}_source_review.jpg')
        np.savez(out/f'{short}_sampled.npz',vertices=vs,joints=js,frames=frames,faces=model.faces)
        records.append(dict(character_id=m['character_id'],manifest=str(path),checks=checks))
    write_json(out/'report.json',dict(source=source,source_parameters={k:z[k].tolist() for k in z.files if k not in ['poses','trans']},
        frames=int(len(z['poses'])),pose_shape=list(z['poses'].shape),source_translation_range_m=np.ptp(z['trans'],axis=0).tolist(),
        status='mesh_action_review_pending',records=records,seconds=time.monotonic()-start,
        policy='All source rotations retained; target shape fixed; target pelvis centered. Source 16 betas/trans retained only as provenance.'))

if __name__=='__main__':main()
