"""Independent parameter, pelvis and fixed 138-row readback of adult candidates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import PARAMETERS, evaluate
from densekp_retrain.io.json import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);a=p.parse_args();torch.set_num_threads(2)
    root=a.assets.resolve();report=json.loads((root/'report.json').read_text());r=np.load(ROOT/'assets/densekp_regressor_R.npy')
    assert r.shape==(138,10475);records=[]
    for row in report['characters']:
        m=json.loads(Path(row['manifest']).read_text());model=load_smplx_model(Path(m['model_dir']),gender=m['gender'],num_betas=30,batch_size=1)
        jobs=[m['reference']]+[d for d in report['derivatives'] if d['character_id']==m['character_id']]
        for job in jobs:
            geom=Path(job['geometry']);z=np.load(geom/'motion_params.npz');params={k:z[k] for k in PARAMETERS}
            beta=np.asarray(params['betas']).reshape(-1);beta=np.pad(beta,(0,30-len(beta)))
            np.testing.assert_allclose(beta,m['betas'],atol=0,rtol=0);params['betas']=beta
            v,j=evaluate(model,params);saved=np.load(geom/'vertices.npy');points=np.load(geom/'densekp_3d.npy')
            np.testing.assert_array_equal(model.faces,np.load(geom/'faces.npy'))
            ve=float(np.abs(v-saved).max());ke=float(np.abs(r@saved[0]-points[0]).max());pe=float(np.abs(j[0,0]).max())
            if ve>2e-5 or ke>1e-6 or pe>2e-6:raise ValueError('Unchanged body/138/pelvis gate failed')
            if job['pose_source_id']!='reference':
                source=np.load(job['pose_source_parameters'])
                for key in PARAMETERS:
                    if key not in ('betas','transl'):np.testing.assert_array_equal(source[key],params[key])
            cloth=Path(job['clothing_geometry']);ids=np.load(cloth/'body_barycentric_indices.npy');weights=np.load(cloth/'body_barycentric_weights.npy')
            anchor=(saved[:,ids]*weights[None,:,:,None]).sum(2)
            np.testing.assert_array_equal(anchor,np.load(cloth/'reference_body_vertices.npy'))
            cv=np.load(cloth/'clothing_vertices.npy');distance=np.linalg.norm(cv-anchor,axis=-1)
            if not np.isfinite(cv).all() or distance.max()>.010002:raise ValueError('Legacy shell offset gate failed')
            records.append(dict(character_id=m['character_id'],pose=job['pose_source_id'],parameter_max_m=ve,
                operator_max_m=ke,pelvis_max_m=pe,clothing_max_offset_m=float(distance.max()),pose_unchanged=True))
    write_json(root/'body_clothing_readback.json',dict(status='passed',cases=records,body_gate_m=2e-5,pelvis_gate_m=2e-6,operator_gate_m=1e-6,
        fixed_rows=138,supervision='in_view & inside_full_crop; independent from visibility',appearance_status='not determined by this audit'))
    print('passed',len(records))

if __name__=='__main__':main()
