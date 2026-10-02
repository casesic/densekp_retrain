"""Read back all candidate parameters and bindings without accepting failed components."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import PARAMETERS, evaluate
from densekp_retrain.io.json import write_json
from densekp_retrain.clothing.binding import pose_rotations
from densekp_retrain.clothing.contracts import validate_binding


def boundary_components(faces):
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    adjacency={}
    for a,b in unique[counts==1]:
        adjacency.setdefault(int(a),set()).add(int(b))
        adjacency.setdefault(int(b),set()).add(int(a))
    remaining=set(adjacency);components=0
    while remaining:
        pending=[remaining.pop()];components+=1
        while pending:
            for vertex in adjacency[pending.pop()]&remaining:
                remaining.remove(vertex);pending.append(vertex)
    return dict(boundary_components=components,nonmanifold_edges=int((counts>2).sum()))


def winding_number(point,vertices,faces):
    a,b,c=(vertices[faces].astype(np.float64)-point).transpose(1,0,2)
    la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
    numerator=(a*np.cross(b,c)).sum(1)
    denominator=la*lb*lc+(a*b).sum(1)*lc+(b*c).sum(1)*la+(c*a).sum(1)*lb
    return float((2*np.arctan2(numerator,denominator)).sum()/(4*np.pi))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('assets',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    report=json.loads((args.assets/'report.json').read_text())
    regressor=np.load(ROOT/'assets/densekp_regressor_R.npy');records=[]
    for character in report['characters']:
        path=Path(character['manifest']);manifest=json.loads(path.read_text())
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=manifest['num_betas'],batch_size=1)
        joints=np.load(path.parent/'rest_joints.npy')
        jobs=[manifest['reference']]+[d for d in report['derivatives'] if d['character_id']==character['character_id']]
        for job in jobs:
            geometry=Path(job['geometry']);clothing=Path(job['clothing_geometry'])
            with np.load(geometry/'motion_params.npz') as archive:
                params={k:archive[k] for k in PARAMETERS}
                expression=archive['expression'] if 'expression' in archive else model.expression.detach().numpy()
            np.testing.assert_array_equal(params['betas'].reshape(-1),np.array(manifest['betas'],np.float32))
            if np.any(expression):raise ValueError('Expression changed identity')
            restored,jj=evaluate(model,params)
            body=np.load(geometry/'vertices.npy');faces=np.load(geometry/'faces.npy');dense=np.load(geometry/'densekp_3d.npy')
            np.testing.assert_array_equal(faces,model.faces)
            mesh_error=float(np.abs(restored-body).max());operator_error=float(np.abs(regressor@body[0]-dense[0]).max())
            pelvis_error=float(np.abs(jj[0,0]).max())
            if mesh_error>2e-5 or operator_error>1e-6 or pelvis_error>2e-6:raise ValueError('Existing body geometry gate failed')
            indices=np.load(clothing/'body_barycentric_indices.npy');weights=np.load(clothing/'body_barycentric_weights.npy')
            np.testing.assert_allclose(weights.sum(1),1,atol=1e-6,rtol=0)
            if (weights<0).any() or indices.min()<0 or indices.max()>=body.shape[1]:raise ValueError('Invalid attachment')
            reference=(body[:,indices]*weights[None,:,:,None]).sum(2)
            cv=np.load(clothing/'clothing_vertices.npy');cf=np.load(clothing/'clothing_faces.npy')
            binding=validate_binding(clothing,reference,cv)
            skin=(model.lbs_weights.numpy()[indices]*weights[:,:,None]).sum(1)
            np.testing.assert_array_equal(skin,np.load(clothing/'skin_weights.npy'))
            expected_rotation=np.einsum('nj,jab->nab',skin,pose_rotations(model,params,joints))
            np.testing.assert_allclose(expected_rotation,np.load(clothing/'binding_rotations.npy'),atol=1e-6,rtol=0)
            info=json.loads((clothing/'clothing_manifest.json').read_text())
            topology=[]
            for component in info['components']:
                start=component['vertex_start'];stop=start+component['vertex_count']
                sub=cf[((cf>=start)&(cf<stop)).all(1)]
                topology.append(dict(name=component['name'],**boundary_components(sub)))
            correction=np.load(clothing/'collision_correction.npy');sample=int(np.linalg.norm(correction,axis=1).argmax())
            seed=cv[0,sample]-correction[sample]
            records.append(dict(character_id=character['character_id'],geometry=str(geometry),clothing=str(clothing),
                body_status='passed',parameter_mesh_max_m=mesh_error,operator_max_m=operator_error,pelvis_max_m=pelvis_error,
                **binding,topology=topology,clothing_numeric_status=info['status'],collision=info['collision'],
                largest_correction_probe=dict(vertex_id=sample,initial_position=seed.tolist(),
                    winding_number=winding_number(seed,body[0],faces),
                    scope='one diagnostic sample; not an exhaustive body or garment self-intersection test')))
    write_json(args.assets/'independent_readback.json',dict(split='integration_only',body_status='passed',
        binding_status='passed',overall_status='needs_correction',cases=records,
        thresholds=dict(parameter_mesh_m=2e-5,operator_m=1e-6,pelvis_m=2e-6,binding_reconstruction_m=2e-6),
        not_checked=['DS actual rendered world mesh and projection','hair binding','exhaustive garment self intersections']))


if __name__=='__main__':main()
