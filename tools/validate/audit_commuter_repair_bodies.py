"""Recheck frozen body parameters/138 points and compare garment evidence bodies."""


def main():
    import argparse
    import json
    from pathlib import Path
    import sys
    import time
    import numpy as np
    import torch

    ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    from densekp_retrain.body.geometry import load_smplx_model
    from densekp_retrain.body.parameters import PARAMETERS, evaluate

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('repair',type=Path)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);regressor=np.load(ROOT/'assets/densekp_regressor_R.npy')
    rows=[]
    for cid in ['cycling_F02_v2','cycling_M02_v2']:
        folder=a.source/'characters'/cid;manifest=json.loads((folder/'manifest.json').read_text())
        model=load_smplx_model(Path(manifest['model_dir']),gender=manifest['gender'],num_betas=manifest['num_betas'],batch_size=1)
        for pose in ['reference','coco_292226_person_1330527','coco_306800_person_225160']:
            geom=folder/'reference/geometry' if pose=='reference' else a.source/'derivatives'/cid/pose/'geometry'
            z=np.load(geom/'motion_params.npz');params={k:z[k] for k in PARAMETERS};body,joints=evaluate(model,params)
            saved=np.load(geom/'vertices.npy');dense=np.load(geom/'densekp_3d.npy');faces=np.load(geom/'faces.npy')
            np.testing.assert_array_equal(faces,model.faces)
            np.testing.assert_array_equal(params['betas'].ravel(),np.array(manifest['betas'],np.float32))
            row=dict(character_id=cid,pose=pose,geometry=str(geom),parameter_mesh_max_m=float(np.abs(body-saved).max()),
                densekp_operator_max_m=float(np.abs(regressor@saved[0]-dense[0]).max()),pelvis_max_m=float(np.abs(joints[0,0]).max()),
                expression_zero=bool(not np.any(model.expression.detach().numpy())),label_shape=list(dense.shape))
            if row['parameter_mesh_max_m']>2e-5 or row['densekp_operator_max_m']>1e-6 or row['pelvis_max_m']>2e-6:
                raise ValueError('Frozen body gate failed')
            rows.append(row)
    evidence=[]
    for path in a.repair.glob('**/repair.json'):
        report=json.loads(path.read_text());geom=Path(report['body_geometry']);z=np.load(path.parent/'evidence.npz')
        same=np.array_equal(z['body'],np.load(geom/'vertices.npy')[0]) and np.array_equal(z['body_faces'],np.load(geom/'faces.npy'))
        if not same:raise ValueError(f'Body changed in {path}')
        evidence.append(dict(path=str(path),body_exactly_equal=True))
    out=dict(split='integration_only',status='passed',cases=rows,component_evidence=evidence,seconds=time.monotonic()-start,
        thresholds=dict(parameter_mesh_m=2e-5,densekp_operator_m=1e-6,pelvis_m=2e-6),
        not_checked=['new DS world transform/projection','final hair-clothing contact','new visibility/loader'])
    (a.repair/'frozen_body_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(body_cases=len(rows),component_evidence=len(evidence),seconds=out['seconds'])))


if __name__ == "__main__":
    main()
