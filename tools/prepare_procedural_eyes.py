"""Generate material coordinates from native SMPL-X shaped rest vertices."""
import json
from pathlib import Path
import numpy as np
import smplx
import torch

root=Path(__file__).resolve().parents[1]
gd=root/'outputs/p0_dressed_dataset/geometry'
out=root/'outputs/eye_fix_v2'; out.mkdir(exist_ok=True)
gm=json.loads((gd/'geometry_manifest.json').read_text())
params=np.load(gd/'motion_params.npz')
model=smplx.SMPLX(gm['model_dir'],gender='male',ext='npz',num_betas=10,use_pca=False,flat_hand_mean=True)
with torch.no_grad():
    rest=model(betas=torch.from_numpy(params['betas'][None]),return_verts=True).vertices[0].numpy()
labels=np.load(root/'outputs/body_texture_audit/body_uv.npz')['component_labels']
faces=np.load(gd/'faces.npy')
assert np.array_equal(model.faces,faces)
weights=model.lbs_weights.numpy()
coordinates=np.zeros_like(rest)
report=[]
for c in np.unique(labels):
    ids=np.where(labels==c)[0]
    joint=int(weights[ids].mean(0).argmax())
    if joint not in (23,24): continue
    center=rest[ids].mean(0)
    coordinates[ids]=rest[ids]-center
    report.append({'component':int(c),'joint':joint,'center':center.tolist(),
                   'gaze_rest_axis':[0,0,1],'iris_radius_m':.0057,'pupil_radius_m':.0022})
assert len(report)==2
np.savez(out/'procedural_eye_coordinates.npz',coordinates=coordinates,faces=faces)
(out/'procedural_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
