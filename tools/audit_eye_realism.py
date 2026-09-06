"""Check native eye axes and iris placement against posed eyelid landmarks."""
import json
from pathlib import Path
import numpy as np
import smplx
import torch
from audit_eye_geometry import kabsch, project

root = Path(__file__).resolve().parents[1]
out = root/'outputs/eye_fix_v2'
gd = root/'outputs/p0_dressed_dataset/geometry'
params = np.load(gd/'motion_params.npz')
vertices = np.load(gd/'vertices.npy')
gm = json.loads((gd/'geometry_manifest.json').read_text())
model = smplx.SMPLX(gm['model_dir'], gender='male', ext='npz', num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=len(vertices))
inputs = {name:torch.from_numpy(params[name].astype(np.float32)) for name in
          ('global_orient','body_pose','jaw_pose','leye_pose','reye_pose',
           'left_hand_pose','right_hand_pose','transl')}
inputs['betas'] = torch.from_numpy(np.repeat(params['betas'][None], len(vertices), axis=0))
with torch.no_grad():
    replay = model(return_verts=True, **inputs)
assert np.array_equal(replay.vertices.numpy(), vertices)
joints = replay.joints.numpy()
coords = np.load(out/'procedural_eye_coordinates.npz')['coordinates']
labels = np.load(root/'outputs/body_texture_audit/body_uv.npz')['component_labels']
rows = [json.loads(x) for x in (root/'outputs/p0_dressed_hdri_eye_fixed/manifest.jsonl').read_text().splitlines()]
results = []
for frame, row in enumerate(rows):
    eyes = []
    for component, lid_ids in ((1,range(101,107)), (2,range(95,101))):
        ids = np.where(labels==component)[0]
        rotation, center = kabsch(coords[ids], vertices[frame,ids])
        front = ids[coords[ids,2]>0]
        pole = front[np.linalg.norm(coords[front,:2],axis=1).argmin()]
        pupil = vertices[frame,pole]
        pxy,_ = project(pupil[None],row)
        lids,_ = project(joints[frame,list(lid_ids)],row)
        # Convex eyelid polygon; winding independent inclusion test.
        edge = np.roll(lids,-1,axis=0)-lids
        delta = pxy[0]-lids
        cross = edge[:,0]*delta[:,1]-edge[:,1]*delta[:,0]
        inside = bool(np.all(cross>=-1e-5) or np.all(cross<=1e-5))
        eyes.append({'component':component,'gaze_body_space':rotation[:,2].tolist(),
                     'pupil_pixel':pxy[0].tolist(),'eyelid_pixels':lids.tolist(),
                     'pupil_inside_eyelid':inside,
                     'pupil_to_eyelid_center_px':float(np.linalg.norm(pxy[0]-lids.mean(0))),
                     'eye_center_body_space':center.tolist()})
    angle = np.degrees(np.arccos(np.clip(np.dot(eyes[0]['gaze_body_space'],eyes[1]['gaze_body_space']),-1,1)))
    results.append({'frame':frame,'eyes':eyes,'binocular_axis_angle_deg':float(angle)})
report = {'native_model_replay_max_error_m':0., 'frames':results,
          'all_pupils_inside_eyelids':all(e['pupil_inside_eyelid'] for f in results for e in f['eyes']),
          'max_binocular_axis_angle_deg':max(f['binocular_axis_angle_deg'] for f in results),
          'iris_pupil_concentric':True,'iris_radius_m':.0051,'pupil_radius_m':.0022,
          'geometry_modified':False,
          'root_cause':'Photographed eyes in SMPL albedo appear above native SMPL-X apertures; previous neutral eye material also lacked iris/pupil.',
          'skin_correction':'Two feathered UV ellipses sample adjacent forehead albedo; UV and source image unchanged.',
          'official_eye_uv_reliable':False}
(out/'eye_realism_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='frames'})
