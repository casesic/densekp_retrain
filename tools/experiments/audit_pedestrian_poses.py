"""Inspect existing full SMPL-X motion frames without inventing action labels."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.body.parameters import rest_body
from densekp_retrain.body.parameters import evaluate
from densekp_retrain.io.json import write_json

SOURCES = [Path('/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz'),
           ROOT / 'third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz']


def parameters(pose, beta, pelvis):
    limits = [('global_orient', 0, 3), ('body_pose', 3, 66), ('jaw_pose', 66, 69),
              ('leye_pose', 69, 72), ('reye_pose', 72, 75),
              ('left_hand_pose', 75, 120), ('right_hand_pose', 120, 165)]
    return dict({key: np.asarray(pose[a:b], np.float32)[None] for key, a, b in limits},
                betas=beta, transl=-pelvis[None])


def mesh_panel(vertices, faces, label, size=(300, 280)):
    canvas = np.full((size[1], size[0], 3), 245, np.uint8)
    for view, axes in enumerate([(0, 1, 2), (2, 1, 0)]):
        xy = vertices[:, list(axes[:2])].copy()
        xy[:, 0] = xy[:, 0] * 115 + 75 + view * 150
        xy[:, 1] = 260 - (xy[:, 1] - vertices[:, 1].min()) * 115
        normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                          vertices[faces[:, 2]] - vertices[faces[:, 0]])
        light = np.abs(normal[:, axes[2]]) / np.maximum(np.linalg.norm(normal, axis=1), 1e-9)
        for i in np.argsort(vertices[faces, axes[2]].mean(1)):
            color = int(90 + 120 * light[i])
            cv2.fillConvexPoly(canvas, np.rint(xy[faces[i]]).astype(np.int32), (color, color, color))
    result = Image.fromarray(canvas)
    ImageDraw.Draw(result).text((5, 5), label, fill='black')
    return result


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); torch.set_num_threads(2)
    model = load_smplx_model(ROOT / 'outputs/experiments/cycling_commuter_pilot_20260930/models',
                             gender='male', num_betas=30, batch_size=1).eval()
    records = []
    for source_id, path in enumerate(SOURCES):
        z = np.load(path); beta = np.pad(z['betas'], (0, 30-len(z['betas']))).astype(np.float32)
        _, _, pelvis = rest_body(model, beta)
        frames = np.unique(np.r_[np.arange(0, len(z['poses']), 10), len(z['poses'])-1])
        sheet = Image.new('RGB', (1200, 280*((len(frames)+3)//4)), 'white')
        verts, joints = [], []
        for i, frame in enumerate(frames):
            v, j = evaluate(model, parameters(z['poses'][frame], beta, pelvis))
            verts.append(v[0]); joints.append(j[0])
            sheet.paste(mesh_panel(v[0], model.faces, f'source {source_id} frame {frame}'), (i%4*300, i//4*280))
        sheet.save(out/f'source_{source_id}_review.jpg')
        np.savez(out/f'source_{source_id}_sampled.npz', vertices=verts, joints=joints, frames=frames, faces=model.faces)
        records.append(dict(source_id=source_id, path=str(path), bytes=path.stat().st_size,
            motion_info=z['motion_info'].tolist(), frames=len(z['poses']), fps=int(z['mocap_frame_rate']),
            gender=str(z['gender']), source_shape=z['betas'].tolist(), sampled_frames=frames.tolist(),
            source_translation_range_m=np.ptp(z['trans'],axis=0).tolist(),
            normalization='source global/body/hand pose unchanged; target pelvis centered; source translation retained in source archive'))
    write_json(out/'inventory.json', dict(sources=records, seconds=time.monotonic()-start, status='action_review_pending'))


if __name__ == '__main__':
    main()
