#!/usr/bin/env python3
"""Recover independent SMPL poses from supplied cyclist boxes using local 4D-Humans.

Exports original predictions and crop-space reprojection checks. These are estimated
poses, not a tracked sequence or verified 3D ground truth, and are not SMPL-X.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import default_collate
import trimesh
from smplx import SMPL


# COCO shoulders through ankles -> the HMR2 wrapper's OpenPose-25 joint order.
BODY_COCO = np.arange(5, 17)
BODY_OPENPOSE = np.array([5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11])


def axis_angle(rotations):
    matrices = np.asarray(rotations).reshape(-1, 3, 3)
    angles = np.stack([cv2.Rodrigues(r.astype(np.float64))[0].ravel() for r in matrices])
    restored = np.stack([cv2.Rodrigues(a)[0] for a in angles])
    np.testing.assert_allclose(restored, matrices, atol=2e-5, rtol=0)
    return angles.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hmr-repo', type=Path, required=True)
    parser.add_argument('--model-assets', type=Path, required=True,
                        help='Extracted official hmr2_data package root')
    parser.add_argument('--smpl-model', type=Path, required=True,
                        help='Neutral SMPL pickle with NumPy arrays, no chumpy dependency')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sys.path.insert(0, str(args.hmr_repo.resolve()))
    from hmr2.configs import get_config
    from hmr2.models import HMR2
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.renderer import Renderer, cam_crop_to_full

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    run = args.model_assets / 'logs/train/multiruns/hmr2/0'
    checkpoint = run / 'checkpoints/epoch=35-step=1000000.ckpt'
    cfg = get_config(str(run / 'model_config.yaml'))
    cfg.defrost()
    for key in ('JOINT_REGRESSOR_EXTRA', 'MEAN_PARAMS'):
        cfg.SMPL[key] = str((args.model_assets / cfg.SMPL[key]).resolve())
    cfg.SMPL.MODEL_PATH = str(args.smpl_model.resolve())
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
    cfg.freeze()
    model = HMR2(cfg, init_renderer=False)
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    # Verify exported parameters with a separate loader, outside checkpoint buffers.
    export_model = SMPL(str(args.smpl_model), gender='neutral').eval()
    np.testing.assert_array_equal(export_model.faces, model.smpl.faces)
    renderer = Renderer(cfg, faces=model.smpl.faces)
    manifest = json.loads((args.input / 'manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'model_config.yaml').write_text(cfg.dump())
    cases, panels = [], []
    for row in manifest['samples']:
        sample_id = row['sample_id']
        folder = args.output / sample_id
        folder.mkdir()
        image = cv2.imread(str(args.input / row['image_file']))
        if image is None:
            raise ValueError(f'Missing image: {sample_id}')
        dataset = ViTDetDataset(cfg, image, np.array([row['bbox_xyxy']], np.float32))
        item = dataset[0]
        batch = {key: value.to(device) for key, value in default_collate([item]).items()}
        with torch.inference_mode():
            prediction = model(batch)
        parameters = {key: value.cpu().numpy() for key, value in prediction['pred_smpl_params'].items()}
        vertices = prediction['pred_vertices'][0].cpu().numpy()
        if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
            raise ValueError(f'Invalid reconstructed SMPL mesh: {sample_id}')
        global_orient = axis_angle(parameters['global_orient']).reshape(1, 3)
        body_pose = axis_angle(parameters['body_pose']).reshape(1, 69)
        with torch.inference_mode():
            restored = export_model(
                global_orient=torch.from_numpy(global_orient),
                body_pose=torch.from_numpy(body_pose),
                betas=torch.from_numpy(parameters['betas']),
            ).vertices[0].numpy()
        np.testing.assert_allclose(restored, vertices, atol=2e-5, rtol=0)
        export_error = float(np.max(np.abs(restored - vertices)))
        full_focal = float(cfg.EXTRA.FOCAL_LENGTH) / cfg.MODEL.IMAGE_SIZE * max(image.shape[:2])
        camera_full = cam_crop_to_full(prediction['pred_cam'], batch['box_center'],
                                       batch['box_size'], batch['img_size'], full_focal)[0].cpu().numpy()
        camera_crop = prediction['pred_cam_t'][0].cpu().numpy()
        projected_crop = prediction['pred_keypoints_2d'][0].cpu().numpy()
        gt = np.array(row['coco_keypoints17'], dtype=np.float32)[BODY_COCO]
        target_crop = (gt[:, :2] - item['box_center']) / item['box_size']
        errors = np.linalg.norm(projected_crop[BODY_OPENPOSE] - target_crop, axis=1) * 256
        visible = gt[:, 2] == 2
        labelled = gt[:, 2] > 0
        scores = {name: dict(count=int(mask.sum()), mean_crop_px=float(errors[mask].mean()) if mask.any() else None)
                  for name, mask in (('visible_body_joints', visible), ('labelled_body_joints', labelled))}
        np.savez_compressed(
            folder / 'smpl_prediction.npz', model_type=np.array('smpl'),
            global_orient=global_orient, body_pose=body_pose, betas=parameters['betas'],
            poses=np.concatenate((global_orient, body_pose), axis=1),
            global_orient_rotmat=parameters['global_orient'], body_pose_rotmat=parameters['body_pose'],
            vertices_model_m=vertices, faces=model.smpl.faces,
            camera_crop_translation=camera_crop, camera_full_translation=camera_full,
            camera_full_focal_px=np.array(full_focal), camera_crop_weak=prediction['pred_cam'][0].cpu().numpy(),
            box_center=item['box_center'], box_size=item['box_size'],
            projected_openpose25_crop_normalized=projected_crop[:25],
            coco_body_gt=gt, coco_body_error_crop_px=errors,
        )
        trimesh.Trimesh(vertices=vertices, faces=model.smpl.faces, process=False).export(folder / 'body.obj')

        crop = item['img'].transpose(1, 2, 0) * np.array(cfg.MODEL.IMAGE_STD) + np.array(cfg.MODEL.IMAGE_MEAN)
        overlay = renderer(vertices, camera_crop.copy(), batch['img'][0], mesh_base_color=(.3, .7, .9))
        side = renderer(vertices, camera_crop.copy(), batch['img'][0], side_view=True,
                        mesh_base_color=(.3, .7, .9), scene_bg_color=(1, 1, 1))
        panel = Image.new('RGB', (768, 286), 'white')
        for column, pixels in enumerate((crop, overlay, side)):
            panel.paste(Image.fromarray((np.clip(pixels, 0, 1)*255).astype(np.uint8)), (column*256, 30))
        score = scores['visible_body_joints']['mean_crop_px']
        score_text = 'N/A' if score is None else f'{score:.2f}px'
        draw = ImageDraw.Draw(panel)
        draw.text((5, 3), f'{sample_id} / visible COCO body reprojection: {score_text}', fill='black')
        draw.text((5, 16), 'Input crop                        SMPL overlay                       SMPL side view', fill='black')
        panel.save(folder / 'review.jpg')
        panels.append(panel)
        cases.append(dict(sample_id=sample_id, source=row, scores=scores,
                          model_type='smpl', vertices=6890, finite_vertices=True,
                          export_max_abs_vertex_error_m=export_error,
                          rotation_roundtrip_passed=True, review_status='pending mesh review'))
        print(sample_id, scores, flush=True)
    sheet = Image.new('RGB', (768, 286*len(panels)), 'white')
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, 286*index))
    sheet.save(args.output / 'reconstruction_review.jpg')
    report = dict(
        method='4D-Humans HMR2', checkpoint=str(checkpoint.resolve()),
        checkpoint_bytes=checkpoint.stat().st_size, source_repo=str(args.hmr_repo.resolve()),
        source_manifest=str((args.input/'manifest.json').resolve()),
        device=str(device), torch=torch.__version__, precision='float32; TF32 off',
        samples=len(cases), sequence=False, smplx_converted=False,
        camera_semantics='estimated perspective camera translation; not measured world motion',
        metric='visible/labelled COCO shoulders-to-ankles vs OpenPose joints; full 256-square crop pixels; no 3D GT',
        scope='pose acquisition pilot; COCO may overlap HMR2 training; not held-out model evaluation',
        cases=cases,
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
