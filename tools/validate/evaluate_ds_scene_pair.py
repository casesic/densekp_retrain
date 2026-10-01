#!/usr/bin/env python3
"""Run an existing checkpoint on the diagnostic paired scenes, without training.

GT lens validity selects evaluation points, grouped by visibility. Channel 2 is saved
unchanged and never interpreted as confidence or used to weight an error.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, points_in_mask
from tools.validate.validate_bedlam_densekp_tiny import audit_cfg
from tools.validate.occlusion_metrics import (
    summarize, point_errors, evaluation_masks, prediction_retention, pair_masks,
)


def panel(item, prediction, visible, in_view, label, mean_error, target_mask):
    canvas = Image.fromarray(cv2.cvtColor(item['img_disp'], cv2.COLOR_BGR2RGB)).resize((512, 512))
    draw = ImageDraw.Draw(canvas)
    crop_mask = cv2.warpAffine(target_mask, item['_trans'], (256, 256), flags=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(map(int, p)) for p in contour[:, 0] * 2]
        if len(points) > 1:
            draw.line(points + points[:1], fill=(0, 220, 255), width=1)
    gt = (item['proj_verts_cropped'][:, :2] + .5) * 512
    pred = (prediction[:, :2] + .5) * 512
    # Synthetic geometry supplies GT even when its surface is occluded.
    for j in np.flatnonzero(in_view):
        a, b = tuple(gt[j]), tuple(pred[j])
        draw.line([a, b], fill=(230, 190, 70), width=1)
        color = (30, 255, 80) if visible[j] else (255, 170, 40)
        draw.ellipse((a[0]-3, a[1]-3, a[0]+3, a[1]+3), fill=color)
        draw.ellipse((b[0]-2, b[1]-2, b[0]+2, b[1]+2), fill=(255, 60, 70))
    draw.line((64, 34, 64, 512), fill=(180, 180, 180))
    draw.line((448, 34, 448, 512), fill=(180, 180, 180))
    draw.rectangle((0, 0, 512, 33), fill=(0, 0, 0))
    draw.text((6, 3), label, fill='white')
    error_text = f'{mean_error:.2f}' if mean_error is not None else 'N/A'
    draw.text((6, 18), f'GT green/orange: visible/hidden; red: pred; mean {error_text} px', fill='white')
    return canvas


def evaluate_scene(directory, output, model, device):
    output.mkdir(parents=True, exist_ok=False)
    validation = json.loads((directory / 'validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Validate generated scene and loader before inference')
    generated = json.loads((directory / 'report.json').read_text())
    annotation_path = directory / 'diagnostic_loader.npz'
    arrays, records, missing = {}, [], []
    for name in ('cam2', 'cam4'):
        with np.load(directory / name / 'annotations.npz') as archive:
            arrays[name] = {k: archive[k] for k in archive.files}
        for box in json.loads((directory / name / 'boxes.json').read_text()):
            row = dict(camera=name, person_id=box['person_id'], bbox_xyxy_exclusive=box['bbox_xyxy_exclusive'])
            if box['bbox_xyxy_exclusive'] is None:
                missing.append(row)
            else:
                records.append(row)
    cfg = audit_cfg('supervision_weight')
    with patch.dict(DATASET_FOLDERS, {'scene-pair-eval': str(directory)}), \
         patch.dict(DATASET_FILES[False], {'scene-pair-eval': str(annotation_path)}):
        dataset = DatasetKeypoints(cfg, 'scene-pair-eval', is_train=False, use_augmentation=False)
    if len(dataset) != len(records):
        raise ValueError('Diagnostic NPZ rows differ from oracle box population')
    ids = arrays['cam2']['person_ids'].tolist()
    np.testing.assert_array_equal(arrays['cam2']['world_densekp'], arrays['cam4']['world_densekp'])
    panels, states, cases = {}, {}, []
    aggregate = {}
    for index, row in enumerate(records):
        camera, person_id = row['camera'], row['person_id']
        data = arrays[camera]
        person_index = data['person_ids'].tolist().index(person_id)
        item = dataset[index]
        if Path(item['imgname']).parent.name != camera:
            raise ValueError('Diagnostic row camera mismatch')
        np.testing.assert_array_equal(item['proj_verts'][:, :2], data['xy'][person_index])
        visible = data['visible'][person_index]
        if dataset.data.get('supervision_policy') != SUPERVISION_POLICY:
            raise ValueError('Regenerate diagnostic_loader.npz with the current scene validator')
        np.testing.assert_array_equal(item['visible'], visible)
        box = np.asarray(row['bbox_xyxy_exclusive'])
        np.testing.assert_allclose(item['box_center'], (box[:2] + box[2:]) / 2)
        np.testing.assert_allclose(item['box_size'], max(box[2:] - box[:2]) * 1.2, rtol=1e-6)
        with torch.inference_mode():
            raw = model({'img': torch.from_numpy(item['img'][None]).to(device)})['pred_keypoints']
        if tuple(raw.shape) != (1, 138, 3) or not torch.isfinite(raw).all():
            np.save(output / f'{camera}_person_{person_id}_failed_output.npy', raw.float().cpu().numpy())
            raise ValueError(f'Invalid prediction for {camera}/{person_id}; evaluation aborted')
        prediction = raw[0].float().cpu().numpy()
        restored = restore_crop_xy(prediction[:, :2], item['_trans'], (256, 256))
        target = item['proj_verts_cropped'][:, :2]
        np.testing.assert_allclose(restore_crop_xy(target, item['_trans'], (256, 256)),
                                   data['xy'][person_index], atol=2e-4, rtol=0)
        width, height = Image.open(directory / camera / 'normal_erp.png').size
        crop_error, erp_error = point_errors(prediction[:, :2], target, restored, data['xy'][person_index], width)
        crop_pixels = (target + .5) * 256
        retained = (crop_pixels[:, 0] >= 32) & (crop_pixels[:, 0] < 224) & \
                   (crop_pixels[:, 1] >= 0) & (crop_pixels[:, 1] < 256)
        other_camera = 'cam4' if camera == 'cam2' else 'cam2'
        other_index = arrays[other_camera]['person_ids'].tolist().index(person_id)
        shared = visible & arrays[other_camera]['visible'][other_index]
        masks = evaluation_masks(data['in_view'][person_index], visible, target)
        masks['shared_visible'] = shared
        lens_mask = np.asarray(Image.open(directory / camera / 'valid_mask.png')) > 0
        prediction_valid = points_in_mask(restored, lens_mask)
        scores = {}
        for group, mask in masks.items():
            aggregate.setdefault(group, dict(crop=[], erp=[]))
            scores[group] = dict(crop_px=summarize(crop_error[mask]), erp_periodic_px=summarize(erp_error[mask]))
            aggregate[group]['crop'].extend(crop_error[mask].tolist())
            aggregate[group]['erp'].extend(erp_error[mask].tolist())
        # A diagnostic hint only: compare the same semantic row on visible neighbors.
        nearest_other = np.full(138, np.inf)
        for j in range(len(data['person_ids'])):
            if j == person_index:
                continue
            delta = restored - data['xy'][j]
            delta[:, 0] = (delta[:, 0] + width / 2) % width - width / 2
            distance = np.linalg.norm(delta, axis=1)
            nearest_other = np.minimum(nearest_other, np.where(data['visible'][j], distance, np.inf))
        closer_other = visible & (nearest_other + 5 * float(item['box_size']) / 256 < erp_error)
        category = next(p['category'] for p in generated['people'] if p['id'] == person_id)
        case = dict(**row, category=category, metrics=scores, finite_output_values=414,
                    prediction_retention={group: prediction_retention(mask, prediction_valid)
                                          for group, mask in masks.items()},
                    visible_outside_retained_input=int((visible & ~retained).sum()),
                    nearer_visible_neighbor_same_row_by_5_crop_px=int(closer_other.sum()))
        cases.append(case)
        np.savez(output / f'{camera}_person_{person_id}.npz',
                 prediction_raw=prediction, prediction_erp_xy=restored, affine=item['_trans'],
                 target_crop_xy=target, target_erp_xy=data['xy'][person_index],
                 retained_input=retained, prediction_valid=prediction_valid, **masks,
                 crop_error_px=crop_error, erp_periodic_error_px=erp_error,
                 nearer_neighbor_hint=closer_other)
        key = (person_id, camera)
        target_mask = np.array(Image.open(directory / camera / f'person_{person_id}_mask.png'))
        panels[key] = panel(item, prediction, visible, masks['all_in_view'],
                            f'{directory.name} / {camera} / person {person_id}',
                            scores['all_in_view']['crop_px']['mean'], target_mask)
        panels[key].save(output / f'{camera}_person_{person_id}.jpg')
        states[key] = dict(error=crop_error, shared=shared, visible=visible,
                           in_view=data['in_view'][person_index], prediction_valid=prediction_valid)
        print(f'{directory.name} {camera} person {person_id}: {scores["visible"]["crop_px"]}', flush=True)
    pairs = []
    for person_id in ids:
        a, b = states.get((person_id, 'cam2')), states.get((person_id, 'cam4'))
        if a is None or b is None:
            pairs.append(dict(person_id=person_id, status='missing oracle box in one view', shared_points=0))
            continue
        shared = a['shared']
        np.testing.assert_array_equal(shared, b['shared'])
        groups = pair_masks(a['in_view'], a['visible'], b['in_view'], b['visible'])
        grouped_scores = {
            name: dict(mean_two_view_crop_px=summarize((a['error'][mask]+b['error'][mask])/2),
                       worse_view_crop_px=summarize(np.maximum(a['error'][mask], b['error'][mask])),
                       prediction_retention=prediction_retention(mask, a['prediction_valid'] & b['prediction_valid']))
            for name, mask in groups.items()
        }
        pairs.append(dict(person_id=person_id, status='scored' if groups['both_in_view'].any() else 'no shared in-view points',
                          metrics=grouped_scores,
                          shared_points=int(shared.sum()),
                          mean_of_two_view_crop_errors_px=summarize((a['error'][shared] + b['error'][shared]) / 2),
                          worse_view_crop_error_px=summarize(np.maximum(a['error'][shared], b['error'][shared]))))
    sheet = Image.new('RGB', (1024, 512 * len(ids)), (25, 25, 25))
    for i, person_id in enumerate(ids):
        for j, camera in enumerate(('cam2', 'cam4')):
            if (person_id, camera) in panels:
                sheet.paste(panels[(person_id, camera)], (512*j, 512*i))
            else:
                ImageDraw.Draw(sheet).text((512*j+8, 512*i+8), f'{camera} / person {person_id}: no oracle box', fill='white')
    sheet.save(output / 'prediction_review.jpg')
    result = dict(scene=directory.name, input_root=str(directory), cases=cases, pairs=pairs,
                  missing_boxes=missing, evaluated_views=len(cases), finite_output_values=414*len(cases),
                  metrics={k: dict(crop_px=summarize(v['crop']), erp_periodic_px=summarize(v['erp']))
                           for k, v in aggregate.items()})
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'assets_external/densekp.ckpt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.pilot_root = args.pilot_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with GPU access or explicitly select --device cpu')
    with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs/hydra')):
        cfg = compose(config_name='train', overrides=['experiment=densekp'])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = 'supervision_weight'
    OmegaConf.save(cfg, args.output / 'model_config.yaml')
    (args.output / 'loader_config.yaml').write_text(audit_cfg('supervision_weight').dump())
    print(f'Loading original checkpoint: {args.checkpoint}', flush=True)
    model = DenseKP.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location='cpu', strict=True)
    model.to(device).eval()
    results = [evaluate_scene(args.pilot_root / name, args.output / name, model, device)
               for name in ('cycling_forest', 'urban_street_assets')]
    result = dict(status='completed', checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_bytes=args.checkpoint.stat().st_size,
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  tracked_worktree_changes=subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines(),
                  evaluator_source=str(Path(__file__).resolve()), torch=torch.__version__,
                  device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  precision='float32, no autocast, TF32 disabled', scenes=results,
                  value_semantics='raw third output preserved; not confidence and never used for scoring',
                  metric_definition='unweighted point mean; periodic ERP x; crop errors in full 256-square pixels',
                  pair_metric='per shared semantic row, mean/max of its two per-view GT errors; not camera-to-camera xy distance',
                  neighbor_hint='same-row visible neighbor is nearer by >5 crop px; diagnostic proximity, not proof of identity swap',
                  scope='12 oracle-box views; approximate synthetic visibility; repeated identity/poses; no training, detector or reconstruction')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Completed: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
