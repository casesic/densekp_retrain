#!/usr/bin/env python3
"""Normalize one already-acquired BEDLAM2 SMPL-X walk for the composition CLI.

This only copies the published pose blocks and relative source root into a
fixed-target parameter contract. It does not fit a new motion or generate
garments, geometry, or an A1 scene.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from densekp_retrain.body.geometry import load_smplx_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--character-manifest', type=Path, required=True)
    ap.add_argument('--first', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with np.load(args.source, allow_pickle=False) as source:
        fps = float(source['mocap_frame_rate'])
        poses = np.asarray(source['poses'])
        trans = np.asarray(source['trans'])
        if str(source['surface_model_type']) != 'smplx' or poses.ndim != 2 or poses.shape[1] != 165:
            raise ValueError('unsupported_source_pose_contract')
        if fps != 30 or trans.shape != (len(poses), 3) or not (0 <= args.first < args.end <= len(poses)):
            raise ValueError('unsupported_source_time_or_root')
        if not np.isfinite(poses).all() or not np.isfinite(trans).all():
            raise ValueError('nonfinite_source')
        source_betas_shape = list(source['betas'].shape)
        source_gender = str(source['gender'])
        root = trans[args.first:args.end] - trans[args.first]
        chosen = poses[args.first:args.end]
    character = json.loads(args.character_manifest.read_text())
    if character['num_betas'] != 30 or character['gender'] not in ('male', 'female'):
        raise ValueError('unsupported_target_shape')
    import torch
    torch.set_num_threads(2)
    betas = np.asarray(character['betas'], np.float32)
    model = load_smplx_model(Path(character['model_dir']), gender=character['gender'], num_betas=30, batch_size=1).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        pelvis = model(betas=torch.from_numpy(betas[None])).joints[0, 0].numpy().astype(np.float32)
    n = len(chosen)
    blocks = {'global_orient': chosen[:, 0:3], 'body_pose': chosen[:, 3:66],
              'jaw_pose': chosen[:, 66:69], 'leye_pose': chosen[:, 69:72],
              'reye_pose': chosen[:, 72:75], 'left_hand_pose': chosen[:, 75:120],
              'right_hand_pose': chosen[:, 120:165]}
    params = {k: np.asarray(v, np.float32) for k, v in blocks.items()}
    params.update(betas=betas, transl=np.asarray(root, np.float32)-pelvis[None],
                  expression=np.zeros((n, 10), np.float32),
                  source_frame_ids=np.arange(args.first, args.end, dtype=np.int64),
                  source_root_relative=root,
                  source_time_ticks=np.arange(args.first, args.end, dtype=np.int64),
                  asset_time_ticks=np.arange(n, dtype=np.int64),
                  time_base=np.array([1, 30], np.int64))
    np.savez(out / 'parameters.npz', **params)
    timeline = [dict(asset_frame=i, source_frame=int(args.first+i), source_time_tick=int(args.first+i),
                     episode_time_tick=i, time_base=[1, 30]) for i in range(n)]
    (out / 'timeline.json').write_text(json.dumps(timeline, indent=2) + '\n')
    manifest = {'source': str(args.source.resolve()), 'character_manifest': str(args.character_manifest.resolve()),
                'source_frame_range': [args.first, args.end], 'source_published_fps': fps,
                'source_gender': source_gender, 'source_betas_shape': source_betas_shape,
                'source_surface_model_type': 'smplx', 'target_gender': character['gender'],
                'target_betas': betas.tolist(), 'target_rest_pelvis_m': pelvis.tolist(),
                'root_reference_source_frame': args.first,
                'root_formula': 'target transl = published trans[source_frame] - published trans[first] - target shaped-rest pelvis',
                'source_shape_used': False, 'pose_policy': 'published 165D axis-angle blocks copied to standard SMPL-X target; no new fitting',
                'expression_policy': 'neutral_zero; source does not supply expression',
                'time_status': 'BEDLAM2 published 30 Hz; original AMASS capture PTS not inferred',
                'status': 'integration_only'}
    (out / 'normalization_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'frames': n, 'root_span_m': float(np.linalg.norm(root[-1]))}))


if __name__ == '__main__':
    main()
