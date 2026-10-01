#!/usr/bin/env python3
"""Package the two audited cycling clips for SMPL-X fitting, in native BVH units.

No retargeting, unit conversion or DenseKP labels are produced here.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from audit_cyclist_bvh import forward_kinematics, read_bvh


ROOT = Path(__file__).resolve().parents[2]


def prepare_clip(audit_dir, output, mapping):
    audit = json.loads((audit_dir / 'bvh_audit.json').read_text())
    selection = json.loads((audit_dir / 'clip_selection.json').read_text())
    source = Path(audit['path'])
    start = selection['selected']['start_frame']
    end = selection['selected']['end_frame_exclusive']
    joints, values, dt = read_bvh(source)
    if not selection['selection_verified'] or not 0 <= start < end <= len(values):
        raise ValueError(f'Invalid audited selection: {audit_dir}')
    if len(values) != audit['frames'] or dt != audit['frame_time'] or joints != audit['joints']:
        raise ValueError(f'Source no longer matches audited structure: {source}')

    channels = values[start:end]
    positions = forward_kinematics(joints, channels)
    previous = np.load(audit_dir / 'joint_positions_native.npy', mmap_mode='r')
    np.testing.assert_allclose(positions, previous[start:end], rtol=0, atol=1e-10)
    names = [joint['name'] for joint in joints]
    source_indices = np.array([names.index(row['bvh_joint']) for row in mapping])
    target_indices = np.array([int(row['smplx_index']) for row in mapping])
    np.testing.assert_array_equal(target_indices, np.arange(22))

    output.mkdir()
    np.savez_compressed(
        output / 'clip_native.npz', channels_native=channels, joint_positions_native=positions,
        joint_names=np.array(names), parents=np.array([joint['parent'] for joint in joints]),
        source_frame_ids=np.arange(start, end), frame_time=np.array(dt),
        mapping_bvh_indices=source_indices, mapping_smplx_indices=target_indices,
    )
    # Keep the exact selected text rows and original hierarchy for other BVH tools.
    hierarchy, motion = source.read_text().split('MOTION', 1)
    lines = [line for line in motion.splitlines() if line.strip()]
    clip_text = (hierarchy + f'MOTION\nFrames: {end-start}\n' + lines[1] + '\n'
                 + '\n'.join(lines[2+start:2+end]) + '\n')
    clip_path = output / 'source_clip.bvh'
    clip_path.write_text(clip_text)
    saved_joints, saved_channels, saved_dt = read_bvh(clip_path)
    if saved_joints != joints or saved_dt != dt:
        raise ValueError('Exported BVH hierarchy or frame time changed')
    np.testing.assert_array_equal(saved_channels, channels)

    for name in ('clip_selection.json', 'kinematic_review.json', 'kinematic_diagnostic.png'):
        shutil.copyfile(audit_dir / name, output / name)
    report = dict(
        source_bvh=str(source.resolve()), source_bytes=source.stat().st_size,
        source_total_frames=len(values), source_frame_range=[start, end],
        frame_range_convention='zero-based, end exclusive', frames=end-start, fps=1/dt,
        channels_shape=list(channels.shape), joints_shape=list(positions.shape),
        source_audit=str(audit_dir.resolve()), joints=joints,
        units='native BVH; 0.01 metres/unit is an uncalibrated anatomical inference only',
        axes='native BVH; inferred +Y up, no frame conversion applied',
        mapping_status='22-joint fitting design, not a correspondence between local rotations',
        source_rows_preserved=True, fk_matches_audited_positions=True,
        bicycle_required=False, smplx_fitted=False, training_ready=False,
    )
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return {key: report[key] for key in (
        'source_bvh', 'source_frame_range', 'frames', 'fps',
        'source_rows_preserved', 'fk_matches_audited_positions', 'smplx_fitted',
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path,
                        default=ROOT / 'outputs/audits/dedicated_cycling_mocap_audit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mapping_path = args.audit_root / 'BVH_TO_SMPLX_JOINT_MAPPING.csv'
    with mapping_path.open(newline='') as stream:
        mapping = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(mapping_path, args.output / mapping_path.name)
    # Preserve downloaded source metadata and attribution without another download.
    shutil.copyfile(args.audit_root / 'zenodo_record.json', args.output / 'zenodo_record.json')
    clips = {name: prepare_clip(args.audit_root / name, args.output / name, mapping)
             for name in ('participant_1', 'participant_2')}
    report = dict(status='prepared for fitting; not SMPL-X motion', clips=clips)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
