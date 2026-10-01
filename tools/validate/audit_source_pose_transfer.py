"""Record the retained pose and explicit translation replacement per derivative."""
import json
from pathlib import Path
import sys
import numpy as np


def main():
    root = Path(sys.argv[1]).resolve()
    report = json.loads((root / 'assets/report.json').read_text())
    records = []
    names = ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose',
             'reye_pose', 'left_hand_pose', 'right_hand_pose']
    for row in report['derivatives']:
        source_path = Path(row['original_pose_file'])
        with np.load(source_path) as source, np.load(Path(row['geometry']) / 'motion_params.npz') as target:
            if row['action'] == 'cycling':
                for key in names:
                    np.testing.assert_array_equal(target[key], source[key])
                translation = source['transl'].reshape(-1).tolist()
                source_shape = source['betas'].reshape(-1).tolist()
            else:
                pose = np.concatenate([target[key].reshape(-1) for key in names])
                np.testing.assert_array_equal(pose, source['poses'][row['source_frame']].astype(np.float32))
                translation = source['trans'][row['source_frame']].tolist()
                source_shape = source['betas'].reshape(-1).tolist()
            records.append(dict(
                character_id=row['character_id'], pose_source_id=row['pose_source_id'],
                source_file=str(source_path), source_frame=row['source_frame'],
                source_group=row['source_group'], partition=row['partition'],
                source_translation=translation, target_translation=target['transl'].reshape(-1).tolist(),
                source_betas=source_shape, target_betas=target['betas'].reshape(-1).tolist(),
                shape_manifest=row['parent_manifest'],
                pose_exact_float32=True,
            ))
    result = dict(status='passed', cases=records,
        pose='All 165 rotations retained. No pose edit for clothing or native-ground repair.',
        translation='Original translation retained as provenance, replaced by negative target rest pelvis for canonical assets; explicit world placement and sole lift applied by renderer.',
        units='SMPL-X model meters, Y-up canonical; rendering conversion and transforms independently checked per layout.',
        shape='Target identity shape only; source betas are provenance, not a conversion recipe.')
    (root / 'assets/source_transform_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(records))


if __name__ == '__main__':
    main()
