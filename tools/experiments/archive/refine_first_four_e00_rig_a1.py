"""Bounded rig-only refinement of the already frozen e00 layout attempt 1.

The 100 camera poses are frozen before any projection.  The existing planner
accepts at most 12 trials per invocation, so proposals are evaluated in
consecutive batches while the unique person layout stays unchanged.
"""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from densekp_retrain.data.mixed_road_pass import plan_mixed, preflight_mixed

BASE = ROOT / 'outputs/experiments/first_four_episodes_a1_20261011/e00'
SOURCE = BASE / 'config.json'
OUT = BASE / 'rig_refinement1'


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    source = json.loads(SOURCE.read_text())
    frozen = OUT / 'frozen_rig_candidates.json'
    if frozen.exists():
        plan = json.loads(frozen.read_text())
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        layout = source['explicit_candidate_parameters'][1]
        nominal = source['explicit_camera_parameters'][1]
        poses = []
        # Offsets are within the original bounded camera placement range.
        for ox, oy in ((0.0, 0.0), (0.12, 0.0), (0.0, 0.12), (-0.12, -0.12)):
            for step in range(25):
                poses.append(dict(origin_x_m=nominal['origin_x_m'] + ox,
                                  origin_y_m=nominal['origin_y_m'] + oy,
                                  yaw_deg=80.0 + 0.5 * step))
        plan = dict(schema='first_four_e00_rig_refinement_v1',
                    source_config=str(SOURCE), source_layout_attempt=1,
                    unique_person_layouts=1, prior_camera_trials=12,
                    camera_trial_limit=128, new_camera_trials_frozen=len(poses),
                    ordered_camera_poses=poses, person_layout=layout,
                    selection='first full preflight pass in frozen order',
                    reason='original 12 trials reached no-domain to all-domain camera transition without simultaneous dual-edge')
        write(frozen, plan)
    if plan['prior_camera_trials'] + len(plan['ordered_camera_poses']) > plan['camera_trial_limit']:
        raise ValueError('camera trial limit exceeded')
    for offset in range(0, len(plan['ordered_camera_poses']), 12):
        batch = OUT / f'batch_{offset // 12:02d}'
        decision = batch / 'decision.json'
        if decision.exists():
            passed = json.loads(decision.read_text())['passed']
        else:
            batch.mkdir(parents=True, exist_ok=False)
            poses = plan['ordered_camera_poses'][offset:offset + 12]
            cfg = copy.deepcopy(source)
            cfg['attempt_limit'] = len(poses)
            cfg['explicit_candidate_parameters'] = [copy.deepcopy(plan['person_layout']) for _ in poses]
            cfg['explicit_camera_parameters'] = poses
            cfg['parameter_ranges'] = {key: [value, value] for key, value in plan['person_layout'].items()}
            cfg['camera_parameter_ranges'] = {key: [min(p[key] for p in poses), max(p[key] for p in poses)] for key in poses[0]}
            write(batch / 'config.json', cfg)
            passed = bool(preflight_mixed(plan_mixed(cfg), batch))
            write(decision, dict(passed=passed, proposal_indices=list(range(offset, offset + len(poses))),
                                 unique_person_layouts=1, total_camera_trials_including_prior=12 + offset + len(poses)))
        if passed:
            write(OUT / 'selection.json', dict(status='selected', batch=str(batch),
                                              total_new_camera_trials_evaluated=offset + len(plan['ordered_camera_poses'][offset:offset+12])))
            print(json.dumps(dict(status='selected', batch=str(batch))))
            return
    write(OUT / 'selection.json', dict(status='no_qualified_rig',
                                      total_new_camera_trials_evaluated=len(plan['ordered_camera_poses'])))
    print(json.dumps(dict(status='no_qualified_rig', new_camera_trials=len(plan['ordered_camera_poses']))))


if __name__ == '__main__':
    main()
