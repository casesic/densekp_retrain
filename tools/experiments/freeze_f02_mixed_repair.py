"""Freeze one bounded F02 + M04 repair layout list before CPU preflight.

The old scene and asset manifests are read-only; this creates only a new plan.
"""
import argparse
import json
from pathlib import Path
from prepare_f02_mixed_scene import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    cfg=prepare(out,20261010)
    first=json.loads((Path(__file__).resolve().parents[2]/'outputs/experiments/f02_mtb_mixed_scene_a1_20261010/preflight_decision.json').read_text())['candidate_parameters'][0]
    target={key:first[key] for key in ('target_x_m','target_y_m','cycling_speed_m_s')}
    # M04's previously successful fixed position had y=1.075 m. Move the old
    # failed near-camera y~0 back to this road depth; vary lateral gap first.
    positions=[(-1.40,1.10),(-1.20,1.10),(-1.63,1.075),(-1.00,1.10),
               (-1.40,1.25),(-1.20,1.25),(-1.00,1.25),(-1.63,1.25),
               (-1.40,1.40),(-1.20,1.40),(-1.00,1.40),(-1.63,1.40)]
    cfg.update(episode_id='f02_m04_repair_00',attempt_limit=len(positions),
               separation_policy='aabb_then_vertex_3d_v1',
               parameter_ranges=dict(target_x_m=[target['target_x_m']]*2,target_y_m=[target['target_y_m']]*2,
                                     cycling_speed_m_s=[target['cycling_speed_m_s']]*2,
                                     walker_x_m=[-1.65,-0.95],walker_y_m=[1.05,1.45]),
               explicit_candidate_parameters=[dict(target,walker_x_m=x,walker_y_m=y) for x,y in positions],
               range_rationale='F02 old candidate0 frozen; M04 historical y=1.075m avoided close-camera divergence. Twelve ordered directed layouts shift its fixed road origin only; -90deg yaw and source-root path unchanged.',
               candidate_limit_includes_all_manual_and_random=12)
    (out/'config.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    (out/'candidate_freeze.json').write_text(json.dumps(dict(seed=20261010,algorithm='explicit_bounded_order_v1',
        parent_rejection='conservative per-person region/union exceeded raster or 7M; no actual visible-instance box existed',
        target_frozen_from='2026-10-10 F02 target-only candidate0; 20/31 physical dual-edge target-only opportunity',
        M04_layout_basis='old accepted M01+M04 fixed road y=1.075m; retain source-root motion and -90deg fixed yaw',
        candidates=cfg['explicit_candidate_parameters'],max_union_roi_pixels=7000000,
        separation_policy=cfg['separation_policy'],
        exact_groom_preflight_required=True),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
