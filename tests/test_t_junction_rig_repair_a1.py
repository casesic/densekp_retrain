import json
from pathlib import Path

import numpy as np

from tools.experiments.prepare_urban_three_multirig_a1 import coverage_pass
from tools.experiments.repair_t_junction_rigs_a1 import (
    CONFIG, ROOT, anchor_positions, headings, light_candidate,
)


def test_historical_back_edge_gate_and_explicit_group_level_policy():
    common = [True] * 31
    view = [True] * 8 + [False] * 23
    edge = [False] * 31
    config = dict(minimum_common_physical_frames=6, minimum_view_common_run=6,
                  minimum_view_common_frames=6)
    assert not coverage_pass(common, view, edge, view_goal="back", config=config, road_pass=False)
    explicit = dict(config, minimum_dual_edge_frames=0, minimum_dual_edge_run=0)
    assert coverage_pass(common, view, edge, view_goal="back", config=explicit, road_pass=False)
    assert not coverage_pass(common, [False] * 31, edge, view_goal="front", config=explicit, road_pass=False)


def test_actual_paired_camera_root_gate_excludes_illusory_edge():
    config = json.loads(CONFIG.read_text())
    src = ROOT / config["source_prepared"]
    world = np.load(src / "world_densekp.npy")
    matrices = np.load(src / "scene_matrices.npy")
    forward = headings(src)
    from PIL import Image
    masks = {c: np.asarray(Image.open(src / "calibration" / f"{c}_valid_mask.png")) > 0
             for c in ("cam2", "cam4")}
    candidate = dict(attempt=152, origin_m=[-0.0773498390045792, 79.94340709067974, 0.],
                     yaw_deg=37., camera_child_seed=1)
    row = light_candidate(candidate, "back", world, matrices, forward, masks, .18, config)
    assert row["edge_window_frames"] >= 6
    assert row["minimum_camera_to_person_root_m"] < .5
    assert not row["placement_passed"]
    assert not row["edge_and_view_scale_passed"]


def test_position_seed_replay_and_light_budget():
    config = json.loads(CONFIG.read_text())
    src = ROOT / config["source_prepared"]
    matrices = np.load(src / "scene_matrices.npy")
    forward = headings(src)
    one = anchor_positions(config, "side", matrices, forward)
    two = anchor_positions(config, "side", matrices, forward)
    assert one == two
    assert len(one[1]) <= config["positions_per_view"]
    assert len(one[1]) * (360 // config["coarse_yaw_step_deg"] +
                          len(config["local_yaw_offsets_deg"])) <= 512
