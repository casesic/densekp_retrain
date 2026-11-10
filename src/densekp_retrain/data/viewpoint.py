"""Horizontal body-to-camera viewing angles for seeded road episodes."""
import math
import numpy as np

VIEW_CLASSES = ('front', 'front_oblique', 'side', 'rear_oblique', 'back')


def classify_view(angle_deg):
    """Bins are [0,45), [45,80), [80,100), [100,135), [135,180]."""
    if not math.isfinite(angle_deg) or not 0 <= angle_deg <= 180:
        raise ValueError('invalid_view_angle')
    return VIEW_CLASSES[sum(angle_deg >= boundary for boundary in (45, 80, 100, 135))]


def view_angle(pelvis_world, forward_world, camera_world):
    pelvis = np.asarray(pelvis_world, dtype=float)
    forward = np.asarray(forward_world, dtype=float)
    camera = np.asarray(camera_world, dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (pelvis, forward, camera)):
        raise ValueError('invalid_view_vector')
    f = forward[:2]
    to_camera = (camera - pelvis)[:2]
    fn = np.linalg.norm(f)
    cn = np.linalg.norm(to_camera)
    if fn <= 1e-9 or cn <= 1e-9:
        raise ValueError('degenerate_horizontal_view_vector')
    return float(np.degrees(np.arccos(np.clip(np.dot(f, to_camera) / (fn * cn), -1., 1.))))


def view_coverage(rows, edge_flags, requested, minimum_count=16, minimum_seconds=.5):
    if requested not in VIEW_CLASSES:
        raise ValueError('invalid_requested_view_class')
    if len(rows) != len(edge_flags):
        raise ValueError('view_edge_length_mismatch')
    flags = [all(row['views'][c]['view_class'] == requested for c in ('cam2', 'cam4')) for row in rows]
    overlap = [a and b for a, b in zip(flags, edge_flags)]
    best = 0
    run = 0
    for value in overlap:
        run = run + 1 if value else 0
        best = max(best, run)
    # Existing episodes are explicitly 10 Hz; endpoints span (N-1)/10 s.
    return dict(requested_view_class=requested, both_camera_flags=flags,
                both_camera_count=sum(flags), both_camera_fraction=sum(flags) / len(flags),
                dual_edge_overlap_flags=overlap, longest_overlap_frames=best,
                longest_overlap_seconds=max(0, best - 1) / 10,
                passed=sum(flags) >= minimum_count and max(0, best - 1) / 10 >= minimum_seconds)


def side_pass_edge_coverage(rows, edge_flags, signed_road_positions, *, minimum_edge_count=16, minimum_overlap_frames=6):
    """Fixed-rider side pass: edge quota, simultaneous side window, road pass.

    This is deliberately separate from the older view_coverage policy, whose
    side-class quota is 16 frames. The strict physical endpoint transition is
    a separate diagnostic, never inferred from a road cross-section pass.
    """
    if not (len(rows) == len(edge_flags) == len(signed_road_positions)) or not rows:
        raise ValueError('side_pass_length_mismatch')
    if minimum_edge_count < 1 or minimum_overlap_frames < 2:
        raise ValueError('invalid_side_pass_threshold')
    side = [all(row['views'][camera]['view_class'] == 'side' for camera in ('cam2', 'cam4')) for row in rows]
    overlap = [bool(a and b) for a, b in zip(side, edge_flags)]
    run = longest = 0
    for flag in overlap:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    positions = np.asarray(signed_road_positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError('nonfinite_signed_road_position')
    road_pass = bool(positions[0] < 0 < positions[-1] or positions[0] > 0 > positions[-1])
    return dict(side_both_flags=side, side_both_count=sum(side),
                side_edge_overlap_flags=overlap, longest_side_edge_run_frames=longest,
                longest_side_edge_span_seconds=max(0, longest - 1) / 10,
                dual_edge_count=int(sum(edge_flags)), signed_road_positions_m=positions.tolist(),
                road_cross_section_passed=road_pass,
                passed=sum(edge_flags) >= minimum_edge_count and longest >= minimum_overlap_frames and road_pass)


def side_view_control(rows, signed_road_positions, *, minimum_common_frames=16, minimum_side_run=3):
    """Side-view control with shared physical support, independent of edge coverage."""
    if not rows or len(rows) != len(signed_road_positions):
        raise ValueError('side_control_length_mismatch')
    if minimum_common_frames < 1 or minimum_side_run < 2:
        raise ValueError('invalid_side_control_threshold')
    common = [int(row['common_physical_points']) >= 8 for row in rows]
    side = [all(row['views'][c]['view_class'] == 'side' for c in ('cam2', 'cam4')) for row in rows]
    both = [a and b for a, b in zip(common, side)]
    run = longest = 0
    for flag in both:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    positions = np.asarray(signed_road_positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError('nonfinite_signed_road_position')
    road_pass = bool(positions[0] < 0 < positions[-1] or positions[0] > 0 > positions[-1])
    return dict(common_physical_flags=common, common_physical_count=sum(common),
                side_common_flags=both, longest_side_common_run_frames=longest,
                longest_side_common_span_seconds=max(0, longest-1)/10,
                signed_road_positions_m=positions.tolist(), road_cross_section_passed=road_pass,
                passed=road_pass and sum(common) >= minimum_common_frames and longest >= minimum_side_run)
