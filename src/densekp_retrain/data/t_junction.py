"""Arc-length road trajectories for the positive-Y city T junction.

The turn has a straight +Y lead, a circular quarter arc, and a +X exit.
Position and tangent are continuous at both joins.  Asset-local pose remains
untouched; only the scene-owned rigid matrix changes with road progress.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .dynamic_scene import checked_matrices


def road_point(s: float, *, turn_y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    if radius < 3.5 or not math.isfinite(s):
        raise ValueError('invalid_turn_progress_or_radius')
    arc = math.pi * radius / 2
    if s <= 0:
        return np.array([0., turn_y + s]), np.array([0., 1.])
    if s < arc:
        theta = math.pi - s / radius
        return np.array([radius + radius * math.cos(theta), turn_y + radius * math.sin(theta)]), np.array([math.sin(theta), -math.cos(theta)])
    return np.array([radius + s - arc, turn_y + radius]), np.array([1., 0.])


def road_supported(x: float, y: float, *, margin: float = 0.) -> bool:
    """Conservative union of the actual main and cross asphalt rectangles."""
    return (abs(x) <= 3.5-margin and -80+margin <= y <= 80.) or (
        abs(x) <= 80-margin and 80. <= y <= 88-margin)


def scene_trajectory(records, *, lane: str, start: float, speed: float,
                     lateral: float, height: float, base_rotation,
                     turn_y: float = 78., radius: float = 4.):
    if lane not in ('turn_main_to_positive_x', 'cross_positive_x', 'cross_negative_x'):
        raise ValueError('unsupported_junction_lane')
    if speed <= 0 or not np.isfinite([start, speed, lateral, height]).all():
        raise ValueError('invalid_junction_trajectory')
    base = np.asarray(base_rotation, dtype=float)
    if base.shape != (3, 3):
        raise ValueError('invalid_base_rotation')
    matrices = []
    for row in records:
        time = float(Fraction(row['episode_time_tick']) * Fraction(*row['time_base']))
        progress = start + speed * time
        if lane == 'turn_main_to_positive_x':
            xy, tangent = road_point(progress, turn_y=turn_y, radius=radius)
            # lateral is an inward lane displacement, perpendicular to the tangent.
            xy = xy + lateral * np.array([-tangent[1], tangent[0]])
        else:
            sign = 1. if lane == 'cross_positive_x' else -1.
            xy = np.array([progress * sign, 84. + lateral])
            tangent = np.array([sign, 0.])
        yaw = -math.atan2(tangent[0], tangent[1])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ base
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [*xy, height]
        matrices.append(matrix)
    return checked_matrices(matrices, len(records))


def check_trajectory(matrices, *, max_yaw_step_deg: float = 8.,
                     max_speed_step_m_s: float = .12):
    roots = np.asarray(matrices)[:, :3, 3]
    steps = np.diff(roots[:, :2], axis=0)
    speeds = np.linalg.norm(steps, axis=1) * 10.
    headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    yaw_steps = np.rad2deg(np.abs(np.diff(headings)))
    if (speeds <= 0).any() or (np.abs(np.diff(speeds)) > max_speed_step_m_s).any() or (yaw_steps > max_yaw_step_deg).any():
        raise ValueError('junction_speed_or_heading_discontinuity')
    return dict(root_displacement_m=float(np.linalg.norm(roots[-1, :2]-roots[0, :2])),
                speed_range_m_s=[float(speeds.min()), float(speeds.max())],
                maximum_sampled_heading_step_deg=float(yaw_steps.max(initial=0.)))
