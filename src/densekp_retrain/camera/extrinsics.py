"""Camera extrinsic construction shared by fisheye entry points."""

import numpy as np


def pose(target, origin, yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    forward = np.array([np.sin(yaw), np.cos(yaw), 0.0])
    p = np.eye(4)
    p[:3, :3] = np.column_stack((right, down, forward))
    p[:3, 3] = origin
    return p
