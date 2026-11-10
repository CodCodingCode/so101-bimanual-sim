"""LeRobot v3 so101_follower state (degrees from the calibration middle pose, gripper % open) -> SO-100 Menagerie
radians, the same first-guess map as so101_twin/pickplace/joints.py before its fit (armfit fits the offsets)."""
import numpy as np

SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS_DEG = np.array([0.0, -90.0, 90.0, 0.0, 0.0])
JAW_RANGE = (-0.174, 1.75)


def real_to_sim(q):
    q = np.asarray(q, float)
    rad = np.deg2rad(SIGNS * q[:5] + OFFSETS_DEG)
    jaw = JAW_RANGE[0] + np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])
