"""LeRobot SO-101 state (degrees, gripper %) -> Menagerie so_arm100 qpos (radians).

LeRobot (>= 0.5, `so100_follower` / SO-101 with use_degrees) reports each joint in degrees relative to the
calibration "middle" pose (arm straight up), gripper in percent open. The Menagerie model's zero is the arm
stretched horizontally along -y. Mapping derived from the dataset's rest pose (shoulder_lift -99, elbow_flex +99,
wrist_flex +75) landing on the Menagerie "rest" keyframe (Pitch -190 deg, Elbow 178 deg, Wrist_Pitch 68 deg), then
refined by the silhouette fit in fit_cameras.py (OFFSETS_DEG are the fitted values).

    sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])        j = pan, lift, elbow, wrist_flex, wrist_roll
    Jaw        = JAW_RANGE[0] + gripper% / 100 * (JAW_RANGE[1] - JAW_RANGE[0])
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
JOINTS_JSON = HERE / "joints.json"

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)       # pan flipped: real +pan swings toward the box (-x)
OFFSETS_DEG = np.array([-14.8, -92.4, 81.9, 6.0, 153.6], dtype=float)   # fitted (joints.json overrides)
JAW_RANGE = (-0.174, 1.75)                                      # Menagerie Jaw joint range (rad)
JAW_PCT_OPEN = 100.0                                            # gripper % that maps to JAW_RANGE[1]

if JOINTS_JSON.exists():                                        # fitted values written by fit_cameras.py
    _j = json.loads(JOINTS_JSON.read_text())
    SIGNS = np.asarray(_j["signs"], float)
    OFFSETS_DEG = np.asarray(_j["offsets_deg"], float)
    JAW_PCT_OPEN = float(_j.get("jaw_pct_open", JAW_PCT_OPEN))


def real_to_sim(q: np.ndarray, signs=None, offsets_deg=None, jaw_pct_open=None) -> np.ndarray:
    """(6,) real state [deg x5, gripper %] -> (6,) sim qpos [rad] in SIM_ORDER."""
    signs = SIGNS if signs is None else np.asarray(signs, float)
    offsets_deg = OFFSETS_DEG if offsets_deg is None else np.asarray(offsets_deg, float)
    jaw_pct_open = JAW_PCT_OPEN if jaw_pct_open is None else float(jaw_pct_open)
    q = np.asarray(q, float)
    rad = np.deg2rad(signs * q[:5] + offsets_deg)
    jaw = JAW_RANGE[0] + np.clip(q[5], 0, 100) / jaw_pct_open * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])
