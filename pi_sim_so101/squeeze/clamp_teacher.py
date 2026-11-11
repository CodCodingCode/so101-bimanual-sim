"""Two-arm TIP clamp into the grey box (no jaws): both grippers lean in inside their arm planes and press the cube's
sides with their fingertips, lift it level, carry it high over the box (just past the cube) and drop it in.

Per-arm tilt / roll / tip offset from squeeze/runs/design_clamp2.json (CMA-ES over 6 cube placements with the real
jaw shapes, friction 0.4, fitted servos, and penalties for any arm-arm (incl. wrist camera mounts), arm-camera post,
arm-box or arm-table contact). Open loop from the cube (and box) position only, so it runs unchanged on the real arms.

    from clamp_teacher import ClampPlan, clamp_quat
    for side in ("left", "right"): env.sim.pad_quat[side] = clamp_quat(side, cube_xy)
    plan = ClampPlan(cube_xyz, box_xy)
    targets = plan.next_targets(env.sim._ee_targets)     # (2, 3), rate-limited, every 40 ms step
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
_D = HERE / "runs" / "design_clamp2.json"
P = json.loads(_D.read_text())["params"] if _D.exists() else {}
TILT = {"left": np.radians(P.get("tilt_l", 32.8)), "right": np.radians(P.get("tilt_r", 30.2))}
ROLL = {"left": np.radians(P.get("roll_l", -97.5)), "right": np.radians(P.get("roll_r", -112.4))}
TIP_DY = {"left": P.get("dy_l_mm", -8.8) / 1000, "right": P.get("dy_r_mm", -4.0) / 1000}
TIP_Z = P.get("tip_mm", 14.9) / 1000                  # tool point (fingertips) above the cube's bottom
GAP = P.get("gap_mm", 0.4) / 1000                     # final commanded tip |x| from the cube centre line
CUBE_HALF = 0.016
CUBE_Y = 0.0372                                       # nominal cube y (world; bases at y = -0.22, 25.9 cm in front)
BOX_DY = -0.10                                        # box centre between the arms and the cube (carry backward:
                                                      # reaching forward to a box past the cube makes the grippers meet)
CARRY_Z = 0.09                                        # tool height while carrying (cube bottom ~7.5 cm, rim 3 cm)
DROP_Z = TIP_Z + 0.035                                # tool height when letting go (cube bottom ~3.5 cm, rim 3 cm)
ARM_BASE = {"left": np.array([-0.116, -0.22]), "right": np.array([0.116, -0.22])}
MAX_STEP = 0.0048                                     # m per 40 ms step (0.12 m/s)
SIDES = ("left", "right")


def clamp_quat(side, cube_xy=(0.0, CUBE_Y)):
    """Fingers tilted TILT[side] toward the cube inside the arm's reaching plane, rolled ROLL[side]."""
    v = np.asarray(cube_xy, float)[:2] - ARM_BASE[side]
    h = np.r_[v / np.linalg.norm(v), 0.0]
    approach = np.array([0, 0, -1.0]) * np.cos(TILT[side]) + h * np.sin(TILT[side])
    side_ax = np.cross(approach, [0, 0, 1.0])
    side_ax /= np.linalg.norm(side_ax)
    up_ax = np.cross(side_ax, approach)
    close = np.cos(ROLL[side]) * side_ax + np.sin(ROLL[side]) * up_ax
    R = np.column_stack([approach, np.cross(close, approach), close])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


class ClampPlan:
    """Waypoints (name, left target, right target, steps), executed with a per-step speed cap."""

    def __init__(self, cube_xyz, box_xy=None, release=True):
        cx, cy = float(cube_xyz[0]), float(cube_xyz[1])
        dyl, dyr = TIP_DY["left"], TIP_DY["right"]
        out, z, close = CUBE_HALF + 0.035, TIP_Z, CUBE_HALF - 0.004

        def pair(xl, xr, y, zz):
            return [xl, y + dyl, zz], [xr, y + dyr, zz]

        W = [("hover", *pair(cx - out, cx + out, cy, z + 0.10), 50),       # high: pass over the box
             ("descend", *pair(cx - out, cx + out, cy, z), 40),
             ("close", *pair(cx - close, cx + close, cy, z), 45),
             ("squeeze", *pair(cx - GAP, cx + GAP, cy, z), 15),
             ("lift", *pair(cx - GAP, cx + GAP, cy, z + 0.04), 45),
             ("hold", *pair(cx - GAP, cx + GAP, cy, z + 0.04), 15)]
        if box_xy is not None:
            bx, by = float(box_xy[0]), float(box_xy[1])
            W += [("raise", *pair(cx - GAP, cx + GAP, cy, CARRY_Z), 25),
                  ("carry", *pair(bx - GAP, bx + GAP, by, CARRY_Z), 70),
                  ("lower", *pair(bx - GAP, bx + GAP, by, DROP_Z), 25)]
            if release:
                W += [("release", *pair(bx - GAP - 0.012, bx + GAP + 0.012, by, DROP_Z), 15),
                      ("retreat", *pair(bx - 0.07, bx + 0.07, by, DROP_Z + 0.06), 40),      # apart AND up
                      ("settle", *pair(bx - 0.07, bx + 0.07, by, DROP_Z + 0.06), 15)]
        elif release:
            W += [("lower", *pair(cx - GAP, cx + GAP, cy, z + 0.003), 40),
                  ("release", *pair(cx - out, cx + out, cy, z + 0.003), 30),
                  ("retreat", *pair(cx - out, cx + out, cy, z + 0.063), 40)]
        self.waypoints = [(n, np.array(l, float), np.array(r, float), k) for n, l, r, k in W]
        self._i, self._k = 0, 0

    @property
    def phase(self):
        return self.waypoints[min(self._i, len(self.waypoints) - 1)][0]

    @property
    def done(self):
        return self._i >= len(self.waypoints)

    def next_targets(self, commanded):
        if self.done:
            return np.asarray(commanded, float).copy()
        _, l, r, k = self.waypoints[self._i]
        step = np.clip(np.stack([l, r]) - commanded, -MAX_STEP, MAX_STEP)
        self._k += 1
        if self._k >= k:
            self._i, self._k = self._i + 1, 0
        return commanded + step
