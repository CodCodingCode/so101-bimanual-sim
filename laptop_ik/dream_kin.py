"""SO-101 hand poses for Dream-1 from the rig's own MuJoCo model (pi_sim), both ways.

Forward: joints (so101_real units: degrees with rig offsets = pi_sim joint angles) -> the tool point's pose in the
arm's base frame, as Dream reads it: eef_xyz in metres, eef_rot6 = the tool frame's x then y axis in the base frame.
Inverse: Dream's answered poses -> joints, with pi_sim's damped least-squares IK (RobotEnv._solve_ik's update rule,
position first, rotation weighted 0.08; the SO-101 has 5 joints, so a full 6-D pose is matched as near as it can be).

Frames (Dream's convention):
  base: the arm's `<side>_base` body: x forward into the workspace, y to the arm's left, z up.
  tool: pi_sim's `<side>_gripperframe` site (the grasp point the sim teacher and so101_real's RealOracle use);
        Dream z = the finger approach axis (the site's x), Dream x = the line the jaws close on, y = z cross x.
"""
from __future__ import annotations

import threading

import mujoco
import numpy as np

SIDES = ("left", "right")
# Dream's tool axes written in the gripperframe site's axes (columns x_d, y_d, z_d). See check() for the sign test.
SITE_TO_DREAM = np.array([[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0]])


def rot6(R):
    return np.r_[R[:, 0], R[:, 1]]


def from_rot6(v):
    v = np.asarray(v, dtype=np.float64)
    x = v[:3] / (np.linalg.norm(v[:3]) + 1e-12)
    y = v[3:6] - x * (x @ v[3:6])
    y /= np.linalg.norm(y) + 1e-12
    return np.stack([x, y, np.cross(x, y)], 1)


def _import_pi_sim():
    """pi_sim from C:\\robot\\pi. Its editable install points through a link in dzkch's profile that the robot
    account may not traverse, so that one finder is skipped and the copy beside this folder is used."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "pi"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
    from pi_sim import RobotEnv
    return RobotEnv


class Kinematics:
    def __init__(self):
        RobotEnv = _import_pi_sim()
        env = RobotEnv(task="pick_place", control_mode="joint", object_color="red", object_shape="cube", control_hz=50)
        env.reset(seed=0, options={"randomize": False})
        self.m = env.model
        self.d = mujoco.MjData(self.m)
        self.lock = threading.Lock()
        self.arm = {}
        for s in SIDES:
            a = env.robot[s]
            self.arm[s] = {"qpos": a.qpos_ids, "dof": a.dof_ids, "site": a.site_id,
                           "base": self.m.body(f"{s}_base").id,
                           "limits": self.m.jnt_range[a.joint_ids].copy()}
        self.env = env              # kept alive: freeing the sim off the thread that built it crashed on Windows
        mujoco.mj_kinematics(self.m, self.d)

    def _set(self, s, units5):
        self.d.qpos[self.arm[s]["qpos"]] = np.deg2rad(np.asarray(units5[:5], dtype=np.float64))

    def _pose(self, s):
        """(xyz in base, R_dream in base) for the current self.d."""
        a = self.arm[s]
        Rb, pb = self.d.xmat[a["base"]].reshape(3, 3), self.d.xpos[a["base"]]
        Rs, ps = self.d.site_xmat[a["site"]].reshape(3, 3), self.d.site_xpos[a["site"]]
        return Rb.T @ (ps - pb), Rb.T @ Rs @ SITE_TO_DREAM

    def fk(self, s, units5):
        with self.lock:
            self._set(s, units5)
            mujoco.mj_kinematics(self.m, self.d)
            xyz, R = self._pose(s)
            return xyz.copy(), rot6(R)

    def ik(self, s, xyz, r6, q0_units, iters=60, rot_w=0.08, limits_deg=None):
        """Joint angles (units, 5) that put the tool at (xyz, rot6) in the base frame, from q0. Returns (units5, err_m).
        limits_deg: (lo5, hi5) to clamp to instead of the sim model's joint ranges (the real servos' calibrated ranges
        are wider: the folded rest pose is outside the sim's)."""
        a = dict(self.arm[s])
        if limits_deg is not None:
            a["limits"] = np.deg2rad(np.stack([np.asarray(limits_deg[0][:5], float),
                                               np.asarray(limits_deg[1][:5], float)], 1))
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        with self.lock:
            mujoco.mj_kinematics(self.m, self.d)
            Rb, pb = self.d.xmat[a["base"]].reshape(3, 3).copy(), self.d.xpos[a["base"]].copy()
            p_t = pb + Rb @ np.asarray(xyz, dtype=np.float64)
            R_t = Rb @ from_rot6(r6) @ SITE_TO_DREAM.T          # target site rotation in the world
            q_t = np.zeros(4)
            mujoco.mju_mat2Quat(q_t, R_t.ravel())
            self._set(s, q0_units)
            q, rot_err = np.zeros(4), np.zeros(3)
            for _ in range(iters):
                mujoco.mj_kinematics(self.m, self.d)
                mujoco.mj_comPos(self.m, self.d)
                pos_err = p_t - self.d.site_xpos[a["site"]]
                mujoco.mju_mat2Quat(q, self.d.site_xmat[a["site"]])
                mujoco.mju_subQuat(rot_err, q_t, q)
                rot_world = self.d.site_xmat[a["site"]].reshape(3, 3) @ rot_err
                if np.linalg.norm(pos_err) < 0.0002 and np.linalg.norm(rot_world) < 0.002:
                    break
                err = np.r_[pos_err, rot_world * rot_w]
                mujoco.mj_jacSite(self.m, self.d, jp, jr, a["site"])
                J = np.vstack([jp[:, a["dof"]], jr[:, a["dof"]] * rot_w])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                qp = self.d.qpos[a["qpos"]] + np.clip(dq, -0.15, 0.15)
                self.d.qpos[a["qpos"]] = np.clip(qp, a["limits"][:, 0], a["limits"][:, 1])
            mujoco.mj_kinematics(self.m, self.d)
            err_m = float(np.linalg.norm(p_t - self.d.site_xpos[a["site"]]))
            return np.rad2deg(self.d.qpos[a["qpos"]]).copy(), err_m


def check(units12):
    """Prints each arm's pose for a 12-value pose and an IK round trip, for the sign test of SITE_TO_DREAM."""
    k = Kinematics()
    for i, s in enumerate(SIDES):
        u = np.asarray(units12[i * 6:i * 6 + 6], dtype=np.float64)
        xyz, r6 = k.fk(s, u)
        R = from_rot6(r6)
        q, err = k.ik(s, xyz, r6, u[:5] + np.array([8, -8, 8, -8, 8]))
        print(f"{s}: xyz {np.round(xyz, 3)}  x_d {np.round(R[:, 0], 2)}  y_d {np.round(R[:, 1], 2)}  "
              f"z_d (fingers) {np.round(R[:, 2], 2)}  | IK from +-8 deg: err {err * 1000:.2f} mm, "
              f"joints off {np.round(q - u[:5], 2)}")


if __name__ == "__main__":
    import sys
    from so101_real import SIM_HOME
    print("SIM_HOME (pi_sim start pose):")
    check(SIM_HOME)
    if len(sys.argv) == 13:
        print("given pose:")
        check([float(x) for x in sys.argv[1:]])
