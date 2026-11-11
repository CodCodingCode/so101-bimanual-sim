"""Fit the sim's servo dynamics to the real arms (system identification).

Replays the exact command stream from dyn_test.py (50 Hz, both arms) in the pi_sim MuJoCo model and fits, per joint
type (shared by both arms): position gain kp, damping gain kv, torque cap, joint friction loss, armature, plus one
command delay (fractional ticks, all joints). The fit uses the LEFT arm only; the RIGHT arm is the held-out check.

    sim/pi_sim_so101/.venv/bin/python rig_tools/fit_dynamics.py [--evals 3000]

Writes rig_tools/data/dyn/dyn_fit.json (parameters + RMS errors before/after, per joint, both arms) and
dyn_fit.png-style traces as dyn_fit.jpg.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi_sim_so101"))
import mujoco                      # noqa: E402
from pi_sim.env import RobotEnv    # noqa: E402

DATA = HERE / "data" / "dyn"
SIDES = ("left", "right")
JN = ("pan", "lift", "elbow", "wflex", "roll", "grip")
SUB = 10                           # 0.02 s control tick / 0.002 s physics step
GRIP_W = 0.3                       # gripper error weight (its units are 0..100 %, not degrees)


class Sim:
    def __init__(self):
        self.env = RobotEnv(task="pick_place", control_mode="joint")
        self.env.reset(seed=0, options={"randomize": False})
        m = self.m = self.env.model
        self.d = mujoco.MjData(m)
        self.act = {s: list(self.env.robot[s].actuator_ids) + [self.env.robot[s].gripper_actuator] for s in SIDES}
        self.jnt = {s: list(self.env.robot[s].joint_ids) + [m.joint(f"{s}_gripper").id] for s in SIDES}
        self.qadr = {s: [m.jnt_qposadr[j] for j in self.jnt[s]] for s in SIDES}
        self.dadr = {s: [m.jnt_dofadr[j] for j in self.jnt[s]] for s in SIDES}
        self.glo, self.ghi = m.actuator_ctrlrange[self.act["left"][5]]
        self.nominal = self.read()
        z = np.load(DATA / "dyn.npz")
        self.cmd, self.meas = z["cmd"], z["meas"]

    def read(self):
        m, a, j = self.m, self.act["left"], self.dadr["left"]
        return {"kp": m.actuator_gainprm[a, 0].copy(), "kv": -m.actuator_biasprm[a, 2].copy(),
                "frc": m.actuator_forcerange[a, 1].copy(), "fric": m.dof_frictionloss[j].copy(),
                "arm": m.dof_armature[j].copy()}

    def apply(self, p):
        m = self.m
        for s in SIDES:
            a, j = self.act[s], self.dadr[s]
            m.actuator_gainprm[a, 0] = p["kp"]
            m.actuator_biasprm[a, 1] = -p["kp"]
            m.actuator_biasprm[a, 2] = -p["kv"]
            m.actuator_forcerange[a, 0], m.actuator_forcerange[a, 1] = -p["frc"], p["frc"]
            m.dof_frictionloss[j] = p["fric"]
            m.dof_armature[j] = p["arm"]

    def to_q(self, u12):
        """rig units (deg, gripper %) -> {side: 6 joint values in rad}."""
        out = {}
        for i, s in enumerate(SIDES):
            u = u12[i * 6:(i + 1) * 6]
            out[s] = np.r_[np.deg2rad(u[:5]), self.glo + np.clip(u[5], 0, 100) / 100 * (self.ghi - self.glo)]
        return out

    def to_units(self, q6):
        return np.r_[np.rad2deg(q6[:5]), (q6[5] - self.glo) / (self.ghi - self.glo) * 100]

    def rollout(self, p, delay):
        """Simulate the logged command stream; returns (N,12) sim joint values in rig units."""
        self.apply(p)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.env.data.qpos
        d.joint("cube_joint").qpos[:3] = [2, 2, 0]
        q0 = self.to_q(self.meas[0])
        for s in SIDES:
            d.qpos[self.qadr[s]] = q0[s]
        mujoco.mj_forward(m, d)
        n = len(self.cmd)
        idx = np.arange(n) - delay
        lo = np.clip(np.floor(idx).astype(int), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        f = (idx - np.floor(idx))[:, None]
        cmd = self.cmd[lo] * (1 - f) + self.cmd[hi] * f
        out = np.zeros((n, 12))
        for k in range(n):
            qc = self.to_q(cmd[k])
            for s in SIDES:
                d.ctrl[self.act[s]] = qc[s]
            mujoco.mj_step(m, d, nstep=SUB)
            for i, s in enumerate(SIDES):
                out[k, i * 6:(i + 1) * 6] = self.to_units(d.qpos[self.qadr[s]])
        return out

    def errors(self, sim):
        e = sim - self.meas
        return np.sqrt((e ** 2).mean(axis=0))          # (12,) RMS per joint


def unpack(x, nominal):
    """x: 6 x (log kp, log kv, log force, log friction, log armature) scales + delay."""
    x = np.asarray(x)
    p = {}
    for k, key in enumerate(("kp", "kv", "frc", "fric", "arm")):
        p[key] = nominal[key] * np.exp(x[k * 6:(k + 1) * 6])
    return p, float(np.clip(x[30] * 2.0 + 3.0, 0.0, 12.0))     # delay in ticks


def loss_from(err12, side="left"):
    o = 0 if side == "left" else 6
    w = np.r_[np.ones(5), GRIP_W]
    return float(np.sqrt(np.mean((err12[o:o + 6] * w) ** 2)))


_S = None


def _init():
    global _S
    _S = Sim()


def _loss(x):
    p, dl = unpack(x, _S.nominal)
    return loss_from(_S.errors(_S.rollout(p, dl)), "left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import cma
    S = Sim()
    print("nominal sim servo:", {k: np.round(v, 4).tolist() for k, v in S.nominal.items()}, flush=True)
    t0 = time.time()
    e0 = S.errors(S.rollout(S.nominal, 0.0))
    print(f"one rollout {time.time() - t0:.2f}s | nominal sim, no delay: RMS left {loss_from(e0, 'left'):.2f}, "
          f"right {loss_from(e0, 'right'):.2f}", flush=True)
    x0 = np.zeros(31)
    x0[30] = 0.0                                               # delay 3 ticks to start
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init)
    es = cma.CMAEvolutionStrategy(x0, 0.5, {"maxfevals": a.evals, "verbose": -9, "seed": 3, "popsize": 16,
                                            "bounds": [[-4] * 30 + [-1.5], [4] * 30 + [4.5]]})
    best, t0 = (1e9, x0), time.time()
    while not es.stop():
        xs = es.ask()
        ls = pool.map(_loss, [np.asarray(x) for x in xs])
        for x, l in zip(xs, ls):
            if l < best[0]:
                best = (l, np.asarray(x).copy())
        es.tell(xs, ls)
        if es.countiter % 10 == 0:
            _, dl = unpack(best[1], S.nominal)
            print(f"  it {es.countiter:3d} ({time.time() - t0:4.0f}s): left RMS {best[0]:.3f} deg, delay {dl * 20:.0f} ms", flush=True)
    pool.close()
    p, dl = unpack(best[1], S.nominal)
    sim = S.rollout(p, dl)
    e1 = S.errors(sim)
    res = {"delay_ms": dl * 20, "params": {k: np.round(v, 5).tolist() for k, v in p.items()},
           "nominal": {k: np.round(v, 5).tolist() for k, v in S.nominal.items()}, "joints": list(JN),
           "rms_nominal": {"left": e0[:6].round(3).tolist(), "right": e0[6:].round(3).tolist()},
           "rms_fitted": {"left": e1[:6].round(3).tolist(), "right": e1[6:].round(3).tolist()},
           "loss_nominal": {"left": loss_from(e0, "left"), "right_heldout": loss_from(e0, "right")},
           "loss_fitted": {"left": loss_from(e1, "left"), "right_heldout": loss_from(e1, "right")}, "x": best[1].tolist()}
    (DATA / "dyn_fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "x"}, indent=1))
    np.save(DATA / "dyn_fit_sim.npy", sim)
    np.save(DATA / "dyn_nominal_sim.npy", S.rollout(S.nominal, 0.0))


if __name__ == "__main__":
    main()
