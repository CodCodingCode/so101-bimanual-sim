"""Refit the servo dynamics in the form pi_sim actually builds: per joint kp, kv, force cap, friction loss, armature
and a first-order command lag (MuJoCo position `timeconst`), no pure delay. Starts from fit_dynamics.py's result.
Fit on the LEFT arm, held-out check on the RIGHT arm. Writes rig_tools/data/dyn/so101_dynamics.json in the
pi_sim/assets/so101_dynamics.json schema.

    sim/pi_sim_so101/.venv/bin/python rig_tools/fit_dynamics_tc.py [--evals 3000]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dyn"
INIT = DATA / "dyn_init_tc.json"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = ("kp", "kv", "forcerange", "frictionloss", "armature", "timeconst")


def write_init():
    r = json.loads((DATA / "dyn_fit.json").read_text())["params"]
    joints = {n: {"kp": r["kp"][i], "kv": r["kv"][i], "forcerange": r["frc"][i], "frictionloss": r["fric"][i],
                  "armature": r["arm"][i], "timeconst": 0.04} for i, n in enumerate(NAMES)}
    INIT.write_text(json.dumps({"joints": joints}, indent=1))


os.environ["PI_SIM_DYNAMICS"] = str(INIT)
if not INIT.exists():
    write_init()
sys.path.insert(0, str(HERE))
import fit_dynamics as FD          # noqa: E402  (imports pi_sim after PI_SIM_DYNAMICS is set)


class SimTC(FD.Sim):
    def read(self):
        p = super().read()
        p["tc"] = self.m.actuator_dynprm[self.act["left"], 0].copy()
        return p

    def apply(self, p):
        super().apply(p)
        for s in FD.SIDES:
            self.m.actuator_dynprm[self.act[s], 0] = p["tc"]


def unpack(x, nominal):
    x = np.asarray(x)
    return {k: nominal[k] * np.exp(x[i * 6:(i + 1) * 6]) for i, k in enumerate(("kp", "kv", "frc", "fric", "arm", "tc"))}


_S = None


def _init():
    global _S
    _S = SimTC()


def _loss(x):
    return FD.loss_from(_S.errors(_S.rollout(unpack(x, _S.nominal), 0.0)), "left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import cma
    S = SimTC()
    assert np.all(S.nominal["tc"] > 0), "model compiled without timeconst"
    e0 = S.errors(S.rollout(S.nominal, 0.0))
    print(f"start (previous fit, timeconst 40 ms, no delay): left {FD.loss_from(e0, 'left'):.3f}, "
          f"right {FD.loss_from(e0, 'right'):.3f}", flush=True)
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init)
    es = cma.CMAEvolutionStrategy(np.zeros(36), 0.3, {"maxfevals": a.evals, "verbose": -9, "seed": 4, "popsize": 16,
                                                      "bounds": [[-3] * 30 + [-2.5] * 6, [3] * 30 + [1.5] * 6]})
    best, t0 = (1e9, np.zeros(36)), time.time()
    while not es.stop():
        xs = es.ask()
        ls = pool.map(_loss, [np.asarray(x) for x in xs])
        for x, l in zip(xs, ls):
            if l < best[0]:
                best = (l, np.asarray(x).copy())
        es.tell(xs, ls)
        if es.countiter % 10 == 0:
            tc = unpack(best[1], S.nominal)["tc"] * 1000
            print(f"  it {es.countiter:3d} ({time.time() - t0:4.0f}s): left RMS {best[0]:.3f} | timeconst ms "
                  + " ".join(f"{v:.0f}" for v in tc), flush=True)
    pool.close()
    p = unpack(best[1], S.nominal)
    e1 = S.errors(S.rollout(p, 0.0))
    out = {"source": "2026-11-10 rig sysid (rig_tools/dyn_test.py + fit_dynamics_tc.py): 50 Hz scripted steps, sines, "
                     "gravity holds and gripper moves on both real arms; fitted on the left arm, right arm held out.",
           "rms_deg_stock": {"left": FD.loss_from(np.array(json.loads((DATA / 'dyn_fit.json').read_text())["rms_nominal"]["left"]
                                                            + [0] * 6), "left")},
           "rms_deg_fitted": {"left": FD.loss_from(e1, "left"), "right_heldout": FD.loss_from(e1, "right")},
           "per_joint_rms": {"left": e1[:6].round(3).tolist(), "right_heldout": e1[6:].round(3).tolist()},
           "joints": {n: {"kp": float(p["kp"][i]), "kv": float(p["kv"][i]), "forcerange": float(p["frc"][i]),
                          "frictionloss": float(p["fric"][i]), "armature": float(p["arm"][i]),
                          "timeconst": float(p["tc"][i])} for i, n in enumerate(NAMES)}}
    (DATA / "so101_dynamics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "joints"}, indent=1))
    for n, v in out["joints"].items():
        print(f"  {n:13s} " + " ".join(f"{k} {v[k]:.4g}" for k in KEYS))


if __name__ == "__main__":
    main()
