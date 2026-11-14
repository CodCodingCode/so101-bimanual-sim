"""Exact save / restore of a ClampRLEnv (physics + every piece of Python-side episode state), so a teacher can be rolled
forward from the student's state to label it and the episode then continues exactly as if nothing happened.

    snap = snapshot(env)
    ... env.step(...) as often as you like ...
    restore(env, snap)          # bit-identical continuation (checked by self_test)

Used by teacher_rollout() to produce coherent action-chunk labels: the teacher's own next K steps from this state.

    python squeeze/sim_snapshot.py      # self-test
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

_SKIP_TYPES = (mujoco.MjModel, mujoco.MjData)
_SKIP_KEYS = {"base", "sim", "model", "data", "_ik_data", "robot", "_renderers", "_camera_plans", "camera_specs",
              "action_space", "observation_space", "_baseline", "_arm_geoms", "_side_of", "_pads", "visual_randomization"}


def _objects(env):
    return (env, env.base, env.base.sim)


def snapshot(env):
    m, d, sim = env.model, env.data, env.base.sim
    phys = mujoco.MjData(m)                  # the whole MjData (state AND derived quantities, which may be one substep
    mujoco.mj_copyData(phys, m, d)           # stale right after reset), so the continuation is bit-identical
    py = []
    for obj in _objects(env):
        saved = {}
        for k, v in obj.__dict__.items():
            if k in _SKIP_KEYS or isinstance(v, _SKIP_TYPES) or callable(v) and not hasattr(v, "__dict__"):
                continue
            saved[k] = copy.deepcopy(v)
        py.append(saved)
    ik = mujoco.MjData(m)
    mujoco.mj_copyData(ik, m, sim._ik_data)
    return phys, py, ik


def restore(env, snap):
    phys, py, ik = snap
    m, d, sim = env.model, env.data, env.base.sim
    mujoco.mj_copyData(d, m, phys)
    for obj, saved in zip(_objects(env), py):
        for k, v in saved.items():
            setattr(obj, k, copy.deepcopy(v))
    mujoco.mj_copyData(sim._ik_data, m, ik)


def teacher_rollout(env, teacher_act, horizon, limiter=None):
    """Roll the teacher forward `horizon` steps from the current state and come back. Returns the commanded tool points
    (horizon, 2, 3) and commanded joint targets (horizon, 10, degrees) after each step; if the episode ends early the
    last values are held. teacher_act(env, obs) -> action (6,); limiter(env, action) -> action (e.g. squeeze limiter)."""
    sim, d = env.base.sim, env.data
    a_ids = [sim.robot[s].actuator_ids for s in ("left", "right")]
    snap = snapshot(env)
    obs = env._observe()
    ee, joints = [], []
    done = False
    for _ in range(horizon):
        if not done:
            act = teacher_act(env, obs)
            if limiter is not None:
                act = limiter(env, act)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        ee.append(sim._ee_targets.copy())
        joints.append(np.rad2deg(np.r_[d.ctrl[a_ids[0]], d.ctrl[a_ids[1]]]))
    restore(env, snap)
    return np.asarray(ee), np.asarray(joints)


def self_test(seed=7, every=2, horizon=8):
    import os
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent)]
    os.environ.setdefault("SQUEEZE_TASK", "clamp")
    os.environ.setdefault("CLAMP_RECOVERY", "1")
    import train as T
    from clamp_rl import ClampRLEnv
    td = here / "runs_clamp_v3" / "grip_teacher"
    teacher = T.BCPPO.load(td / "best.zip", device="cpu")
    ts = np.load(td / "obs_stats.npz")
    act = lambda env, o: np.asarray(teacher.predict(np.clip((o - ts["mean"]) / np.sqrt(ts["var"] + 1e-8), -10, 10)
                                                    .astype(np.float32), deterministic=True)[0], np.float32)

    def run(with_rollouts):
        env = ClampRLEnv(seed=seed)
        o, _ = env.reset(seed=seed)
        trace, rolls = [], []
        for t in range(env.base.max_steps):
            if with_rollouts and t % every == 0:
                rolls.append(teacher_rollout(env, act, horizon))
                if t % (every * 20) == 0:              # the same rollout twice must agree too
                    again = teacher_rollout(env, act, horizon)
                    assert np.array_equal(again[0], rolls[-1][0]) and np.array_equal(again[1], rolls[-1][1]), t
            o, r, te, tr, info = env.step(act(env, o))
            trace.append(np.r_[o, r, env.data.qpos])
            if te or tr:
                break
        return np.asarray(trace), info, rolls
    a, ia, _ = run(False)
    b, ib, rolls = run(True)
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"plain episode {len(a)} steps, with {len(rolls)} teacher rollouts {len(b)} steps; bit-identical: {same}; "
          f"success {ia['is_success']} / {ib['is_success']}, clean {ia['bad_free_steps'] == 0} / {ib['bad_free_steps'] == 0}")
    if not same and a.shape == b.shape:
        i = int(np.argmax(np.abs(a - b).max(1) > 0))
        print("first difference at step", i, "max", float(np.abs(a - b).max()))
    return same


if __name__ == "__main__":
    import sys
    ok = all(self_test(seed=s) for s in (7, 8))
    sys.exit(0 if ok else 1)
