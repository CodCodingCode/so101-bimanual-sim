"""Command safety filter for joint-target policies (the SmolVLA): never SEND a pose the real guard's commanded-pose
check would reject; send the farthest point toward it that passes, or hold the last command.

Why (2026-11-13): with rig_tools/safety.py's guard checked every tick in sim (eval_smolvla.py, CLAMP_GUARD=1
CLAMP_GUARD_CMD=1), the clamp_mix 5k SmolVLA was stopped in 12 of its first 14 push + rotation runs, almost all
"arm-arm 0.0 mm (cmd)": during the squeeze its predicted joint targets put the two grippers INTO each other (the
cube is between them, so nothing actually touches; the measured pose stayed clear). The teacher never does this:
its rig rules keep >= 1 mm of commanded gap (clamp_wide._squeeze_limit). This is the same rule for joint commands.
The measured-pose guard still stops the run on any real approach.

    from vla_safety import CmdFilter
    filt = CmdFilter(env)            # after env.reset(): uses env._guard (CLAMP_GUARD=1) and the detected box
    q_send, kept = filt(q_prev, q_new)   # dicts side -> 6 values (5 joints deg, gripper %); kept = fraction 0..1
"""
from __future__ import annotations

import numpy as np

BUFFER = {"arm-arm": 0.0005}        # m added to the guard's cmd margins (arm-arm 0.5 -> 1.0 mm, like the teacher)


class CmdFilter:
    """relax=True (2026-11-14, opt-in): when a command breaks a margin, first pull it back toward the MEASURED pose
    (q = new + b * (meas - new), smallest b that passes) instead of shortening the step from the last command. Why: the
    ACT (validated rig, 20k / 30k) commands its squeeze slightly deeper than the 1 mm commanded-gap rule while it
    carries the cube; the segment rule then rejects the WHOLE step (squeeze kept median 0.00, 75 % of commands cut),
    so the arms froze holding the cube and never delivered it. During a squeeze the command and the measured pose
    differ mostly along the squeeze (the cube holds the grippers apart), so pulling toward the measured pose removes
    the too-deep squeeze and keeps the carry. Same safety rule: whatever is sent passes every cmd margin (+ buffer);
    if the measured pose itself does not, it falls back to the segment rule. rate_deg: per-tick joint step limit
    from the last command (the sim's joint_rate_limit * dt), re-checked after clipping."""

    def __init__(self, env, iters=7, relax=False, rate_deg=None):
        self.env, self.iters, self.relax, self.rate_deg = env, iters, relax, rate_deg
        self.g, self.SF = env._guard, env._SF
        env._guard_check()                                   # sets the guard's box from the detection, as the rig does
        self.margin = {k: m + BUFFER.get(k, 0.0) for k, m in self.SF.MARGIN["cmd"].items()}
        self.n_cut, self.n_hold, self.n, self.n_relax = 0, 0, 0, 0
        self.cut_kinds = {}                                  # diagnostics: which margin each rejected command broke

    def meas(self):
        """The measured pose in the guard's units (5 joints deg, gripper %), as clamp_rl._guard_check builds it."""
        e, d = self.env, self.env.data
        q = {}
        for s in e._gg:
            adr, lo, hi = e._gg[s]
            q[s] = np.r_[np.rad2deg(d.qpos[e._gq[s]]), 100.0 * (d.qpos[adr] - lo) / (hi - lo)]
        return q

    def _relaxed(self, prev, new):
        meas = self.meas()
        if not self.ok(meas):
            return None
        lo, hi = 0.0, 1.0                                    # b: 0 = the policy's command, 1 = the measured pose
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            if self.ok({s: new[s] + mid * (meas[s] - new[s]) for s in new}):
                hi = mid
            else:
                lo = mid
        out = {s: new[s] + hi * (meas[s] - new[s]) for s in new}
        if self.rate_deg is not None:
            out = {s: np.r_[np.clip(out[s][:5], prev[s][:5] - self.rate_deg, prev[s][:5] + self.rate_deg), out[s][5:]]
                   for s in out}
            if not self.ok(out):
                return None
        return out, 1.0 - hi

    def bad(self, pose):
        dist = self.g.distances(pose)
        return [k for k, m in self.margin.items() if k in dist and dist[k][0] < m]

    def ok(self, pose):
        return not self.bad(pose)

    def __call__(self, prev, new):
        self.n += 1
        broke = self.bad(new)
        if not broke:
            return new, 1.0
        for k in broke:
            self.cut_kinds[k] = self.cut_kinds.get(k, 0) + 1
        if self.relax:
            out = self._relaxed(prev, new)
            if out is not None:
                self.n_relax += 1
                return out
        if not self.ok(prev):                                # already inside a margin: hold (the guard decides)
            self.n_hold += 1
            return prev, 0.0
        lo, hi = 0.0, 1.0
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            if self.ok({s: prev[s] + mid * (new[s] - prev[s]) for s in new}):
                lo = mid
            else:
                hi = mid
        self.n_cut += 1
        return {s: prev[s] + lo * (new[s] - prev[s]) for s in new}, lo
