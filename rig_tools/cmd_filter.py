"""Command filter for joint-target policies on the real rig (SmolVLA): never SEND a pose that the guard's commanded-pose
check would reject. Instead send the farthest point toward it that passes, or hold the last command.

Rig-side port of sim-74's squeeze/vla_safety.CmdFilter (2026-11-13). In sim, with the guard checked every tick, the
SmolVLA was stopped in 12 of 14 runs at "arm-arm 0.0 mm (cmd)": its squeeze targets put the grippers INTO each other
(the cube is between them, so the measured pose stays clear). The teacher never commands < 1 mm of gap
(clamp_wide._squeeze_limit), so the arm-arm margin here is the guard's 0.5 mm + 0.5 mm = 1.0 mm.
The measured-pose guard (safety.Guard.check_pose(q, "meas")) still stops a run on any real approach.

    filt = CmdFilter(guard)                  # a safety.Guard with the box set (set_box(xy, yaw)) when known
    q_send, kept = filt(q_prev, q_new)       # dicts side -> 6 values (5 joints deg, gripper %); kept = 0..1
"""
from __future__ import annotations

import numpy as np

import safety as SF

BUFFER = {"arm-arm": 0.0005}


# ---- the guard's GJK distance (safety.convex_distance), with the 1-3 unknown simplex solves in closed form instead of
# numpy lstsq (97% of the filter's time was numpy call overhead on 1x1..3x3 systems). Checked against
# safety.convex_distance in the filter's self-test at start-up. safety.py itself is untouched: the guard still checks
# every commanded and measured pose with its own code.
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _solve(M, r):
    n = len(r)
    if n == 1:
        return None if M[0][0] <= 1e-30 else [r[0] / M[0][0]]
    if n == 2:
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if abs(det) <= 1e-12 * M[0][0] * M[1][1]:
            return None
        return [(r[0] * M[1][1] - M[0][1] * r[1]) / det, (M[0][0] * r[1] - r[0] * M[1][0]) / det]
    (a, b, c), (d, e, f), (g, h, i) = M
    co = (e * i - f * h, f * g - d * i, d * h - e * g)
    det = a * co[0] + b * co[1] + c * co[2]
    if abs(det) <= 1e-12 * a * e * i:
        return None
    inv = ((co[0], c * h - b * i, b * f - c * e), (co[1], a * i - c * g, c * d - a * f), (co[2], b * g - a * h, a * e - b * d))
    return [sum(inv[k][j] * r[j] for j in range(3)) / det for k in range(3)]


def _closest_on_simplex(S):
    best = None
    for sub in SF._SUBSETS[len(S)]:
        p0 = S[sub[0]]
        if len(sub) == 1:
            x = p0
        else:
            A = [(S[j][0] - p0[0], S[j][1] - p0[1], S[j][2] - p0[2]) for j in sub[1:]]
            t = _solve([[_dot(u, v) for v in A] for u in A], [-_dot(u, p0) for u in A])
            if t is None or min(t) < -1e-12 or sum(t) > 1 + 1e-12:
                continue                          # singular: covered by its sub-faces; else the closest point is outside
            x = tuple(p0[c] + sum(t[k] * A[k][c] for k in range(len(A))) for c in range(3))
        n = _dot(x, x)
        if best is None or n < best[0]:
            best = (n, x, sub)
    return best[1], [S[j] for j in best[2]]


def convex_distance(XA, XB, cap):
    """safety.convex_distance, same algorithm and stopping rules."""
    return convex_distance_dir(XA, XB, cap)[0]


def convex_distance_dir(XA, XB, cap):
    """(distance, v): v = the last GJK direction; -v/|v| separates A from B by >= the bound GJK proved."""
    v = XA.mean(0) - XB.mean(0)
    if not v.any():
        v = np.array([1.0, 0.0, 0.0])
    w0 = XA[np.argmax(XA @ -v)] - XB[np.argmax(XB @ v)]
    S, v = [tuple(w0)], tuple(w0)
    for _ in range(64):
        vn = _dot(v, v) ** 0.5
        if vn < 1e-9:
            return 0.0, v
        va = np.array(v)
        w = tuple(XA[np.argmax(XA @ -va)] - XB[np.argmax(XB @ va)])
        lb = _dot(v, w) / vn
        if lb >= cap:
            return cap, v
        if vn - lb <= 1e-7 + 1e-6 * vn:
            return min(vn, cap), v
        v, S = _closest_on_simplex(S + [w])
    return min(_dot(v, v) ** 0.5, cap), v


def min_dist(guard, A, B, cap):
    """Guard._min_dist (bounding spheres, nearest pairs first) with the fast distance."""
    WA = [guard._world(p) for p in A]
    WB = [guard._world(p) for p in B]
    cand = []
    for na, XA, ca, ra in WA:
        for nb, XB, cb, rb in WB:
            lb = float(np.linalg.norm(ca - cb)) - ra - rb
            if lb < cap:
                cand.append((lb, na, XA, nb, XB))
    best, pair = cap, None
    for lb, na, XA, nb, XB in sorted(cand, key=lambda x: x[0]):
        if lb >= best:
            break
        dist = convex_distance(XA, XB, best)
        if dist < best:
            best, pair = dist, (na, nb)
    return best, pair


def self_test(n=300, seed=0):
    """The fast distance against safety.convex_distance on random convex point sets near contact; raises if off."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        A = rng.normal(size=(rng.integers(4, 40), 3)) * rng.uniform(0.005, 0.03, 3)
        B = rng.normal(size=(rng.integers(4, 40), 3)) * rng.uniform(0.005, 0.03, 3) + rng.normal(size=3) * 0.03
        cap = rng.choice([0.001, 0.006, 0.05])
        worst = max(worst, abs(convex_distance(A, B, cap) - SF.convex_distance(A, B, cap)))
    if worst > 2e-6:
        raise SF.SafetyStop(f"cmd filter distance self-test failed ({worst * 1000:.4f} mm off the guard's): refusing to run")
    return worst


class CmdFilter:
    """Speed (2026-11-14; 47-73 ms -> see the benchmark in the docstring of `__call__`): every check is capped AT ITS
    MARGIN (the bounding-sphere test drops almost every link pair and GJK stops at the first proof of clearance; the
    guard's distances() caps at 5 cm), and the bisection tests only the LINK PAIRS that are too close at the new pose.
    The chosen pose always gets the full check over every kind; a pair failing there joins the tested set and the
    bisection continues below it (3 rounds, then hold the last command). The last sent pose is remembered as passing."""

    def __init__(self, guard, iters=7):
        self.g, self.iters = guard, iters
        self.margin = {k: m + BUFFER.get(k, 0.0) for k, m in SF.MARGIN["cmd"].items()}
        arm = guard.arm
        mount = {s: [p for p in arm[s] if p[1].endswith("camera_mount")] for s in SF.SIDES}
        rest = {s: [p for p in arm[s] if not p[1].endswith("camera_mount")] for s in SF.SIDES}
        # (kind, [(A, B), ...]) in the same groupings as Guard.distances
        self.pairs = [("arm-arm", [(rest["left"], rest["right"])]),
                      ("camera-mount", [(mount["left"], arm["right"]), (rest["left"], mount["right"])]),
                      ("arm-box", [(arm[s], guard.box) for s in SF.SIDES]),
                      ("arm-camera", [(arm[s], guard.post) for s in SF.SIDES])]
        self.part = {p[1]: p for s in SF.SIDES for p in arm[s]}
        self.part.update({p[1]: p for p in guard.box + guard.post})
        self._last = None                                  # the last pose returned (known to pass)
        self._axis, self._w = {}, {}                       # separating direction per link pair; world points per pose
        self.n = self.n_cut = self.n_hold = 0
        self.self_test_mm = round(self_test() * 1000, 5)

    # Each link pair is cleared by the cheapest proof that works, all exact (a direction u with
    # min_B(u.b) - max_A(u.a) >= margin PROVES the hulls are at least that far apart): bounding spheres; the direction
    # that separated this pair last time (poses change little tick to tick and within the bisection); the centre-to-
    # centre direction; only then GJK (whose final direction is remembered for next time).
    def _set_pose(self, pose):
        self.g._pose(pose)
        self._w = {}

    def _world(self, part):
        w = self._w.get(part[1])
        if w is None:
            w = self._w[part[1]] = self.g._world(part)
        return w

    def _pair_ok(self, A, B, cap):
        na, XA, ca, ra = self._world(A)
        nb, XB, cb, rb = self._world(B)
        dc = cb - ca
        nd = float(np.sqrt(dc @ dc))
        if nd - ra - rb >= cap:
            return True
        u = self._axis.get((na, nb))
        if u is not None and float((XB @ u).min() - (XA @ u).max()) >= cap:
            return True
        if nd > 1e-9:
            u = dc / nd
            if float((XB @ u).min() - (XA @ u).max()) >= cap:
                self._axis[(na, nb)] = u
                return True
        dist, v = convex_distance_dir(XA, XB, cap)
        if dist >= cap:
            v = np.asarray(v, float)
            self._axis[(na, nb)] = -v / max(float(np.sqrt(v @ v)), 1e-12)
            return True
        return False

    def too_close(self, pose, first=True):
        """[(margin, part A, part B)] for every link pair that violates its commanded-pose margin (first=True: the
        first one found)."""
        self._set_pose(pose)
        bad = []
        for kind, groups in self.pairs:
            if kind not in self.margin or (kind == "arm-box" and not self.g.has_box):
                continue
            cap = self.margin[kind]
            for As, Bs in groups:
                for A in As:
                    for B in Bs:
                        if not self._pair_ok(A, B, cap):
                            bad.append((cap, A, B))
                            if first:
                                return bad
        return bad

    def ok(self, pose):
        return not self.too_close(pose)

    def _pairs_ok(self, pose, pairs):
        self._set_pose(pose)
        return all(self._pair_ok(A, B, cap) for cap, A, B in pairs)

    def _same(self, a, b):
        return b is not None and all(np.array_equal(a[s], b[s]) for s in a)

    def __call__(self, prev, new):
        """Benchmark 2026-11-14 (Mac under load, 444 logged real squeeze transitions incl. x4/x8/x20 overshoots, box
        set): ticks that need a cut 76 ms median (max 132) -> 7 ms (max 27); other ticks < 1 ms; ok() agrees with the
        guard's distances on 4440/4440 poses; every output passes guard.check_pose(.., "cmd")."""
        self.n += 1
        prev = {s: np.asarray(v, float) for s, v in prev.items()}
        new = {s: np.asarray(v, float) for s, v in new.items()}
        pairs = self.too_close(new, first=False)
        if not pairs:
            self._last = new
            return new, 1.0
        if not self._same(prev, self._last) and not self.ok(prev):
            self.n_hold += 1                            # already inside a margin: hold, the guard decides
            self._last = None
            return prev, 0.0
        lerp = lambda t: {s: prev[s] + t * (new[s] - prev[s]) for s in new}
        lo, hi = 0.0, 1.0
        for _ in range(3):
            for _ in range(self.iters):
                mid = 0.5 * (lo + hi)
                if self._pairs_ok(lerp(mid), pairs):
                    lo = mid
                else:
                    hi = mid
            more = self.too_close(lerp(lo), first=False) if lo > 0 else []
            if not more:
                break
            pairs, hi, lo = pairs + more, lo, 0.0       # another pair too close below the cut: search below it
        else:
            lo = 0.0
        self.n_cut += 1
        out = lerp(lo)
        self._last = out
        return out, lo
