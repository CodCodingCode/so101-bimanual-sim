"""Global random search for the fixed top camera (independent of the builder's init): look-at parametrisation
(target on the table, distance, elevation, azimuth, roll, fovy), IoU on 32 even-episode frames at 160x120, then CMA
refinement of the best few at 320x240. Writes out/global_top.json (fit-json format, roll offset as given)."""
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
import arm as A  # noqa: E402

roll_off = float(sys.argv[1]) if len(sys.argv) > 1 else -90.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
sc = F.load_scene("legocup")
sc.CAMERAS = {"top": sc.CAMERAS["top"]}
sc.OBJ_TRIM = 0.15
fs = F.real_set(sc)
real = F.load_masks("legocup", "top")
ep = np.asarray(fs.labels["episode"]).astype(int)
area = real.reshape(fs.n, -1).mean(1)
ok = (area >= 0.004) & (ep % 2 == 0)
X = fs.state[:, :6] / 30.0
cand = np.nonzero(ok)[0]
idx = cand[F.fps_select(X[cand], 32)]
small = real[idx].reshape(len(idx), 120, 4, 160, 4).mean((2, 4)) >= 0.5
P = F.init_params(sc)
P["offsets_deg"] = [0.0, 0.0, 0.0, 0.0, roll_off]
rig = F.Rig(sc, P)
cid = rig.arm.cam_id["top"]
states = [fs.state[i] for i in idx]


def cam_from(v):
    tx, ty, d, el, az, rl, fov = v
    tgt = np.array([tx, ty, 0.0])
    e, a = math.radians(el), math.radians(az)
    pos = tgt + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return pos, A.look_quat(pos, tgt, rl), fov


def score_small(v):
    pos, quat, fov = cam_from(v)
    rig.arm.m.cam_pos[cid] = pos
    rig.arm.m.cam_quat[cid] = quat
    rig.arm.m.cam_fovy[cid] = fov
    out = []
    for st, r in zip(states, small):
        rig.arm.set_state(st)
        s = rig.arm.arm_mask("top", 160, 120)
        u = (s | r).sum()
        out.append((s & r).sum() / u if u else np.nan)
    v_ = np.sort(np.asarray(out)[np.isfinite(out)])
    return float(v_[int(0.15 * len(v_)):].mean()) if len(v_) else 0.0


# builder camera in this parametrisation (for reference)
g = P["cams"]["top"]["groups"]["all"]
R = A.mat_from_quat(g["quat"])
fwd = -R[:, 2]
t = -g["pos"][2] / fwd[2]
tgt = np.asarray(g["pos"]) + t * fwd
dvec = np.asarray(g["pos"]) - tgt
d = np.linalg.norm(dvec)
el = math.degrees(math.asin(dvec[2] / d))
az = math.degrees(math.atan2(dvec[1], dvec[0]))
v0 = np.array([tgt[0], tgt[1], d, el, az, 0.0, P["cams"]["top"]["fovy"]])
# roll: find the look_quat roll that reproduces the builder quat
best_r = min(np.arange(-30, 30.1, 0.5), key=lambda r: np.linalg.norm(A.mat_from_quat(cam_from(np.r_[v0[:5], r, v0[6]])[1]) - R))
v0[5] = best_r
s0 = score_small(v0)
print(f"roll offset {roll_off}: builder camera as look-at {np.round(v0, 3).tolist()} score {s0:.3f}", flush=True)
rng = np.random.default_rng(0)
lo = np.array([v0[0] - 0.08, v0[1] - 0.08, 0.30, 35.0, v0[4] - 30, -15, 25.0])
hi = np.array([v0[0] + 0.08, v0[1] + 0.08, 0.90, 88.0, v0[4] + 30, 15, 65.0])
t0 = time.time()
res = []
for k in range(N):
    v = lo + rng.random(7) * (hi - lo)
    res.append((score_small(v), v))
    if (k + 1) % 500 == 0:
        res.sort(key=lambda x: -x[0])
        res = res[:50]
        print(f"{k + 1} samples {time.time() - t0:.0f}s best {res[0][0]:.3f} {np.round(res[0][1], 3).tolist()}", flush=True)
res.sort(key=lambda x: -x[0])
# local refinement of the top 6 (Nelder-Mead-ish via CMA in look-at space)
import cma  # noqa: E402

top = []
for s, v in res[:6] + [(s0, v0)]:
    sc_ = np.array([0.01, 0.01, 0.03, 3.0, 3.0, 2.0, 2.0])
    es = cma.CMAEvolutionStrategy(np.zeros(7), 0.7, dict(maxfevals=160, verbose=-9, seed=1, popsize=10))
    best = (s, v)
    while not es.stop():
        xs = es.ask()
        fs_ = [-score_small(v + sc_ * x) for x in xs]
        es.tell(xs, fs_)
        j = int(np.argmin(fs_))
        if -fs_[j] > best[0]:
            best = (-fs_[j], v + sc_ * xs[j])
    top.append(best)
    print(f"refined {s:.3f} -> {best[0]:.3f} {np.round(best[1], 3).tolist()}", flush=True)
top.sort(key=lambda x: -x[0])
s, v = top[0]
pos, quat, fov = cam_from(v)
P["cams"]["top"]["groups"]["all"] = dict(pos=[float(x) for x in pos], quat=[float(x) for x in quat])
P["cams"]["top"]["fovy"] = float(fov)
(HERE / "out" / f"global_top_{int(roll_off)}.json").write_text(json.dumps(dict(params=P, score=s, lookat=v.tolist()), indent=1))
print(f"best {s:.3f} lookat {np.round(v, 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
