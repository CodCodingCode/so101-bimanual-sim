"""Per-frame brick state for the pickplace twin: 'table' (episode start pose), 'held' (between the jaw tips, follows
the twin gripper), 'plate' (on the box floor). Same method as ricky_objects/block_state.py: Grounding-DINO + SAM
brick detections in every frame of both cameras vs the three hypotheses projected through the frozen cameras,
monotone dynamic programme per episode, an open gripper (reading < 3 %) cannot hold. Writes work/block_state.json."""
import json, pathlib, sys
import mujoco
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camgeo, objfit  # noqa
import build_pickplace as B  # noqa

T = B.make(colours={})
vo = B.view_of(T)
H, W = T.H, T.W
z = np.load(B.WORK / "seg_brick_all.npz", allow_pickle=True)
det = {}
for k in range(len(z["frame"])):
    m = np.unpackbits(z["bits"][k], axis=-1)[..., :W].astype(bool)
    det[(int(z["frame"][k]), str(z["cam"][k]))] = (float(z["score"][k]), np.argwhere(m).mean(0)[::-1] if m.any() else None, int(m.sum()))
first = [int(np.flatnonzero(B.EP == e)[0]) for e in np.unique(B.EP)]
AREA = {c: float(np.median([det[(i, c)][2] for i in first if det[(i, c)][2] > 0])) for c in ("top", "side")}
for (i, c), (s_, cen, a) in list(det.items()):
    if cen is not None and not (0.3 * AREA[c] <= a <= 3.0 * AREA[c]):
        det[(i, c)] = (0.0, None, a)
print("brick area medians", AREA)
P = {v: T.F.p[v] for v in T.F.views}
zo = B.OBJ["z_objects"]
hx, hy, hz = B.OBJ["shapes"]["brick"]
sid = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
bid_g = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_BODY, "arm_gripper")
bid_m = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_BODY, "arm_moving_jaw_so101_v1")
gm = [g for g in range(T.m.ngeom) if T.m.geom_bodyid[g] == bid_m and T.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and T.m.geom_group[g] == 2][0]
mid = T.m.geom_dataid[gm]
MV = T.m.mesh_vert[T.m.mesh_vertadr[mid]:T.m.mesh_vertadr[mid] + T.m.mesh_vertnum[mid]].copy()
n = T.F.fs.n
tcp = np.zeros((n, 3))
for i in range(n):
    q = T.F.q(i)
    for a, x in zip(T.qadr, q):
        T.d.qpos[a] = x
    mujoco.mj_kinematics(T.m, T.d)
    tcp[i] = T.d.site_xpos[sid]


def jaw_offset(i):
    q = T.F.q(i)
    for a, x in zip(T.qadr, q):
        T.d.qpos[a] = x
    mujoco.mj_kinematics(T.m, T.d)
    site = T.d.site_xpos[sid].copy()
    wrist = T.d.xpos[bid_g].copy()
    dirv = (site - wrist) / np.linalg.norm(site - wrist)
    w = MV @ T.d.geom_xmat[gm].reshape(3, 3).T + T.d.geom_xpos[gm]
    tip_m = w[np.argmax((w - wrist) @ dirv)]
    c = 0.5 * (site + tip_m) - dirv * max(hy, hz)
    return T.d.site_xmat[sid].reshape(3, 3).T @ (c - site)


def proj(v, X):
    u, w, _ = camgeo.project(P[v], np.atleast_2d(X), H, W)
    return np.array([u[0], w[0]])


def in_poly(pt, poly):
    x, y = pt
    inside = False
    for k in range(len(poly)):
        x1, y1 = poly[k]; x2, y2 = poly[(k + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


SCALE = 40.0
states = ("table", "held", "plate")
out = {}
for e in np.unique(B.EP):
    idx = np.flatnonzero(B.EP == e)
    idx = idx[np.argsort(B.LAB["t"][idx])]
    ep = B.OBJ["episodes"][str(int(e))]
    kx, ky, _, kz = ep["brick"]["pose"]
    bx, by, byaw = ep["rim"]["pose"]
    sx, sy = B.OBJ["shapes"]["rim"]
    c_, s_ = np.cos(byaw), np.sin(byaw)
    corners = [(bx + c_ * a - s_ * b, by + s_ * a + c_ * b, zo + 0.01) for a, b in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
    C = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        for c in ("top", "side"):
            s, cen, npx = det[(i, c)]
            v = vo[c][int(e)]
            if cen is None or s < 0.3:
                C[k] += [0.8, 0.5, 0.5]
                continue
            d_tab = np.linalg.norm(cen - proj(v, [kx, ky, kz]))
            d_tcp = np.linalg.norm(cen - proj(v, tcp[i]))
            poly = [proj(v, X) for X in corners]
            pc = np.mean(poly, 0)
            d_box = 0.0 if in_poly(cen, poly) else min(np.linalg.norm(cen - pc), 3 * SCALE)
            C[k] += np.minimum([d_tab, d_tcp, d_box], 3 * SCALE) / SCALE
        C[k, 1] += 1.0 if B.LAB["gripper"][i] < 3 else (-0.2 if B.LAB["gripper"][i] >= 5 else 0.0)
    C[0, 1:] += 5.0
    D = np.full((len(idx), 3), np.inf); bp = np.zeros((len(idx), 3), int)
    D[0] = C[0]
    for k in range(1, len(idx)):
        for s in range(3):
            j = int(np.argmin(D[k - 1, :s + 1]))
            D[k, s], bp[k, s] = D[k - 1, j] + C[k, s], j
    s = int(np.argmin(D[-1])); seq = [s]
    for k in range(len(idx) - 1, 0, -1):
        s = bp[k, s]; seq.append(s)
    seq = seq[::-1]
    for k, i in enumerate(idx):
        rec = dict(state=states[seq[k]], cost=[round(float(x), 2) for x in C[k]])
        if seq[k] == 1:
            rec["offset"] = jaw_offset(int(i)).tolist()
        if seq[k] == 2:
            for c in ("top", "side"):
                s_, cen, _ = det[(int(i), c)]
                if cen is not None and s_ >= 0.3:
                    X = camgeo.cast_plane_z(P[vo[c][int(e)]], [cen[0]], [cen[1]], H, W, zo + 2 * B.BOX_WALL + hz)[0]
                    lx = c_ * (X[0] - bx) + s_ * (X[1] - by); ly = -s_ * (X[0] - bx) + c_ * (X[1] - by)
                    if abs(lx) < sx / 2 - hx and abs(ly) < sy / 2 - hy:
                        rec["xy"] = [float(X[0]), float(X[1])]
                        break
        out[str(int(i))] = rec
cnt = {s: sum(r["state"] == s for r in out.values()) for s in states}
print("states", cnt)
ph = B.LAB["phase"]
for s in states:
    sel = [int(k) for k, r in out.items() if r["state"] == s]
    u, c = np.unique(ph[sel], return_counts=True)
    print(s, dict(zip(u.tolist(), c.tolist())))
(B.WORK / "block_state.json").write_text(json.dumps(out, indent=0))
