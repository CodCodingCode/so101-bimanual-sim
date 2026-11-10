"""Per-frame block state for the ricky twin: 'table' (episode start pose), 'held' (between the jaw tips, follows
the twin gripper), 'plate' (on the bowl floor). Decided from Grounding-DINO + SAM block detections in every frame of
both cameras against the three hypotheses projected through the frozen cameras, with the order table -> held ->
plate enforced per episode (dynamic programming). The held offset (block centre in the twin gripper-site frame) is
the median of two-view triangulations over confidently held frames. Writes work/block_state.json."""
import json, pathlib, sys
import mujoco
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camgeo, objfit  # noqa
import build_ricky as B  # noqa

T = B.make(colours={})
vo = B.view_of(T)
H, W = T.H, T.W
z = np.load(B.WORK / "seg_block_all.npz", allow_pickle=True)
det = {}
for k in range(len(z["frame"])):
    m = np.unpackbits(z["bits"][k], axis=-1)[..., :W].astype(bool)
    det[(int(z["frame"][k]), str(z["cam"][k]))] = (float(z["score"][k]), np.argwhere(m).mean(0)[::-1] if m.any() else None, int(m.sum()))
# plausible block size: 0.3x .. 2.5x the median first-frame block area per camera (Grounding DINO falls back to the
# chair, the table or the bowl when the block is hidden; those boxes are far larger)
first = [int(np.flatnonzero(B.EP == e)[0]) for e in np.unique(B.EP)]
AREA = {c: float(np.median([det[(i, c)][2] for i in first])) for c in ("top", "side")}
for (i, c), (s_, cen, a) in list(det.items()):
    if cen is not None and not (0.3 * AREA[c] <= a <= 2.5 * AREA[c]):
        det[(i, c)] = (0.0, None, a)
print("block area medians", AREA)
P = {v: T.F.p[v] for v in T.F.views}
zo = B.OBJ["z_objects"]
hx, hy, hz = B.OBJ["shapes"]["block"]
sid = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
n = T.F.fs.n
tcp, Rg = np.zeros((n, 3)), np.zeros((n, 3, 3))
for i in range(n):
    q = T.F.q(i)
    for a, x in zip(T.qadr, q):
        T.d.qpos[a] = x
    mujoco.mj_kinematics(T.m, T.d)
    tcp[i], Rg[i] = T.d.site_xpos[sid], T.d.site_xmat[sid].reshape(3, 3)


def proj(v, X):
    u, w, _ = camgeo.project(P[v], np.atleast_2d(X), H, W)
    return np.array([u[0], w[0]])


def tri(i):
    """Least-squares intersection of the two detection rays (None unless both views detect confidently)."""
    rows = []
    for c in ("top", "side"):
        s, cen, npx = det[(i, c)]
        if cen is None or s < 0.3:
            return None
        v = vo[c][int(B.EP[i])]
        d = camgeo.rays(P[v], [cen[0]], [cen[1]], H, W)[0]
        rows.append((np.asarray(P[v][:3]), d))
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rows:
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ o
    return np.linalg.solve(A, b)


SCALE = 60.0     # px: a detection this far from a hypothesis costs 1
states = ("table", "held", "plate")
out = {}
held_off = []
for e in np.unique(B.EP):
    idx = np.flatnonzero(B.EP == e)
    idx = idx[np.argsort(B.LAB["t"][idx])]
    ep = B.OBJ["episodes"][str(int(e))]
    kx, ky, _ = ep["block"]["pose"]
    bx, by = ep["bowl"]["pose"]
    rb, rt, hb = B.OBJ["shapes"]["bowl"]
    C = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        for c in ("top", "side"):
            s, cen, npx = det[(i, c)]
            v = vo[c][int(e)]
            if cen is None or s < 0.3:
                C[k] += [0.8, 0.5, 0.5]   # unseen: hidden by the jaws or inside the bowl more often than on the table
                continue
            d_tab = np.linalg.norm(cen - proj(v, [kx, ky, zo + hz]))
            d_tcp = np.linalg.norm(cen - proj(v, tcp[i]))
            # inside the bowl: distance to the bowl axis at mid height, minus the projected inner radius
            pc = proj(v, [bx, by, zo + 0.5 * hb])
            pr = np.linalg.norm(proj(v, [bx + 0.8 * rt, by, zo + 0.5 * hb]) - pc)
            d_bowl = max(0.0, np.linalg.norm(cen - pc) - pr)
            C[k] += np.minimum([d_tab, d_tcp, d_bowl], 3 * SCALE) / SCALE
    for k, i in enumerate(idx):      # an open gripper (reading < 8) cannot hold the block; a closed one favours held
        C[k, 1] += 1.0 if B.LAB["gripper"][i] < 8 else (-0.2 if B.LAB["gripper"][i] >= 10 else 0.0)
    C[0, 1:] += 5.0                  # every episode starts with the block on the table
    # DP with monotone states
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
            X = tri(int(i))
            if X is not None:
                held_off.append(Rg[i].T @ (X - tcp[i]))
                rec["tri"] = X.tolist()
        if seq[k] == 2:
            s_, cen, _ = det[(int(i), "side")]
            if cen is not None and s_ >= 0.3:
                X = camgeo.cast_plane_z(P[vo["side"][int(e)]], [cen[0]], [cen[1]], H, W, zo + B.BOWL_WALL + 2 * hz)[0]
                if np.hypot(X[0] - bx, X[1] - by) < rb - hx:      # inside the bowl floor
                    rec["xy"] = [float(X[0]), float(X[1])]
        out[str(int(i))] = rec
bid_g = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_BODY, "arm_gripper")
bid_m = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_BODY, "arm_moving_jaw_so101_v1")
gm = [g for g in range(T.m.ngeom) if T.m.geom_bodyid[g] == bid_m and T.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and T.m.geom_group[g] == 2][0]
mid = T.m.geom_dataid[gm]
MV = T.m.mesh_vert[T.m.mesh_vertadr[mid]:T.m.mesh_vertadr[mid] + T.m.mesh_vertnum[mid]].copy()


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
    c = 0.5 * (site + tip_m) - dirv * max(hx, hy)
    return T.d.site_xmat[sid].reshape(3, 3).T @ (c - site)


held_off = np.array(held_off)
med = np.median(held_off, 0) if len(held_off) else np.zeros(3)
mad = np.median(np.abs(held_off - med), 0) if len(held_off) else np.zeros(3)
print("held frames triangulated", len(held_off), "offset in gripper-site frame (m)", np.round(med, 4), "MAD", np.round(mad, 4))
geo = []
for k in out:
    if out[k]["state"] == "held":
        o_ = jaw_offset(int(k))
        geo.append(o_)
        out[k]["offset"] = o_.tolist()
if geo:
    print("geometric held offsets (site frame, m): median", np.round(np.median(geo, 0), 4), "spread", np.round(np.std(geo, 0), 4))
cnt = {s: sum(r["state"] == s for r in out.values()) for s in states}
print("states", cnt)
ph = B.LAB["phase"]
for s in states:
    sel = [int(k) for k, r in out.items() if r["state"] == s]
    u, c = np.unique(ph[sel], return_counts=True)
    print(s, dict(zip(u.tolist(), c.tolist())))
(B.WORK / "block_state.json").write_text(json.dumps(dict(held_offset=med.tolist(), held_offset_mad=mad.tolist(), n_tri=int(len(held_off)), frames=out) | out, indent=0))
