"""Convert the fitkit session's best pos5 fit (legacy/pos5_so101_gain.json, armfit.py conventions: so101_old_calib.xml,
SO-100-style joint map with gains about the mean pose, cx/cy in pixels, base placed like the SO-100 Menagerie base)
into this folder's scene description (arm.py / fit.py conventions: so101_new_calib.xml, q = deg2rad(scale*state+off),
cx/cy as fractions, ArmSpec base pose), and check that both twins render the same silhouettes.

    .venv/bin/python so101_twin/pos5/convert_legacy.py [--json legacy/pos5_so101_gain.json] [--out out/init_from_legacy.json]

Everything is read from copies in pos5/legacy/ (fitkit/ is read-only for this session); the legacy armfit module is
imported from the copy with its asset paths pointed at so101_twin/assets/so101_official.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
TWIN = HERE.parent
ROOT = TWIN.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
import arm as A  # noqa: E402


def load_legacy():
    sys.path.insert(0, str(HERE / "legacy"))
    spec = importlib.util.spec_from_file_location("legacy_armfit", HERE / "legacy" / "armfit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODELS["so101"]["xml"] = TWIN / "assets" / "so101_official" / "so101_old_calib.xml"
    mod.MODELS["so101"]["meshdir"] = TWIN / "assets" / "so101_official" / "assets"
    mod.MODELS["so100"]["xml"] = TWIN / "assets" / "so_arm100" / "so_arm100.xml"
    mod.MODELS["so100"]["meshdir"] = TWIN / "assets" / "so_arm100"
    import _v3joints  # legacy copy
    return mod, _v3joints


def geom_table(m, d, prefix_bodies):
    """{(body, mesh, k): (pos, mat)} for visual mesh geoms."""
    out, seen = {}, {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if b not in prefix_bodies:
            continue
        mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
        k = seen.get((b, mesh), 0)
        seen[(b, mesh)] = k + 1
        out[(b, mesh, k)] = (d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy())
    return out


BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "legacy" / "pos5_so101_gain.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "init_from_legacy.json"))
    a = ap.parse_args()
    L, J = load_legacy()
    st = json.loads(pathlib.Path(a.json).read_text())
    from sets import Store

    fs = Store().frames("so101_pos5_real")
    W, H = 640, 480
    leg = L.ArmScene("so101", (0.0, 0.0, 0.0), "0 0 0 1", ["top", "side"], W, H, {}, dict(servos=True), False)
    leg.full_w = W
    for v in ("top", "side"):
        leg.set_cam(v, np.array([st["views"][v][k] for k in L.CAM_KEYS], float))
    sign, off, gain, qm = (np.asarray(st[k], float) for k in ("joint_sign", "joint_off_rad", "joint_gain", "q_mean"))

    def q_leg(state):
        qb = np.asarray(J.real_to_sim(state), float)
        return sign * ((1 + gain) * (qb - qm) + qm) + off

    # ---- my arm at the default placement
    spec0 = A.ArmSpec(calib="new", model="so101", wrist_mount=False)
    cams = [A.CamSpec("top"), A.CamSpec("side")]
    me = A.Arm(A.build_mjcf(spec0, cams), "new", "so101")
    leg.set_q(np.zeros(6))
    mujoco.mj_forward(leg.m, leg.d)
    T_leg = geom_table(leg.m, leg.d, BODIES)
    me.d.qpos[:] = 0
    mujoco.mj_forward(me.m, me.d)
    T_me = geom_table(me.m, me.d, BODIES)
    kb = ("base", "base_so101_v2", 0)
    pL, RL = T_leg[kb]
    pM, RM = T_me[kb]
    Rz = RL @ RM.T
    yaw = math.degrees(math.atan2(Rz[1, 0], Rz[0, 0]))
    print("base rotation (should be a z rotation):\n", np.round(Rz, 5), "yaw", round(yaw, 4))
    t = pL - Rz @ pM
    print("base_pos", np.round(t, 6))
    spec = A.ArmSpec(calib="new", model="so101", wrist_mount=False, base_pos=tuple(map(float, t)), base_yaw_deg=yaw)
    me = A.Arm(A.build_mjcf(spec, cams), "new", "so101")
    me.d.qpos[:] = 0
    mujoco.mj_forward(me.m, me.d)
    T_me = geom_table(me.m, me.d, BODIES)
    print("base geom match err (m)", float(np.abs(T_me[kb][0] - pL).max()), "rot", float(np.abs(T_me[kb][1] - RL).max()))
    keys = sorted(set(T_leg) & set(T_me))
    print(len(keys), "matched geoms of", len(T_leg), len(T_me))

    # ---- joint map: for random legacy poses find the new-calib qpos that reproduces every geom pose
    def fk_res(qn, Tl):
        me.d.qpos[me.qadr] = qn
        mujoco.mj_kinematics(me.m, me.d)
        Tm = geom_table(me.m, me.d, BODIES)
        r = []
        for k in keys:
            r.append(Tm[k][0] - Tl[k][0])
            r.append(0.05 * (Tm[k][1] - Tl[k][1]).ravel())
        return np.concatenate(r)

    rng = np.random.default_rng(0)
    QO, QN = [], []
    lo = np.array([-1.5, -1.5, -1.5, -1.5, -2.5, -0.1])
    hi = np.array([1.5, 1.5, 1.5, 1.5, 2.5, 1.6])
    guess = None
    for s in range(16):
        qo = rng.uniform(lo, hi)
        leg.set_q(qo)
        mujoco.mj_forward(leg.m, leg.d)
        Tl = geom_table(leg.m, leg.d, BODIES)
        best = None
        inits = [guess(qo)] if guess is not None else []
        inits += [rng.uniform(-3, 3, 6) for _ in range(40 if guess is None else 3)]
        for q0 in inits:
            r = least_squares(fk_res, q0, args=(Tl,), xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        QO.append(qo)
        QN.append((best.x + math.pi) % (2 * math.pi) - math.pi)
        if s == 3:   # after a few solves, fit sign/offset to speed up the rest
            qo_a, qn_a = np.array(QO), np.array(QN)
            sg = np.sign([np.corrcoef(qo_a[:, j], qn_a[:, j])[0, 1] for j in range(6)])
            c = np.array([np.median(((qn_a[:, j] - sg[j] * qo_a[:, j]) + math.pi) % (2 * math.pi) - math.pi) for j in range(6)])
            guess = lambda q, sg=sg, c=c: sg * q + c  # noqa: E731
        print(f"sample {s}: residual {math.sqrt(2 * best.cost):.2e}", flush=True)
    QO, QN = np.array(QO), np.array(QN)
    sg = np.sign([np.corrcoef(QO[:, j], QN[:, j])[0, 1] for j in range(6)])
    c = np.array([np.median(((QN[:, j] - sg[j] * QO[:, j]) + math.pi) % (2 * math.pi) - math.pi) for j in range(6)])
    resid = ((QN - (sg * QO + c)) + math.pi) % (2 * math.pi) - math.pi
    print("q_new = sign * q_old + c:  sign", sg.tolist(), "c (deg)", np.round(np.degrees(c), 4).tolist(),
          "max resid (deg)", np.round(np.degrees(np.abs(resid).max(0)), 4).tolist())

    # ---- legacy (state -> q_old) is affine per joint: q_old = a * state + b
    a_leg = sign[:5] * (1 + gain[:5]) * np.deg2rad(J.SIGNS)
    b_leg = sign[:5] * ((1 + gain[:5]) * (np.deg2rad(J.OFFSETS_DEG) - qm[:5]) + qm[:5]) + off[:5]
    scales = np.degrees(sg[:5] * a_leg)
    offsets = np.degrees(sg[:5] * b_leg + c[:5])
    # gripper: legacy jaw = JAW_LO + g/100 (JAW_HI - JAW_LO) (sign/gain/off of joint 5 are 1/0/0 in this fit)
    assert sign[5] == 1 and gain[5] == 0 and off[5] == 0
    jlo, jhi = J.JAW_RANGE
    # mine: q = GRIP_LO + gs * g/100 (GRIP_HI - GRIP_LO) [+ grip_off];  new = sg5 * old + c5
    gs = sg[5] * (jhi - jlo) / (A.GRIP_HI - A.GRIP_LO)
    goff = sg[5] * jlo + c[5] - A.GRIP_LO
    print("scales (deg/unit)", np.round(scales, 4).tolist(), "offsets (deg)", np.round(offsets, 3).tolist(),
          "grip_scale", round(gs, 4), "grip_off (rad)", round(goff, 4))

    # ---- cameras: same world frame (the arm was moved onto the legacy base), lens units converted
    P = dict(cams={}, offsets_deg=offsets.tolist(), scales=scales.tolist(), grip_scale=float(gs), grip_off=float(goff))
    for v in ("top", "side"):
        p = st["views"][v]
        P["cams"][v] = dict(mount="world", fovy=float(p["fovy"]),
                            groups={"all": dict(pos=[p["x"], p["y"], p["z"]],
                                                quat=A.ypr_quat(p["yaw"], p["pitch"], p["roll"]).tolist())},
                            lens=dict(k1=float(p["k1"]), k2=float(p["k2"]), cx=float(p["cx"]) / W, cy=float(p["cy"]) / H),
                            dt_frames=float(st["lag_frames"]))

    # ---- check: render both twins for frames of each camera and compare silhouettes
    from fullrate import FullRate

    fr = FullRate(fs)
    print("fullrate ok", fr.ok)
    rig_cams = [A.CamSpec(v, pos=tuple(P["cams"][v]["groups"]["all"]["pos"]), quat=tuple(P["cams"][v]["groups"]["all"]["quat"]),
                          fovy=P["cams"][v]["fovy"]) for v in ("top", "side")]
    me = A.Arm(A.build_mjcf(spec, rig_cams, offwidth=2048, offheight=1600), "new", "so101")
    me.offsets_deg, me.scales = offsets, scales
    me.grip_scale = gs
    lens = {v: A.Lens(**P["cams"][v]["lens"]) for v in ("top", "side")}
    gl = me.qadr[5]
    for v in ("top", "side"):
        ious = []
        for i in range(0, fs.n, 11):
            s = fr.state_at(i, st["lag_frames"])
            leg.set_q(q_leg(s))
            ml = leg.mask(v) > 0.5
            me.set_state(s)
            me.d.qpos[gl] += goff
            mujoco.mj_forward(me.m, me.d)
            mm = me.arm_mask(v, W, H, lens[v])
            u = (ml | mm).sum()
            if u > 100:
                ious.append((ml & mm).sum() / u)
        print(f"{v}: legacy vs converted silhouette IoU over {len(ious)} frames: median {np.median(ious):.4f} "
              f"min {np.min(ious):.4f}")
    out = dict(params=P, arm=dict(calib="new", model="so101", base_pos=list(map(float, t)), base_yaw_deg=float(yaw)),
               source=str(a.json), joint_map_new_from_old=dict(sign=sg.tolist(), c_deg=np.degrees(c).tolist()))
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
