"""JAR: object placement for the RGB render (never used by the metric) from the twin-independent DINO+SAM object
masks of the d1 run (masks/jar_<cam>_d1_jar.npy, _d1_cube.npy), through the FITTED cameras.

    .venv/bin/python so101_twin/jar/objects.py --fit out/<run>_fit.json   # -> out/objects.json

Jar (per episode, static within an episode): white cylinder radius R, height H; R, H from a Nelder-Mead fit of the
rendered jar silhouette to the jar masks (both views) on 8 episodes (median), then (x, y) per episode with R, H fixed
(start: top-view mask centroid ray at z = H / 2). Cube (per frame): 'table' = the episode's top-view cube detection
(earliest frame where it is clean) ray-cast to z = CUBE_H / 2 with its yaw from the min-area box; 'held' = at the
tool point while the gripper is around the cube (grasp / release phase, gripper 12..60 and the tool point > 3 cm
above the table, or the cube mask touching the gripper); 'jar' = inside the jar after the release (gripper < 12 in the
release phase); otherwise hidden.
"""
import argparse
import dataclasses
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from scipy import optimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from geom import cam_ray, hit_z  # noqa: E402

W, H = 160, 120


def load(tag):
    return np.unpackbits(np.load(HERE / "masks" / f"jar_{tag}.npy"), axis=-1)[..., :640].astype(bool)


def small(m):
    return m.reshape(-1, H, 4, W, 4).mean((2, 4)) >= 0.5


def min_box_yaw(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
    best = (1e18, 0.0)
    for a in np.deg2rad(np.arange(0, 90, 3)):
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q = pts @ R
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    a = ap.parse_args()
    import scene_jar as S  # noqa: F401  (module-level constants only)

    sc = F.load_scene("jar")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    sc.OBJ = None
    rig = F.Rig(sc, P, full=True)
    m, d = rig.arm.m, rig.arm.d
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    jar_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("jar_body", "jar_rim", "jar_in")]
    lut = np.zeros(m.ngeom + 1, bool)
    lut[np.asarray(jar_ids) + 1] = True
    jb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jar")]
    cb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")]
    d.mocap_pos[cb] = (0, 0, -2)
    real = {c: small(load(f"{c}_d1_jar")) for c in ("top", "side")}
    lens = {c: F.A.Lens(**P["cams"][c]["lens"]) for c in ("top", "side")}
    for c in ("top", "side"):
        g = P["cams"][c]["groups"]["all"]
        cid = rig.arm.cam_id[c]
        m.cam_pos[cid] = g["pos"]
        m.cam_quat[cid] = np.asarray(g["quat"]) / np.linalg.norm(g["quat"])
        m.cam_fovy[cid] = P["cams"][c]["fovy"]
    # hide the arm far away so it does not occlude the jar silhouette
    m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_root")] = (0, 0, -5)
    for gname in ("table",):
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)][3] = 0.0
        m.geom_group[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)] = 4   # not rendered (group 4 hidden)

    def set_shape(R_, H_):
        m.geom_size[jar_ids[0]] = [R_, H_ / 2, 0]
        m.geom_pos[jar_ids[0]] = [0, 0, H_ / 2]
        m.geom_size[jar_ids[1]] = [R_ * 0.97, 0.0012, 0]
        m.geom_pos[jar_ids[1]] = [0, 0, H_ + 0.0006]
        m.geom_size[jar_ids[2]] = [R_ * 0.86, 0.0012, 0]
        m.geom_pos[jar_ids[2]] = [0, 0, H_ + 0.0016]

    def jar_mask(c):
        mujoco.mj_kinematics(m, d)
        seg = lens[c].render(rig.arm, c, W, H, seg=True)
        return lut[seg[..., 0] + 1] & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))

    def ep_frames(e):
        idx = np.nonzero(ep == e)[0]
        out = {}
        for c in ("top", "side"):
            ar = real[c][idx].reshape(len(idx), -1).mean(1)
            ok = idx[(ar > 0.003) & (ar < 0.3)]
            out[c] = ok[np.argsort(-real[c][ok].reshape(len(ok), -1).sum(1))][:3] if len(ok) else []
        return out

    def loss(x, frames, R_=None, H_=None):
        if R_ is None:
            R_, H_ = x[2], x[3]
        set_shape(R_, H_)
        d.mocap_pos[jb] = (x[0], x[1], 0.0)
        v = []
        for c, fr in frames.items():
            if not len(fr):
                continue
            sim = jar_mask(c)
            for i in fr:
                r = real[c][i]
                u = (sim | r).sum()
                v.append((sim & r).sum() / u if u else 0.0)
        return -float(np.mean(v)) if v else 0.0

    def init_xy(e, H_):
        fr = ep_frames(e)
        for i in fr["top"]:
            ys, xs = np.nonzero(real["top"][i])
            o, dd = cam_ray(P, "top", xs.mean() * 4 + 1.5, ys.mean() * 4 + 1.5)
            return hit_z(o, dd, H_ / 2)[:2]
        return np.array([0.15, 0.35])

    eps = sorted(set(ep.tolist()))
    shp = []
    for e in eps[::5]:
        fr = ep_frames(e)
        xy = init_xy(e, S.JAR_H)
        r = optimize.minimize(loss, [xy[0], xy[1], S.JAR_R, S.JAR_H], args=(fr,), method="Nelder-Mead",
                              options=dict(xatol=1e-3, fatol=1e-4, maxfev=250, initial_simplex=np.array(
                                  [[xy[0], xy[1], S.JAR_R, S.JAR_H], [xy[0] + 0.03, xy[1], S.JAR_R, S.JAR_H],
                                   [xy[0], xy[1] + 0.03, S.JAR_R, S.JAR_H], [xy[0], xy[1], S.JAR_R + 0.01, S.JAR_H],
                                   [xy[0], xy[1], S.JAR_R, S.JAR_H + 0.02]])))
        shp.append((r.x[2], r.x[3], -r.fun))
        print(f"shape ep{e}: R {r.x[2]:.4f} H {r.x[3]:.4f} IoU {-r.fun:.3f}", flush=True)
    R_ = float(np.median([s[0] for s in shp]))
    H_ = float(np.median([s[1] for s in shp]))
    print(f"jar R {R_:.4f} H {H_:.4f}", flush=True)
    jar_xy, jar_iou = {}, {}
    for e in eps:
        fr = ep_frames(e)
        if not len(fr["top"]) and not len(fr["side"]):
            jar_xy[str(e)] = None
            continue
        xy = init_xy(e, H_)
        r = optimize.minimize(loss, xy, args=(fr, R_, H_), method="Nelder-Mead",
                              options=dict(xatol=5e-4, fatol=1e-4, maxfev=120,
                                           initial_simplex=np.array([xy, xy + [0.03, 0], xy + [0, 0.03]])))
        jar_xy[str(e)] = [round(float(r.x[0]), 4), round(float(r.x[1]), 4)]
        jar_iou[str(e)] = round(-float(r.fun), 3)
    print("jar IoU per episode median", np.median(list(jar_iou.values())), flush=True)
    # ---- cube
    cube_top = load("top_d1_cube")
    arm_top = load("top_d1c")
    g = np.asarray(fs.labels["gripper"], float)
    ph = np.asarray(fs.labels["phase"]).astype(str)
    rig2 = F.Rig(sc, P)
    sid = mujoco.mj_name2id(rig2.arm.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cube = {}
    table_pos = {}
    for e in eps:
        idx = np.nonzero(ep == e)[0]
        for i in idx:
            cm = cube_top[i]
            if 150 < cm.sum() < 8000 and (cm & arm_top[i]).sum() < 0.05 * cm.sum():
                ys, xs = np.nonzero(cm)
                o, dd = cam_ray(P, "top", xs.mean(), ys.mean())
                p = hit_z(o, dd, S.CUBE_H / 2)
                ang = min_box_yaw(cm)
                table_pos[e] = ([round(float(p[0]), 4), round(float(p[1]), 4), round(S.CUBE_H / 2, 4)], float(ang))
                break
    for i in range(fs.n):
        e = int(ep[i])
        rig2.arm.set_state(F.frame_state(sc, P, "top", i))
        tcp = rig2.arm.d.site_xpos[sid].copy()
        held = ph[i] in ("grasp", "release") and 12 <= g[i] <= 60 and tcp[2] > 0.03
        if ph[i] == "grasp" and 12 <= g[i] <= 60 and tcp[2] <= 0.03:
            held = True   # closing on the cube at the table
        if held:
            cube[str(i)] = ["held"]
        elif ph[i] == "release" and g[i] < 12 and jar_xy.get(str(e)) is not None:
            jx, jy = jar_xy[str(e)]
            cube[str(i)] = ["jar", [jx, jy, round(H_ - 0.045, 4)], 0.3]
        elif e in table_pos and ph[i] in ("approach", "grasp"):
            p, ang = table_pos[e]
            cube[str(i)] = ["table", p, ang]
        else:
            cube[str(i)] = None
    # ---- wall / door (RGB backdrop): the table-wall edge read off the side-view background median at pixels
    # (0, 340) and (420, 240), the door frame at (440, 250), ray-cast to z = 0 through the fitted side camera
    p1 = hit_z(*cam_ray(P, "side", 0, 340), 0.0)
    p2 = hit_z(*cam_ray(P, "side", 420, 240), 0.0)
    p3 = hit_z(*cam_ray(P, "side", 440, 250), 0.0)
    dvec = (p2 - p1) / np.linalg.norm(p2 - p1)
    nrm = np.array([-dvec[1], dvec[0], 0.0])
    if nrm @ (np.asarray(P["cams"]["side"]["groups"]["all"]["pos"]) - p1) > 0:
        nrm = -nrm   # normal pointing away from the side camera (behind the wall surface)
    yaw = math.atan2(dvec[1], dvec[0])
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mid = 0.5 * (p1 + p2)
    wall = dict(pos=list(map(float, mid + 0.006 * nrm + np.array([0, 0, 0.6]))), quat=q, size=[4.0, 0.005, 0.6])
    t3 = float((p3 - mid) @ dvec)
    door_c = mid + (t3 + 0.45 * np.sign((p2 - p1) @ dvec)) * dvec - 0.002 * nrm + np.array([0, 0, 0.6])
    door = dict(pos=list(map(float, door_c)), quat=q, size=[0.45, 0.004, 0.6])
    print("wall p1", np.round(p1, 3), "p2", np.round(p2, 3), "door at", np.round(p3, 3), flush=True)
    # door-coloured backdrop facing the side camera 2.5 m along its axis (fills the view right of the wall)
    gs = P["cams"]["side"]["groups"]["all"]
    Rs = F.A.mat_from_quat(np.asarray(gs["quat"]) / np.linalg.norm(gs["quat"]))
    back = dict(pos=list(map(float, np.asarray(gs["pos"]) - 2.5 * Rs[:, 2])), quat=list(map(float, gs["quat"])),
                size=[3.0, 3.0, 0.005])
    out = dict(fit=a.fit, jar_r=round(R_, 4), jar_h=round(H_, 4), jar_xy=jar_xy, jar_iou=jar_iou, cube=cube,
               layout=dict(wall=wall, door=door, back=back))
    (HERE / "out" / "objects.json").write_text(json.dumps(out, indent=1))
    print("cube states:", {k: sum(1 for v in cube.values() if (v is None and k == "none") or (v is not None and v[0] == k))
                           for k in ("table", "held", "jar", "none")})


if __name__ == "__main__":
    main()
