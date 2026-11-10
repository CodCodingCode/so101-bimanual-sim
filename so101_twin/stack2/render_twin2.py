"""Write so101_stack_twin2 (exact twin-set layout) from a fit json, plus out/before_after.jpg.

    .venv/bin/python so101_twin/stack2/render_twin2.py --fit-json so101_twin/common/out/stack_r2_fit.json
        --mask-tag _s2 [--limit N] [--out DIR]

Per frame: real state (shifted by each camera's fitted latency, full-rate parquet) -> official SO-101 new_calib with the
fitted joint offsets / scales / gripper scale; the builder's objects (disc, wooden cube, green block; cube on the disc /
in the jaws / on the block by the builder's gripper rule); every camera rendered at 640x480 with its fitted pose for the
frame's session, fovy and lens warp (k1, k2, cx, cy applied to the RGB exactly as to the masks: arm.Lens); backdrop
sheets only in the side view (builder); then a per-camera per-channel colour gain (real / sim mean colour, 16 frames).
before_after.jpg: 8 odd-episode (held-out) frames, per camera: real | old twin (so101_stack_twin) | new twin | overlay of
the SAM mask vs the new twin silhouette (white both, red real only, blue twin only).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

SETS = F.ROOT / "neuroscope" / "molmoact2" / "store" / "sets"
COMMON_MASKS = F.TWIN / "masks"


def render_frame(rig, sc, fs, i, groups, gain):
    out = {}
    for c in sc.CAMERAS:
        rig.pose(c, groups[c][i], F.frame_state(sc, rig.p, c, i))
        sc.frame_setup(rig, fs, i)
        img = rig.arm.render(c, 640, 480, rig.lens[c])
        if gain is not None:
            img = np.clip(img.astype(np.float32) * np.asarray(gain[c], np.float32) + 0.5, 0, 255).astype(np.uint8)
        out[c] = img
    return out


def objects_from_fk(sc, P) -> dict:
    """Arm-consistent object placement: the disc / resting cube where the twin's tool centre point is when the gripper
    closes on the cube, the green block under the twin's tcp when it opens to release it (medians over the EVEN
    episodes at full rate, the fitted joint map and latency); the table plane = grasp height - half cube. Sizes, colours
    and the cube timeline stay the builder's."""
    import mujoco
    import pyarrow.parquet as pq

    fs = F.real_set(sc)
    fr = F.fullrate(sc)
    rig = F.Rig(sc, P)
    sid = mujoco.mj_name2id(rig.arm.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    lag = float(np.mean([P["cams"][c].get("dt_frames", 0.0) for c in P["cams"]]))
    G, R = [], []
    for e, st in fr.seq.items():
        if e % 2:
            continue
        g = st[:, 5]
        tg = next((t for t in range(1, len(g)) if g[t - 1] > 18 >= g[t] and st[t, 1] > -60), None)
        tr = next((t for t in range((tg or 0) + 10, len(g)) if g[t - 1] <= 18 < g[t]), None) if tg else None
        for t, L in ((tg, G), (tr, R)):
            if t is None:
                continue
            x = np.clip(t + lag, 0, len(st) - 1)
            a0 = int(np.floor(x)); b0 = min(a0 + 1, len(st) - 1); w = x - a0
            rig.arm.set_state((1 - w) * st[a0].astype(float) + w * st[b0].astype(float))
            L.append(rig.arm.d.site_xpos[sid].copy())
    ng, nr = len(G), len(R)
    G, R = np.median(np.array(G), 0), np.median(np.array(R), 0)
    half = sc.B.OBJECTS["cube"]["size"][2]
    return dict(disc_xy=G[:2].tolist(), block_xy=R[:2].tolist(), table_z=float(G[2] - half),
                block_top_z=float(R[2] - half), n_grasp_episodes=ng, n_release_episodes=nr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-json", required=True)
    ap.add_argument("--mask-tag", required=True, help="SAM masks so101_twin/masks/stack_<cam><tag>.npy used for the IoU")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--builder-objects", action="store_true",
                    help="keep the builder's object xy (inconsistent with the refit arm by 10-17 cm) instead of "
                         "placing disc / block at the twin's grasp / release points")
    a = ap.parse_args()
    sc = F.load_scene("stack")
    fpath = pathlib.Path(a.fit_json).resolve()
    fj = json.loads(fpath.read_text())
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    n = fs.n if not a.limit else min(a.limit, fs.n)
    placement = None
    if not a.builder_objects:
        placement = objects_from_fk(sc, P)
        print("object placement from the arm's grasp / release points:", placement, flush=True)
        dz = placement["table_z"]
        side_z = P["cams"]["side"]["groups"]["all"]["pos"][2] if "side" in P["cams"] else 1.0
        if dz > side_z - 0.015:     # the low side camera must stay above the table plane (RGB scene only)
            print(f"table plane {dz:.3f} lowered to {side_z - 0.015:.3f} (side camera height {side_z:.3f})", flush=True)
            dz = side_z - 0.015
        placement["table_z_used"] = dz
        sc.B.DISC = dict(sc.B.DISC, xy=tuple(placement["disc_xy"]))
        sc.B.OBJECTS = {k: dict(v) for k, v in sc.B.OBJECTS.items()}
        sc.B.OBJECTS["cube"]["xy"] = tuple(placement["disc_xy"])
        sc.B.OBJECTS["green"]["xy"] = tuple(placement["block_xy"])
        sc.B.TABLE_HALF = tuple(sc.B.TABLE_HALF)
        sc.TABLE_Z = dz
    rig = F.Rig(sc, P, full=True)
    rig.arm.geomgroups = sc.GEOMGROUPS
    if placement is not None:   # shift table / disc / block / resting cube to the fitted table plane
        import mujoco
        m, d = rig.arm.m, rig.arm.d
        for gname in ("table_top", "disc"):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            m.geom_pos[gid][2] += dz
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "green")
        d.mocap_pos[m.body_mocapid[bid]][2] += dz
        sc.TABLE_Z = dz
    groups = {c: F.frame_groups(sc, c) for c in sc.CAMERAS}
    t0 = time.time()
    probe = np.linspace(0, fs.n - 1, 16).astype(int)
    sim_m, real_m = {c: [] for c in sc.CAMERAS}, {c: [] for c in sc.CAMERAS}
    for i in probe:
        fr = render_frame(rig, sc, fs, int(i), groups, None)
        for c in sc.CAMERAS:
            sim_m[c].append(fr[c].reshape(-1, 3).mean(0))
            real_m[c].append(np.asarray(fs.obs(int(i))["images"][c]).reshape(-1, 3).mean(0))
    gain = {c: list(np.clip(np.mean(real_m[c], 0) / np.maximum(np.mean(sim_m[c], 0), 1), 0.3, 2.0)) for c in sc.CAMERAS}
    print("gain", {c: np.round(v, 3).tolist() for c, v in gain.items()}, flush=True)
    imgs = {c: np.zeros((n, 480, 640, 3), np.uint8) for c in sc.CAMERAS}
    for i in range(n):
        fr = render_frame(rig, sc, fs, i, groups, gain)
        for c in sc.CAMERAS:
            imgs[c][i] = fr[c]
        if (i + 1) % 80 == 0:
            print(f"{i + 1}/{n} frames {time.time() - t0:.0f}s", flush=True)
    out = pathlib.Path(a.out) if a.out else SETS / "so101_stack_twin2"
    out.mkdir(parents=True, exist_ok=True)
    for c in sc.CAMERAS:
        np.save(out / f"images_{c}.npy", imgs[c])
    if n == fs.n:
        shutil.copyfile(fs.path / "state.npy", out / "state.npy")
        shutil.copyfile(fs.path / "prompt.json", out / "prompt.json")
    else:
        np.save(out / "state.npy", fs.state[:n])
        (out / "prompt.json").write_text(json.dumps(fs.prompt[:n]))
    labels = {k: np.asarray(v)[:n] for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.full(n, "sim")
    labels["source"] = np.full(n, "stack_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    np.save(out / "cube_where.npy", sc.cube_where(fs)[:n])
    # IoU stats on the full-res real grid (same as fit.evaluate) + excluded-frame counts
    real = {c: np.unpackbits(np.load(COMMON_MASKS / f"stack_{c}{a.mask_tag}.npy"), axis=-1)[..., :640].astype(bool)
            for c in sc.CAMERAS}
    rig_eval = F.Rig(sc, P)          # arm only, exactly as fit.py's evaluate (objects never occlude the scored mask)
    ev = F.evaluate(rig_eval, sc, real, None)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    excl = {}
    for c in sc.CAMERAS:
        area = real[c].reshape(fs.n, -1).mean(1)
        vis = area >= F.VISIBLE
        excl[c] = dict(excluded_all=int((~vis).sum()), excluded_held_out=int((~vis & (ep % 2 == 1)).sum()),
                       held_out_frames=int((ep % 2 == 1).sum()), all_frames=int(fs.n))
    cams = {}
    for c, p in P["cams"].items():
        cams[c] = dict(mount=p["mount"], frame="world (builder's: table top z=0, pan axis at (0, 0.0452), reach +y)",
                       fovy_deg=p["fovy"], poses_by_group=p["groups"], distortion=p["lens"],
                       latency_frames=p.get("dt_frames", 0.0), group_rule=sc.GROUP_RULE[c])
    rmeta = fs.meta
    stats = {c: dict(held_out=ev[c]["new_heldout"], all_frames=ev[c]["new"]) for c in sc.CAMERAS}
    meta = dict(name="so101_stack_twin2", n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source="stack_twin2",
                twin_of="so101_stack_real", image_shape=[480, 640, 3], dataset=rmeta.get("dataset"),
                feature_keys=rmeta.get("feature_keys"), episodes=rmeta.get("episodes"),
                per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"), built=time.strftime("%Y-%m-%d %H:%M:%S"),
                arm_model="TheRobotStudio SO-ARM100 Simulation/SO101 (official), so101_new_calib.xml + official wrist "
                          "camera mount (so101_new_calib_camera.xml)",
                calibration="new (LeRobot so101_follower)",
                joint_map=dict(formula="q_rad[j] = deg2rad(scale[j] * state[j] + offset[j]), j = shoulder_pan, "
                                       "shoulder_lift, elbow_flex, wrist_flex, wrist_roll (official new_calib signs, all +1)",
                               offset_deg=P["offsets_deg"], scale=P.get("scales", [1.0] * 5),
                               gripper=f"q = {F.A.GRIP_LO} + {P.get('grip_scale', 1.0):.4f} * gripper/100 * "
                                       f"({F.A.GRIP_HI} - {F.A.GRIP_LO}) rad",
                               latency="each camera posed with the full-rate parquet state at t + latency_frames"),
                cameras_fit=cams,
                lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal point offset (cx, cy) "
                           "as fractions of (width, height); applied as a warp of a padded pinhole render onto the real "
                           "640x480 grid (masks and RGB alike)",
                gain_rgb=gain,
                iou_metric=("whole-arm silhouette IoU, twin arm mask vs SAM 2.1 mask of the real arm (so101_twin/masks/"
                            f"stack_<cam>{a.mask_tag}.npy), full-res real pixel grid; frames with real arm area < 0.1% of "
                            "the image (arm not in view) excluded; fit on EVEN episodes only, held_out = ODD episodes"),
                held_out={c: stats[c]["held_out"] for c in sc.CAMERAS},
                all_frames={c: stats[c]["all_frames"] for c in sc.CAMERAS},
                excluded_frames=excl,
                objects=dict(note=("builder's object models / sizes / cube timeline (so101_twin/stack/build_scene.py); "
                                   + ("builder's xy (--builder-objects)" if placement is None else
                                      "xy moved to the twin's grasp (disc, resting cube) and release (green block) points, "
                                      "table plane from the grasp height (kept 1.5 cm under the side camera)")),
                             placement=placement,
                             disc=dict(xy=list(sc.B.DISC["xy"]), r=sc.B.DISC["r"]),
                             **{k: dict(xy=list(v["xy"]), half_size=list(v["size"])) for k, v in sc.B.OBJECTS.items()}),
                scene_code="so101_twin/common/scenes/stack.py", fit_json=str(fpath.relative_to(F.ROOT)),
                render_code="so101_twin/stack2/render_twin2.py")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print("stats", json.dumps(stats, default=float), "excluded", excl, flush=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    # before / after on held-out frames
    from sets import Store

    old = Store().frames("so101_stack_twin")
    odd = np.nonzero(ep[:n] % 2 == 1)[0]
    vis_both = [i for i in odd if all(real[c][i].mean() >= F.VISIBLE for c in sc.CAMERAS)]
    pool = vis_both if len(vis_both) >= 8 else (list(odd) if len(odd) else list(range(n)))
    idx = [pool[k] for k in np.linspace(0, len(pool) - 1, 8).astype(int)]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            r_img = np.asarray(fs.obs(int(i))["images"][c])
            s = rig_eval.mask(c, groups[c][i], F.frame_state(sc, P, c, int(i)), 640, 480)
            r = real[c][i]
            ov = (r_img.astype(np.float32) * 0.45).astype(np.uint8)
            ov[r & s] = (255, 255, 255)
            ov[r & ~s] = (255, 40, 40)
            ov[~r & s] = (40, 110, 255)
            tiles += [r_img, np.asarray(old.obs(int(i))["images"][c]), imgs[c][i], ov]
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width // 4, im.height // 4), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // (4 * len(sc.CAMERAS))
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(("real", "old twin", "new twin", "SAM vs twin")):
            d.text(((4 * j + t) * w + 3, 2), f"{c}: {lab}", fill=(255, 255, 0))
    for k, i in enumerate(idx):
        d.text((3, k * (im.height // len(idx)) + 12), f"#{i} ep{ep[i]}", fill=(255, 255, 0))
    im.save(HERE / "out" / "before_after.jpg", quality=88)
    print("wrote", HERE / "out" / "before_after.jpg")


if __name__ == "__main__":
    main()
