"""Write neuroscope/molmoact2/store/sets/so101_legocup_twin2 from the legocup2 fit (adapted from
so101_twin/common/render_set.py; scene so101_twin/legocup2/scene_legocup.py, params out/<fit>_fit.json).

    .venv/bin/python so101_twin/legocup2/render_lc2.py --fit out/legocup_final_fit.json [--limit N] [--masks-tag _lc2]

Per frame: real state (shifted by the camera's fitted latency, full-rate parquet) -> official SO-101 new_calib with
the wrist-camera mount (fitted joint offsets, gripper scale / offset), cup / brick per episode (builder placement,
re-ray-cast through the refitted top camera), each camera rendered at 640x480 with its fitted pose, fovy and lens warp
(k1, k2, principal point; the same warp as the fitted masks), then a per-camera per-channel gain (real / sim mean
colour over 16 spread frames). Writes the set in the existing twin layout + out/before_after.jpg (8 odd-episode
frames: per camera real | old twin | new twin | mask overlay (white both, red real only, blue twin only)).
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
NAME = "so101_legocup_twin2"


def render_frame(rig, sc, fs, i, gain=None) -> dict:
    out = {}
    for c in sc.CAMERAS:
        rig.pose(c, "all", F.frame_state(sc, rig.p, c, i))
        sc.frame_setup(rig, fs, i)
        img = rig.arm.render(c, 640, 480, rig.lens[c])
        if gain is not None:
            img = np.clip(img.astype(np.float32) * np.asarray(gain[c], np.float32) + 0.5, 0, 255).astype(np.uint8)
        out[c] = img
    return out


def estimate_gain(rig, sc, fs) -> dict:
    probe = np.linspace(0, fs.n - 1, 16).astype(int)
    sim = {c: [] for c in sc.CAMERAS}
    real = {c: [] for c in sc.CAMERAS}
    for i in probe:
        fr = render_frame(rig, sc, fs, int(i))
        obs = fs.obs(int(i))["images"]
        for c in sc.CAMERAS:
            sim[c].append(fr[c].reshape(-1, 3).mean(0))
            real[c].append(np.asarray(obs[c]).reshape(-1, 3).mean(0))
    return {c: [float(x) for x in np.clip(np.mean(real[c], 0) / np.maximum(np.mean(sim[c], 0), 1), 0.3, 2.5)]
            for c in sc.CAMERAS}


def before_after(sc, fs, rig, imgs: dict, real_masks: dict, path: pathlib.Path):
    from sets import Store

    old = Store().frames("so101_legocup_twin")
    ep = np.asarray(fs.labels["episode"]).astype(int)
    odd = np.nonzero(ep % 2 == 1)[0]
    # 8 odd-episode frames spread over episodes and phases, arm visible in the top camera
    vis = [i for i in odd if real_masks["top"][i].mean() > 0.01]
    idx = [vis[k] for k in np.linspace(0, len(vis) - 1, 8).astype(int)]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            r = np.asarray(fs.obs(int(i))["images"][c])
            s = rig.mask(c, "all", F.frame_state(sc, rig.p, c, int(i)), 640, 480)
            m = real_masks[c][i]
            ov = r.astype(np.float32) * 0.45
            col = np.zeros((480, 640, 3), np.float32)
            col[m & s] = 255
            col[m & ~s] = (255, 40, 40)
            col[~m & s] = (40, 90, 255)
            a = (m | s)[..., None]
            ov = np.where(a, 0.3 * ov + 0.7 * col, ov).astype(np.uint8)
            tiles += [r, np.asarray(old.obs(int(i))["images"][c]), imgs[c][i], ov]
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width * 2 // 5, im.height * 2 // 5), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // 8
    labs = ["real", "old twin", "new twin", "IoU mask (W both R real B twin)"]
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(labs):
            d.text(((4 * j + t) * w + 4, 2), f"{c}: {lab}", fill=(255, 255, 0))
    hgt = im.height // 8
    for k, i in enumerate(idx):
        d.text((4, k * hgt + 14), f"#{i} ep{ep[i]} t{int(fs.labels['t'][i])} {fs.labels['phase'][i]}", fill=(255, 255, 0))
    im.save(path, quality=88)
    return [int(i) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--masks-tag", default="", help="SAM masks used for the overlay column")
    a = ap.parse_args()
    sc = F.load_scene("legocup")
    fj = json.loads((HERE / a.fit if not pathlib.Path(a.fit).is_absolute() else pathlib.Path(a.fit)).read_text())
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    n = fs.n if not a.limit else min(a.limit, fs.n)
    rig = F.Rig(sc, P, full=True)
    moved = sc.ray_cast_objects(rig)
    mv = np.array([v["moved_cm"] for v in moved.values()])
    print(f"objects re-ray-cast through the refitted top camera: cup moved {mv[:, 0].mean():.2f} cm (max "
          f"{mv[:, 0].max():.2f}), brick {mv[:, 1].mean():.2f} cm (max {mv[:, 1].max():.2f})", flush=True)
    t0 = time.time()
    gain = estimate_gain(rig, sc, fs)
    print("gain", {c: np.round(v, 3).tolist() for c, v in gain.items()}, flush=True)
    imgs = {c: np.zeros((n, 480, 640, 3), np.uint8) for c in sc.CAMERAS}
    for i in range(n):
        fr = render_frame(rig, sc, fs, i, gain)
        for c in sc.CAMERAS:
            imgs[c][i] = fr[c]
        if (i + 1) % 80 == 0:
            print(f"{i + 1}/{n} frames {time.time() - t0:.0f}s", flush=True)
    out = pathlib.Path(a.out) if a.out else SETS / NAME
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
    labels["source"] = np.full(n, "legocup_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {}
    for c, p in P["cams"].items():
        g = p["groups"]["all"]
        cams[c] = dict(mount=p["mount"],
                       frame="world (table top z=0, SO-101 pan axis at (0, 0.0452), arm reaches +y at pan 0)"
                       if p["mount"] == "world" else "body 'gripper' of the official SO-101 (wrist-roll link that carries "
                       "the fixed jaw and the wrist-camera mount); camera module + bracket hidden from this camera",
                       pos=g["pos"], quat_wxyz=g["quat"], fovy_deg=p["fovy"], distortion=p["lens"],
                       latency_frames=p.get("dt_frames", 0.0),
                       convention="MuJoCo camera: looks along -z, +y up in the image; pos/quat in the mount frame")
    st = fj["stats"]
    rmeta = fs.meta
    meta = dict(
        name=NAME, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source="legocup_twin2", twin_of=sc.REAL_SET,
        image_shape=[480, 640, 3], dataset=rmeta.get("dataset"), feature_keys=rmeta.get("feature_keys"),
        episodes=rmeta.get("episodes"), per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"),
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
        arm_model="TheRobotStudio SO-ARM100 Simulation/SO101 (official), so101_new_calib.xml + wrist camera mount and "
                  "camera module from so101_new_calib_camera.xml (so101_twin/assets/so101), all parts black",
        calibration="so101_follower new calibration, arm joints recorded in LeRobot NORMALISED units (range_m100_100, "
                    "use_degrees=False: the dataset's action stats hit exactly -100.0 / +100.0), gripper range_0_100",
        joint_map=dict(formula="q_rad[j] = deg2rad(scale[j] * state[j] + offset_deg[j]) for shoulder_pan, shoulder_lift, "
                               "elbow_flex, wrist_flex, wrist_roll (all signs +1, official so101_new_calib joint frames)",
                       scale=[round(float(x), 4) for x in P.get("scales", [1.0] * 5)],
                       offset_deg=[round(float(x), 3) for x in P["offsets_deg"]],
                       gripper=f"q_grip = {F.A.GRIP_LO} + grip_off + grip_scale * clip(state[5], 0, 100) / 100 * "
                               f"({F.A.GRIP_HI} - {F.A.GRIP_LO})",
                       grip_scale=round(float(P.get("grip_scale", 1.0)), 4), grip_off=round(float(P.get("grip_off", 0.0)), 4),
                       notes="wrist_roll offset ~-90 deg: the camera module faces the top camera as in the real "
                             "frames (0 / 180 put it at the side); state taken latency_frames earlier/later from the "
                             "30 fps parquet per camera"),
        cameras_fit=cams,
        lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4) in focal-length units, principal point offset (cx, cy) as fractions "
                   "of (width, height); applied as a warp of the pinhole render, to both the RGB images and the masks",
        gain_rgb=gain,
        objects="builder placement (so101_twin/legocup/out/objects.json: per-episode cup / brick pixel centroids) "
                "re-ray-cast through the refitted top camera; brick on the table (approach), between the jaws (grasp / "
                "transfer_open), in the cup (release)",
        iou_protocol=fj.get("protocol", "whole-arm silhouette IoU, twin arm mask vs SAM 2.1 mask of the real arm, per "
                     "camera, frames with the real arm visible (>= 0.1% of the image); fit on even episodes, "
                     "held_out = odd episodes"),
        sam_masks=fj.get("mask_files"),
        held_out={c: st[c]["new_heldout"] for c in sc.CAMERAS},
        all_frames={c: st[c]["new"] for c in sc.CAMERAS},
        old_twin_held_out={c: st[c]["old_heldout"] for c in sc.CAMERAS},
        old_twin_all_frames={c: st[c]["old"] for c in sc.CAMERAS},
        extra_stats=fj.get("extra_stats"),
        code="so101_twin/legocup2/ (scene_legocup.py, arm.py, fit.py, segment_lc2.py, finalize.py, score_lc2.py, "
             "render_lc2.py; README.md)",
        fit_json="so101_twin/legocup2/" + str(pathlib.Path(a.fit)) if not pathlib.Path(a.fit).is_absolute() else a.fit)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        real_masks = {c: F.load_masks("legocup", c, a.masks_tag) for c in sc.CAMERAS}
        idx = before_after(sc, fs, rig, imgs, real_masks, HERE / "out" / "before_after.jpg")
        print("wrote", HERE / "out" / "before_after.jpg", "frames", idx)
    (HERE / "out" / "objects_twin2.json").write_text(json.dumps(moved, indent=1))


if __name__ == "__main__":
    main()
