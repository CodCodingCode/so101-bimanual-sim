"""GRAYBLOCK copy of jar/render_set.py: write the twin set so101_grayblock_twin2 (JAR: copy of so101_twin/ballcup/render_set.py adapted; changes marked JAR / GRAYBLOCK).

    .venv/bin/python so101_twin/grayblock/render_set.py --fit out/<run>_fit.json --score out/score_final.json [--limit N]

Per frame: real state -> fitted arm (joint scales / offsets / gripper map, per-camera latency from the full-rate
parquet), objects placed (scene_grayblock.frame_setup: box per episode, gray block per frame from out/objects.json), every
camera rendered at 640x480 with its fitted pose, fovy and lens warp (the same warp as the masks), then a per-camera
per-channel gain (real mean colour / sim mean colour over 16 spread frames). Output
neuroscope/molmoact2/store/sets/so101_grayblock_twin2/ in the existing twin layout; meta.json carries the fitted params and
the score json's held_out / all_frames stats (strict reading, and the real-occluder-excluded reading if scored).
Also out/before_after.jpg: 8 odd-episode (held-out) frames, per camera real | twin | IoU overlay (white = both, red =
real arm only, blue = twin arm only).
"""
from __future__ import annotations

import argparse
import dataclasses
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
NAME = "so101_grayblock_twin2"


def session(fs, i):
    """GRAYBLOCK: the side camera's exposure drops from episode 23 on (static_check: the camera does not move), so the
    colour gain is estimated per camera and session (episodes <= 22 / >= 23)."""
    return 0 if int(fs.labels["episode"][i]) <= 22 else 1


def render_frame(rig, sc, fs, i, gain):
    out = {}
    for c in sc.CAMERAS:
        rig.pose(c, "all", F.frame_state(sc, rig.p, c, i))
        sc.frame_setup(rig, fs, i)
        img = rig.arm.render(c, 640, 480, rig.lens[c])
        if gain is not None:
            g = gain[c][session(fs, i)]   # GRAYBLOCK: per session
            img = np.clip(img.astype(np.float32) * np.asarray(g, np.float32) + 0.5, 0, 255).astype(np.uint8)
        out[c] = img
    return out


def estimate_gain(rig, sc, fs):
    """per camera and session: real mean colour / sim mean colour over 12 spread frames of the session."""
    out = {c: [None, None] for c in sc.CAMERAS}
    sess = np.array([session(fs, i) for i in range(fs.n)])
    for s_ in (0, 1):
        idx = np.nonzero(sess == s_)[0]
        probe = idx[np.linspace(0, len(idx) - 1, 12).astype(int)]
        sim = {c: [] for c in sc.CAMERAS}
        real = {c: [] for c in sc.CAMERAS}
        for i in probe:
            fr = render_frame(rig, sc, fs, int(i), None)
            obs = fs.obs(int(i))["images"]
            for c in sc.CAMERAS:
                sim[c].append(fr[c].reshape(-1, 3).mean(0))
                real[c].append(np.asarray(obs[c]).reshape(-1, 3).mean(0))
        for c in sc.CAMERAS:
            out[c][s_] = [float(x) for x in np.clip(np.mean(real[c], 0) / np.maximum(np.mean(sim[c], 0), 1), 0.3, 2.5)]
    return out


def before_after(sc, fs, imgs, mrig, real_masks, per_frame, path, k=8):
    ep = np.asarray(fs.labels["episode"]).astype(int)   # GRAYBLOCK: 8 odd episodes, a different within-episode
    oe = sorted(set(ep[ep % 2 == 1].tolist()))           # frame each (progress 1..6 of 0..7), so the rest pose
    pick = [oe[j] for j in np.linspace(0, len(oe) - 1, k).astype(int)]   # (34 % of the frames) does not fill the sheet
    idx = np.array([np.nonzero(ep == e)[0][(j % 6) + 1] for j, e in enumerate(pick)])
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            re = np.asarray(fs.obs(int(i))["images"][c])
            s = mrig.mask(c, "all", F.frame_state(sc, mrig.p, c, int(i)), 640, 480)
            r = real_masks[c][i]
            ov = np.zeros((480, 640, 3), np.float32)
            ov[r & s] = 255
            ov[r & ~s] = (255, 40, 40)
            ov[~r & s] = (40, 90, 255)
            a = (r | s)[..., None]
            base = re.astype(np.float32) * 0.5
            tiles += [re, imgs[c][i], np.where(a, 0.35 * base + 0.65 * ov, base).astype(np.uint8)]
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width // 3, im.height // 3), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // (3 * len(sc.CAMERAS))
    hh = im.height // len(idx)
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(("real", "twin", "IoU overlay")):
            d.text(((3 * j + t) * w + 4, 2), f"{c}: {lab}", fill=(255, 255, 0))
    for r, i in enumerate(idx):
        for j, c in enumerate(sc.CAMERAS):
            v = per_frame[c]["strict"][i]
            d.text(((3 * j + 2) * w + 4, r * hh + 12), f"frame {i} ep{fs.labels['episode'][i]} IoU "
                   f"{'n/a' if v is None else f'{v:.3f}'}", fill=(255, 255, 0))
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--score", required=True, help="score.py json (strict and, with --occ-tag, occluder-excluded)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sc = F.load_scene("grayblock")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sj = json.loads(pathlib.Path(a.score).read_text())
    model = fj.get("model", sc.ARM.model)
    sc.ARM = dataclasses.replace(sc.ARM, model=model)
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    n = fs.n if not a.limit else min(a.limit, fs.n)
    rig = F.Rig(sc, P, full=True)
    t0 = time.time()
    gain = estimate_gain(rig, sc, fs)
    print("gain", {c: np.round(v, 3).tolist() for c, v in gain.items()}, flush=True)   # [session 0, session 1]
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
    labels["source"] = np.full(n, "grayblock_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: dict(mount=p["mount"], frame="world (table top z = 0, pan axis = z axis, arm reaches +y at pan 0)",
                    fovy_deg=p["fovy"], pose=p["groups"]["all"], distortion=p["lens"], group_rule="single pose (static)",
                    latency_frames=p.get("dt_frames", 0.0), dataset_feature=fs.meta["feature_keys"][c])
            for c, p in P["cams"].items()}
    A = F.A
    st = sj["stats"]
    rmeta = fs.meta
    meta = dict(
        name=NAME, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source="grayblock_twin2", twin_of=sc.REAL_SET,
        image_shape=[480, 640, 3], dataset=rmeta.get("dataset"), feature_keys=rmeta.get("feature_keys"),
        episodes=rmeta.get("episodes"), per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"),
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
        arm_model=(f"TheRobotStudio SO-ARM100 Simulation/SO101 (official), so101_{sc.ARM.calib}_calib.xml"
                   if model == "so101" else "mujoco_menagerie trs_so_arm100 (SO-100 meshes)"),   # GRAYBLOCK
        arm=model, calibration=sc.ARM.calib,
        joint_units="DEGREES: LeRobot v2.1 OLD calibration (robot 'so101', feature names main_*; shoulder_lift 187.9 and "
                    "elbow_flex 179.4 at the folded rest pose, impossible in the normalised -100..100 mode; no joint at "
                    "exactly +-100), so joint scales are fixed at 1 deg per unit; gripper 0..~50 (0 = closed), mapped "
                    "with a fitted grip_scale / grip_off",   # GRAYBLOCK
        joint_map=("q_rad = deg2rad(sign * scale * state + nominal + offset)",
                   dict(sign=(A.SIGNS[sc.ARM.calib] if model == "so101" else A.SIGNS_SO100[sc.ARM.calib]).tolist(),
                        nominal=(np.zeros(5) if model == "so101" else A.NOMINAL_SO100[sc.ARM.calib]).tolist(),
                        joints=list(A.JOINTS[:5]))),
        joint_scales_deg_per_unit=P.get("scales", [1.0] * 5), joint_scales=P.get("scales", [1.0] * 5),
        joint_offsets_deg=P["offsets_deg"], joint_compliance_deg_per_Nm=P.get("compliance", [0.0] * 5),
        grip_scale=P.get("grip_scale", 1.0), grip_off_rad=P.get("grip_off", 0.0),
        gripper_map=f"q = {A.GRIP_LO} + grip_off + grip_scale * gripper/100 * ({A.GRIP_HI} - {A.GRIP_LO}) rad",
        cameras_fit=cams, lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal point offset "
                                     "(cx, cy) as fractions of (width, height); applied as a warp of a padded pinhole render",
        camera_latency_frames={c: P["cams"][c].get("dt_frames", 0.0) for c in sc.CAMERAS}, gain_rgb=gain,
        objects="so101_twin/grayblock/out/objects.json (blue box per episode, gray block per frame: table -> jaws -> box; "
                "from the DINO+SAM object masks of the real frames through the fitted cameras; RGB only, never in the "
                "metric)",   # GRAYBLOCK
        iou_protocol=("whole-arm silhouette IoU, twin ARM-ONLY mask (padded pinhole render warped by the fitted lens onto "
                      "the real 640x480 grid) vs SAM 2.1 mask of the real arm (so101_twin/grayblock/masks/grayblock_<cam>"
                      + sj["mask_tag"] + ".npy, never resampled), frames with the real arm >= 0.1 % of the image; fit on "
                      "EVEN episodes only, held_out = ODD episodes"),
        held_out={c: st[c]["strict_arm_only"]["held_out"] for c in sc.CAMERAS},
        all_frames={c: st[c]["strict_arm_only"]["all_frames"] for c in sc.CAMERAS},
        excluded_not_in_view={c: st[c]["excluded_not_in_view"] for c in sc.CAMERAS},
        scene_code="so101_twin/grayblock/scene_grayblock.py",
        fit_json=str(pathlib.Path(a.fit).resolve().relative_to(F.ROOT)),
        score_json=str(pathlib.Path(a.score).resolve().relative_to(F.ROOT)))
    if sj.get("occ_tag"):
        meta["held_out_real_occluder_excluded"] = {c: st[c]["occluder_excluded"]["held_out"] for c in sc.CAMERAS}
        meta["all_frames_real_occluder_excluded"] = {c: st[c]["occluder_excluded"]["all_frames"] for c in sc.CAMERAS}
        meta["occluder_excluded_fraction_of_union"] = {c: st[c]["excluded_fraction_of_union"] for c in sc.CAMERAS}
        meta["occluder_excluded_fraction_of_twin_mask"] = {c: st[c]["excluded_fraction_of_twin_mask"] for c in sc.CAMERAS}
    extra = HERE / "out" / "meta_extra.json"
    if extra.exists():
        meta.update(json.loads(extra.read_text()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        mrig = F.Rig(sc, P)
        real_masks = {c: F.load_masks("grayblock", c, sj["mask_tag"]) for c in sc.CAMERAS}
        before_after(sc, fs, imgs, mrig, real_masks, sj["per_frame"], HERE / "out" / "before_after.jpg")
        print("wrote", HERE / "out" / "before_after.jpg")


if __name__ == "__main__":
    main()
