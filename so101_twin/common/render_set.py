"""Write the upgraded twin set <scene>_twin2 from a fitted scene (fit.py output) in exactly the existing twin layout.

    .venv/bin/python so101_twin/common/render_set.py lemon [--limit N] [--out DIR]
    .venv/bin/python so101_twin/common/render_set.py lemon --before-after     # only the comparison figure

Per frame: real state -> official SO-101 (fitted offsets / gripper scale), scene.frame_setup() moves objects, every
camera rendered at 640x480 with its fitted pose for the frame's episode group, fovy and lens warp (radial k1, k2 +
principal point, applied to the RGB exactly as to the masks), then a per-camera per-channel colour gain (scene.GAIN or
estimated: real mean colour / sim mean colour over 16 spread frames).
Output neuroscope/molmoact2/store/sets/so101_<scene>_twin2/: images_<cam>.npy uint8 [n,480,640,3], state.npy and
prompt.json copied byte-identical from the real set, labels.npz = real labels + real_or_sim='sim',
source='<scene>_twin2', pair=arange(n); meta.json (twin_of, cameras, fitted params, distortion, IoU stats).
Also so101_twin/common/out/<scene>_before_after.jpg: 8 frames, per camera real | old twin | new twin.
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


def render_frame(rig: F.Rig, sc, fs, i: int, groups: dict, gain: dict | None) -> dict:
    out = {}
    setup = getattr(sc, "frame_setup", None)
    for c in sc.CAMERAS:
        rig.pose(c, groups[c][i], F.frame_state(sc, rig.p, c, i))
        if setup is not None:
            setup(rig, fs, i)
        img = rig.arm.render(c, 640, 480, rig.lens[c])
        if gain is not None:
            img = np.clip(img.astype(np.float32) * np.asarray(gain[c], np.float32) + 0.5, 0, 255).astype(np.uint8)
        out[c] = img
    return out


def estimate_gain(rig, sc, fs, groups) -> dict:
    g = getattr(sc, "GAIN", None)
    if g is not None:
        return {c: list(np.broadcast_to(np.asarray(g[c], float), 3)) for c in sc.CAMERAS}
    probe = np.linspace(0, fs.n - 1, 16).astype(int)
    sim = {c: [] for c in sc.CAMERAS}
    real = {c: [] for c in sc.CAMERAS}
    for i in probe:
        fr = render_frame(rig, sc, fs, int(i), groups, None)
        obs = fs.obs(int(i))["images"]
        for c in sc.CAMERAS:
            sim[c].append(fr[c].reshape(-1, 3).mean(0))
            real[c].append(np.asarray(obs[c]).reshape(-1, 3).mean(0))
    return {c: list(np.clip(np.mean(real[c], 0) / np.maximum(np.mean(sim[c], 0), 1), 0.3, 2.0)) for c in sc.CAMERAS}


def before_after(sc, fs, new_imgs: dict, old_set: str, path: pathlib.Path, k: int = 8):
    from sets import Store

    old = Store().frames(old_set)
    idx = np.linspace(0, fs.n - 1, k + 2).astype(int)[1:-1]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            tiles += [np.asarray(fs.obs(int(i))["images"][c]), np.asarray(old.obs(int(i))["images"][c]), new_imgs[c][i]]
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width // 3, im.height // 3), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // (3 * len(sc.CAMERAS))
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(("real", "old twin", "new twin")):
            d.text(((3 * j + t) * w + 4, 2), f"{c}: {lab}", fill=(255, 255, 0))
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-tag", default="", help="use out/<scene><tag>_fit.json (e.g. _so101 / _so100)")
    a = ap.parse_args()
    sc = F.load_scene(a.scene)
    fj = json.loads((F.OUT / f"{a.scene}{a.fit_tag}_fit.json").read_text())
    if fj.get("model") and fj["model"] != sc.ARM.model:
        import dataclasses

        sc.ARM = dataclasses.replace(sc.ARM, model=fj["model"])
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    n = fs.n if not a.limit else min(a.limit, fs.n)
    rig = F.Rig(sc, P, full=True)
    groups = {c: F.frame_groups(sc, c) for c in sc.CAMERAS}
    t0 = time.time()
    gain = estimate_gain(rig, sc, fs, groups)
    print("gain", {c: np.round(v, 3).tolist() for c, v in gain.items()}, flush=True)
    imgs = {c: np.zeros((n, 480, 640, 3), np.uint8) for c in sc.CAMERAS}
    for i in range(n):
        fr = render_frame(rig, sc, fs, i, groups, gain)
        for c in sc.CAMERAS:
            imgs[c][i] = fr[c]
        if (i + 1) % 80 == 0:
            print(f"{i + 1}/{n} frames {time.time() - t0:.0f}s", flush=True)
    name = f"so101_{a.scene}_twin2"
    out = pathlib.Path(a.out) if a.out else SETS / name
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
    labels["source"] = np.full(n, f"{a.scene}_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {}
    for c, p in P["cams"].items():
        cams[c] = dict(mount=p["mount"], frame="world" if p["mount"] == "world" else f"{p['mount']} body of the SO-101",
                       fovy_deg=p["fovy"], poses_by_group=p["groups"], distortion=p["lens"],
                       group_rule=getattr(sc, "GROUP_RULE", {}).get(c, "single pose") if hasattr(sc, "GROUP_RULE") else
                       ("single pose" if len(p["groups"]) == 1 else "per episode group (see scene file)"))
    # held-out comparison of every arm model fitted for this scene with the same masks / split (out/<scene>_*_fit.json)
    comparison = {}
    for f in sorted(F.OUT.glob(f"{a.scene}_*_fit.json")):
        try:
            o = json.loads(f.read_text())
        except Exception:
            continue
        if o.get("split") == fj.get("split") and o.get("mask_tag") == fj.get("mask_tag") and o.get("model") and \
                "_on_" not in f.name and "_nolag" not in f.name:
            comparison[o["model"]] = dict(fit_json=f"so101_twin/common/out/{f.name}",
                                          held_out_median={c: round(o["stats"][c]["new_heldout"]["median"], 4)
                                                           for c in sc.CAMERAS},
                                          held_out_p10={c: round(o["stats"][c]["new_heldout"]["p10"], 4)
                                                        for c in sc.CAMERAS})
    rmeta = fs.meta
    meta = dict(name=name, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source=f"{a.scene}_twin2",
                twin_of=sc.REAL_SET, image_shape=[480, 640, 3], dataset=rmeta.get("dataset"),
                feature_keys=rmeta.get("feature_keys"), episodes=rmeta.get("episodes"),
                per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"), built=time.strftime("%Y-%m-%d %H:%M:%S"),
                arm_model=("TheRobotStudio SO-ARM100 Simulation/SO101 (official), " + f"so101_{sc.ARM.calib}_calib.xml"
                           + (" + wrist camera mount" if sc.ARM.wrist_mount else "")) if sc.ARM.model == "so101" else
                "mujoco_menagerie trs_so_arm100 (SO-100 meshes, as the builder twins)",
                calibration=sc.ARM.calib,
                joint_map=("q_rad = deg2rad(sign * state + nominal + offset)", dict(
                    sign=(F.A.SIGNS[sc.ARM.calib] if sc.ARM.model == "so101" else F.A.SIGNS_SO100[sc.ARM.calib]).tolist(),
                    nominal=(np.zeros(5) if sc.ARM.model == "so101" else F.A.NOMINAL_SO100[sc.ARM.calib]).tolist())),
                joint_offsets_deg=P["offsets_deg"], joint_scales=P.get("scales", [1.0] * 5),
                grip_scale=P.get("grip_scale", 1.0), arm=sc.ARM.model,
                gripper_map=f"q = {F.A.GRIP_LO} + grip_scale * gripper/100 * ({F.A.GRIP_HI} - {F.A.GRIP_LO}) rad",
                cameras_fit=cams, lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal "
                "point offset (cx, cy) as fractions of (width, height); applied as a warp of the pinhole render",
                camera_latency_frames={c: P["cams"][c].get("dt_frames", 0.0) for c in sc.CAMERAS},
                gain_rgb=gain,
                iou_protocol=("arm IoU vs SAM 2.1 masks (so101_twin/masks/<scene>_<cam>" + fj.get("mask_tag", "")
                              + ".npy), frames with the real arm visible (>= 0.1% of the image); fit on even "
                              "episodes only, held_out = odd episodes"),
                held_out={c: fj["stats"][c]["new_heldout"] for c in sc.CAMERAS},
                all_frames={c: fj["stats"][c]["new"] for c in sc.CAMERAS},
                old_twin_held_out={c: fj["stats"][c]["old_heldout"] for c in sc.CAMERAS},
                old_twin_all_frames={c: fj["stats"][c]["old"] for c in sc.CAMERAS},
                arm_model_comparison=comparison or None, arm_model_chosen=sc.ARM.model,
                scene_code=f"so101_twin/common/scenes/{a.scene}.py",
                fit_json=f"so101_twin/common/out/{a.scene}{a.fit_tag}_fit.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        before_after(sc, fs, imgs, f"so101_{a.scene}_twin", F.OUT / f"{a.scene}_before_after.jpg")
        print("wrote", F.OUT / f"{a.scene}_before_after.jpg")


if __name__ == "__main__":
    main()
