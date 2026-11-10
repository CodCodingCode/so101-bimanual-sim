"""orangebox COPY of so101_twin/common/render_set.py (ORANGEBOX: fit json / score json / mask tag as arguments, no old
builder twin, before_after = 8 held-out (odd-episode) frames x per camera real | twin | IoU overlay, meta.json with the
normalised-unit joint map, held_out / all_frames stats from score.py and the excluded-frame counts).

    .venv/bin/python so101_twin/orangebox/render_set.py --fit out/<fit>.json --score out/<fit>_score<tag>.json
        [--limit N] [--out DIR]

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


def before_after(rig, sc, fs, new_imgs: dict, real_masks: dict, ious: dict, path: pathlib.Path, k: int = 8):
    """ORANGEBOX: 8 held-out (odd-episode) frames spread over the set; per camera: real | twin | IoU overlay
    (white = both, red = real SAM mask only, blue = twin only), IoU printed on the overlay."""
    ep = np.asarray(fs.labels["episode"]).astype(int)
    odd = np.nonzero(ep % 2 == 1)[0]
    idx = odd[np.linspace(0, len(odd) - 1, k + 2).astype(int)[1:-1]]
    rows, labels = [], []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            re = np.asarray(fs.obs(int(i))["images"][c])
            grp = F.frame_groups(sc, c)[i]
            s = rig.mask(c, grp, F.frame_state(sc, rig.p, c, int(i)), 640, 480)
            r = real_masks[c][i]
            ov = re.astype(np.float32) * 0.45
            col = np.zeros((480, 640, 3), np.float32)
            col[r & s] = 255
            col[r & ~s] = (255, 40, 40)
            col[~r & s] = (40, 90, 255)
            a = (r | s)[..., None]
            ov = np.where(a, 0.3 * ov + 0.7 * col, ov).astype(np.uint8)
            tiles += [re, new_imgs[c][i], ov]
            labels.append((c, int(i), ious[c][int(i)]))
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width // 3, im.height // 3), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // (3 * len(sc.CAMERAS))
    h = im.height // len(idx)
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(("real", "twin", "IoU overlay")):
            d.text(((3 * j + t) * w + 4, 2), f"{c}: {lab}", fill=(255, 255, 0))
    for r_, i in enumerate(idx):
        for j, c in enumerate(sc.CAMERAS):
            v = ious[c][int(i)]
            d.text(((3 * j + 2) * w + 4, r_ * h + 14), f"frame {int(i)} ep {ep[i]}  IoU {v:.3f}" if v is not None else
                   f"frame {int(i)} not in view", fill=(255, 255, 0))
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="orangebox")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--fit", required=True, help="fit json (orangebox/out/...)")
    ap.add_argument("--score", required=True, help="score json from score.py (held_out / all_frames stats)")
    ap.add_argument("--alt-scores", default="", help="comma-separated further score jsons (other mask sets / models)")
    a = ap.parse_args()
    sc = F.load_scene(a.scene)
    fit_path = pathlib.Path(a.fit) if pathlib.Path(a.fit).is_absolute() else HERE / a.fit
    fj = json.loads(fit_path.read_text())
    sj = json.loads((pathlib.Path(a.score) if pathlib.Path(a.score).is_absolute() else HERE / a.score).read_text())
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
        cams[c] = dict(mount=p["mount"], frame="world (arm frame: base bottom z = 0 = table top, pan axis = z, pan 0 "
                       "reaches +y)", fovy_deg=p["fovy"], poses_by_group=p["groups"], distortion=p["lens"],
                       latency_frames=p.get("dt_frames", 0.0), group_rule=getattr(sc, "GROUP_RULE", {}).get(c, "single pose"))
    alt = {}
    for f in [x for x in a.alt_scores.split(",") if x]:
        o = json.loads((pathlib.Path(f) if pathlib.Path(f).is_absolute() else HERE / f).read_text())
        alt[pathlib.Path(f).name] = dict(model=o.get("model"), masks=o.get("masks"),
                                         held_out={c: o["cams"][c]["held_out"] for c in o["cams"]},
                                         all_frames={c: o["cams"][c]["all_frames"] for c in o["cams"]})
    rmeta = fs.meta
    sign = (F.A.SIGNS[sc.ARM.calib] if sc.ARM.model == "so101" else F.A.SIGNS_SO100[sc.ARM.calib]).tolist()
    nominal = (np.zeros(5) if sc.ARM.model == "so101" else F.A.NOMINAL_SO100[sc.ARM.calib]).tolist()
    meta = dict(name=name, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source=f"{a.scene}_twin2",
                twin_of=sc.REAL_SET, image_shape=[480, 640, 3], dataset=rmeta.get("dataset"),
                feature_keys=rmeta.get("feature_keys"), episodes=rmeta.get("episodes"),
                per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"), built=time.strftime("%Y-%m-%d %H:%M:%S"),
                model=sc.ARM.model,
                arm_model=("TheRobotStudio SO-ARM100 Simulation/SO101 (official), " + f"so101_{sc.ARM.calib}_calib.xml"
                           + (" + wrist camera mount (so101_new_calib_camera.xml)" if sc.ARM.wrist_mount else ""))
                if sc.ARM.model == "so101" else "mujoco_menagerie trs_so_arm100 (SO-100 meshes)",
                calibration=sc.ARM.calib,
                joint_units="LeRobot NORMALISED units (use_degrees=False): -100..100 of each joint's calibrated travel "
                            "(range_m100_100), gripper 0..100 (range_0_100); evidence: meta/stats.json action min "
                            "shoulder_lift = -100.0 and max elbow_flex = +100.0 exactly (the clamp of the normalised mode)",
                joint_map=dict(formula="q_rad[j] = deg2rad(sign[j] * scale[j] * state[j] + nominal[j] + offset[j]), "
                               "j = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll",
                               sign=sign, nominal_deg=nominal, scale_deg_per_unit=P.get("scales", [1.0] * 5),
                               offset_deg=P["offsets_deg"],
                               gripper=f"q = {F.A.GRIP_LO} + grip_off + grip_scale * gripper/100 * "
                                       f"({F.A.GRIP_HI} - ({F.A.GRIP_LO})) rad",
                               grip_scale=P.get("grip_scale", 1.0), grip_off_rad=P.get("grip_off", 0.0)),
                joint_offsets_deg=P["offsets_deg"], joint_scales=P.get("scales", [1.0] * 5),
                grip_scale=P.get("grip_scale", 1.0), grip_off=P.get("grip_off", 0.0), arm=sc.ARM.model,
                cameras_fit=cams, lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal "
                "point offset (cx, cy) as fractions of (width, height); applied as a warp of a padded pinhole render "
                "onto the real 640x480 grid (pixels the render cannot fill count as background)",
                camera_latency_frames={c: P["cams"][c].get("dt_frames", 0.0) for c in sc.CAMERAS},
                gain_rgb=gain,
                iou_protocol=("whole-arm silhouette IoU, twin arm mask (lens-warped onto the real grid) vs SAM 2.1 mask "
                              "of the real arm (so101_twin/orangebox/masks/orangebox_<cam>" + sj["masks"] + ".npy, "
                              "unwarped), per camera; frames with the real arm < 0.1% of the image excluded; fit on "
                              "EVEN episodes only, held_out = ODD episodes"),
                masks=sj["masks"],
                held_out={c: sj["cams"][c]["held_out"] for c in sc.CAMERAS},
                all_frames={c: sj["cams"][c]["all_frames"] for c in sc.CAMERAS},
                excluded_not_in_view={c: sj["cams"][c]["excluded_not_in_view"] for c in sc.CAMERAS},
                excluded_held_out={c: sj["cams"][c]["excluded_held_out"] for c in sc.CAMERAS},
                other_scores=alt or None,
                scene_code="so101_twin/orangebox/scene_orangebox.py",
                fit_json=str(fit_path.relative_to(F.ROOT)))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        real_masks = {c: F.load_masks(a.scene, c, sj["masks"]) for c in sc.CAMERAS}
        ious = {c: sj["cams"][c]["iou"] for c in sc.CAMERAS}
        before_after(rig, sc, fs, imgs, real_masks, ious, F.OUT / "before_after.jpg")
        print("wrote", F.OUT / "before_after.jpg")


if __name__ == "__main__":
    main()
