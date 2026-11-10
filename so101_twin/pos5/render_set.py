"""pos5 COPY of so101_twin/common/render_set.py (POS5: fit json / score jsons as arguments, both IoU readings and
the excluded fractions in meta.json, before_after = real | twin | IoU overlay on 8 odd-episode frames).

Write the upgraded twin set <scene>_twin2 from a fitted scene (fit.py output) in exactly the existing twin layout.

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


def before_after(sc, fs, imgs: dict, masks: dict, twin: dict, path: pathlib.Path, k: int = 8):   # POS5
    """8 odd-episode frames; per camera: real | twin render | IoU overlay (white both, red real only, blue twin only)."""
    odd = np.flatnonzero(np.asarray(fs.labels["episode"]).astype(int) % 2 == 1)
    idx = odd[np.linspace(0, len(odd) - 1, k + 2).astype(int)[1:-1]]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            real = np.asarray(fs.obs(int(i))["images"][c])
            R, T = masks[c][i], twin[c][i]
            ov = (real * 0.35).astype(np.uint8)
            ov[R & T] = 255
            ov[R & ~T] = (255, 40, 40)
            ov[~R & T] = (40, 90, 255)
            u = (R | T).sum()
            t = Image.fromarray(np.concatenate([real, imgs[c][i], ov], 1))
            ImageDraw.Draw(t).text((1290, 6), f"IoU {(R & T).sum() / max(u, 1):.3f}", fill=(255, 255, 0))
            ImageDraw.Draw(t).text((6, 6), f"{c} frame {int(i)} ep {int(fs.labels['episode'][i])}", fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        rows.append(np.concatenate(tiles, 1))
    im = Image.fromarray(np.concatenate(rows, 0))
    im = im.resize((im.width // 3, im.height // 3), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    w = im.width // (3 * len(sc.CAMERAS))
    for j, c in enumerate(sc.CAMERAS):
        for t, lab in enumerate(("real", "twin", "IoU: white both / red real only / blue twin only")):
            d.text(((3 * j + t) * w + 4, im.height - 12), f"{c}: {lab}", fill=(255, 255, 0))
    im.save(path, quality=88)


def main():   # POS5
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="fit json (pos5/out/...)")
    ap.add_argument("--score", required=True, help="score json of that fit on the primary masks (score.py)")
    ap.add_argument("--extra-scores", default="", help="comma list of more score jsons (other mask sets) for meta")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import p5

    sc = F.load_scene("pos5")
    fp = pathlib.Path(a.fit) if pathlib.Path(a.fit).is_absolute() else HERE / a.fit
    fj = json.loads(fp.read_text())
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
    name = "so101_pos5_twin2"
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
    labels["source"] = np.full(n, "pos5_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {}
    for c, p in P["cams"].items():
        cams[c] = dict(mount=p["mount"], frame="world", fovy_deg=p["fovy"], poses_by_group=p["groups"],
                       distortion=p["lens"], principal_point_shift_px=[p["lens"]["cx"] * 640, p["lens"]["cy"] * 480],
                       latency_frames=p.get("dt_frames", 0.0), group_rule=sc.GROUP_RULE.get(c, "single pose"))
    sj = json.loads(pathlib.Path(a.score).read_text())
    tag = sj["masks"].split(",")[0]
    R = sj["results"][tag]
    readings = {}
    for f in [a.score] + [x for x in a.extra_scores.split(",") if x]:
        o = json.loads(pathlib.Path(f).read_text())
        for t, rr in o["results"].items():
            readings[t] = {c: {k: v for k, v in rr[c].items() if not k.startswith("per_frame")} for c in rr}
    rmeta = fs.meta
    meta = dict(name=name, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source="pos5_twin2",
                twin_of=sc.REAL_SET, image_shape=[480, 640, 3], dataset=rmeta.get("dataset"),
                feature_keys=rmeta.get("feature_keys"), episodes=rmeta.get("episodes"),
                per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"), built=time.strftime("%Y-%m-%d %H:%M:%S"),
                model=("TheRobotStudio SO-ARM100 Simulation/SO101 (official), so101_new_calib.xml" if sc.ARM.model ==
                       "so101" else "mujoco_menagerie trs_so_arm100 (SO-100 meshes)"),
                arm=sc.ARM.model, calibration=sc.ARM.calib,
                arm_base=dict(pos=list(sc.ARM.base_pos), yaw_deg=sc.ARM.base_yaw_deg,
                              note="normalised arm frame (pan axis = z through the origin, base bottom z = 0) placed "
                                   "at this world pose; world = the fitkit session's frame"),
                joint_units="LeRobot so101_follower NORMALISED: -100..100 of each joint's calibrated travel (gripper "
                            "0..100)",
                joint_map=("q_rad = deg2rad(scale * state + offset_deg) per joint (shoulder_pan, shoulder_lift, "
                           "elbow_flex, wrist_flex, wrist_roll), all signs +1 for so101_new_calib.xml" if
                           sc.ARM.model == "so101" else "q_rad = deg2rad(sign_so100 * scale * state + nominal + offset)"),
                joint_scales_deg_per_unit=P.get("scales", [1.0] * 5), joint_offsets_deg=P["offsets_deg"],
                grip_scale=P.get("grip_scale", 1.0), grip_off_rad=P.get("grip_off", 0.0),
                gripper_map=f"q = {F.A.GRIP_LO} + grip_scale * gripper/100 * ({F.A.GRIP_HI} - {F.A.GRIP_LO}) + grip_off rad",
                cameras_fit=cams, lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal "
                "point offset (cx, cy) as fractions of (width, height); the pinhole render (padded) is warped onto the "
                "real 640 x 480 grid; the real masks are never resampled",
                camera_latency_frames={c: P["cams"][c].get("dt_frames", 0.0) for c in sc.CAMERAS},
                gain_rgb=gain,
                iou_protocol=("whole-arm silhouette IoU: twin ARM-ONLY segmentation render (lens-warped onto the real "
                              "grid) vs the real arm mask pos5/masks/pos5_<cam>" + tag + ".npy; frames with the real arm "
                              "visible (>= 0.1% of the image); fit on EVEN episodes only, held_out = ODD episodes"),
                held_out={c: R[c]["strict"]["held_out"] for c in sc.CAMERAS},
                all_frames={c: R[c]["strict"]["all_frames"] for c in sc.CAMERAS},
                held_out_occluder_excluded={c: R[c]["excluded"]["held_out"] for c in sc.CAMERAS} if "excluded" in
                R[sc.CAMERAS.__iter__().__next__()] else None,
                all_frames_occluder_excluded={c: R[c]["excluded"]["all_frames"] for c in sc.CAMERAS} if "excluded" in
                R[sc.CAMERAS.__iter__().__next__()] else None,
                occluder_excluded_fraction={c: R[c].get("excluded_fraction") for c in sc.CAMERAS},
                readings_by_mask_set=readings,
                pass_criterion="held-out median >= 0.80 on every camera",
                pass_claim={c: ("strict" if R[c]["strict"]["held_out"]["median"] >= 0.8 else
                                ("occluder_excluded" if "excluded" in R[c] and R[c]["excluded"]["held_out"]["median"]
                                 >= 0.8 else "FAIL")) for c in sc.CAMERAS},
                arm_parts_dropped=list(sc.ARM.drop_meshes),
                masks=dict(primary=f"so101_twin/pos5/masks/pos5_<cam>{tag}.npy (packed bits, np.unpackbits(a, -1))",
                           how="chroma key against the green screen (twin-free) outside the static box / wall / mat "
                               "pixels, SAM 2.1 (prompted by the even-episode twin, chroma consensus, hard negatives) "
                               "inside them; minus the cyan cube, the pale wall corner and the side view's table "
                               "clamp (combine.py)",
                           occluders="so101_twin/pos5/masks/pos5_<cam>_occ.npy: real cube pixels (hue), visible white "
                                     "box pixels (box region of the median image, unchanged in the frame), side-view "
                                     "table clamp; from the real frames only, fit frozen (occ_masks.py)"),
                guard_rails={c: dict(k1=P["cams"][c]["lens"]["k1"], k1_over_0p4=abs(P["cams"][c]["lens"]["k1"]) > 0.4,
                                     principal_point_shift_frac=[P["cams"][c]["lens"]["cx"], P["cams"][c]["lens"]["cy"]],
                                     pp_over_10pct=max(abs(P["cams"][c]["lens"]["cx"]), abs(P["cams"][c]["lens"]["cy"]))
                                     > 0.1) for c in sc.CAMERAS},
                scene_code="so101_twin/pos5/scene_pos5.py", fit_json=str(fp.relative_to(F.ROOT)),
                score_json=str(pathlib.Path(a.score).resolve().relative_to(F.ROOT)))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        masks = {c: p5.load(c, tag) for c in sc.CAMERAS}
        twin = p5.twin_masks(P, list(sc.CAMERAS), sc=sc)
        before_after(sc, fs, imgs, masks, twin, F.OUT / "before_after.jpg")
        print("wrote", F.OUT / "before_after.jpg")


if __name__ == "__main__":
    main()
