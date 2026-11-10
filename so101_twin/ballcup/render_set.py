"""Write the twin set so101_ballcup_twin2 from a fitted ballcup twin (ballcup copy of so101_twin/common/render_set.py).

    .venv/bin/python so101_twin/ballcup/render_set.py --fit out/<run>_fit.json --score out/score_final.json \
        [--score-occ out/score_final_occ.json] [--limit N]

Per frame: real state -> fitted arm (offsets / gripper scale / latency per camera), objects placed (scene
frame_setup: cup at its episode position, ball at its frame position), every camera rendered at 640x480 with its
fitted pose, fovy and lens warp (the same warp as the masks), then a per-camera per-channel gain (real mean colour /
sim mean colour over 16 spread frames). Output neuroscope/molmoact2/store/sets/so101_ballcup_twin2/ in the existing
twin layout; meta.json carries the fitted params and the score json's held_out / all_frames stats.
Also out/before_after.jpg: 8 odd-episode (held-out) frames, per camera real | twin | IoU overlay
(white = both, red = real arm only, blue = twin arm only).
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
NAME = "so101_ballcup_twin2"


def render_frame(rig, sc, fs, i, gain):
    out = {}
    for c in sc.CAMERAS:
        rig.pose(c, "all", F.frame_state(sc, rig.p, c, i))
        sc.frame_setup(rig, fs, i)
        img = rig.arm.render(c, 640, 480, rig.lens[c])
        if gain is not None:
            img = np.clip(img.astype(np.float32) * np.asarray(gain[c], np.float32) + 0.5, 0, 255).astype(np.uint8)
        out[c] = img
    return out


def estimate_gain(rig, sc, fs):
    probe = np.linspace(0, fs.n - 1, 16).astype(int)
    sim = {c: [] for c in sc.CAMERAS}
    real = {c: [] for c in sc.CAMERAS}
    for i in probe:
        fr = render_frame(rig, sc, fs, int(i), None)
        obs = fs.obs(int(i))["images"]
        for c in sc.CAMERAS:
            sim[c].append(fr[c].reshape(-1, 3).mean(0))
            real[c].append(np.asarray(obs[c]).reshape(-1, 3).mean(0))
    return {c: [float(x) for x in np.clip(np.mean(real[c], 0) / np.maximum(np.mean(sim[c], 0), 1), 0.3, 2.0)]
            for c in sc.CAMERAS}


def before_after(sc, fs, imgs, mrig, real_masks, per_frame, path, k=8):
    odd = np.nonzero(np.asarray(fs.labels["episode"]).astype(int) % 2 == 1)[0]
    idx = odd[np.linspace(0, len(odd) - 1, k + 2).astype(int)[1:-1]]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            re = np.asarray(fs.obs(int(i))["images"][c])
            s = mrig.mask(c, "all", F.frame_state(sc, mrig.p, c, int(i)), 640, 480, i=int(i))
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
            v = per_frame[c][i]
            d.text(((3 * j + 2) * w + 4, r * hh + 12), f"frame {i} ep{fs.labels['episode'][i]} IoU "
                   f"{'n/a' if v is None else f'{v:.3f}'}", fill=(255, 255, 0))
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--score", required=True, help="score.py json for the primary metric (held_out / all_frames)")
    ap.add_argument("--score-plain", default="", help="score.py json WITHOUT --occ (unoccluded arm silhouette)")
    ap.add_argument("--score-indep", default="", help="score.py --occ json against the twin-independent masks (_d1o)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sj = json.loads(pathlib.Path(a.score).read_text())
    import dataclasses

    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    n = fs.n if not a.limit else min(a.limit, fs.n)
    rig = F.Rig(sc, P, full=True)
    t0 = time.time()
    ob = rig.obj   # the placements actually used for the RGB (and the visible-arm masks)
    (HERE / "out" / "objects_final.json").write_text(json.dumps(dict(
        fit=a.fit, cup_shape_m=P.get("cup"), ball_r_m=sc.BALL_R,
        cup_xy_by_episode={str(e): [round(float(v), 4) for v in xy] for e, xy in ob["cup"].items()},
        ball_by_frame={str(i): (None if b is None else ("top-ray on table or held at the tool point: " +
                                                         str([round(float(v), 4) for v in b[1]])) if isinstance(b, tuple)
                                else [round(float(v), 4) for v in b]) for i, b in ob["ball"].items()}), indent=1))
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
    labels["source"] = np.full(n, "ballcup_twin2")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: dict(mount=p["mount"], frame="world (table top z = 0, pan axis = z axis, arm reaches +y at pan 0)",
                    fovy_deg=p["fovy"], pose=p["groups"]["all"], distortion=p["lens"], group_rule="single pose (static)",
                    latency_frames=p.get("dt_frames", 0.0)) for c, p in P["cams"].items()}
    A = F.A
    model = fj.get("model", sc.ARM.model)
    rmeta = fs.meta
    st = sj["stats"]
    meta = dict(
        name=NAME, n=n, cameras=list(sc.CAMERAS), real_or_sim="sim", source="ballcup_twin2", twin_of=sc.REAL_SET,
        image_shape=[480, 640, 3], dataset=rmeta.get("dataset"), feature_keys=rmeta.get("feature_keys"),
        episodes=rmeta.get("episodes"), per_episode=rmeta.get("per_episode"), fps=rmeta.get("fps"),
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
        arm_model=("TheRobotStudio SO-ARM100 Simulation/SO101 (official), so101_old_calib.xml" if model == "so101" else
                   "mujoco_menagerie trs_so_arm100 (SO-100 meshes)"),
        arm=model, calibration=sc.ARM.calib,
        joint_map=("q_rad = deg2rad(sign * scale * state + nominal + offset)  (state = LeRobot v2.1 old-calibration "
                   "degrees)", dict(sign=(A.SIGNS[sc.ARM.calib] if model == "so101" else A.SIGNS_SO100[sc.ARM.calib]).tolist(),
                                    nominal=(np.zeros(5) if model == "so101" else A.NOMINAL_SO100[sc.ARM.calib]).tolist())),
        joint_offsets_deg=P["offsets_deg"], joint_scales=P.get("scales", [1.0] * 5), grip_scale=P.get("grip_scale", 1.0),
        grip_off_rad=P.get("grip_off", 0.0),
        gripper_map=f"q = {A.GRIP_LO} + grip_off + grip_scale * gripper/100 * ({A.GRIP_HI} - {A.GRIP_LO}) rad "
                    "(gripper 0 = closed)",
        cameras_fit=cams, lens_model="x_d = x_u (1 + k1 r^2 + k2 r^4), x in focal-length units; principal point offset "
                                     "(cx, cy) as fractions of (width, height); applied as a warp of a padded pinhole render",
        camera_latency_frames={c: P["cams"][c].get("dt_frames", 0.0) for c in sc.CAMERAS}, gain_rgb=gain,
        objects=dict(source="so101_twin/ballcup/out/objects_final.json (cup per episode, ball per frame; from the "
                            "DINO+SAM object masks of both cameras through the fitted cameras, scene_ballcup.update_objects)",
                     cup_shape_m=P.get("cup"), ball_r_m=sc.BALL_R),
        iou_protocol=("whole-arm silhouette IoU, twin arm mask (padded pinhole render warped by the fitted lens onto the "
                      "real 640x480 grid) vs SAM 2.1 mask of the real arm (so101_twin/ballcup/masks/ballcup_<cam>"
                      + sj["mask_tag"] + ".npy), frames with the real arm >= 0.1 % of the image; fit on EVEN episodes "
                      "only, held_out = ODD episodes" + ("; twin mask with cup/ball occlusion" if sj.get("occluders") else
                                                         "; twin mask = unoccluded arm silhouette")),
        held_out={c: st[c]["held_out"] for c in sc.CAMERAS}, all_frames={c: st[c]["all_frames"] for c in sc.CAMERAS},
        excluded_not_in_view={c: st[c]["excluded_not_in_view"] for c in sc.CAMERAS},
        scene_code="so101_twin/ballcup/scene_ballcup.py",
        fit_json=str(pathlib.Path(a.fit).resolve().relative_to(F.ROOT)), score_json=a.score)
    cmpf = HERE / "out" / "model_comparison.json"
    if cmpf.exists():   # mesh / round comparison and which round final_fit.json is (finalize.py)
        cm = json.loads(cmpf.read_text())
        meta["fit_source"] = cm.get("chosen_fit")
        meta["arm_model_comparison"] = dict(metric=cm.get("metric"), mask_tag=cm.get("mask_tag"), rows=cm.get("rows"),
                                            rule="within a mesh: better EVEN-episode score; between meshes: better held-out")
    meta["metric_reading"] = ("held_out / all_frames above: VISIBLE-arm reading (the twin's arm mask rendered in the twin "
                              "scene, the twin's cup / ball hide the arm pixels behind them, as the real cup / ball hide "
                              "them from SAM)" if sj.get("occluders") else "unoccluded arm silhouette")
    if a.score_plain:
        so = json.loads(pathlib.Path(a.score_plain).read_text())
        meta["unoccluded_arm_silhouette"] = dict(
            note="same twin and masks, arm rendered alone (no cup / ball occlusion), as the lemon / legocup fits scored",
            held_out={c: so["stats"][c]["held_out"] for c in sc.CAMERAS},
            all_frames={c: so["stats"][c]["all_frames"] for c in sc.CAMERAS})
    if a.score_indep:
        so = json.loads(pathlib.Path(a.score_indep).read_text())
        meta["twin_independent_mask_check"] = dict(
            note="same twin (visible arm) vs the purely twin-independent DINO+SAM masks _d1o (ball / cup removed; they "
                 "still contain the power plug / cable, so they understate any twin)", mask_tag=so["mask_tag"],
            held_out={c: so["stats"][c]["held_out"] for c in sc.CAMERAS},
            all_frames={c: so["stats"][c]["all_frames"] for c in sc.CAMERAS})
    extra = HERE / "out" / "meta_extra.json"
    if extra.exists():
        meta.update(json.loads(extra.read_text()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    if n == fs.n:
        sc2 = F.load_scene("ballcup")
        sc2.ARM = sc.ARM
        sc2.MASK_OCCLUDERS = bool(sj.get("occluders"))
        mrig = F.Rig(sc2, P)
        real_masks = {c: F.load_masks("ballcup", c, sj["mask_tag"]) for c in sc.CAMERAS}
        before_after(sc2, fs, imgs, mrig, real_masks, sj["per_frame"], HERE / "out" / "before_after.jpg")
        print("wrote", HERE / "out" / "before_after.jpg")


if __name__ == "__main__":
    main()
