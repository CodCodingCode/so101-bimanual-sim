"""Replay every frame of so101_drawer_real in the twin scene and write so101_drawer_twin in the same layout.

    .venv/bin/python so101_twin/drawer/replay.py [--out neuroscope/molmoact2/store/sets/so101_drawer_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); the purple "left block" placed by the real set's
'phase' label (approach: on the table at its start position; grasp / transfer_open: between the jaws; release:
on the drawer tray floor); the drawer tray pulled out when the real overhead frame shows it open
(drawer_open(): brightness of the strip in front of the drawer unit relative to the unit's top, see README);
render "top" (wrist camera) and "side" (overhead) with the fitted cameras, apply a per-camera brightness gain
(GAIN) so the mean intensity roughly matches the real exposure. Output: meta.json, images_top.npy,
images_side.npy (uint8 [n,480,640,3]), state.npy / prompt.json identical to the real set, labels.npz = real
labels with real_or_sim='sim', source='drawer_twin', plus pair = index into the real set.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from twin import ROOT, Sim, real_set  # noqa: E402

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_drawer_twin"
GAIN = {"top": 1.06, "side": 1.66}      # per-camera brightness gain (brightness_gain.py: real/sim mean intensity)
BLOCK = "purple"                        # the "left block" that is moved
# overhead pixel regions (y0:y1, x0:x1) used by drawer_open(): the unit's top plate and the strip in front of it
DRAWER_TOP_PX = (slice(10, 50), slice(200, 300))
DRAWER_FRONT_PX = (slice(70, 130), slice(200, 300))
DRAWER_OPEN_THRESH = 0.5


def drawer_open(fs) -> np.ndarray:
    """bool [n]: the real overhead frame shows the drawer tray pulled out (the strip in front of the unit is as
    bright as the unit's top). Normalised by the top plate so the dark-lit episodes work too."""
    out = np.zeros(fs.n, bool)
    for i in range(fs.n):
        im = fs.obs(i)["images"]["side"].astype(int).min(2)
        ref = np.percentile(im[DRAWER_TOP_PX], 50)
        out[i] = (im[DRAWER_FRONT_PX] > 0.55 * ref).mean() > DRAWER_OPEN_THRESH
    return out


def block_pose(sim: Sim, phase: str, tray_y: float) -> np.ndarray:
    """World position of the purple block centre for this frame's phase."""
    hz = build_scene.OBJECTS[BLOCK]["size"][2]
    if phase in ("grasp", "transfer_open"):
        sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        return sim.d.site_xpos[sid].copy()
    if phase == "release":
        cx, cy = build_scene.DRAWER_UNIT["center"]
        ty = build_scene.DRAWER_TRAY["size"][1]
        return np.array([cx, tray_y - build_scene.DRAWER_UNIT["size"][1] / 2 + ty / 2, 0.012 + hz])
    x, y = build_scene.OBJECTS[BLOCK]["xy"]
    return np.array([x, y, hz])


def set_mocap(sim: Sim, body: str, pos: np.ndarray) -> None:
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, body)
    sim.d.mocap_pos[sim.m.body_mocapid[bid]] = pos


def render_frame(sim: Sim, state: np.ndarray, phase: str, opened: bool = False) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    cx, cy = build_scene.DRAWER_UNIT["center"]
    tray_y = cy - (build_scene.DRAWER_TRAVEL if opened else 0.0)
    set_mocap(sim, "tray", np.array([cx, tray_y, 0.0]))
    set_mocap(sim, BLOCK, block_pose(sim, phase, tray_y))
    mujoco.mj_forward(sim.m, sim.d)
    out = {}
    for c in twin.CAMS:
        img = sim.render(c).astype(np.float32) * GAIN[c]
        out[c] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    fs = real_set()
    sim = Sim()
    n = fs.n if not a.limit else min(a.limit, fs.n)
    imgs = {c: np.zeros((n, twin.H, twin.W, 3), np.uint8) for c in twin.CAMS}
    t0 = time.time()
    opened = drawer_open(fs)
    print(f"drawer open in {opened.sum()}/{fs.n} real frames", flush=True)
    for i in range(n):
        fr = render_frame(sim, fs.state[i], str(fs.labels["phase"][i]), bool(opened[i]))
        for c in twin.CAMS:
            imgs[c][i] = fr[c]
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{n} frames, {time.time() - t0:.0f}s", flush=True)
    sim.close()

    out.mkdir(parents=True, exist_ok=True)
    for c in twin.CAMS:
        np.save(out / f"images_{c}.npy", imgs[c])
    np.save(out / "state.npy", fs.state[:n].astype(np.float32))
    (out / "prompt.json").write_text(json.dumps(fs.prompt[:n]))
    labels = {k: np.asarray(v)[:n] for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.full(n, "sim")
    labels["source"] = np.full(n, "drawer_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    for c in twin.CAMS:
        cams[c]["frame"] = build_scene.CAM_BODY[c] or "world"
        cams[c]["angles"] = "yaw/pitch/roll deg, see build_scene.cam_xyaxes"
    meta = dict(name="so101_drawer_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="drawer_twin",
                twin_of="so101_drawer_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                feature_keys=fs.meta.get("feature_keys"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_map=dict(order=list(twin.LEROBOT_ORDER), sim_order=list(twin.SIM_ORDER),
                               signs=twin.SIGNS[:5].tolist(), offsets_deg=twin.OFFSETS_DEG[:5].tolist(),
                               jaw_range=list(twin.JAW_RANGE),
                               formula="sim_rad = deg2rad(signs*real_deg + offsets_deg); jaw = -0.174 + gripper/100*1.924"),
                cameras_fit=cams, gain=GAIN, drawer_open_frames=int(opened.sum()),
                notes="'top' is the SO-101 wrist camera (moves with the gripper); 'side' is the static overhead camera")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
