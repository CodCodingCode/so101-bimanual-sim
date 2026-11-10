"""Replay every frame of so101_ringpole_real in the twin scene and write so101_ringpole_twin in the same layout.

    .venv/bin/python so101_twin/ringpole/replay.py [--out neuroscope/molmoact2/store/sets/so101_ringpole_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); pole and ring placed for the frame's episode
(objects.json from place_objects.py); the ring follows the real set's 'phase' label (approach: on the table at its
start position; grasp / transfer_open: between the jaws; release: dropped over the pole); render "top" (static
overhead) and "side" (shoulder camera on the pan frame) with the fitted cameras and apply a per-camera brightness
gain (GAIN). Output: meta.json, images_top.npy, images_side.npy (uint8 [n,480,640,3]), state.npy / prompt.json
identical to the real set, labels.npz = real labels with real_or_sim='sim', source='ringpole_twin', plus
pair = index into the real set, and sim_qpos.npy (the qpos actually rendered).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from fit_cameras import episode_layout, load_objects  # noqa: E402
from twin import ROOT, Sim, real_set  # noqa: E402

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_ringpole_twin"
GAIN = {"top": 1.09, "side": 1.12}    # per-camera brightness gain: twin mean intensity -> real mean (118 / 128 over the set)


def ring_pose(sim: Sim, phase: str, lay: dict) -> tuple[np.ndarray, np.ndarray]:
    """World position / quaternion of the ring centre for this frame's phase."""
    h = build_scene.RING["h"]
    if phase in ("grasp", "transfer_open"):
        # held between the jaws: ring centre ~1 cm below the tcp, ring plane horizontal (the real ring hangs
        # flat from the vertical jaws), slightly offset toward the fingers
        p = sim.d.site_xpos[sim.tcp].copy()
        p[2] = max(p[2] - 0.012, h / 2)
        return p, np.array([1.0, 0, 0, 0])
    if phase == "release":
        px, py = lay["pole"]
        return np.array([px, py, build_scene.POLE["base_h"] + h / 2]), np.array([1.0, 0, 0, 0])
    rx, ry = lay["ring"]
    return np.array([rx, ry, h / 2]), np.array([1.0, 0, 0, 0])


def render_frame(sim: Sim, objects: dict | None, state: np.ndarray, episode: int, phase: str) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    lay = episode_layout(objects, episode)
    sim.set_mocap("pole", [lay["pole"][0], lay["pole"][1], 0.0])
    # the first episodes have a bare pole (no purple disc): shrink the base geom to nothing for those
    base = sim.m.geom("pole_base").id
    sim.m.geom_size[base, 0] = 1e-4 if lay.get("pole_kind") == "rod" else build_scene.POLE["base_r"]
    pos, quat = ring_pose(sim, phase, lay)
    sim.set_mocap("ring", pos, quat)
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
    objects = load_objects()
    n = fs.n if not a.limit else min(a.limit, fs.n)
    imgs = {c: np.zeros((n, twin.H, twin.W, 3), np.uint8) for c in twin.CAMS}
    qpos = np.zeros((n, 6), np.float32)
    t0 = time.time()
    for i in range(n):
        fr = render_frame(sim, objects, fs.state[i], int(fs.labels["episode"][i]), str(fs.labels["phase"][i]))
        qpos[i] = sim.d.qpos[:6]
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
    np.save(out / "sim_qpos.npy", qpos)
    labels = {k: np.asarray(v)[:n] for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.full(n, "sim")
    labels["source"] = np.full(n, "ringpole_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    cams["top"]["frame"] = "world"
    cams["side"]["frame"] = "pan_frame (child of Rotation_Pitch: x = arm forward, z = up, origin at the pan axis)"
    fit = {}
    for c in twin.CAMS:
        p = twin.HERE / "out" / f"camera_fit_{c}.json"
        if p.exists():
            d = json.loads(p.read_text())
            fit[c] = dict(iou_640x480=d["iou_640x480"], iou_320x240=d["iou_320x240"], frames=d["frames"])
    meta = dict(name="so101_ringpole_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="ringpole_twin",
                twin_of="so101_ringpole_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                feature_keys=fs.meta.get("feature_keys"), episodes=fs.meta.get("episodes"),
                per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_mapping="sim_rad = deg2rad(signs * real_deg + offsets_deg); jaw = jaw_range[0] + gripper/100 * (jaw_range[1] - jaw_range[0])",
                joint_order_real=list(twin.LEROBOT_ORDER), joint_order_sim=list(twin.SIM_ORDER),
                joint_signs=twin.SIGNS[:5].tolist(), joint_offsets_deg=twin.OFFSETS_DEG[:5].tolist(),
                jaw_range=list(twin.JAW_RANGE), cameras_fit=cams, camera_fit_iou=fit, gain=GAIN,
                objects_per_episode=objects is not None)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
