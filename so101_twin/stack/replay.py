"""Replay every frame of so101_stack_real in the twin scene and write so101_stack_twin in the same layout.

    .venv/bin/python so101_twin/stack/replay.py [--out neuroscope/molmoact2/store/sets/so101_stack_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); the wooden cube placed per episode from the gripper
value: on the disc until the gripper first closes on it (5 <= gripper <= 18 = holding the 4 cm cube), between
the jaws (site `tcp` on Fixed_Jaw, oriented with the jaw) while holding, on top of the green block after the
last holding frame; render "top" and "side" with the fitted cameras and apply a per-camera brightness gain so
the mean intensity matches the real exposure (computed from 8 frames at start-up). Output: meta.json,
images_top.npy, images_side.npy (uint8 [n,480,640,3]), state.npy / prompt.json identical to the real set,
labels.npz = real labels with real_or_sim='sim', source='stack_twin', plus pair = index into the real set;
cube_where.npy (extra) = where the cube was rendered per frame.
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

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_stack_twin"
GAIN = {"top": 1.0, "side": 1.0}        # per-camera brightness gain, set in main() from the real/sim mean intensity
EPISODE_CAMS = ROOT / "so101_twin" / "stack" / "out" / "episode_cams.json"   # per-episode top camera (episode_cams.py)
HOLD = (5.0, 18.0)                      # gripper % range = jaws closed on the cube


def cube_where(fs) -> np.ndarray:
    """Per frame: 'disc' | 'jaws' | 'block' from the gripper timeline of each episode (frames sorted by t)."""
    g, ep, t = fs.state[:, 5], np.asarray(fs.labels["episode"]), np.asarray(fs.labels["t"])
    where = np.full(fs.n, "disc", dtype="<U5")
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(t[idx])]
        hold = (g[idx] >= HOLD[0]) & (g[idx] <= HOLD[1])
        if hold.any():
            first, last = np.argmax(hold), len(idx) - 1 - np.argmax(hold[::-1])
        else:                                   # no held frame sampled: open jaws bracket the transfer
            op = g[idx] > HOLD[1]
            first, last = np.argmax(op) + 1, len(idx) - 1 - np.argmax(op[::-1]) - 1
        for k, i in enumerate(idx):
            where[i] = "disc" if k < first else ("jaws" if k <= last else "block")
    return where


def cube_pose(sim: Sim, where: str) -> tuple[np.ndarray, np.ndarray]:
    """World position + quaternion of the cube centre for this frame."""
    h = build_scene.OBJECTS["cube"]["size"][2]
    ident = np.array([1.0, 0, 0, 0])
    if where == "jaws":
        sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, sim.d.site_xmat[sid].copy())
        return sim.d.site_xpos[sid].copy(), q
    if where == "block":
        gx, gy = build_scene.OBJECTS["green"]["xy"]
        gh = build_scene.OBJECTS["green"]["size"][2]
        return np.array([gx, gy, 2 * gh + h]), ident
    x, y = build_scene.OBJECTS["cube"]["xy"]
    return np.array([x, y, h]), ident


def set_mocap(sim: Sim, body: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, body)
    sim.d.mocap_pos[sim.m.body_mocapid[bid]] = pos
    if quat is not None:
        sim.d.mocap_quat[sim.m.body_mocapid[bid]] = quat


def render_frame(sim: Sim, state: np.ndarray, where: str) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    set_mocap(sim, "cube", *cube_pose(sim, where))
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
    where = cube_where(fs)
    ep_cams = json.loads(EPISODE_CAMS.read_text()) if EPISODE_CAMS.exists() else None
    episode = np.asarray(fs.labels["episode"])

    def use_episode_cam(i: int) -> None:
        if ep_cams is None:
            return
        q = ep_cams["episodes"][str(int(episode[i]))]["params"]
        sim.set_camera(ep_cams["cam"], (q["x"], q["y"], q["z"]), q["yaw"], q["pitch"], q["roll"], q["fovy"])
    t0 = time.time()
    # brightness gain: real / sim mean intensity over 8 spread frames (rendered with gain 1)
    probe = np.linspace(0, fs.n - 1, 8).astype(int)
    for c in twin.CAMS:
        GAIN[c] = 1.0
    real_mean = {c: np.mean([fs.obs(i)["images"][c].mean() for i in probe]) for c in twin.CAMS}
    sim_mean = {}
    for c in twin.CAMS:
        vals = []
        for i in probe:
            use_episode_cam(i)
            vals.append(render_frame(sim, fs.state[i], str(where[i]))[c].mean())
        sim_mean[c] = np.mean(vals)
    for c in twin.CAMS:
        GAIN[c] = float(np.clip(real_mean[c] / max(sim_mean[c], 1.0), 0.5, 1.6))
    print("gain", {c: round(GAIN[c], 3) for c in twin.CAMS}, "real mean", {c: round(float(real_mean[c]), 1) for c in twin.CAMS},
          "sim mean", {c: round(float(sim_mean[c]), 1) for c in twin.CAMS}, flush=True)
    for i in range(n):
        use_episode_cam(i)
        fr = render_frame(sim, fs.state[i], str(where[i]))
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
    labels["source"] = np.full(n, "stack_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    np.save(out / "cube_where.npy", where[:n])          # extra: where the cube was rendered (disc / jaws / block)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    meta = dict(name="so101_stack_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="stack_twin",
                twin_of="so101_stack_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_signs=twin.SIGNS.tolist(), joint_offsets_deg=twin.OFFSETS_DEG.tolist(),
                jaw_range=list(twin.JAW_RANGE), cameras_fit=cams, gain=GAIN,
                objects={k: dict(xy=list(v["xy"]), size=list(v["size"])) for k, v in build_scene.OBJECTS.items()},
                disc=dict(xy=list(build_scene.DISC["xy"]), r=build_scene.DISC["r"]), cube_hold_gripper=list(HOLD),
                joint_scales=twin.SCALES[:5].tolist(),
                episode_cameras=(None if ep_cams is None else {k: v["params"] for k, v in ep_cams["episodes"].items()}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
