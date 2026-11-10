"""Replay every frame of so101_lemon_real in the twin scene and write so101_lemon_twin in the same layout.

    .venv/bin/python so101_twin/lemon/replay.py [--out neuroscope/molmoact2/store/sets/so101_lemon_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim), lemon placed by the real set's 'phase' label
(approach: on the table at its start position; grasp / transfer_open: between the jaws; release: in the bowl),
render "top" and "side" with the fitted cameras, apply a per-camera brightness gain (GAIN) so the mean
intensity roughly matches the real camera exposure. Output: meta.json, images_top.npy, images_side.npy
(uint8 [n,480,640,3]), state.npy / prompt.json identical to the real set, labels.npz = real labels with
real_or_sim='sim', source='lemon_twin', plus pair = index into the real set.
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

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_lemon_twin"
GAIN = {"top": 0.92, "side": 0.67}      # per-camera brightness gain (set by brightness_gain.py)


def lemon_pose(sim: Sim, phase: str) -> np.ndarray:
    """World position of the lemon centre for this frame's phase."""
    r = build_scene.OBJECTS["lemon"]["size"][2]
    if phase in ("grasp", "transfer_open"):
        sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        return sim.d.site_xpos[sid].copy()
    if phase == "release":
        bx, by = build_scene.BOWL["xy"]
        return np.array([bx, by, r + 0.004])
    x, y = build_scene.OBJECTS["lemon"]["xy"]
    return np.array([x, y, r])


def set_mocap(sim: Sim, body: str, pos: np.ndarray) -> None:
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, body)
    sim.d.mocap_pos[sim.m.body_mocapid[bid]] = pos


def render_frame(sim: Sim, state: np.ndarray, phase: str) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    set_mocap(sim, "lemon", lemon_pose(sim, phase))
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
    for i in range(n):
        fr = render_frame(sim, fs.state[i], str(fs.labels["phase"][i]))
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
    labels["source"] = np.full(n, "lemon_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    meta = dict(name="so101_lemon_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="lemon_twin",
                twin_of="so101_lemon_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_signs=twin.SIGNS.tolist(), joint_offsets_deg=twin.OFFSETS_DEG.tolist(),
                jaw_range=list(twin.JAW_RANGE), cameras_fit=cams, gain=GAIN)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
