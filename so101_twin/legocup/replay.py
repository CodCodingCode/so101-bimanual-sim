"""Replay every frame of so101_legocup_real in the twin scene and write so101_legocup_twin in the same layout.

    .venv/bin/python so101_twin/legocup/replay.py [--out neuroscope/molmoact2/store/sets/so101_legocup_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); the cup at the episode's position and the block
placed by the real set's 'phase' label (approach: on the table at the episode's start position; grasp /
transfer_open: between the jaws, following the gripper; release: in the cup), both from out/objects.json
(place_objects.py); render "top" (fixed overhead camera) and "side" (wrist camera on the Fixed_Jaw body) with the
fitted camera parameters, apply a per-camera brightness gain (GAIN) so the mean intensity roughly matches the
real camera exposure. Output: meta.json, images_top.npy, images_side.npy (uint8 [n,480,640,3]), state.npy /
prompt.json identical to the real set, labels.npz = real labels with real_or_sim='sim', source='legocup_twin',
plus pair = index into the real set.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from twin import HERE, ROOT, Sim, real_set  # noqa: E402

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_legocup_twin"
GAIN = {"top": 1.03, "side": 0.92}      # per-camera brightness gain (set by brightness_gain.py)
OBJECTS = json.loads((HERE / "out" / "objects.json").read_text()) if (HERE / "out" / "objects.json").exists() else {}


def episode_objects(ep: int) -> tuple[np.ndarray, np.ndarray]:
    o = OBJECTS.get(str(int(ep)))
    cup = np.array(o["cup"] if o else build_scene.CUP["xy"], float)
    blk = np.array(o["block"] if o else build_scene.BLOCK["xy"], float)
    return cup, blk


def set_mocap(sim: Sim, body: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, body)
    k = sim.m.body_mocapid[bid]
    sim.d.mocap_pos[k] = pos
    if quat is not None:
        sim.d.mocap_quat[k] = quat


def block_pose(sim: Sim, phase: str, cup_xy: np.ndarray, blk_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World position / quaternion of the block centre for this frame's phase."""
    hz = build_scene.BLOCK["half"][2]
    ident = np.array([1.0, 0, 0, 0])
    if phase in ("grasp", "transfer_open"):
        sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, sim.d.xmat[bid])
        return sim.d.site_xpos[sid].copy(), q
    if phase == "release":
        return np.array([cup_xy[0], cup_xy[1], 0.006 + hz]), ident
    return np.array([blk_xy[0], blk_xy[1], hz]), ident


def render_frame(sim: Sim, state: np.ndarray, phase: str, ep: int) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    cup_xy, blk_xy = episode_objects(ep)
    set_mocap(sim, "cup", np.array([cup_xy[0], cup_xy[1], 0.0]))
    mujoco.mj_forward(sim.m, sim.d)                      # tcp site / jaw pose for this state
    pos, quat = block_pose(sim, phase, cup_xy, blk_xy)
    set_mocap(sim, "block", pos, quat)
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
        fr = render_frame(sim, fs.state[i], str(fs.labels["phase"][i]), int(fs.labels["episode"][i]))
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
    labels["source"] = np.full(n, "legocup_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    for c in twin.CAMS:
        cams[c]["parent"] = build_scene.CAM_PARENT[c]
    meta = dict(name="so101_legocup_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="legocup_twin",
                twin_of="so101_legocup_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                feature_keys=fs.meta.get("feature_keys"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_signs=twin.SIGNS.tolist(), joint_offsets_deg=twin.OFFSETS_DEG.tolist(),
                jaw_range=list(twin.JAW_RANGE), cameras_fit=cams, gain=GAIN,
                notes="'side' is a wrist camera (child of the Fixed_Jaw body); objects per episode from out/objects.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
