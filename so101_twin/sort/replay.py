"""Replay every frame of so101_sort_real in the twin scene and write so101_sort_twin in the same layout.

    .venv/bin/python so101_twin/sort/replay.py [--out neuroscope/molmoact2/store/sets/so101_sort_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); cubes at the episode's layout from
out/cube_layouts.json (place_objects.py), the target cube placed by the real set's 'phase' label (approach: on the
table; grasp / transfer_open: between the jaws at site tcp; release: in the cardboard box); the real "top" image
rendered from the session's static camera ("top" for episodes 4..55, "top_b" for 66..101), the real "side" image
from the wrist camera "side" on Fixed_Jaw; per-camera brightness gain GAIN. Output: meta.json, images_top.npy,
images_side.npy (uint8 [n,480,640,3]), state.npy / prompt.json identical to the real set, labels.npz = real
labels with real_or_sim='sim', source='sort_twin', plus pair = index into the real set.
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
from twin import HERE, ROOT, Sim, real_set  # noqa: E402

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_sort_twin"
GAIN = {"top": 0.72, "side": 0.69}      # per-camera brightness gain (set by brightness_gain.py)
LAYOUTS = HERE / "out" / "cube_layouts.json"
PARK = (0.0, -0.6)                     # unused cube slots go under the table behind the base


def load_layouts() -> dict:
    if LAYOUTS.exists():
        return json.loads(LAYOUTS.read_text())["layouts"]
    return {}


def set_mocap(sim: Sim, body: str, pos) -> None:
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, body)
    sim.d.mocap_pos[sim.m.body_mocapid[bid]] = np.asarray(pos, float)


def place_cubes(sim: Sim, layout: dict | None, phase: str) -> None:
    h = build_scene.CUBE_HALF
    cubes = layout["cubes"] if layout else [list(c) for c in build_scene.CUBES_DEFAULT]
    target = layout["target"] if layout else 0
    for i in range(build_scene.N_CUBES):
        if i >= len(cubes):
            set_mocap(sim, f"cube{i}", (*PARK, -0.2))
            continue
        x, y = cubes[i]
        pos = np.array([x, y, h])
        if i == target:
            if phase in ("grasp", "transfer_open"):
                sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
                pos = sim.d.site_xpos[sid].copy()
            elif phase == "release":
                bx, by = build_scene.BOX["xy"]
                pos = np.array([bx, by, build_scene.BOX["wall"] + h])
        set_mocap(sim, f"cube{i}", pos)


def render_frame(sim: Sim, state: np.ndarray, phase: str, top_cam: str, layout: dict | None) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    place_cubes(sim, layout, phase)
    mujoco.mj_forward(sim.m, sim.d)
    out = {}
    for c, sim_cam in (("top", top_cam), ("side", "side")):
        img = sim.render(sim_cam).astype(np.float32) * GAIN[c]
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
    layouts = load_layouts()
    n = fs.n if not a.limit else min(a.limit, fs.n)
    imgs = {c: np.zeros((n, twin.H, twin.W, 3), np.uint8) for c in twin.CAMS}
    qpos = np.zeros((n, 6), np.float32)
    t0 = time.time()
    for i in range(n):
        ep = int(fs.labels["episode"][i])
        fr = render_frame(sim, fs.state[i], str(fs.labels["phase"][i]), build_scene.top_cam_for_episode(ep),
                          layouts.get(str(ep)))
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
    (out / "prompt.json").write_text((fs.path / "prompt.json").read_text() if n == fs.n else json.dumps(fs.prompt[:n]))
    labels = {k: np.asarray(v)[:n] for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.full(n, "sim")
    labels["source"] = np.full(n, "sort_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    np.save(out / "sim_qpos.npy", qpos)
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.SIM_CAMS}
    cams["side"]["frame"] = "Fixed_Jaw body (wrist camera); fingers along -y"
    cams["top"]["frame"] = "world, episodes 4..55 (session A)"
    cams["top_b"]["frame"] = "world, episodes 66..101 (session B)"
    fits = {}
    for c in twin.SIM_CAMS:
        p = HERE / "out" / f"camera_fit_{c}.json"
        if p.exists():
            d = json.loads(p.read_text())
            fits[c] = dict(iou_640x480=d["iou_640x480"], iou_per_frame=d["iou_per_frame"], frames=d["frames"])
    meta = dict(name="so101_sort_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="sort_twin",
                twin_of=twin.REAL_SET, image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                feature_keys=fs.meta.get("feature_keys"), episodes=fs.meta.get("episodes"),
                per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                joint_map=dict(order=list(twin.LEROBOT_ORDER), sim_order=list(twin.SIM_ORDER),
                               signs=twin.SIGNS[:5].tolist(), offsets_deg=twin.OFFSETS_DEG[:5].tolist(),
                               jaw_range=list(twin.JAW_RANGE),
                               formula="sim_rad = deg2rad(signs*real_deg + offsets_deg); jaw = -0.174 + gripper/100*1.924"),
                cameras_fit=cams, camera_iou=fits, top_camera_by_episode={"top": "episodes < 60", "top_b": "episodes >= 60"},
                gain=GAIN, cube_layouts=str(LAYOUTS.relative_to(ROOT)) if LAYOUTS.exists() else None)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
