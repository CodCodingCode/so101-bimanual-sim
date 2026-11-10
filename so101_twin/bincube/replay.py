"""Replay every frame of so101_bincube_real in the twin scene and write so101_bincube_twin in the same layout.

    .venv/bin/python so101_twin/bincube/replay.py [--out neuroscope/molmoact2/store/sets/so101_bincube_twin] [--limit N]

Per frame: sim joints from the real state (twin.real_to_sim); lid and cube placed by a per-episode state machine
on the gripper reading (episode_states): the lid handle is thin, so gripper < PINCH means the jaws are pinching the
lid (lid closed up to the first pinch frame, open afterwards; a second pinch run late in the episode = closing it
again), a gripper in CUBE_GRIP holds the cube (cube between the jaws), after that it is in the bin. Episodes with
no pinch start with the lid open. The cube's start position on the table is per episode (out/cube_xy.json from
place_objects.py; falls back to build_scene.CUBE). Render "top" (wrist) and "side" with the fitted cameras, apply
a per-camera brightness gain (GAIN). Output: meta.json, images_top.npy, images_side.npy (uint8 [n,480,640,3]),
state.npy / prompt.json identical to the real set, labels.npz = real labels with real_or_sim='sim',
source='bincube_twin', plus pair = index into the real set.
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

DEFAULT_OUT = ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_bincube_twin"
GAIN = {"top": 1.03, "side": 0.76}      # per-camera brightness gain (set by brightness_gain.py)
PINCH = 12.0                          # gripper below this = pinching the lid handle
CUBE_GRIP = (15.0, 26.5)              # gripper in this band = holding the cube
CUBE_XY_FILE = HERE / "out" / "cube_xy.json"
LID_OPEN_DIR = (0.97, -0.26)          # table direction (unit) from the bin centre to where the open lid lies: image-left of
                                      # the bin at the same image height in the fitted side view (checked by projection)


def episode_states(g: np.ndarray) -> tuple[list[str], list[str]]:
    """Per frame (in episode order) lid state 'closed'/'open' and cube state 'table'/'jaws'/'bin' from the gripper.
    First pinch run (gripper < PINCH): the lid is closed for its first two frames (grab, lift) and open afterwards;
    a second pinch run = closing the lid again (closed from its second frame). The cube goes table -> jaws when the
    gripper enters CUBE_GRIP from outside it while the lid is open, jaws -> bin when the gripper opens again."""
    g = np.asarray(g, float)
    pinch = g < PINCH
    lid = "closed" if pinch.any() else "open"
    cube = "table"
    lids, cubes = [], []
    run, k = 0, 0                                   # pinch run number, index within the run
    for i in range(len(g)):
        if pinch[i]:
            k = k + 1 if (i > 0 and pinch[i - 1]) else 0
            run += 0 if k else 1
            if run == 1:
                lid = "closed" if k < 2 else "open"
            else:
                lid = "open" if k < 1 else "closed"
                if cube == "jaws":
                    cube = "bin"
        else:
            if run == 1:                            # first pinch run is over: the lid has been flipped open
                lid = "open"
        if not pinch[i] and lid == "open":
            in_band = CUBE_GRIP[0] <= g[i] <= CUBE_GRIP[1]
            prev_in_band = i > 0 and CUBE_GRIP[0] <= g[i - 1] <= CUBE_GRIP[1]
            if cube == "table" and in_band and not prev_in_band:
                cube = "jaws"
            elif cube == "jaws" and g[i] > CUBE_GRIP[1]:
                cube = "bin"
        lids.append(lid)
        cubes.append(cube)
    return lids, cubes


def all_states(fs) -> tuple[np.ndarray, np.ndarray]:
    lids = np.empty(fs.n, dtype=object)
    cubes = np.empty(fs.n, dtype=object)
    for e in np.unique(fs.labels["episode"]):
        idx = fs.where(episode=e)
        idx = idx[np.argsort(fs.labels["t"][idx])]
        L, C = episode_states(fs.state[idx, 5])
        lids[idx], cubes[idx] = L, C
    return lids, cubes


def load_cube_xy() -> dict[int, tuple[float, float]]:
    if CUBE_XY_FILE.exists():
        return {int(k): tuple(v) for k, v in json.loads(CUBE_XY_FILE.read_text()).items()}
    return {}


def lid_poses(sim: Sim) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World pos / quat of the lid body for 'closed' (on the bin) and 'open' (flipped over the bin's edge in the
    LID_OPEN_DIR direction, lying upside down on the table)."""
    bx, by = build_scene.BIN["xy"]
    hz = build_scene.BIN["half"][2]
    lh = build_scene.LID["half"]
    closed = (np.array([bx, by, 2 * hz + lh[2]]), np.array([1.0, 0, 0, 0]))
    d = np.array(LID_OPEN_DIR, float)
    d /= np.linalg.norm(d)
    yaw = np.arctan2(d[1], d[0]) - np.pi / 2            # lid +y axis points along d
    q_yaw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0, 0, 1.0]), yaw)
    q_flip = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_flip, np.array([1.0, 0, 0]), np.pi)   # flipped about the hinge (local x)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_flip)
    # hinge on the bin's edge toward the camera: use the bin half-extent along d
    hx, hy = build_scene.BIN["half"][:2]
    cy, sy = np.cos(build_scene.BIN["yaw"]), np.sin(build_scene.BIN["yaw"])
    ext = abs(d[0] * cy + d[1] * sy) * hx + abs(-d[0] * sy + d[1] * cy) * hy
    centre = np.array([bx, by]) + d * (ext + lh[1] + 0.01)
    opened = (np.array([centre[0], centre[1], lh[2]]), q)
    return {"closed": closed, "open": opened}


def cube_pos(sim: Sim, state: str, xy: tuple[float, float]) -> np.ndarray:
    h = build_scene.CUBE["half"]
    if state == "jaws":
        sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        return sim.d.site_xpos[sid].copy()
    if state == "bin":
        bx, by = build_scene.BIN["xy"]
        return np.array([bx, by, h + 0.006])
    return np.array([xy[0], xy[1], h])


def render_frame(sim: Sim, state: np.ndarray, lid: str, cube: str, cube_xy, lidp) -> dict[str, np.ndarray]:
    sim.set_joints(state)
    sim.set_mocap("lid", *lidp[lid])
    sim.set_mocap("cube", cube_pos(sim, cube, cube_xy))
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
    lidp = lid_poses(sim)
    lids, cubes = all_states(fs)
    cube_xy = load_cube_xy()
    n = fs.n if not a.limit else min(a.limit, fs.n)
    imgs = {c: np.zeros((n, twin.H, twin.W, 3), np.uint8) for c in twin.CAMS}
    t0 = time.time()
    for i in range(n):
        e = int(fs.labels["episode"][i])
        fr = render_frame(sim, fs.state[i], lids[i], cubes[i], cube_xy.get(e, build_scene.CUBE["xy"]), lidp)
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
    labels["source"] = np.full(n, "bincube_twin")
    labels["pair"] = np.arange(n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    np.savez(HERE / "out" / "twin_states.npz", lid=lids[:n].astype(str), cube=cubes[:n].astype(str),
             cube_xy=np.array([cube_xy.get(int(e), build_scene.CUBE["xy"]) for e in fs.labels["episode"][:n]]))
    cams = {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in build_scene.CAMS[c].items()} for c in twin.CAMS}
    fit = {}
    for c in twin.CAMS:
        f = HERE / "out" / f"camera_fit_{c}.json"
        if f.exists():
            d = json.loads(f.read_text())
            fit[c] = dict(iou_640x480=d["iou_640x480"], iou_per_frame=d["iou_per_frame"], frames=d["frames"])
    meta = dict(name="so101_bincube_twin", n=n, cameras=list(twin.CAMS), real_or_sim="sim", source="bincube_twin",
                twin_of="so101_bincube_real", image_shape=[twin.H, twin.W, 3], dataset=fs.meta.get("dataset"),
                feature_keys=fs.meta.get("feature_keys"), episodes=fs.meta.get("episodes"),
                per_episode=fs.meta.get("per_episode"), fps=fs.meta.get("fps"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene=str(twin.XML.relative_to(ROOT)),
                camera_note="'top' is a wrist camera on the Fixed_Jaw body (params in that body's frame); 'side' is static (world frame)",
                joint_signs=twin.SIGNS.tolist(), joint_offsets_deg=twin.OFFSETS_DEG.tolist(),
                jaw_range=list(twin.JAW_RANGE), cameras_fit=cams, camera_fit_iou=fit, gain=GAIN,
                lid_cube_rule=dict(pinch_below=PINCH, cube_grip=list(CUBE_GRIP)),
                bin=dict(xy=list(build_scene.BIN["xy"]), yaw=build_scene.BIN["yaw"], half=list(build_scene.BIN["half"])))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
