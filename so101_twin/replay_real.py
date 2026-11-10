"""Phase 1 of the twin: replay real bimanual SO-101 joint states in the sim and render the three cameras.

Input : a LeRobot dataset directory (v2.1 or v3 layout) recorded on the real bimanual rig.
Output: out/replay/ep*.npz with, per kept frame, the sim renders of cam_top / left wrist / right wrist,
        the sim qpos used, and (episode_index, frame_index, timestamp) so the real frames can be joined later
        by the existing loaders in neuroscope/pi05 (make_sets.py reads the same parquet files with pyarrow).

    python so101_twin/replay_real.py --dataset data_real/<my_so101_bimanual> --stride 5 --episodes 0 1 2

Joint convention. LeRobot >= 0.5 stores calibrated SO-101 joints in DEGREES under observation.state with
names like left_shoulder_pan.pos ... left_gripper.pos (gripper in percent open). The sim joints are the
Menagerie names in radians. SIGNS / OFFSETS_DEG below are the first guess from the LeRobot MolmoAct2 doc
(shoulder_lift flipped, shoulder_lift and elbow_flex shifted by 90 deg); Phase 2 calibrates them against
the real frames by arm-silhouette overlap. Until then treat the renders as approximate.

Objects are NOT placed by this script: real object poses are unknown. Per-episode initial object poses are
entered by hand in OBJECT_INIT after looking at the first real frame; attaching a grasped object to the
closing jaw is the next step.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE / "bimanual_so100_workcell.xml"
CAMS = ("cam_top", "left_cam_wrist", "right_cam_wrist")

# LeRobot motor order for one SO-101 arm, and the matching Menagerie joint for each.
LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=float)          # calibrate in Phase 2
OFFSETS_DEG = np.array([0, 90, 90, 0, 0, 0], dtype=float)   # calibrate in Phase 2
JAW_RANGE = (-0.174, 1.75)                                  # Menagerie Jaw joint range (rad); gripper % -> rad

# Per-episode initial object poses (world xyz + quat wxyz) keyed by body name; filled in by hand.
OBJECT_INIT: dict[int, dict[str, list[float]]] = {}


def real_to_sim(q_deg: np.ndarray) -> np.ndarray:
    """(12,) real joints [deg, gripper %] -> (12,) sim qpos [rad]."""
    out = np.zeros(12)
    for a in range(2):
        s = q_deg[a * 6:(a + 1) * 6]
        rad = np.deg2rad(SIGNS[:5] * (s[:5] + OFFSETS_DEG[:5]))
        jaw = JAW_RANGE[0] + np.clip(s[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
        out[a * 6:(a + 1) * 6] = np.concatenate([rad, [jaw]])
    return out


def load_frames(dataset: pathlib.Path) -> dict[str, np.ndarray]:
    """All data rows of the dataset as {column: array}; observation.state expanded to (N, 12) in LeRobot order."""
    files = sorted(glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset}/data")
    tabs = [pq.read_table(f) for f in files]
    cols = {}
    for name in tabs[0].column_names:
        cols[name] = np.concatenate([t.column(name).to_numpy(zero_copy_only=False) for t in tabs])
    names = None
    info = dataset / "meta" / "info.json"
    if info.exists():
        names = json.load(open(info))["features"].get("observation.state", {}).get("names")
    if "observation.state" in cols:                     # v2.1 / v3: one array column
        state = np.stack([np.asarray(x, dtype=float) for x in cols["observation.state"]])
    else:                                               # already one column per joint
        if not names:
            names = [c[len("observation.state."):] for c in cols if c.startswith("observation.state.")]
        state = np.stack([cols[f"observation.state.{n}"] for n in names], axis=1).astype(float)
    if not names or len(names) != state.shape[1]:
        names = [f"j{i}" for i in range(state.shape[1])]
    order = []
    for side in ("left", "right"):
        for j in LEROBOT_ORDER:
            hits = [i for i, n in enumerate(names) if f"{side}_{j}" in n]
            if not hits:
                raise KeyError(f"no state entry for {side}_{j}; names: {names}")
            order.append(hits[0])
    cols["state12"] = state[:, order]
    for k in ("episode_index", "frame_index"):
        if k not in cols:
            raise KeyError(f"dataset has no {k} column")
    if "timestamp" not in cols:
        cols["timestamp"] = np.zeros(len(cols["frame_index"]))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--size", type=int, nargs=2, default=(640, 480))
    ap.add_argument("--preview", action="store_true", help="also write a PNG strip per episode")
    a = ap.parse_args()

    cols = load_frames(pathlib.Path(a.dataset))
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    w, h = a.size
    r = mujoco.Renderer(m, h, w)
    out_dir = HERE / "out" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = a.episodes or sorted(set(cols["episode_index"].tolist()))
    for ep in episodes:
        idx = np.flatnonzero(cols["episode_index"] == ep)
        idx = idx[np.argsort(cols["frame_index"][idx])][::a.stride]
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for body, pose in OBJECT_INIT.get(ep, {}).items():
            jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)]
            adr = m.jnt_qposadr[jid]
            d.qpos[adr:adr + 7] = pose
        imgs = {c: [] for c in CAMS}
        qpos_log = []
        for i in idx:
            q = real_to_sim(cols["state12"][i])
            d.qpos[:12] = q
            d.ctrl[:12] = q
            mujoco.mj_forward(m, d)
            qpos_log.append(q)
            for c in CAMS:
                r.update_scene(d, camera=c)
                imgs[c].append(r.render().copy())
        np.savez_compressed(
            out_dir / f"ep{ep:04d}.npz",
            episode_index=ep,
            frame_index=cols["frame_index"][idx],
            timestamp=cols["timestamp"][idx],
            qpos=np.array(qpos_log, dtype=np.float32),
            **{f"sim_{c}": np.array(v, dtype=np.uint8) for c, v in imgs.items()},
        )
        if a.preview:
            k = min(6, len(idx))
            strip = np.concatenate([np.concatenate([imgs[c][j] for c in CAMS], axis=1) for j in
                                    np.linspace(0, len(idx) - 1, k).astype(int)], axis=0)
            Image.fromarray(strip).save(out_dir / f"ep{ep:04d}_preview.png")
        print(f"episode {ep}: {len(idx)} frames -> {out_dir / f'ep{ep:04d}.npz'}")
    r.close()


if __name__ == "__main__":
    main()
