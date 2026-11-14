#!/usr/bin/env python3
"""Build the LeRobot dataset for SmolVLA co-training from gen_vla_sim.py (sim) + convert_vla_real.py (real) episodes.

Runs on the robot laptop in the separate training env (C:\\robot\\vla\\venv):
    python build_vla_dataset.py --sim C:/robot/vla/episodes/sim --real C:/robot/vla/episodes/real \
        --root C:/robot/vla/data/so101_clamp_cotrain --real-repeat 8

Every episode: 25 fps; state = measured joints, action = commanded joints (5 per arm, sim degrees = real units);
three cameras at 320x240. Real episodes are added --real-repeat times (they are ~4 % of the ticks otherwise).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

CAMS = ("overhead", "left_wrist", "right_wrist")
JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="local/so101_clamp_cotrain")
    ap.add_argument("--real-repeat", type=int, default=8)
    ap.add_argument("--max-sim", type=int, default=0, help="0 = all")
    ap.add_argument("--cam-keys", default="overhead,left_wrist,right_wrist",
                    help="dataset image key names for overhead,left_wrist,right_wrist (observation.images.<name>)")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    keys = dict(zip(CAMS, a.cam_keys.split(",")))
    features = {f"observation.images.{keys[c]}": {"dtype": "video", "shape": (240, 320, 3),
                                                  "names": ["height", "width", "channels"]} for c in CAMS}
    features["observation.state"] = {"dtype": "float32", "shape": (10,), "names": JOINTS}
    features["action"] = {"dtype": "float32", "shape": (10,), "names": JOINTS}
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=25, features=features, root=a.root, robot_type="bi_so101",
                               use_videos=True, image_writer_threads=8)

    sim = sorted(glob.glob(str(Path(a.sim) / "ep_*.npz")))
    if a.max_sim:
        sim = sim[:a.max_sim]
    real = sorted(glob.glob(str(Path(a.real) / "ep_*.npz")))
    plan = [(p, "sim") for p in sim] + [(p, "real") for p in real for _ in range(a.real_repeat)]
    t0, ticks = time.time(), {"sim": 0, "real": 0}
    index = []
    for n, (p, kind) in enumerate(plan):
        z = np.load(p, allow_pickle=True)
        dec = {c: [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in z[f"jpg_{c}"]] for c in CAMS}
        task = str(z["task"])
        for t in range(len(z["state"])):
            k = int(z["frame_idx"][t])
            frame = {f"observation.images.{keys[c]}": dec[c][k] for c in CAMS}
            frame["observation.state"] = z["state"][t].astype(np.float32)
            frame["action"] = z["action"][t].astype(np.float32)
            frame["task"] = task
            ds.add_frame(frame)
        ds.save_episode()
        ticks[kind] += len(z["state"])
        index.append({"episode": n, "source": kind, "file": Path(p).name, "ticks": int(len(z["state"]))})
        if n % 10 == 0 or n == len(plan) - 1:
            print(f"{time.strftime('%H:%M:%S')} episode {n + 1}/{len(plan)} ({kind}) | ticks sim {ticks['sim']} real "
                  f"{ticks['real']} | {(time.time() - t0) / (n + 1) * (len(plan) - n - 1) / 60:.0f} min left", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    Path(a.root, "episode_sources.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(plan)} episodes ({len(sim)} sim, {len(real)} real x{a.real_repeat}), ticks {ticks}", flush=True)


if __name__ == "__main__":
    main()
