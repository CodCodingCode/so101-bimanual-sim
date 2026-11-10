"""Replay every frame of so101_pickplace_real in the sim and write the twin set so101_pickplace_twin.

    .venv/bin/python so101_twin/pickplace/make_twin.py [--overwrite]

Output (neuroscope/molmoact2/store/sets/so101_pickplace_twin/), same layout as make_sets.py main():
  images_top.npy, images_side.npy  uint8 [n, 480, 640, 3]   sim renders at the real frame's joint state
  state.npy                        identical to the real set's state.npy
  prompt.json                      identical to the real set's prompt.json
  labels.npz                       the real set's arrays, with real_or_sim='sim', source='pickplace_twin', plus
                                   pair = index into the real set (0..n-1)
  meta.json                        name, n, cameras, real_or_sim, source, image_shape, twin_of, camera/joint params
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene  # noqa: E402
import joints  # noqa: E402
from twin import CAMS, REAL_SET, Twin, real_frames  # noqa: E402

TWIN_SET = "so101_pickplace_twin"
SOURCE = "pickplace_twin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    fs = real_frames()
    out = fs.path.parent / TWIN_SET
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} exists (use --overwrite)")
    tw = Twin()
    t0 = time.time()
    imgs = {c: np.zeros((fs.n, tw.h, tw.w, 3), np.uint8) for c in CAMS}
    qpos = np.zeros((fs.n, 6), np.float32)
    for i in range(fs.n):
        qpos[i] = tw.set_state(fs.state[i])
        for c in CAMS:
            imgs[c][i] = tw.render(c)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{fs.n} frames, {time.time() - t0:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for c in CAMS:
        np.save(out / f"images_{c}.npy", imgs[c])
    np.save(out / "state.npy", fs.state.astype(np.float32))
    shutil.copyfile(fs.path / "prompt.json", out / "prompt.json")
    labels = {k: np.asarray(v) for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.array(["sim"] * fs.n)
    labels["source"] = np.array([SOURCE] * fs.n)
    labels["pair"] = np.arange(fs.n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    np.save(out / "sim_qpos.npy", qpos)
    cams = build_scene.camera_params()
    meta = dict(name=TWIN_SET, n=int(fs.n), cameras=list(CAMS), real_or_sim="sim", source=SOURCE,
                twin_of=REAL_SET, image_shape=[tw.h, tw.w, 3], dataset=fs.meta.get("dataset"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), scene_xml=str(build_scene.OUT_XML.relative_to(HERE.parent.parent)),
                cameras_params=cams,
                joint_map=dict(order=list(joints.LEROBOT_ORDER), sim_order=list(joints.SIM_ORDER),
                               signs=joints.SIGNS.tolist(), offsets_deg=joints.OFFSETS_DEG.tolist(),
                               jaw_range=list(joints.JAW_RANGE), jaw_pct_open=joints.JAW_PCT_OPEN))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}: n={fs.n}, {size:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
