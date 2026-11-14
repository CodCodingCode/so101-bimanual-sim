#!/usr/bin/env python3
"""Sim demonstrations of the clamp task for VLA training, written as a LeRobot v2.1 dataset (same layout as the real
SO-101 sets in sim/data_so101: data/chunk-000/episode_*.parquet, videos/chunk-000/<camera>/episode_*.mp4, meta/).

Driver: the clamp_wide teacher (scripted IK plan + residual PPO, cube / box layouts from a clamp_wide.SCHEDULE stage,
servo spread, grip randomisation as in training). Only successful episodes are kept (the outcome is also written).
  observation.images.{overhead,left_wrist,right_wrist}: the calibrated cameras (lens distortion included), 240 x 320
  observation.state (12): left / right shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg, sim
    convention = real units incl. joint offsets), gripper (0 shut .. 100 open)
  action (12): the commanded joint targets of the same joints, same units (what the real runner sends after IK)
  task: one language instruction
A VLA trained on this predicts action CHUNKS (SmolVLA / pi0.5 / MolmoAct2 all do natively).

    CLAMP_GRIP_DR=1 python squeeze/record_vla.py --teacher runs_clamp_wide/rotation_servo_done.zip --stages 4 \
        --episodes 200 --out runs_vla/clamp_rotation --workers 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

FPS = 25
H, W = 240, 320
CAMS = ("overhead", "left_wrist", "right_wrist")
TASK = "Squeeze the red cube between both grippers and drop it into the grey box."   # same for push-then-clamp
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NAMES = [f"{s}_{j}" for s in ("left", "right") for j in JOINTS]


def _grip_pct(env, side, ctrl=False):
    m, d = env.model, env.data
    arm = env.sim.robot[side]
    lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
    v = d.ctrl[arm.gripper_actuator] if ctrl else d.qpos[m.joint(f"{side}_gripper").qposadr[0]]
    return float(np.clip((v - lo) / (hi - lo), 0, 1) * 100)


def state_vec(env):
    out = []
    for s in ("left", "right"):
        arm = env.sim.robot[s]
        out += list(np.rad2deg(env.data.qpos[arm.qpos_ids][:5])) + [_grip_pct(env, s)]
    return np.array(out, np.float32)


def action_vec(env):
    out = []
    for s in ("left", "right"):
        arm = env.sim.robot[s]
        out += list(np.rad2deg(env.data.ctrl[arm.actuator_ids][:5])) + [_grip_pct(env, s, ctrl=True)]
    return np.array(out, np.float32)


def frames(env):
    from PIL import Image
    out = {}
    for c in CAMS:
        rgb = env.sim.render_camera(c, native=True)
        out[c] = np.asarray(Image.fromarray(rgb).resize((W, H), Image.Resampling.BILINEAR))
    return out


def write_video(path, imgs):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as box:
        st = box.add_stream("libx264", rate=FPS)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "23", "g": "2"}       # short GOP: random frame access for training
        for im in imgs:
            for p in st.encode(av.VideoFrame.from_ndarray(im, format="rgb24")):
                box.mux(p)
        for p in st.encode():
            box.mux(p)


def _randomizer(env):
    """VLA_DR=1 (--dr): sim-e5's per-episode appearance randomisation (distill.Randomizer: arm grey-blue..white, table
    / cube / box / room colours, lights + shadows, overhead mount and every camera pose jittered by mm). Applied AFTER
    env.reset(): pi_sim's reset restores the nominal look (applied before, it was a silent no-op). The first VLA data
    (2026-11-12/13) had NO appearance variation at all."""
    if os.environ.get("VLA_DR") != "1":
        return None
    import distill as D
    return D.Randomizer(env)


def _worker(job):
    teacher_path, stages, seeds, tmp = job[:4]
    pushclamp = len(job) > 4 and job[4]
    import clamp_wide as W_
    import chunk_student as CS
    teacher = CS.Teacher(teacher_path)
    if pushclamp:
        return _worker_pushclamp(teacher, seeds, tmp, pushclamp)
    env, rnd = None, None
    kept = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        stage = int(rng.choice(stages))
        lvl, yaw, bias = W_.SCHEDULE[stage]
        if env is None:
            env = W_.make_env_cls()(seed=int(seed), level=lvl, yaw_max=yaw)
            rnd = _randomizer(env)
        env.set_level(lvl), env.set_yaw(yaw), env.set_bias(bias)
        o, _ = env.reset(seed=int(seed))
        if rnd is not None:
            rnd.apply(np.random.default_rng([int(seed), 99]))
        S, A, F, info, Q = [], [], {c: [] for c in CAMS}, {}, []
        for t in range(env.base.max_steps):
            fr = frames(env)
            qpos_before = env.data.qpos.copy()
            s = state_vec(env)
            o, _, term, trunc, info = CS.teacher_step(env, teacher, o)
            S.append(s)
            Q.append(qpos_before)
            A.append(action_vec(env))                       # the command issued this tick
            for c in CAMS:
                F[c].append(fr[c])
            if term or trunc:
                break
        ok = bool(info.get("is_success"))
        if ok:
            out = Path(tmp) / f"seed{int(seed)}.npz"
            np.savez_compressed(out, state=np.array(S), action=np.array(A), qpos=np.array(Q, np.float32), stage=stage, clean=int(info.get("bad_contact_steps", 0) == 0),
                                **{f"img_{c}": np.array(F[c], np.uint8) for c in CAMS})
            kept.append(str(out))
        why = "" if ok else (f" guard {info.get('guard_stop')}" if info.get("guard_stop") else
                             " box tipped" if info.get("can_tipped") else "")
        print(f"seed {seed} stage {stage}: {'kept' if ok else 'failed, dropped'} ({len(S)} frames){why}", flush=True)
    return kept


def _worker_pushclamp(teacher, seeds, tmp, band_cm):
    """Push-then-clamp demonstrations (push_clamp_wide.PushClamp: box-aware scripted push into the clamp's spot, then
    the clamp policy into the detected box). Cube start anywhere in x +-12 cm, y +-band_cm around the nominal spot,
    box at the clamp design spacing from where the reset put the cube. Recording stops once the cube has settled in
    the box; only successes are kept."""
    import clamp_teacher as C
    import clamp_wide as W_
    import push_clamp_wide as P
    env = W_.make_env_cls()(seed=int(seeds[0]), level=0.0, yaw_max=45.0, episode_seconds=P.EPISODE_S)
    env.set_bias(1.0)
    rnd = _randomizer(env)
    kept = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        o, _ = env.reset(seed=int(seed))
        start = (rng.uniform(-0.12, 0.12), C.CUBE_Y + rng.uniform(-band_cm, band_cm) / 100)
        o = P._set_start(env, start)
        if rnd is not None:
            rnd.apply(np.random.default_rng([int(seed), 99]))
        pc = P.PushClamp(env, teacher)
        pc.start(o)
        S, A, F, info, Q = [], [], {c: [] for c in CAMS}, {}, []
        for t in range(env.base.max_steps):
            fr = frames(env)
            qpos_before = env.data.qpos.copy()
            s = state_vec(env)
            _, _, term, trunc, info = pc.step()
            S.append(s)
            Q.append(qpos_before)
            A.append(action_vec(env))
            for c in CAMS:
                F[c].append(fr[c])
            if term or trunc or info.get("is_success"):
                break
        ok = bool(info.get("is_success"))
        if ok:
            out = Path(tmp) / f"pc_seed{int(seed)}.npz"
            np.savez_compressed(out, state=np.array(S), action=np.array(A), qpos=np.array(Q, np.float32), stage=-1,
                                clean=int(info.get("bad_contact_steps", 0) == 0),
                                **{f"img_{c}": np.array(F[c], np.uint8) for c in CAMS})
            kept.append(str(out))
        print(f"push+clamp seed {seed} start ({start[0] * 100:+.1f}, {(start[1] - C.CUBE_Y) * 100:+.1f}) cm: "
              f"{'kept' if ok else 'failed, dropped'} ({len(S)} frames)"
              + ("" if ok or not info.get("guard_stop") else f" guard {info.get('guard_stop')}"), flush=True)
    env.close()
    return kept


HOME_DEG = np.array([0.0, -45.84, 45.84, 89.95, -2.79])   # both arms' start pose (pi_sim reset), deg


def timer_hold_keep(A, keep=5, tol=0.05):
    """Frame indices to keep after cutting the scripted teacher's TIMER waits down to `keep` + 1 frames.

    In push-then-clamp demos the joint targets freeze while the teacher waits on a clock, with nothing in the camera
    images saying when the wait ends (2026-11-13: 64/102 demos, 13.7 % of all frames): (1) the re-home before the clamp
    waits until every joint is within 2 deg of the start pose -- with the +-1 deg servo spread and gravity sag that
    rarely happens, so both arms sit at the start pose until the 100-tick timeout (~40 still frames); (2) a blocked
    pusher holds until its 600-tick push timeout (up to 475 still frames). A chunked policy (SmolVLA: 50-step chunks,
    25 executed, then re-plan) that sees a still scene and learned "still scene -> hold" holds again at every re-plan:
    the VLA's parked-arms stall. Cut: holds >= 10 ticks with both arms at the start pose (re-home waits), holds
    >= 25 ticks that end before the last such re-home wait (push-phase stalls), and any hold >= 100 ticks (the blocked
    pusher waiting for its 600-tick timeout; every one of them ended at tick 599). Squeeze / carry dwells (34-47
    ticks, arms in the clamp pose) and the rotation demos' end-of-task rest (<= 89 ticks) are kept: real waits."""
    J = np.concatenate([A[:, 0:5], A[:, 6:11]], 1)
    still = np.abs(np.diff(J, axis=0)).max(1) < tol
    segs, s = [], None
    for t, v in enumerate(still):
        if v and s is None:
            s = t
        if not v and s is not None:
            segs.append((s, t))
            s = None
    if s is not None:
        segs.append((s, len(still)))
    home = np.tile(HOME_DEG, 2)
    home_holds = [(a, b) for a, b in segs if b - a >= 10 and np.abs(J[a] - home).max() < 6.0]
    last_home = max((a for a, _ in home_holds), default=-1)
    push_holds = [(a, b) for a, b in segs if (b - a >= 25 and b <= last_home) or b - a >= 100]
    drop = set()
    for a, b in home_holds + push_holds:            # frames a..b share one target: keep a..a+keep
        drop.update(range(a + keep + 1, b + 1))
    return np.array([t for t in range(len(A)) if t not in drop], int)


def build_dataset(out, files, keep_fn=None):
    """keep_fn(state, action) -> frame indices to keep (e.g. timer_hold_keep); None keeps every frame."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    (out / "meta").mkdir(parents=True, exist_ok=True)
    idx, eps, ep_stats = 0, [], []
    allS, allA = [], []
    cut = 0
    for e, f in enumerate(sorted(files)):
        z = np.load(f)
        S, A = z["state"], z["action"]
        k = keep_fn(S, A) if keep_fn else np.arange(len(S))
        cut += len(S) - len(k)
        S, A = S[k], A[k]
        n = len(S)
        for c in CAMS:
            write_video(out / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{e:06d}.mp4", z[f"img_{c}"][k])
        tbl = pa.table({
            "action": pa.array(list(A), type=pa.list_(pa.float32(), 12)),
            "observation.state": pa.array(list(S), type=pa.list_(pa.float32(), 12)),
            "timestamp": pa.array((np.arange(n) / FPS).astype(np.float32)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, e, np.int64)),
            "index": pa.array(np.arange(idx, idx + n, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, np.int64)),
        })
        (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        eps.append({"episode_index": e, "tasks": [TASK], "length": int(n)})

        def st(x):
            return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
                    "std": x.std(0).tolist(), "count": [int(len(x))]}
        es = {"action": st(A), "observation.state": st(S)}
        for c in CAMS:
            im = z[f"img_{c}"][::10].astype(np.float32) / 255.0                       # (k, H, W, 3)
            es[f"observation.images.{c}"] = {k: np.asarray(v).reshape(3, 1, 1).tolist() for k, v in
                                             (("min", im.min((0, 1, 2))), ("max", im.max((0, 1, 2))),
                                              ("mean", im.mean((0, 1, 2))), ("std", im.std((0, 1, 2))))} | {"count": [int(len(im))]}
        ep_stats.append({"episode_index": e, "stats": es})
        allS.append(S), allA.append(A)
        idx += n
    vid = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"],
           "info": {"video.height": H, "video.width": W, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    info = {"codebase_version": "v2.1", "robot_type": "bi_so101", "total_episodes": len(eps), "total_frames": idx,
            "total_tasks": 1, "total_videos": len(eps) * len(CAMS), "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(eps)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"action": {"dtype": "float32", "shape": [12], "names": NAMES},
                         "observation.state": {"dtype": "float32", "shape": [12], "names": NAMES},
                         **{f"observation.images.{c}": vid for c in CAMS},
                         "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                         "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                         "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                         "index": {"dtype": "int64", "shape": [1], "names": None},
                         "task_index": {"dtype": "int64", "shape": [1], "names": None}}}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    print(f"dataset: {len(eps)} episodes, {idx} frames -> {out}" + (f" ({cut} timer-wait frames cut)" if keep_fn else ""),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--stages", default="4", help="comma-separated clamp_wide.SCHEDULE indices to sample layouts from")
    ap.add_argument("--episodes", type=int, default=200, help="episodes to RUN (failures are dropped)")
    ap.add_argument("--out", default="runs_vla/clamp")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--pushclamp", type=float, default=0.0,
                    help="push-then-clamp demos with the cube anywhere in x +-12 cm, y +-this many cm (0 = clamp only)")
    ap.add_argument("--rebuild-trim", default="",
                    help="no recording: rebuild --out from the episode .npz files in this folder with the teacher's "
                         "timer waits cut (timer_hold_keep)")
    ap.add_argument("--dr", action="store_true", help="per-episode appearance randomisation (distill.Randomizer)")
    a = ap.parse_args()
    if a.dr:
        os.environ["VLA_DR"] = "1"                     # inherited by the spawned workers
    if a.rebuild_trim:
        files = sorted(str(p) for p in Path(a.rebuild_trim).expanduser().glob("*.npz"))
        out_dir = Path(a.out).expanduser()
        build_dataset(out_dir if out_dir.is_absolute() else HERE / a.out, files, keep_fn=lambda S, A: timer_hold_keep(A))
        return
    out = HERE / a.out
    tmp = out / "_episodes"
    tmp.mkdir(parents=True, exist_ok=True)
    stages = [int(s) for s in a.stages.split(",")]
    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    jobs = [(str(HERE / a.teacher), stages, [int(x) for x in c], str(tmp), a.pushclamp) for c in np.array_split(seeds, a.workers) if len(c)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        files = [f for part in pool.map(_worker, jobs) for f in part]
    build_dataset(out, files)


if __name__ == "__main__":
    main()
