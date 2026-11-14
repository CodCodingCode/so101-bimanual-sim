#!/usr/bin/env python3
"""Export the live-run cube-check frames (the 105-frame set of rig_cube_check.py) for a camera/kinematics fitter.
Per frame: run id, tick, the 12 measured joints (rig units = sim degrees with rig.json offsets: 5 + gripper per arm),
cube x, y, yaw from the OVERHEAD only (vision.detect_cube_pose on the 320x240 save upscaled to 640x480, with the
CURRENT overhead model), the real crops at their saved resolution (320x240: overhead, left_wrist, right_wrist) and
which wrists' red-blob detector sees the cube (cube_locator.detect_red, min 8 px at 320x240).
Selection: cube untouched at its start spot, both fingertips > 5 cm from it by forward kinematics (CURRENT kinematics)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]


RIG = HERE.parents[1] / "rig_tools" / "data" / "so101_camera_rig.calibrated_full.json"   # the CURRENT calibrated overhead


def main(runs_dir, out):
    import os
    os.environ.setdefault("SQUEEZE_TASK", "clamp")
    import mujoco
    from PIL import Image
    import env as E, cube_locator as CL, vision as V
    from rig_cube_check import detect_red_merged
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    base_sim = E._Sim

    class _RigSim(base_sim):                 # 2026-11-13: the first export used pi_sim's default rig file, whose
        def __init__(self, *a_, **k_):       # overhead is an older pose (~14 mm constant offset; inference-testing-04)
            k_.setdefault("camera_rig", str(RIG))
            super().__init__(*a_, **k_)
    E._Sim = _RigSim
    env = E.SqueezeEnv(); env.reset(seed=0); sim, m, d = env.sim, env.model, env.data
    cq = m.jnt_qposadr[m.joint("cube_joint").id]
    cam = V.OverheadCamera.from_env(sim)
    rows = {k: [] for k in ("run", "tick", "joints", "cube_xy_yaw", "cube_xy_synth", "overhead", "left_wrist", "right_wrist", "seen")}
    for run in sorted(Path(runs_dir).glob("real_policy_*")):
        mp, lp = run / "meta.json", run / "log.json"
        if not (mp.exists() and lp.exists() and (run / "frames").is_dir()):
            continue
        meta = json.loads(mp.read_text())
        if not meta.get("live") or "cube" not in meta:
            continue
        L = json.loads(lp.read_text()); cx, cy = meta["cube"][:2]
        for f in sorted((run / "frames").glob("t*.jpg")):
            k = int(f.stem[1:5])
            if k >= len(L) or k > 200:
                continue
            u = L[k]["units"]
            for s in ("left", "right"):
                d.qpos[sim.robot[s].qpos_ids] = np.deg2rad(u[s][:5])
            mujoco.mj_kinematics(m, d)
            if min(np.linalg.norm(sim.robot[s].position[:2] - np.array([cx, cy])) for s in ("left", "right")) < 0.05:
                continue
            strip = np.asarray(Image.open(f).convert("RGB"))
            crops = {c: strip[:, 320 * i:320 * (i + 1)] for i, c in enumerate(("overhead", "left_wrist", "right_wrist"))}
            ov = np.asarray(Image.fromarray(crops["overhead"]).resize((640, 480), Image.Resampling.BILINEAR))
            pose = V.detect_cube_pose(ov, cam)
            if pose is None:
                continue
            # analysis by synthesis (sim overhead render at the estimate, same merged-blob detector, table-plane shift)
            for s_ in ("left", "right"):
                d.qpos[sim.robot[s_].qpos_ids] = np.deg2rad(u[s_][:5])
            px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
            ruv, _ = detect_red_merged(ov)
            for _ in range(6):
                d.qpos[cq:cq + 7] = [px, py, 0.016, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_forward(m, d)
                suv, _ = detect_red_merged(sim.render_camera("overhead", native=True))
                if ruv is None or suv is None:
                    break
                pr, ps = cam.to_plane(*ruv, 0.016), cam.to_plane(*suv, 0.016)
                if pr is None or ps is None:
                    break
                px, py = px + float(pr[0] - ps[0]), py + float(pr[1] - ps[1])
                if np.hypot(pr[0] - ps[0], pr[1] - ps[1]) < 2e-4:
                    break
            rows["run"].append(run.name); rows["tick"].append(k)
            rows["joints"].append(np.r_[u["left"][:6], u["right"][:6]].astype(np.float32))
            rows["cube_xy_yaw"].append(np.array(pose[:3], np.float32))
            rows["cube_xy_synth"].append(np.array([px, py, yaw], np.float32))
            for c in ("overhead", "left_wrist", "right_wrist"):
                rows[c].append(crops[c])
            rows["seen"].append(np.array([detect_red_merged(crops[c], min_px=8)[0] is not None for c in ("left_wrist", "right_wrist")]))
    np.savez_compressed(out, run=np.array(rows["run"]), tick=np.array(rows["tick"], np.int32),
                        joints=np.stack(rows["joints"]), cube_xy_yaw=np.stack(rows["cube_xy_yaw"]),
                        cube_xy_yaw_synth=np.stack(rows["cube_xy_synth"]),
                        overhead=np.stack(rows["overhead"]), left_wrist=np.stack(rows["left_wrist"]),
                        right_wrist=np.stack(rows["right_wrist"]), seen_left_right=np.stack(rows["seen"]),
                        readme=np.array(__doc__))
    print(f"{len(rows['run'])} frames from {len(set(rows['run']))} runs -> {out}; wrists seeing the cube: "
          f"left {int(np.sum([s[0] for s in rows['seen']]))}, right {int(np.sum([s[1] for s in rows['seen']]))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
