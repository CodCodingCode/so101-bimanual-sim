"""SmolVLA (vision-language-action) on the REAL rig -- DRY CHECK: nothing moves, torque is never switched on.

The VLA runs in its own process on the laptop GPU (C:\\robot\\vla\\venv, LeRobot + CUDA: vla_server.py, 127.0.0.1); this
script (robot venv) reads the measured joints and the three camera frames, asks for one action chunk (50 joint-target
steps) and checks it the way a live run would be checked, so the user sees what the model WOULD do:
  - every predicted pose through the kinematic shadow and the safety guard (real joint ranges, camera mounts / other
    arm / camera post / box clearances in the commanded pose), the largest jump from the current pose
  - where the fingertips would go (forward kinematics) relative to the triangulated cube
Log + frames in C:\\robot\\real\\runs\\real_vla_<time>.

    python real_vla.py --dry [--ckpt C:/robot/vla/out/smolvla_clamp_v1/checkpoints/010000/pretrained_model]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(r"C:\robot\real")
SIM = HERE / "sim_mac"
RT = SIM / "squeeze"
sys.path[:0] = [str(RT), str(SIM), str(HERE)]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
from PIL import Image              # noqa: E402

import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402
import cube_locator as CL          # noqa: E402

SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
VLA_PY = r"C:\robot\vla\venv\Scripts\python.exe"
SERVER = r"C:\robot\vla\vla_server.py"
PORT = 8791
DEFAULT_CKPT = "C:/robot/vla/out/smolvla_bb1b28/checkpoints/020000/pretrained_model"
# the calibration the bb1b28 models were trained on (cams 6349a0 + kinematics bb1b28), kept next to the VLA so the
# shared rig.json / camera rig other runners use are not changed by this runner (2026-11-14)
CALIB = Path(r"C:\robot\vla\calib_bb1b28")


def ask(frames320, state):
    buf = io.BytesIO()
    np.savez(buf, camera1=frames320["overhead"], camera2=frames320["left_wrist"], camera3=frames320["right_wrist"],
             state=np.asarray(state, np.float32))
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/chunk", data=buf.getvalue(), method="POST")
    z = np.load(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()))
    return z["chunk"], float(z["ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="(the only mode: nothing moves)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--show", type=int, default=25, help="chunk steps to check / report")
    ap.add_argument("--state-dim", type=int, default=10, help="10 = 5 joints per arm; 12 = + gripper %% per arm")
    ap.add_argument("--task", default=None, help="the instruction this checkpoint was trained with")
    ap.add_argument("--offsets", default="new", choices=("new", "current"),
                    help="joint offsets: 'new' = bb1b28 round 6 (what the bb1b28 models were trained with), "
                         "'current' = rig.json's (A/B)")
    ap.add_argument("--camera-rig", default=str(CALIB / "so101_camera_rig.json"))
    a = ap.parse_args()

    os.environ["SQUEEZE_TASK"] = "clamp"
    import env as E
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    rig_cams = Path(a.camera_rig)
    tag = json.loads(rig_cams.read_text()).get("calibration_tag")
    print(f"camera rig: {rig_cams} (calibration {tag})", flush=True)
    base_sim = E._Sim

    class _RigSim(base_sim):
        def __init__(self, *a_, **k_):
            k_.setdefault("camera_rig", str(rig_cams))
            super().__init__(*a_, **k_)
    E._Sim = _RigSim
    shadow = E.SqueezeEnv()
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    intr = {c: CL.cam_intrinsics(sim.camera_specs[c]) for c in CAMS}

    run = HERE / "runs" / f"real_vla_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    print(f"starting the VLA server on the GPU: {a.ckpt}", flush=True)
    srv_log = open(run / "vla_server.log", "w")
    env = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONWARNINGS="ignore")
    srv = subprocess.Popen([VLA_PY, "-u", SERVER, a.ckpt, "--port", str(PORT)] + (["--task", a.task] if a.task else []),
                           stdout=srv_log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if srv.poll() is not None:
                raise SystemExit(f"VLA server exited ({srv.returncode}); see {run / 'vla_server.log'}")
            if "serving on" in (run / "vla_server.log").read_text(errors="ignore"):
                break
            time.sleep(1)
        else:
            raise SystemExit("VLA server did not come up in 4 min")
        print(f"VLA server ready ({time.time() - t0:.0f} s)", flush=True)

        rig = json.loads((HERE / "rig.json").read_text())
        off = rig.get("joint_offsets_deg", {})
        if a.offsets == "new":
            off = json.loads((CALIB / "joint_offsets.json").read_text())["new"]
        print(f"joint offsets ({a.offsets}): L {off['left'][:5]} R {off['right'][:5]}", flush=True)
        arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
        for s, arm in arms.items():
            if arm.fingerprint != R.FINGERPRINTS[s]:
                raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
        cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
        try:
            time.sleep(2.0)
            u = {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}
            for s in SIDES:
                d.qpos[sim.robot[s].qpos_ids] = np.deg2rad(u[s][:5])
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
            mujoco.mj_forward(m, d)
            frames = {c: cams[c].latest() for c in CAMS}
        finally:
            for c in cams.values():
                c.stop.set()
        # camera roles: the USB indices in rig.json get renumbered when the laptop reconnects (2026-11-13 04:45). The
        # overhead must see the box; if not, the views are probably swapped -> refuse (the VLA would get wrong views)
        import vision as V
        box = V.detect_box(frames["overhead"], V.OverheadCamera.from_env(sim))
        if box is None:
            Image.fromarray(np.concatenate([np.asarray(Image.fromarray(frames[c]).resize((320, 240))) for c in CAMS], 1)) \
                .save(run / "frames_check.jpg", quality=90)
            raise SystemExit("the 'overhead' camera does not see the box: camera indices in rig.json may be swapped "
                             f"(USB renumbering) -- see {run / 'frames_check.jpg'}")
        f320 = {c: np.asarray(Image.fromarray(frames[c]).resize((320, 240), Image.Resampling.BILINEAR)) for c in CAMS}
        Image.fromarray(np.concatenate([f320[c] for c in CAMS], 1)).save(run / "frames.jpg", quality=90)
        # cube, triangulated (for reference only: the VLA sees the raw images)
        rays = []
        for c in CAMS:
            uv, _ = CL.detect_red(frames[c])
            if uv is not None:
                rays.append(CL.ray(intr[c], *CL.camera_pose(m, d, c), *uv))
        cube, how = CL.locate(rays)
        state = (np.r_[u["left"][:5], u["right"][:5]] if a.state_dim == 10 else
                 np.r_[u["left"][:5], np.clip(u["left"][5], 0, 100), u["right"][:5], np.clip(u["right"][5], 0, 100)])
        tips0 = {s: sim.robot[s].position.copy() for s in SIDES}
        chunk, ms = ask(f320, state)
        print(f"VLA chunk {chunk.shape} in {ms:.0f} ms; measured joints L {np.round(u['left'][:5], 1).tolist()} "
              f"R {np.round(u['right'][:5], 1).tolist()}", flush=True)
        if chunk.shape[1] not in (10, 12):
            raise SystemExit(f"unexpected action size {chunk.shape[1]}")
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=None)
        guard.set_box(box[:2], yaw=box[2])            # the box the overhead sees (arm-box margins are checked too)
        from cmd_filter import CmdFilter
        filt = CmdFilter(guard)
        print(f"guard self-test {guard.self_test_mm} mm, cmd filter self-test {filt.self_test_mm} mm", flush=True)
        t_filt, t_guard = [], []
        q_prev = {s: np.r_[u[s][:5], np.clip(u[s][5], 0, 100)] for s in SIDES}
        rows, worst = [], {}
        for k in range(min(a.show, len(chunk))):
            a_ = chunk[k]
            q = ({"left": np.r_[a_[0:5], 0.0], "right": np.r_[a_[5:10], 0.0]} if len(a_) == 10 else
                 {"left": np.r_[a_[0:5], a_[5]], "right": np.r_[a_[6:11], a_[11]]})
            bad = guard.check_pose(q, "cmd")
            if len(a_) == 10:                           # no gripper in the action: keep the measured opening
                q = {s_: np.r_[q[s_][:5], q_prev[s_][5]] for s_ in SIDES}
            t0_ = time.perf_counter()
            qf, kept = filt(q_prev, q)                  # what a live run would SEND (sim-74's cmd filter)
            t1_ = time.perf_counter()
            bad_f = guard.check_pose(qf, "cmd")
            t_filt.append((t1_ - t0_) * 1e3)
            t_guard.append((time.perf_counter() - t1_) * 1e3)
            q_prev = qf
            lim = [f"{s} joint {j} {q[s][j]:.0f} deg outside {lo:.0f}..{hi:.0f}" for s in SIDES
                   for j, (lo, hi) in enumerate(zip(*guard.limits[s])) if not lo <= q[s][j] <= hi]
            for s in SIDES:
                d.qpos[sim.robot[s].qpos_ids] = np.deg2rad(q[s][:5])
            mujoco.mj_forward(m, d)
            tips = {s: sim.robot[s].position.copy() for s in SIDES}
            jump = max(float(np.abs(q[s][:5] - u[s][:5]).max()) for s in SIDES)
            dist = guard.distances(q)
            rows.append({"k": k, "q": {s: np.round(q[s], 2).tolist() for s in SIDES}, "max_jump_deg": round(jump, 1),
                         "tips_cm": {s: np.round(tips[s] * 100, 1).tolist() for s in SIDES},
                         "closest_mm": {kd: round(v[0] * 1000, 1) for kd, v in dist.items()},
                         "guard": [str(b) for b in bad] + lim,
                         "filtered": {"kept": round(float(kept), 2), "guard": [str(b) for b in bad_f],
                                      "closest_mm": {kd: round(v[0] * 1000, 1) for kd, v in guard.distances(qf).items()}}})
            for kd, v in dist.items():
                worst[kd] = min(worst.get(kd, 1e9), v[0] * 1000)
        json.dump({"ckpt": a.ckpt, "offsets": a.offsets, "joint_offsets": off, "camera_rig": str(rig_cams),
                   "calibration_tag": tag, "filter_ms": t_filt, "guard_ms": t_guard, "ms": ms, "measured": {s: u[s].tolist() for s in SIDES}, "cube": None if cube is None
                   else cube.tolist(), "cube_how": how, "chunk": chunk.tolist(), "rows": rows}, open(run / "dry_vla.json", "w"))
        flagged = [r for r in rows if r["guard"]]
        print(f"cube (triangulated, reference): {None if cube is None else np.round(cube * 100, 1).tolist()} cm ({how})", flush=True)
        for r in rows[::6] + [rows[-1]]:
            print(f"  step {r['k']:2d}: tips L {r['tips_cm']['left']} R {r['tips_cm']['right']} cm | jump from now "
                  f"{r['max_jump_deg']} deg | closest {r['closest_mm']} mm{' | GUARD: ' + '; '.join(r['guard'][:2]) if r['guard'] else ''}",
                  flush=True)
        print(f"tips now: L {np.round(tips0['left'] * 100, 1).tolist()} R {np.round(tips0['right'] * 100, 1).tolist()} cm", flush=True)
        print(f"closest over the first {len(rows)} steps (mm): {({k: round(v, 1) for k, v in worst.items()})}", flush=True)
        ff = [r for r in rows if r["filtered"]["guard"]]
        kept = np.array([r["filtered"]["kept"] for r in rows])
        print(f"with the cmd filter: {int((kept < 1).sum())}/{len(rows)} steps cut (kept median {np.median(kept):.2f}, "
              f"min {kept.min():.2f}); " + (f"GUARD WOULD STILL STOP at step {ff[0]['k']}: {ff[0]['filtered']['guard'][0]}"
                                            if ff else "guard OK on every filtered step"), flush=True)
        print(f"timing per 40 ms tick: cmd filter median {np.median(t_filt):.1f} ms, max {max(t_filt):.1f} ms; guard cmd check "
              f"median {np.median(t_guard):.1f} ms, max {max(t_guard):.1f} ms", flush=True)
        print(f"{'RAW (unfiltered) GUARD WOULD STOP at step ' + str(flagged[0]['k']) + ': ' + flagged[0]['guard'][0] if flagged else 'guard: every checked step OK'}",
              flush=True)
        print(f"DRY: nothing moved. log {run}", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
