"""Port an existing builder twin (Menagerie SO-100 arm) to a starting scene description for the official SO-101.

    .venv/bin/python so101_twin/common/port_scene.py <scene> --calib new|old [--frame 0]

Runs the scene's own code read-only (so101_twin/<scene>/twin.py), poses it with real frame --frame, and prints:
  - ArmSpec base_pos / base_yaw_deg that put the SO-101 pan axis where the Menagerie pan axis is,
  - every camera's init pose: world cameras as world pos/quat; cameras on Fixed_Jaw -> mount 'gripper', cameras on
    the pan body -> mount 'shoulder', converted into the SO-101 body frame at that same real state
    (arm.mount_pose_from_world), fovy copied,
  - OFFSETS_INIT translated from the scene's Menagerie joint map (pan sign flips; lift/elbow/wrist minus the nominal
    -90/+90/0 of the new calibration; roll copied).
Paste the printed block into so101_twin/common/scenes/<scene>.py and add extras()/frame_setup() for the objects.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TWIN = HERE.parent
ROOT = TWIN.parent
NOMINAL = {"new": dict(signs=[-1, 1, 1, 1, 1], off=[0, -90, 90, 0, 0]),
           "old": dict(signs=[-1, -1, 1, 1, -1], off=[0, 0, 0, 0, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--calib", required=True, choices=("new", "old"))
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import arm as A

    sdir = TWIN / a.scene
    sys.path.insert(0, str(sdir))
    sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
    from sets import Store

    twin = importlib.import_module("twin")
    fs = Store().frames(f"so101_{a.scene}_real")
    sim = twin.Twin() if a.scene == "pickplace" else twin.Sim()
    pose = sim.set_state if a.scene == "pickplace" else sim.set_joints
    m, d = sim.m, sim.d
    state = fs.state[a.frame]
    # pan axis + reach direction of the Menagerie arm at zero joint angles
    qsave = d.qpos.copy()
    rot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Rotation")
    jaw_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    anchor = d.xanchor[rot].copy()
    reach = d.xpos[jaw_b] - anchor
    base_z = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Base")][2]
    yaw = math.degrees(math.atan2(reach[1], reach[0])) - 90.0
    d.qpos[:] = qsave
    pose(state)
    spec = A.ArmSpec(calib=a.calib, base_pos=(float(anchor[0]), float(anchor[1]), float(base_z)), base_yaw_deg=yaw)
    # joint map of the old twin
    signs = np.asarray(getattr(twin, "SIGNS", NOMINAL[a.calib]["signs"]), float)[:5]
    offs = np.asarray(getattr(twin, "OFFSETS_DEG", NOMINAL[a.calib]["off"]), float)[:5]
    if a.scene == "pickplace":
        jj = json.loads((sdir / "joints.json").read_text())
        signs, offs = np.asarray(jj["signs"], float)[:5], np.asarray(jj["offsets_deg"], float)[:5]
    nom = NOMINAL[a.calib]
    if not np.allclose(signs, nom["signs"]):
        print(f"# WARNING: old twin signs {signs.tolist()} differ from the nominal {nom['signs']}: check visually")
    new_off = []
    for j in range(5):
        dv = offs[j] - nom["off"][j]
        new_off.append(round(float(-dv if j == 0 else dv), 2))   # only the pan coordinate is mirrored
    # cameras
    cams = {}
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i): i for i in range(m.ncam)}
    for name, cid in names.items():
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.cam_bodyid[cid]) or "world"
        mount = "world" if m.cam_bodyid[cid] == 0 else ("gripper" if "Jaw" in body or "Wrist" in body else "shoulder")
        cams[name] = dict(body=body, mount=mount, wpos=d.cam_xpos[cid].copy(), wmat=d.cam_xmat[cid].copy(),
                          fovy=float(m.cam_fovy[cid]))
    ws = [A.CamSpec(n, "world", tuple(c["wpos"]), tuple(A.quat_from_mat(c["wmat"].reshape(3, 3))), c["fovy"])
          for n, c in cams.items()]
    new = A.Arm(A.build_mjcf(spec, ws), a.calib, spec.model)
    new.offsets_deg = np.asarray(new_off)
    new.set_state(state)
    print(f"# ported from so101_twin/{a.scene} at real frame {a.frame}; old joint map signs {signs.tolist()} "
          f"offsets {offs.tolist()}")
    print(f"ARM = A.ArmSpec(calib={a.calib!r}, wrist_mount=False, base_pos=({anchor[0]:.4f}, {anchor[1]:.4f}, "
          f"{base_z:.4f}), base_yaw_deg={yaw:.2f}, colors={{...}})")
    print(f"OFFSETS_INIT = {new_off}")
    print("CAMERAS = {")
    for n, c in cams.items():
        pos, quat = A.mount_pose_from_world(new, c["mount"], c["wpos"], c["wmat"])
        print(f"    {n!r}: dict(mount={c['mount']!r}, pos={tuple(np.round(pos, 5).tolist())}, "
              f"quat={tuple(np.round(quat, 6).tolist())}, fovy={c['fovy']:.3f}),   # old parent body: {c['body']}")
    print("}")


if __name__ == "__main__":
    main()
