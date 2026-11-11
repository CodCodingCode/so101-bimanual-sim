"""Open the calibrated rig sim in the MuJoCo viewer: 232 mm mount layout, calibrated overhead camera,
both arms at SIM_HOME. Press [ / ] in the viewer to cycle cameras (overhead = the calibrated real camera).

    sim/pi_sim_so101/.venv/bin/python rig_tools/view_sim.py [--rig rig_tools/data/so101_camera_rig.calibrated.json]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi_sim_so101"))
from pi_sim.env import RobotEnv   # noqa: E402

SIM_HOME = np.deg2rad([0.0, -45.8, 45.9, 90.0, -2.8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", type=Path, default=HERE / "data" / "so101_camera_rig.calibrated.json")
    a = ap.parse_args()
    env = RobotEnv(task="pick_place", control_mode="joint", camera_rig=str(a.rig) if a.rig.exists() else None)
    env.reset(seed=0, options={"randomize": False})
    m, d = env.model, env.data
    for s in ("left", "right"):
        d.qpos[env.robot[s].qpos_ids] = SIM_HOME
    mujoco.mj_forward(m, d)
    xml = HERE / "data" / "rig_sim_scene.xml"
    mujoco.mj_saveLastXML(str(xml), m)
    print("scene saved to", xml, "| camera rig:", a.rig if a.rig.exists() else "default (CAD)")
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
