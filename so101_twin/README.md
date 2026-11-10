# so101_twin: digital twin of the bimanual SO-101 bench

Implements Phase 1-2 of `../SIM2REAL_DIGITAL_TWIN_PLAN.md` (bimanual SO-101 variant). Started 2026-11-06.

## What is here

| file                | does                                                                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`    | writes `bimanual_so100_workcell.xml`: two Menagerie SO-100 arms in the YAM workcell, top camera, a wrist camera on each fixed jaw, placeholder objects. All bench numbers are parameters at the top. |
| `render_cams.py`    | renders cam_top / left wrist / right wrist (and a free view) at a pose; `out/cams_*.png`                                                                                                             |
| `replay_real.py`    | Phase 1: replays real LeRobot joint states in the sim, renders the three cameras, writes `out/replay/ep*.npz` aligned by (episode, frame)                                                            |
| `MOLMOACT2_PORT.md` | hook points for porting the pi0.5 recorder / K-V patcher to MolmoAct2-BimanualYAM; where the adapter goes                                                                                            |
| `assets/so_arm100/` | Menagerie SO-100 MJCF + meshes (Apache-2.0)                                                                                                                                                          |

Verified: scene builds (nq 26, nu 12, 3 cameras); renders offscreen on the Mac; replay runs end to end
on a synthetic LeRobot-layout dataset.

## Decisions taken

- **Arm model is SO-100, not SO-101.** Menagerie has no SO-101 MJCF and the community SO-101 repo checked
  ships none. Link lengths match; motors and brackets differ. Swap meshes when one exists, keep names.
- **Camera rig = MolmoAct2 YAM order**: top, left wrist, right wrist. Wrist camera sits on the outer face
  of the fixed jaw looking along the fingers with a 20 deg tilt; fingertips at the bottom of the frame.
  Real mount pose replaces these numbers in Phase 2.
- **Compute**: M3 Max 64 GB, no CUDA. MolmoAct2 bf16 inference fits locally (when the pi0.5 server is
  not loaded); fine-tunes rent an 80 GB GPU.
- **Objects are placeholders** (two 35 mm cubes, a bin). Replace with the 2-3 real objects once chosen.

## Next

1. Choose the 2-3 objects and the 2 tasks (handover, two-hand lift). Everything below depends on it.
2. Phase 2 calibration: record ~20 real frames with the arms in spread poses; fit cam_top and the two
   wrist camera poses by arm-silhouette overlap; fit `SIGNS` / `OFFSETS_DEG` in `replay_real.py`.
3. Scripted sim demos for the sim-only fine-tune (adapt `yam_sim/bimanual_pick_place.py`'s DLS IK to the
   5-DoF arm: position IK + wrist pitch held vertical).
4. Recorder port per `MOLMOACT2_PORT.md`; first run of MolmoAct2-BimanualYAM on twin frames.
