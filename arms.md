# SO-101 arms: rig facts and findings

What we know about the bimanual SO-101 rig and its MuJoCo sim (`pi_sim_so101/`), and what was learned
building the two-arm squeeze. Compiled 2026-11-10 from several Claude sessions. Where a number came from
another session, the source is named. Check the code before relying on any number here.

## Calibration status

| Item                 | Status                                                                                                                                                                                                  | Where                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Arm spacing          | 232 mm between bases (TheRobotStudio overhead mount), in both the Mac and laptop sims                                                                                                                   | `pi_sim/scene.py`: `ARM_X = 0.116`                             |
| Joint offsets        | Fitted for both arms (sim = real + offset). Left `[5.1, 8.5, -6.6, 2.2, -72.7, 0]`, right `[4.0, -0.9, -6.7, -1.2, -90, 0]` (degrees)                                                                   | `C:\robot\real\rig.json` on dzkchen                            |
| Overhead camera      | Calibrated lens: fx = fy = 610.7, cx 285.1, cy 242.2, k1 -0.9328, k2 0.9382. Pose is still the CAD pose. The old CAD-only file was deleted; only this one exists                                        | `pi_sim/assets/so101_camera_rig.json`                          |
| Servo dynamics       | Fitted per joint (kp, kv, forcerange, timeconst, frictionloss, armature), on by default. Tracking RMS 1.79° stock → 0.52° fitted (left), 0.54° held-out right. `PI_SIM_DYNAMICS=nominal` restores stock | `pi_sim/assets/so101_dynamics.json` (fitted by session sim-e5) |
| Wrist cameras        | Not calibrated (inference-testing-04 was fitting them)                                                                                                                                                  |                                                                |
| Wrist roll           | Not observable from the overhead camera; offset unverified                                                                                                                                              |                                                                |
| Jaw-to-cube friction | **Not measured.** The squeeze depends on it (see below)                                                                                                                                                 |                                                                |

The fitted servos are much softer than stock. Proportional gain kp: shoulder_lift 27, elbow_flex 21, shoulder_pan 245, wrist_flex 399 (stock was 998 on every joint). Torque limits are 0.6–2.0 N·m (stock 2.94).

Checking the camera: with the sim posed from the joint readings of a real overhead frame, the calibrated camera puts
the right arm on top of the real one and the left arm close. The CAD camera drew both arms too small and off-centre.

## Props

- **Red cube:** 3.2 cm edge (sim-e5, from the rig's grip test), 3D printed and glue-filled, **10–20 g** (user).
  `squeeze/env.py` draws the mass from 10–20 g each episode. pi_sim's scene default is 3 cm and 15 g, and other
  tasks still use that.
- **Target:** the real open grey box, 10 × 10 cm outside, 3 cm tall, 4 mm walls (user, via sim-e5). It replaced an
  earlier soda can. The code still uses `can_*` / `lid_*` names for the box.

## Gripper geometry (applies to the real gripper too)

- **The fixed jaw is a wedge.** Its outer face is 31 mm in from the tool point (`gripperframe` site) at the tip,
  33.5 mm at 1–2 cm up and 47 mm at 5–6 cm up: it leans in about 19°. The visual mesh
  (`wrist_roll_follower_so101_v1`) and the collision hull (`..._gripper_part0_v1`) agree, so this is the real
  part's shape, not a convex-hull artifact.
- **The tool point is at the fingertips.** The jaw's lowest point is about 4 mm below it.
- **The flat jaw plates aren't what touches.** The `fixed_jaw_box*` geoms model the jaw's inner gripping face.
  An outside contact hits the hull first. A box-only estimate of where the outside face is was about 7 mm off.
- **Free yaw wanders.** pi_sim's `ee_delta` IK leaves yaw free. In the middle of the table that yaw drifted up to
  90° between poses, moving the gripper's inner surface by 1.5–3 cm. Any task that depends on which face meets an
  object must fix the yaw (wrist roll can do this; IK error < 0.3 mm over the squeeze workspace).
- **No sideways tilt.** The 5-joint arm can't tip its fingertips sideways, out of the arm's own vertical plane.
  The IK quietly drops a requested sideways tilt (the contact normal didn't change from 0° to 35° tilt).
- **Yaw options for pressing a cube from the side** (swept per yaw: first contact, height, contact normal):
  - Fixed jaw facing the cube (left closing axis -x, right +x): contact at the cube's top edge, normal leans 20°
    down (|n_z| = 0.34). Gripper bodies stay narrow in x. **This is the one that works.**
  - Sideways (closing axis along y): nearly horizontal contact (|n_z| ≤ 0.1), but each gripper body sticks out
    about 5 cm toward the middle and the two collide. Fails even with a 5 cm cube.
  - Closed moving jaw facing the cube: the jaw overhangs and touches the cube from the top (n_z = 1).
  - Right arm at yaw -90°: outside wrist-roll range (8 mm IK error).

## Two-sided squeeze

Both grippers keep their jaws shut and press opposite faces of the cube, lift it together, carry it, set it
down and back off sideways. Code is in `pi_sim_so101/squeeze/`:

| File                    | Contents                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `env.py`                | Scene, squeeze and success checks; measures each pad's contact offset from the model at start-up |
| `teacher.py`            | Scripted two-arm plan: approach, descend, close, lift, carry, lower, release, retreat            |
| `run.py`                | Evaluation; `--video` saves a GIF, `--noise` adds xy error to the positions the plan gets        |
| `rl_env.py`, `train.py` | RL: demos → BC warm start → PPO with a decaying BC term                                          |

The plan only needs the cube centre, the target centre and the rim height, which an overhead camera can supply.

### Why the grip height is 7 mm

With vertical fingers, the cube's top edge meets each jaw first, and the two jaw tops clear each other only if
that contact is high enough on the wedge. With the tool 17 mm above the cube's bottom, a 3 cm cube left about
1.3 mm of clearance per side, and the jaws hit each other about 4 cm above the cube with 15 N. Almost none of
the push reached the cube. With the tool 7 mm up (fingertips 3 mm off the table) the tops clear by about 6 mm.
`GRASP_Z` is a geometry limit, not a free parameter. A larger cube only adds clearance.

Cube size at the old 17 mm grip height (10 episodes each): 3 cm lifted 0/10, 4 cm lifted 10/10 but placed
3/10, 5 cm placed 9/10.

### Contact physics

- **Creep:** MuJoCo's soft contacts let a squeezed cube slide. At 8 N per pad it slid 7 mm down while lifting
  4 cm, and the squeeze relaxed to 0.8 N. `model.opt.noslip_iterations = 10` removes it. Real friction does not
  creep.
- **Edge grip:** the cube hangs from its two top edges against a 20° slope. Holding it needs a friction
  coefficient above about tan 20° = 0.36. Printed plastic on printed plastic is borderline, so rubber or tape
  on the outside of the fixed jaws would make the real grip safer.
- **Pendulum:** held only at its top edges, the cube swings when the arms accelerate.
- **Saturated base joint (stock dynamics):** the squeeze force came from position error. Under the stock gains the
  left `shoulder_pan` sat at its 2.94 N·m limit, 8.8 mm short of its target, while the right arm was almost on its
  target. A saturated joint has no torque left to follow motion.

### Tuning results (stock dynamics, 3 cm cube, can target, before sim-e5's calibration)

- Squeeze depth: 1–4 mm lifted then dropped; 8 mm best; 12 mm 20/30.
- Carry speed (8 mm squeeze, 30 episodes): 0.025 m/s placed 5/50, 0.04 → 27/30,
  0.07 → 30/30.
- A staged route (carry low, rise in front of the can) did worse: 11/50.
- Final: 8 mm squeeze, 0.07 m/s carry, 2 cm clearance → 100/100 on new seeds, can never tipped.
- Position error given to the plan (std): 2 mm 50/50, 4 mm 50/50, 7 mm 43/50. **The real rig must locate the
  cube and target to within about 4 mm.**
- Cube mass 40–60 g (an earlier, later-replaced estimate): 89/100.

These constants were tuned before the fitted servo dynamics, the 3.2 cm cube and the box target. sim-e5 is
retuning them.

### RL results

- **One-arm pinch (deleted, wrong task), for reference:** pure PPO never learned the grasp (0 after 580k steps);
  a BC warm start plus PPO with a BC term, and later DAgger, worked (0.85).
- **Squeeze, stock dynamics:** BC warm start alone, 0.92 greedy success (48 episodes). PPO was not run: the user
  held it until calibration.
- **Squeeze, calibrated dynamics, can target** (sim-e5, `squeeze/runs_can_rigcal/`): BC 0.81 → PPO 0.73 at 250k,
  0.92 at 750k, 1.00 at 1M steps (greedy, 48 episodes).
- **Squeeze, calibrated dynamics, box target** (sim-e5, `squeeze/runs/`): BC 0.83 greedy; PPO was running as
  of 21:23 (`squeeze/runs/pipeline_log.txt`).
- The scripted plan in the RL env scores about 2,550 per episode; squeezing and then hovering about 1,740;
  doing nothing 27. Hovering can't beat finishing.

## Open items

1. Measure jaw-to-cube friction on the real rig, and consider rubber pads on the fixed jaws.
2. Locate the cube and box from the overhead camera to within about 4 mm.
3. Wrist cameras and wrist-roll calibration.
4. All squeeze policies so far see true object positions from the simulator. They won't transfer to the real rig
   until positions come from the camera.
5. The comment above `PAD_YAW` in `squeeze/env.py` still describes the sideways orientation that was tried and
   dropped. The value (fixed jaw facing the cube) is correct; the comment is stale.

## Working with the rig

See `DZKCHEN_REMOTE.md` for connecting to the robot laptop (`dzkchen`, Windows PowerShell). Never start a live
run or switch on torque from an agent; the user does that with a hand on the e-stop.
