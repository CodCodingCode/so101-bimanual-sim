# so101_legocup_twin2 — refit of the legocup digital twin (official SO-101)

Twin of `neuroscope/molmoact2/store/sets/so101_legocup_real` (320 frames = 40 episodes x 8, LeRobot v2.1 dataset
`data_so101/Rayenghali__so101_follower_main_put_the_red_lego_block_in_the_black_cup_eps100_fps30_480x640`).
Output set: `neuroscope/molmoact2/store/sets/so101_legocup_twin2/` (same layout as `so101_legocup_twin`).
All code here; nothing in `so101_twin/common/`, `so101_twin/fitkit/` or `so101_twin/legocup/` was edited (the builder's
`legocup/build_scene.py` and `out/objects.json` are read, the common toolkit files were copied and the copies edited).

## Result (agreed metric; pass = held-out median >= 0.80 on every camera)

Primary SAM masks `so101_twin/masks/legocup_<cam>_lc3.npy` (third SAM round, see below):

| camera / frames | n | median | p10 | mean |
|---|---|---|---|---|
| **top** held_out (odd episodes) | 159 | 0.697 | 0.373 | 0.641 |
| top all_frames | 318 | 0.692 | 0.394 | 0.643 |
| top old builder twin, held_out | 159 | 0.585 | 0.431 | 0.583 |
| **side** held_out (odd episodes) | 160 | 0.908 | 0.691 | 0.841 |
| side all_frames | 320 | 0.913 | 0.656 | 0.843 |
| side old builder twin, held_out | 160 | 0.782 | 0.604 | 0.747 |

Same twin scored against the `orig_masks` set:

| camera / frames | n | median | p10 | mean |
|---|---|---|---|---|
| top held_out | 159 | 0.652 | 0.355 | 0.597 |
| top all_frames | 317 | 0.639 | 0.342 | 0.593 |
| top old builder twin, held_out | 159 | 0.623 | 0.507 | 0.621 |
| side held_out | 160 | 0.908 | 0.638 | 0.827 |
| side all_frames | 320 | 0.912 | 0.635 | 0.834 |
| side old builder twin, held_out | 160 | 0.782 | 0.640 | 0.753 |

Same twin scored against the `_lc2_masks` set:

| camera / frames | n | median | p10 | mean |
|---|---|---|---|---|
| top held_out | 159 | 0.707 | 0.332 | 0.632 |
| top all_frames | 319 | 0.692 | 0.351 | 0.628 |
| top old builder twin, held_out | 159 | 0.595 | 0.396 | 0.581 |
| side held_out | 160 | 0.908 | 0.672 | 0.841 |
| side all_frames | 320 | 0.912 | 0.635 | 0.842 |
| side old builder twin, held_out | 160 | 0.782 | 0.558 | 0.747 |

**side (wrist) passes; top (fixed overhead) does not** (see Limits).

## Metric (as agreed)

Whole-arm silhouette IoU between the twin's rendered arm mask (640x480, rendered as a padded pinhole image and warped
by the fitted lens onto the real pixel grid; pixels the padded render cannot fill count as background) and a SAM 2.1
mask of the real arm, per camera, over frames where the real arm covers >= 0.1 % of the image. The fit only ever sees
EVEN episodes; `held_out` = ODD episodes, `all_frames` = all visible frames. `score_lc2.py` recomputes it.

## Model and joint mapping

- Arm: official TheRobotStudio SO-101, `so101_new_calib.xml` + the wrist-camera bracket and camera module of
  `so101_new_calib_camera.xml` (via the `arm.py` copy), every part black. Base: pan axis at world (0, 0.0452, 0), arm
  reaching +y (the builder twin's world frame, so its table / mat / cup / brick coordinates apply).
- The real wrist camera sits where the official module is: the builder's wrist-camera fit, ported into the SO-101
  gripper frame, lands within 2 cm of the official `wrist_camera` body with the same viewing direction.
- Wrist roll: offset ~ -90 deg (checked in `out/check_roll2.jpg`: only then does the camera module face the top camera,
  centred on the gripper with the jaws below it, as in every real top frame; 0 / 180 put it out to the side).
- The arm joints are LeRobot NORMALISED units, not degrees (the dataset's action stats hit exactly -100.0 on
  shoulder_lift and +100.0 on wrist_flex: the range_m100_100 clamp of use_degrees=False). So each joint gets
  q_deg = scale * state + offset (scale = calibrated travel / 200 deg per unit, prior 1 +- 0.25).
- Gripper (range_0_100): q = -0.1745 + grip_off + grip_scale * state/100 * 1.9199 rad.

| joint | scale (deg per state unit) | offset (deg) |
|---|---|---|
| shoulder_pan | 1.235 | -0.82 |
| shoulder_lift | 0.958 | +0.95 |
| elbow_flex | 1.025 | -3.13 |
| wrist_flex | 1.095 | +4.05 |
| wrist_roll | 1.016 | -85.75 |
| gripper | grip_scale 2.204 | grip_off -0.110 rad |

## Cameras

| camera | mount | pos (m) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |
|---|---|---|---|---|---|---|---|---|---|
| top | world | -0.0630, 0.4549, 0.4808 | -0.0756, -0.0103, 0.1619, 0.9839 | 39.88 | 0.0959 | 0.0057 | 0.0117 | 0.0024 | -0.14 |
| side | gripper | -0.0129, 0.0676, -0.0180 | -0.0611, -0.0237, 0.2106, 0.9754 | 60.12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.30 |

`top` is in the world frame; `side` (the wrist camera) is in the SO-101 `gripper` body frame (MuJoCo camera convention: looks along -z, +y up). All |k1| < 0.4 and |cx|, |cy| < 0.1, so no lens-cropping effect is possible on the score (the side lens stayed pinhole: every lens perturbation lowered its IoU).

## What was done

1. `check_init.py`, `check_roll.py`, `check_roll2.py`: joint map / roll check against real frames.
2. `fit.py` (copy of common/fit.py): per-camera fits, `--cams top|side`; the wrist fit (pose in the gripper frame,
   fovy, lens, gripper scale / offset, latency) and the top fit (pose, fovy, lens, joint offsets + scales, latency).
   `gsearch.py`, `global_top.py`, `roll_test.py`, `subset_test.py`: broader searches / diagnostics for the top camera.
3. `export_prompts.py` (twin silhouettes, lens-warped, for every frame) -> `so101_twin/masks/prompts_twin2/legocup_*.npy`
   (round 2) and `prompts_twin2_r3/legocup_*.npy` (round 3); `segment_lc2.py` (copy of common/segment.py): SAM 2.1
   prompted by consensus of the builder twin (`so101_twin/masks/prompts`) and this twin, plus hard negatives on dark
   pixels 18-70 px outside the twin silhouette (the black cup) and on the red brick, and the brick's pixels removed from
   the mask -> `so101_twin/masks/legocup_{top,side}_lc2.npy` (round 2) and `_lc3.npy` (round 3, primary), sheets
   `so101_twin/masks/sheets/legocup_lc2.jpg` / `legocup_lc3.jpg`. Run only after `pgrep -fl segment` showed no other
   SAM process (one SAM process at a time).
4. Refit both cameras on each new mask set; `finalize.py` merges them (cameras from their own runs, joint offsets /
   scales from the top run, gripper from the wrist run) -> `out/legocup_final_fit.json` and scores it with
   `score_lc2.py` on all mask sets -> `out/score_final.json` (per-frame IoU included); `render_lc2.py` -> the set +
   `out/before_after.jpg`; `write_readme.py` fills the tables of this file. Fit logs: `out/*.stdout`,
   `out/legocup_r*_fit.{json,log}`.

Reproduce (from the repo root, MuJoCo venv `.venv`, SAM venv `neuroscope/molmoact2/.venv`):
```
.venv/bin/python so101_twin/legocup2/fit.py legocup --cams side --mask-tag _lc3 --init-json so101_twin/legocup2/out/legocup_final_r2_fit.json --stages BCD --joint-params grip,gripoff --out-tag _r3side --evals 0.5
.venv/bin/python so101_twin/legocup2/fit.py legocup --cams top  --mask-tag _lc3 --init-json so101_twin/legocup2/out/legocup_final_r2_fit.json --stages BCD --joint-params off --out-tag _r3top --evals 0.5
.venv/bin/python so101_twin/legocup2/finalize.py --top out/legocup_r3top_fit.json --side out/legocup_r3side_fit.json
.venv/bin/python so101_twin/legocup2/render_lc2.py --fit out/legocup_final_fit.json --masks-tag _lc3
```

Rounds (held-out median, top / side; masks of that round):

| step | top | side |
|---|---|---|
| builder twin vs original SAM masks | 0.623 | 0.782 |
| r1: separate fits on the original masks (wrist: pose, fovy, lens, gripper, latency; top from the ported builder camera, stuck at ~0.55 on its fit frames, stopped) | - | 0.901 |
| r1b top: + normalised-unit joint scales, then lens / offsets / latency (`legocup_r1btop`, original masks) | 0.623 | - |
| r2: SAM round `_lc2` (consensus prompts r1 twin + builder twin, cup / brick negatives), refit | 0.694 | 0.908 |
| r3: SAM round `_lc3` (prompts from the r2 twin), refit = **final** | **0.697** | **0.908** |

Top-camera diagnostics (all on even episodes unless stated): wrist-roll candidates with a refitted camera each
(`roll_test.py`, trimmed-mean IoU on 48 frames): 0 -> 0.604, -90 -> 0.611, 180 -> 0.607 (all ~equal; -90 kept on
the visual evidence above); global look-at search of the top camera from scratch (`global_top.py`, 2500 random
cameras + CMA refinement of the best 6): nothing better than the builder's basin (0.562 vs 0.540 start, low res);
a free 6-DOF offset of the wrist-camera bracket (`gsearch.py ... mount`): 0.531 -> 0.549, not kept; per-joint
scales for the normalised units: 0.531 -> 0.594 (rest-pose frames 0.43 -> 0.56), kept; camera latency: best at
dt ~ 0 for the top camera (fast frames peak at dt 0..+1), -0.3 frames for the wrist camera; a motion-blur
silhouette (union / majority of sub-frame poses +-0.5 or +-1 frame): no gain (0.649 -> 0.643..0.649).

## Limits

**The fixed overhead camera ("top") stays at held-out median 0.697 (all frames 0.692) after an honest attempt;
the wrist camera ("side") passes at 0.908.** What limits the top camera:

1. Silhouette shape, not pose. With the best per-frame 2D shift of the twin mask (up to +-24 px) the held-out
   median only rises 0.640 -> 0.675 (r1 params): the residual is the outline itself. The top camera sees the arm
   from above and in front with the base outside the image, so only forearm, wrist, gripper, camera board and
   cables are visible, and the outline differs from the official meshes: the servo leads and the camera's USB cable
   run along the left of the forearm / wrist and SAM keeps them where they touch the arm; the real wrist / gripper
   housing outline differs from the official one on the right side; the camera board is a bare PCB on a plate.
   Only 7 % of held-out top frames reach 0.80; the median is flat across phases (rest 0.72, slow moving 0.73,
   fast moving 0.65, approach 0.69, grasp 0.72, release 0.69, transfer 0.70), i.e. a systematic model mismatch,
   not a few bad frames.
2. Gripper conflict between the cameras. The wrist camera needs grip_scale ~2.2 / grip_off -0.11 rad (closed
   frames want the jaws slightly past the model's closed limit, open frames a fast opening; at the physically
   plausible ~1.46 the wrist drops to 0.72). The top camera prefers ~1.0 (+0.01..0.02 for top): with 2.2 the
   twin's moving jaw sticks out to the left at gripper ~30-35 while the real jaw is closed on the brick. One
   physical mapping is used (the wrist's).
3. Motion blur: 47 % of held-out top frames move >= 1 deg/frame; they score 0.65 vs 0.72-0.73 for still frames
   (blurred arm, softer SAM masks).
4. Small visible area: in ~1/3 of the frames (arm folded at rest) the arm is a ~4000-px sliver at the top edge,
   where a few pixels of error cost a lot of IoU.
5. Kinematics: the dataset is in normalised units, so per-joint scales/offsets had to be fitted from the top
   silhouettes alone (pan 1.23, lift 0.96, elbow 1.03, wrist flex 1.09, roll 1.02 deg/unit); rest and moving
   frames still pull the solution in slightly different directions.
6. SAM: the third-round masks no longer swallow the black cup or the brick, but motion-blurred arms and arm parts
   over the black cup still segment imperfectly (p10 0.37).

Getting the overhead camera to >= 0.80 would need a better arm model for this view (cables, the actual camera
board / plate, gripper housing as printed) or a scene whose fixed camera sees the whole arm; the parent may prefer
to swap the scene for a replacement dataset.

Other notes: the brick held in the gripper is placed midway between the two jaw tips (official meshes), oriented
with the gripper; per-episode cup / brick positions are the builder's detections (objects.json pixel centroids)
re-ray-cast through the refitted top camera (moved 2.2 cm cup / 1.1 cm brick on average); RGB images get the same
lens warp as the masks and a per-camera per-channel gain (real / sim mean colour over 16 frames). The SO-101's
camera module and bracket are hidden from the wrist camera (geom group 3) since the camera sits inside them.
