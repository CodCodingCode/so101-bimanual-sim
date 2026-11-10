# so101_stack_twin — MuJoCo digital twin of `so101_stack_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_stack_real`
(LeRobot v3.0 dataset `data_so101/ichbinblau__so101_stack2cubes_dataset`, `so101_follower`). Each twin frame
uses the real frame's joint state, a fitted camera pose and an approximate scene layout. **The camera fit is weak
(arm-silhouette IoU 0.22 top / 0.14 side, well below the 0.5 target); read "What is approximate" before using
the twin for anything that needs pixel alignment.**

Output: `neuroscope/molmoact2/store/sets/so101_stack_twin/`, same layout as the real set: `meta.json`,
`images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json` identical to the real
set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='stack_twin'` and `pair[i] = i` (index
into the real set). Extra file `cube_where.npy` = where the wooden cube was rendered per frame (disc / jaws / block).
`meta.json` has `twin_of: "so101_stack_real"`, the joint mapping, the fitted cameras (global + per episode),
object positions and brightness gains.

## Files

Copied from `so101_twin/lemon/` and adapted (plus two new scripts):

| file               | what                                                                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`   | white SO-101 scene -> `so101_stack.xml`: table, blue cloth + white panel backdrop, paper disc, wooden cube, green block                                                                 |
| `twin.py`          | `real_to_sim()` (signs / offsets / scales; `joints.json` overrides), `Sim` renderer, segmentation masks, object setters                                                                 |
| `check_joints.py`  | real vs sim side by side -> `out/joints_check.png`                                                                                                                                      |
| `init_cameras.py`  | (new) keypoint least-squares camera initialisation; did not converge to anything useful, kept for the record                                                                            |
| `fit_cameras.py`   | silhouette-IoU fit: `--cam`, `--global` (wide random search), `--joint` (both cams + green block + joint offsets/scales), `--apply`                                                     |
| `episode_cams.py`  | (new) per-episode correction of the top camera (it moves between episodes) -> `out/episode_cams.json`                                                                                   |
| `place_objects.py` | disc / cube / block (x, y) by ray-casting hand-picked pixel centroids of real frame 0 through both cameras                                                                              |
| `replay.py`        | replays all 320 states and writes the twin set (per-episode top camera, cube placement, brightness gain)                                                                                |
| `make_sheet.py`    | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                                                                                     |
| `joints.json`      | fitted joint mapping                                                                                                                                                                    |
| `out/`             | `pairs_sheet.jpg`, `fit_top.png` / `fit_side.png` (real / sim / mask overlay: red real-only, blue sim-only, white both), `iou_report.json`, `camera_fit_*.json`, `episode_cams.*`, logs |

Run order (all with `.venv/bin/python`): `build_scene.py` -> `fit_cameras.py --cam top --global`, `--cam side`
-> `fit_cameras.py --joint --rounds 3` -> `--apply` -> `place_objects.py` (paste into DISC / OBJECTS) ->
`episode_cams.py` -> `replay.py` -> `make_sheet.py`.

## Joint mapping

The dataset is LeRobot v3.0 `so101_follower` (degrees relative to the calibration middle pose, arm straight up),
i.e. the same convention as `so101_twin/pickplace`, so the pickplace mapping was the starting point. Menagerie
`so_arm100` order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (rad):

```
sim_rad[j] = deg2rad(SIGNS[j] * SCALES[j] * real[j] + OFFSETS_DEG[j])     j = pan, lift, elbow, wrist_flex, wrist_roll
SIGNS       = [-1,  +1,      +1,      +1,      +1]
OFFSETS_DEG = [ 0,  -92.28,  +64.30,  +29.58,   0]      (derived first guess [0, -90, +90, 0, 0]; fitted)
SCALES      = [ 1,   0.923,   1.107,   0.988,   1]      (fitted; 1 = state already in degrees)
Jaw         = -0.174 + gripper/100 * (1.75 + 0.174)
```

How it was checked: the rest pose (lift -98.5, elbow +98.7) lands on the Menagerie rest keyframe; offsets and
scales of lift / elbow / wrist_flex from the joint silhouette fit (`fit_cameras.py --joint`, 18 frames); visual
check in `out/joints_check.png` and `out/pairs_sheet.jpg` (the arm folds, reaches and swings with the real one, but
the twin arm looks smaller / more folded in the top view). The pan sign is the pickplace one (-1); +1 was tried in
an earlier camera search and was not better, but with this weak signal the pan sign is not firmly established. Caveats: **this mapping is much less certain than lemon/pickplace.** The fit moved the elbow offset
by 26 deg and the wrist by 30 deg from the derived guess, which partly compensates for camera error; a refit from
a 1.5x-scaled scene gave quite different offsets ([-73.0, +58.1, +11.2], scales [0.92, 0.95, 1.15]) at lower
IoU. Pan offset and wrist roll were not fitted (kept 0 / sign +1 as in pickplace; roll barely changes the
silhouette). The gripper is mostly 1-2 (closed, empty), ~12 (holding the 4 cm cube), 24-40 (open); the linear
jaw map renders "holding" as almost closed.

## Cameras (world frame: table top z = 0, arm base at origin, arm reaches +y at pan = 0)

Neither real camera is on the wrist. "top" is an overhead camera ~0.5 m up looking down at the arm base region,
tilted ~33 deg about its axis; it **moves between episodes** (the green block shifts by up to ~50 px). "side" is
a static, low camera across the table looking at the green block with the blue cloth behind; the arm only enters
its view from the top-left when it is near the cubes. Both were fitted as static cameras (pinhole, no distortion).

| cam  | pos (m)                | yaw / pitch / roll (deg) | fovy (deg) | arm IoU 640x480 (18 frames) | green-block IoU |
| ---- | ---------------------- | ------------------------ | ---------- | --------------------------- | --------------- |
| top  | (-0.351, 0.230, 0.522) | -2.7 / -67.4 / 33.3      | 49.6       | **0.222** (0.13..0.32)      | 0.37            |
| side | (0.135, 0.534, 0.161)  | -128.0 / -19.2 / -0.3    | 33.4       | **0.138** (0.00..0.47)      | 0.87            |

Per-episode top camera (`episode_cams.py`, used by `replay.py`, all in `meta.json` `episode_cameras`): a small
(dx, dy, dz, dyaw, dpitch, droll) correction per episode, fitted on that episode's 8 frames. Over all 320 frames:
green-block IoU 0.31 -> **0.58**, arm IoU 0.19 -> 0.19 (unchanged; the arm term is the noisy one).

Fit objective: mean over 18 farthest-point-sampled frames of the mean of (arm IoU, green-block IoU); random/global
search then Nelder-Mead at 320x240, a joint 17-parameter Nelder-Mead over both cameras + block position/size,
alternated with Nelder-Mead over the lift/elbow/wrist offsets and scales. Masks:

- real arm: the arm is WHITE on a light-grey table and a blue cloth, so colour alone does not work. Top: pixels
  > 18 brighter than the per-pixel median background with low saturation, plus near-black motor pixels; a band
  > where the blue cloth edge moves between episodes is ignored. Side: bright low-saturation pixels where the
  > background is not white, plus black. Large parts of the real arm (over the white table/panel) are missed, the
  > black cable and shadows leak in.
- real green block: hue mask (excluding the green PCB near the base in the top view).
- sim: segmentation render of the arm's visual geoms / the block geom.

Why the IoU is low: (1) the white-arm mask is poor (see above), so even a perfect pose would score maybe ~0.5;
(2) the top camera moves between episodes; (3) in the side view the arm is out of frame or only the jaw tips are
visible in ~half the frames (IoU 0 there pulls the mean down); (4) camera pose, scene scale and the joint offsets
are partly degenerate with this little signal — the fit found several local optima of similar score (the logs
`out/fit_joint.log` (restart from a different top-camera optimum, combined 0.13) and `out/fit_joint2.log`
(1.5x-scaled scene, 0.32) are the discarded runs; the kept fit (combined 0.40) is recorded in `out/camera_fit_*.json`
and `out/iou_report.json`). The side camera is constrained mostly by the green block (IoU 0.87).
`init_cameras.py` (keypoint least squares from hand-read cube/fingertip pixels) did not give a usable start.

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes, white body, black motors, white jaws; no wires, PCB or camera mount.
- Table: light grey box; blue cloth sheet and a white panel as vertical backdrops for the side view (geom group 4,
  hidden when rendering "top"). The blue cloth lying on the table in the real top view (lower right), the
  electronics / cardboard under the base and the cable are not modelled.
- Objects (from `place_objects.py`, ray cast through both fitted cameras, averaged): paper disc r 4.2 cm at
  (-0.106, 0.212); wooden cube 3.4 cm at (-0.151, 0.196); green block 5.2 cm at (-0.082, 0.318) (block position
  and size from the camera fit). The real cube is ~4 cm and the block ~5-6 cm; the twin layout is the same for
  all 40 episodes (the real cube start pose varies a little).
- Cube during replay (`cube_where.npy`): on the disc until the first "holding" frame of the episode (gripper
  5..18), attached to the `tcp` site of the fixed jaw while holding, on top of the green block after the last
  holding frame. Only 8 frames per episode are sampled, so the transitions are approximate.
- Exposure: per-camera brightness gain computed from 8 frames so the mean intensity matches the real images
  (top 0.65, side 1.21; in `meta.json`); no white balance, noise or blur.

## Not done / could be improved

- A better real arm mask (e.g. a segmentation model) is the main lever; then refit cameras and joint offsets.
- Per-episode cube / layout, per-episode side camera, lens distortion, the SO-101 base shape.
- Fit pan offset and wrist roll.
