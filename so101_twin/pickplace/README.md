# so101_twin/pickplace — digital twin of `so101_pickplace_real`

A MuJoCo scene with one SO-101 arm (Menagerie `trs_so_arm100` meshes) on a white table, a transparent box with
an orange rim to the arm's left and a pink lego brick in front of it, rendered from two fixed cameras `top`
and `side` that were fitted to the real `up` / `side` cameras of `data_so101/lerobot__svla_so101_pickplace`.
Every frame of the real set `neuroscope/molmoact2/store/sets/so101_pickplace_real` (300 frames, 50 episodes x 6)
was replayed at its stored joint state and rendered, producing the twin set
`neuroscope/molmoact2/store/sets/so101_pickplace_twin` with the identical layout (`labels['pair'][i] == i`).

```
.venv/bin/python so101_twin/pickplace/build_scene.py            # -> so101_pickplace.xml
.venv/bin/python so101_twin/pickplace/fit_cameras.py [--resume]  # -> cameras.json joints.json fit_report.json out/fit_*.jpg
.venv/bin/python so101_twin/pickplace/make_twin.py --overwrite   # -> store/sets/so101_pickplace_twin/
.venv/bin/python so101_twin/pickplace/verify.py                  # layout checks + out/pairs_sheet.jpg
```

Files: `build_scene.py` (scene + all bench parameters at the top), `joints.py` (real -> sim joint map),
`twin.py` (compile, pose, render, silhouette helpers), `fit_cameras.py` (silhouette-IoU fit), `make_twin.py`
(replay + set writer), `verify.py`, `assets/so_arm100/` (copy of the Menagerie arm), `cameras.json` /
`joints.json` (fitted values; they override the defaults in the .py files), `fit_report.json` (per-frame IoU).
All MuJoCo work uses the repo-root venv (`.venv/bin/python`, MuJoCo 3.3.0).

## World frame and scene

Arm base plate at the origin on the table top (z = 0). The Menagerie arm reaches along -y at qpos = 0, which is
toward the `top` camera (the real `up` camera is a front-elevated view: the gripper points at it in the rest
pose). The `side` camera sits on the box side (-x, slightly in front) and looks across the table toward +x.

| object    | pose / size                                                                                                   | how it was set                               |
| --------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| table top | white box, far edge at y = +0.10 (arm clamped near the far edge), 0.9 m wide                                  | by eye, back-projected from the top view     |
| box       | centre (-0.15, 0.02), outer 0.13 x 0.11 x 0.06 m, 2 mm transparent walls, 6 mm orange rim on the bottom edges | by eye on frame-0 overlays                   |
| brick     | 2x4 lego 31.8 x 15.8 x 11 mm (+studs), at (-0.075, -0.23), long axis along y                                  | by eye on frame-0 overlays; **static**       |
| arm       | all links black (`0.07`), fixed jaw white, moving jaw black                                                   | matches the real printed parts               |
| lights    | one weak shadow-casting key light + fill, headlight ambient 0.42 | tuned so the table reads ~0.8 with faint shadows, not saturated |

## Joint mapping (LeRobot SO-101 state -> Menagerie qpos)

`state = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] deg, gripper 0-100` (LeRobot
`so100_follower` degrees mode: zero = calibration "middle" pose), sim joints
`[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]` rad.

```
sim_rad[j] = deg2rad(SIGNS[j] * state[j] + OFFSETS_DEG[j])
SIGNS       = [-1,    +1,    +1,    +1,   +1  ]
OFFSETS_DEG = [-14.8, -92.4, +81.9, +6.0, +153.6]      (fitted; the derived first guess was [0, -90, +90, 0, 0])
Jaw         = -0.174 + gripper/100 * (1.75 + 0.174)   (linear onto the Menagerie Jaw range)
```

Derivation: the dataset rest pose (lift -99, elbow +99, wrist +75) has to land on the Menagerie `rest` keyframe
(Pitch -190 deg, Elbow +178 deg, Wrist_Pitch +68 deg): Pitch = lift - 90, Elbow = elbow + 90, Wrist_Pitch = wrist.
Positive real pan swings the arm over the box at -x while positive Menagerie `Rotation` goes to +x, so pan is
flipped. The offsets were then refined (Nelder-Mead on mean IoU of both cameras, 18 frames) and the wrist-roll
sign/offset chosen among {+-1} x {0, +-90, 180} deg (best: +1, 180 -> refined to 153.6). The replayed arm folds,
reaches and swings like the real one in all 300 frames (see `out/pairs_sheet.jpg`).

Caveats: the pan offset of -14.8 deg is what the fit wants given the camera placement; it may partly absorb a
yaw of the real base relative to the cameras. The elbow goes to 189 deg at rest (Menagerie limit 180; `mj_forward`
does not clamp). Wrist roll only affects the silhouette weakly, so 153.6 deg is the least certain number. The
gripper percent -> jaw angle is a linear guess (the real gripper opens to ~33% at most in this data).

## Cameras (640x480, fitted by silhouette IoU)

Real arm silhouette: `max(RGB) < 70`, components >= 15% of the largest, holes < 2000 px filled. Sim silhouette:
segmentation render of the arm's dark geoms (all meshes except the white fixed jaw). 18 frames chosen by
farthest-point sampling on the joint vector (indices in `fit_report.json`). Random coarse search (2000 samples,
6 frames, 160x120) then Nelder-Mead over pos / lookat / roll / fovy (8 params), alternating 5 rounds with the joint
offsets; final IoU evaluated at 320x240.

| camera | pos (m)                 | lookat (m)              | roll     | fovy     | IoU mean  | IoU min |
| ------ | ----------------------- | ----------------------- | -------- | -------- | --------- | ------- |
| top    | (-0.027, -0.416, 0.558) | (0.010, -0.083, 0.041)  | -3.4 deg | 42.0 deg | **0.699** | 0.609   |
| side   | (-0.386, -0.257, 0.146) | (-0.007, -0.068, 0.024) | +5.5 deg | 46.3 deg | **0.533** | 0.421   |

Overlays: `out/fit_top.jpg`, `out/fit_side.jpg` (green = sim silhouette edge, red = real-only pixels). Fit log:
`out/fit_log.txt`, `out/fit_log2.txt`. The side camera scores lower because the real arm fills and often leaves
the frame there, the real SO-101 base/shoulder housing is bulkier than the SO-100 mesh (an extra base box was
tried and lowered IoU, so it is off: `BASE_BLOCK = None`), and the real lens has visible distortion that a pinhole
camera cannot reproduce.

## Twin set

`neuroscope/molmoact2/store/sets/so101_pickplace_twin/`: `meta.json` (name, n=300, cameras [top, side],
real_or_sim=sim, source=pickplace_twin, image_shape [480, 640, 3], twin_of=so101_pickplace_real, plus the camera
and joint parameters used), `images_top.npy` / `images_side.npy` uint8 [300, 480, 640, 3], `state.npy` and
`prompt.json` byte-identical to the real set, `labels.npz` = the real arrays with `real_or_sim='sim'`,
`source='pickplace_twin'` and `pair = arange(300)`, and `sim_qpos.npy` (the qpos actually rendered).
`verify.py` asserts all of this and writes `out/pairs_sheet.jpg` (8 random pairs: real top | twin top | real side |
twin side). Not recorded through the model here (`record_set.py` is the parent's job).

## What is approximate

- Object poses are static and by eye: the brick moves every episode in reality (and is carried by the gripper);
  the twin always shows it at its first-frame spot. The box is a fixed guess (size, wall transparency, rim).
- The table's far edge / background: a grey wall; the real side view has cardboard boxes and a lit wall.
- The arm is the Menagerie SO-100 (same link lengths as the SO-101, different motor housings / base), uniformly
  black with a white fixed jaw; no wires, stickers, teal fingertip or camera mount.
- Pinhole cameras, no lens distortion, no motion blur / exposure / white-balance matching; shadows are from a
  single key light.
- Silhouette IoU (0.70 / 0.53) is a first-pass fit under the 2 h budget; residual misalignment is mostly in the
  wrist/gripper region and at the base.
