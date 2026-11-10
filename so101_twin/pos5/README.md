# so101_pos5_twin2: pixel-aligned MuJoCo twin of so101_pos5_real

This is the twin of `neuroscope/molmoact2/store/sets/so101_pos5_real`: 320 frames, 40 episodes x 8. The source is the
LeRobot v3.0 dataset `data_so101/hbseong__record-pick-and-place-pos5-so101` (`so101_follower`). A white SO-101 in front
of a green screen picks a cyan cube and drops it into a white open box.

- Camera `top` = `observation.images.top` (static, overhead, strong barrel distortion).
- Camera `side` = `observation.images.right` (static, low, across the bench).

Output set: `neuroscope/molmoact2/store/sets/so101_pos5_twin2/`, in the existing twin layout.

The fitkit session handed this scene over. Its best fit was `fitkit/out/pos5_so101_gain.json`: held-out 0.773 top /
0.729 side, on its own masks, with ignore regions, on 48 frames per camera. All code here is in `so101_twin/pos5/`.

- `so101_twin/common/*.py` were copied here and the copies edited (`POS5` marks).
- `fitkit/` was only read; its pos5 files were copied to `legacy/`.
- Nothing in `fitkit/`, `jar/`, `neuroscope/pi05` or `interp/` was touched.
- The MolmoAct2 policy was never run.

## Result (agreed metric; pass = held-out median >= 0.80 on every camera)

The pass is claimed on two different readings:

- **top: strict.**
- **side: occluder-excluded only.** The side strict reading is 0.776, which fails.

Final twin: `out/pos5_r4_fit.json`, official SO-101. Primary masks: `masks/pos5_<cam>_c3.npy`.

Both readings use the same frozen twin and the same masks. All 320 frames of both cameras have the arm visible.

| camera | reading           | held_out n | median    | p10   | mean  | all_frames n | median | p10   | mean  |
| ------ | ----------------- | ---------- | --------- | ----- | ----- | ------------ | ------ | ----- | ----- |
| top    | strict            | 160        | **0.846** | 0.673 | 0.818 | 320          | 0.843  | 0.674 | 0.817 |
| top    | occluder-excluded | 160        | 0.852     | 0.673 | 0.823 | 320          | 0.851  | 0.675 | 0.822 |
| side   | strict            | 160        | 0.776     | 0.715 | 0.771 | 320          | 0.775  | 0.706 | 0.757 |
| side   | occluder-excluded | 160        | **0.806** | 0.773 | 0.807 | 320          | 0.805  | 0.769 | 0.792 |

The excluded fraction is |O & (twin | real)| / |twin | real| per frame, i.e. the share of the strict union that the
exclusion removes:

| camera | held_out mean | held_out median | all_frames mean | all_frames median |
| ------ | ------------- | --------------- | --------------- | ----------------- |
| top    | 0.018         | 0.001           | 0.019           | 0.002             |
| side   | 0.045         | 0.025           | 0.043           | 0.025             |

Held-out medians by phase:

- top, strict: approach 0.849, grasp 0.835, release 0.865.
- side, occluder-excluded: approach 0.806, grasp 0.814, release 0.787.

Share of held-out frames at or above 0.80:

- top strict: 76 %.
- side strict: 22 %.
- side occluder-excluded: 61 %.

Per-frame values are in `out/pos5_r4_fit_score_c3_ck_s2_lsam_occ.json`. All tables are in `out/pos5_r4_tables.md`.

### The same final twin on every mask set (held-out median, strict / occluder-excluded)

| masks               | how they were made                                    | top           | side              |
| ------------------- | ----------------------------------------------------- | ------------- | ----------------- |
| `_c3` (**primary**) | chroma key + SAM in the static regions (below)        | 0.846 / 0.852 | 0.776 / **0.806** |
| `_ck`               | twin-free chroma key only (no SAM, no twin)           | 0.821 / 0.839 | 0.777 / 0.796     |
| `_s2`               | SAM only (prompted by the r2 twin + chroma consensus) | 0.852 / 0.861 | 0.766 / 0.787     |
| `_lsam`             | the fitkit session's SAM masks                        | 0.790 / 0.816 | 0.668 / 0.720     |

The top passes on every mask set except `_lsam`. The side pass holds only on `_c3`, and with the exclusion. On the
twin-free `_ck` masks it is 0.796. On `_ck` the white arm in front of the white box or pale wall is lost, and the
table clamp counts as arm.

### Before / after on the final masks `_c3` (held-out median, strict / excluded)

| twin                                                                        | top               | side              |
| --------------------------------------------------------------------------- | ----------------- | ----------------- |
| fitkit best `pos5_so101_gain.json`, converted exactly (`convert_legacy.py`) | 0.808 / 0.813     | 0.725 / 0.767     |
| Menagerie SO-100 mesh, same pipeline (`out/pos5_r2so100_fit.json`)          | 0.808 / 0.814     | 0.747 / 0.779     |
| **final, official SO-101 (`out/pos5_r4_fit.json`)**                         | **0.846 / 0.852** | **0.776 / 0.806** |

The official SO-101 beats the SO-100 mesh on both cameras, both on the fit set and on the held-out frames.

`out/before_after.jpg` shows 8 odd-episode frames. Per camera: real | twin render | IoU overlay (white both, red real
only, blue twin only). `out/joint_check.jpg` shows the lowest and highest state of every joint, both cameras, with
the real (green) and twin (red) outlines.

## How the real masks are made (`masks/pos5_<cam>_c3.npy`)

The scene is a white arm in front of a green screen. The boundary against the screen comes from a chroma key; SAM
decides only where the key is blind.

1. **`bg.py`.** Per camera, the per-pixel median of the 320 real frames: `masks/bg_<cam>.npy`, `out/bg_<cam>.png`.
2. **`ck_masks.py`: twin-free, no SAM.**
   - Green = PIL-HSV hue 80-119 with saturation >= 50. The screen has hue 100-112 and saturation 60-160; the arm has
     saturation ~17.
   - Arm = not green, minus the cyan cube (hue 120-150).
   - Also minus the "static" pixels where the frame is unchanged from the median. Static = not green in >= 97 % of
     frames: box, wall corner, mat marks.
   - Small blobs are dropped; the rest must link to the arm-base box or the image border.
   - Result: `_ck`.
   - Higher saturation thresholds (58, 66) were tested and were worse: parts of the screen became "arm".
3. **`segment.py`: SAM 2.1 small, one process at a time, queued by `sam_wait.sh`.**
   - Prompted by the refitted twin's silhouette.
   - Positives only where the twin AND the chroma key agree.
   - Hard negatives on static objects away from the arm, and on the cube.
   - The candidate is chosen by agreement with the twin and with `_ck`.
   - Result: `_s2`, prompted by the r2 twin. This is the one re-segmentation; `_s1` was prompted by the converted
     fitkit twin.
4. **`combine.py`.**
   - Outside the static pixels, the chroma key decides.
   - Inside them, SAM decides: the white arm in front of the white box or pale wall, and arm shadows on the box. A SAM
     blob counts only if it touches the chroma arm. Outside the box it must also differ from the median background.
   - Removed in every frame:
     - the cube;
     - the pale magenta wall corner (R-G >= 10, B-G >= 8, bright; the arm is neutral, median B-G -5);
     - the side view's black table clamp + orange pad at the base (static, from the median image).
   - Result: `_c3`.

The fitkit masks had two main faults:

- **The white box was glued to the arm.** On the side view this cost ~0.1 IoU.
- **The pale wall corner leaked into the side masks.** Rule 4 fixes this. The arm does pass in front of that corner,
  so a fixed ignore region was not allowed. No ignore region is used in either mask.

**Occluder masks for the excluded reading** come from `occ_masks.py`. They use the real frames only, never the twin:

- the cube's pixels (hue);
- the white box's pixels where the box is visible: the box region of the median image, unchanged in the frame;
- the side view's table clamp.

The fit is frozen; both masks lose these pixels. On the side view the excluded pixels are mostly:

- the gripper lowered behind the box's front wall;
- the held cube in front of the gripper / base;
- the clamp.

## Model, joint mapping, units

**Model.** Official SO-101 (TheRobotStudio), `so101_new_calib.xml`, white body and black servos, no wrist camera.

- The Waveshare driver-board mounting plate at the back of the base is left out (`ArmSpec.drop_meshes`). The real side
  view shows the pale wall and the table clamp where the model puts that plate.
- Leaving it out raises the even-episode (fit-set) side mean IoU by +0.003 for every parameter set tried (r2, r3,
  r3k3). The top is unchanged.
- `scene_pos5.DROP_PLATE = False` restores the full model.
- Arm frame placed at world (0, 0.045164, 0), yaw 0. This is the fitkit world: floor z = 0 under the base, pan axis
  at (0, 0.0452), reach +y.

**Joint units.** LeRobot `so101_follower` NORMALISED: -100..100 of each joint's calibrated travel; gripper 0..100.

**Joint map.** `q_rad = deg2rad(scale * state + offset)`, all signs +1 for `so101_new_calib.xml`:

| joint         | scale (deg / unit) | offset (deg) |
| ------------- | ------------------ | ------------ |
| shoulder_pan  | 1.032              | -0.16        |
| shoulder_lift | 1.053              | -2.83        |
| elbow_flex    | 0.963              | -6.35        |
| wrist_flex    | 1.000              | 3.62         |
| wrist_roll    | 1.110              | 5.00         |

A scale is travel / 200, so these mean calibrated travels of 193-222 deg. That is what normalised units explain;
there is no extra gain.

**Gripper.** `q = -0.1745 + 0.55 * g/100 * 1.9199 + 0.15` rad, i.e. -1.4 deg at g = 0 and 59.1 deg at g = 100. The
model's default map is -10..100 deg.

- The calibrated gripper travel is about 60 deg, not 110: a gain of 0.55 relative to the default.
- It was set by a grid on even episodes (`grip_grid.py`). Stage C's CMA had only reached 0.745 / -0.05 from the
  fitkit's 1.0 / 0, which never fitted the gripper.
- Visual check: the grip min/max rows of `out/joint_check.jpg`.

**Conversion from fitkit.** `convert_legacy.py` maps the fitkit parametrisation exactly. The fitkit used
`so101_old_calib.xml`, an SO-100-style map, gains about the mean pose, and cx/cy in px. The relation found by forward
kinematics is q_new = q_old + (0, 90, -90, 0, 2.79, 0) deg. Silhouettes, legacy vs converted: median IoU 0.996 top,
0.990 side.

## Cameras, lens, latency (guard rails)

| camera | world pos (m)             | fovy (deg) | k1     | k2     | principal point shift          | latency (frames) |
| ------ | ------------------------- | ---------- | ------ | ------ | ------------------------------ | ---------------- |
| top    | (0.0440, 0.2554, 0.5233)  | 46.42      | -0.313 | -0.046 | (7.6, 5.6) px = 1.2 %, 1.2 %   | -1.0             |
| side   | (0.4403, 0.2569, -0.0205) | 50.68      | -0.191 | -0.069 | (25.4, 11.9) px = 4.0 %, 2.5 % | -2.0             |

Quaternions are in the fit json and `meta.json`. The lens is x_d = x_u (1 + k1 r^2 + k2 r^4). The pinhole render is
padded and warped onto the real 640 x 480 grid; the real masks are never resampled.

Guard rails:

- |k1| < 0.4 on both cameras; the top's -0.313 is the largest.
- Principal-point shifts are under 10 %.
- No visual check was required, but the overlays were checked anyway.
- A k3 (r^6) term was tried (`out/pos5_r3k3_fit.json`). It tied on the fit set (mean IoU 0.7784 vs 0.7784) and was not
  kept.

Latency (image time minus state time, 30 fps frames): -1 top, -2 side, from the grid. The CMA refinements (r2:
-0.74 / -1.52) did not survive the r3 refit.

## Fit procedure (even episodes only; held-out = odd episodes)

`fit.py` is the common fitter: per-camera pose, fovy, lens, latency, joint offsets / scales, gripper scale. The POS5
additions are:

- the gripper offset;
- a configurable scale prior (sd 0.2 about 1 deg/unit);
- a trimmed objective (the worst 10 % of fit frames dropped);
- `--skip-c2`, `--k3`.

It uses 72 farthest-point fit frames per camera from even episodes, at 320 x 240. Evaluation is at full resolution.

1. `convert_legacy.py`: the fitkit best as the start (`out/init_from_legacy.json`).
2. r1: BCD on `_ck`.
3. `_s1` -> `_c1` -> `_c2` (wall rule).
4. r2: BCD on `_c2`, plus the same for the SO-100 mesh.
5. `_s2` from r2 -> `_c3` (clamp rule).
6. Gripper grid on `_c3`.
7. r3: BCD without C2, with and without k3. The simpler one was kept.
8. Waveshare plate dropped (fit-set evidence).
9. r4: stage-D polish -> **final** `out/pos5_r4_fit.json`.

The median background, static map, wall / clamp / cube rules and occluder masks use real-image statistics over all
frames, never twin parameters. The twin enters the masks only through the SAM prompts and the SAM candidate choice.

## Limits

1. **Side strict reading 0.776 < 0.80.** The side passes only with the agreed occluder exclusion: 0.806, with a
   held-out excluded fraction of mean 4.5 % and median 2.5 %. The margin is small (0.006).
2. **The side pass depends on the masks.** It holds on `_c3` but not on the twin-free chroma mask `_ck` (0.796
   excluded) or SAM-only `_s2` (0.787). `_c3` uses twin-prompted SAM inside the static regions. That carries the usual
   twin-prompt bias (lemon measured about +0.016).
3. **Base geometry.** Even without the plate, the official base reaches ~1.5 cm further back than the real one in the
   side view. Twin-only base pixels are ~6 % of the side union. k3 did not remove it, so it looks like a real
   difference in the printed base, not lens.
4. **Gripper.** Top and side prefer different jaw calibrations: about 40 deg and 57 deg at g ~ 97. The fit is a
   compromise; the jaw geometry / roll coupling is not modelled further.
5. **Not modelled:**
   - Servo cables (red/black loops) stay in the real masks as real-only error, mostly on top.
   - Top-view frames with the gripper at the bottom image edge (strongest distortion, arm partly out of view) are
     the worst (IoU 0.2-0.5).
6. **RGB is approximate.** The metric is arm-only. The box is fitted to the real box pixels (IoU 0.85 top / 0.90
   side). The cube is placed from the top-view cube pixels, in the jaws while closed in the grasp phase, else in the
   box. The green screen is a flat skybox + floor. Lighting and colour gain are rough.

## Files

| file                                                                           | what                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene_pos5.py`                                                                | scene: arm spec (plate dropped), cameras (init = converted fitkit fit), priors, RGB objects, `frame_setup`                                                                                                                                     |
| `arm.py`, `fit.py`, `fullrate.py`, `segment.py`, `render_set.py`               | copies of `so101_twin/common/` (`POS5` edits)                                                                                                                                                                                                  |
| `convert_legacy.py`, `legacy/`                                                 | exact conversion of the fitkit fit; read-only copies of its pos5 files, `armfit.py` and masks                                                                                                                                                  |
| `bg.py`, `ck_masks.py`, `combine.py`, `occ_masks.py`, `objects.py`             | median backgrounds, chroma masks, final masks, occluder masks, box / cube                                                                                                                                                                      |
| `grip_grid.py`, `score.py`, `tables.py`, `sheet.py`, `joint_check.py`, `p5.py` | gripper grid, metric, tables, diagnostics, helpers                                                                                                                                                                                             |
| `sam_wait.sh`                                                                  | runs `segment.py` when no other python segmentation process is alive                                                                                                                                                                           |
| `masks/`                                                                       | `pos5_<cam><tag>.npy` packed masks (`np.unpackbits(a, -1)[..., :640]`). Tags: `_c3` primary, `_ck`, `_s1/_s2` SAM, `_c1/_c2` earlier, `_occ` occluders, `_cube`, `_static`, `_tw*` twin silhouettes (SAM prompts), `_lsam/_ldino` fitkit masks |
| `out/`                                                                         | fits (`pos5_r*_fit.json`, logs, overlays), score jsons with per-frame IoU, `before_after.jpg`, `joint_check.jpg`, `objects.json`                                                                                                               |

## Reproduce

Run from the repo root. The MuJoCo venv is `.venv`; the SAM venv is used through `sam_wait.sh`.

```
.venv/bin/python so101_twin/pos5/convert_legacy.py
.venv/bin/python so101_twin/pos5/bg.py && .venv/bin/python so101_twin/pos5/ck_masks.py
.venv/bin/python so101_twin/pos5/objects.py --fit out/init_from_legacy.json   # box regions for occ_masks
cp so101_twin/pos5/out/objects.json so101_twin/pos5/out/objects_init.json
.venv/bin/python so101_twin/pos5/occ_masks.py
# r1 / r2 as above, then the one re-segmentation from the r2 twin:
.venv/bin/python so101_twin/pos5/score.py out/pos5_r2_fit.json --masks _c2 --save-twin _tw2
so101_twin/pos5/sam_wait.sh --prompts _tw2 --ck _ck --tag _s2
.venv/bin/python so101_twin/pos5/combine.py --ck _ck --sam _s2 --tag _c3
.venv/bin/python so101_twin/pos5/grip_grid.py out/pos5_r2_fit.json --masks _c3 --out out/pos5_r2_grip.json
.venv/bin/python so101_twin/pos5/fit.py pos5 --mask-tag _c3 --split even --init-json so101_twin/pos5/out/pos5_r2_grip.json \
    --stages BCD --skip-c2 --evals 0.8 --nfit 72 --obj-trim 0.1 --out-tag _r3
.venv/bin/python so101_twin/pos5/fit.py pos5 --mask-tag _c3 --split even --init-json so101_twin/pos5/out/pos5_r3_fit.json \
    --stages D --evals 1.0 --nfit 72 --obj-trim 0.1 --out-tag _r4          # scene with DROP_PLATE = True
.venv/bin/python so101_twin/pos5/score.py out/pos5_r4_fit.json --masks _c3,_ck,_s2,_lsam --excl _occ --save-twin _twF
.venv/bin/python so101_twin/pos5/objects.py --fit out/pos5_r4_fit.json --start out/objects_init.json
.venv/bin/python so101_twin/pos5/render_set.py --fit out/pos5_r4_fit.json \
    --score so101_twin/pos5/out/pos5_r4_fit_score_c3_ck_s2_lsam_occ.json
```
