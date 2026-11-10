# so101_orangebox_twin2: pixel-aligned MuJoCo twin of so101_orangebox_real

This is the twin of `neuroscope/molmoact2/store/sets/so101_orangebox_real`: 320 frames, 40 episodes x 8. The source is
the LeRobot v3.0 dataset `data_so101/AbdullahRasul__so101_blue_black_pick_place` (`so101_follower`). The task is: an
orange SO-101 picks a yellow-green block and places it in a black open box on a white table.

- Camera `top` = `observation.images.top` (static, overhead).
- Camera `side` = `observation.images.side` (static, high, across the table).
- The dataset's `front` camera is on the wrist and is not used.

Output set: `neuroscope/molmoact2/store/sets/so101_orangebox_twin2/`, in the existing twin layout.

All code is in this folder. Nothing was edited in `so101_twin/common/`, `fitkit/`, `stack2/`, `ballcup/`,
`neuroscope/pi05` or `interp/`; the common toolkit files were copied here and the copies edited (`ORANGEBOX` marks).
The MolmoAct2 policy was never run.

## Result (agreed metric; pass = held-out median >= 0.80 on every camera): PASS on both cameras

Final twin: `out/orangebox_r4_so101_fit.json`, official SO-101. Primary masks: `masks/orangebox_<cam>_c4.npy`.

| camera / frames                  | n   | median    | p10   | mean  | excluded (real arm < 0.1 % of image) |
| -------------------------------- | --- | --------- | ----- | ----- | ------------------------------------ |
| **top held_out** (odd episodes)  | 128 | **0.908** | 0.875 | 0.906 | 0                                    |
| top all_frames                   | 320 | 0.908     | 0.872 | 0.907 | 0                                    |
| **side held_out** (odd episodes) | 128 | **0.845** | 0.786 | 0.837 | 0                                    |
| side all_frames                  | 320 | 0.848     | 0.780 | 0.837 | 0                                    |

- The arm is in view in all 320 frames of both cameras, so no frame is excluded. The smallest real mask covers
  3.4 % of the image on top and 2.1 % on side.
- Held-out medians by phase:
  - top: approach 0.916, grasp 0.890, release 0.925.
  - side: approach 0.867, grasp 0.825, release 0.839.
- Side held-out median by camera group: e00-09 0.789, e10-48 0.859. 82 % of held-out side frames reach 0.80; all
  held-out top frames do.

### The same final twin on every mask set

Held-out / all-frames median, plus held-out p10:

| masks               | how they were made                                                                                               | top                           | side                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------------------- |
| `_c4` (**primary**) | SAM consensus prompted by the anchored even-episode twin (js4), dark negatives                                   | **0.908** / 0.908 (p10 0.875) | **0.845** / 0.848 (p10 0.786) |
| `_c2b`              | `_c2` (prompted by the NON-anchored twin bs_c1) + the twin-free static-base rule (below), minus the static clamp | 0.908 / 0.907 (0.873)         | 0.803 / 0.804 (0.729)         |
| `_c3`               | SAM consensus prompted by the non-anchored twin bs_c1, dark negatives                                            | 0.908 / 0.908 (0.878)         | 0.791 / 0.794 (0.710)         |
| `_c1`               | SAM consensus prompted by the first joint seed (camera board misplaced)                                          | 0.907 / 0.907 (0.871)         | 0.781 / 0.786 (0.698)         |
| `_dino`             | twin-INDEPENDENT: Grounding-DINO box + hue points, no twin                                                       | 0.904 / 0.902 (0.851)         | 0.757 / 0.768 (0.684)         |

The top score barely depends on the masks: 0.904-0.908 everywhere, including the twin-free `_dino` set.

The side score depends on one thing: whether the masks contain the arm's fixed base, which the side camera sees at its
left image edge (next section).

- `_c4` contains the base in 310 of 320 frames.
- `_c2b` adds it with a deterministic, twin-free rule.
- `_c3`, `_c1` and `_dino` mostly leave it out (`_dino` keeps it in only 68 frames). `_dino` also swallows the black
  clamp under the base in 29 side frames (20 of them in episodes 0-9), and parts of the black box when the gripper is
  over it.

On masks that omit the base, the twin's correctly placed base counts as false positive. That is the gap between 0.845
and 0.76-0.79. Read the side pass as conditional on the base being part of the arm silhouette; the evidence that it
is follows.

### Why the base is part of the arm (and what anchoring it did)

The side view shows two static orange pieces at x = 0..40 px, y = 220..335:

- a hole-grid block: the SO-101 base;
- above it, the shoulder bracket where the arm enters the image.

Below them is a black clamp/stand strip, which is not part of the arm.

The first fits put the twin's base about 50-100 px outside the left edge. Neither camera sees the base in the
twin-prompted masks, so the start of the kinematic chain was unconstrained: joint offsets and camera poses traded off.

Anchoring it means adding the static base pixels to the side masks (`static_parts.py`) and refitting on even episodes.
The result:

1. The twin's fixed `base` body lands exactly on the real hole-grid piece. The arm outline follows the shoulder piece
   above it. See `out/base_anchor_check.png`: the top row is the anchored twin, the bottom row the non-anchored
   control.
2. The **top** camera improved by +0.03 on every mask set, although top masks never contain the base. As a matched
   control I refined the non-anchored solution equally, on `_c3` (`out/ctl_bs_c3.json`). Its top camera stays at
   0.872-0.880 on all sets.

Control vs anchored, held-out medians (top / side):

| twin                                   | `_c4`             | `_c2b`        | `_c3`         | `_c1`         | `_dino`       |
| -------------------------------------- | ----------------- | ------------- | ------------- | ------------- | ------------- |
| non-anchored control (`ctl_bs_c3`)     | 0.880 / 0.722     | 0.878 / 0.694 | 0.879 / 0.807 | 0.879 / 0.807 | 0.872 / 0.791 |
| anchored, final (`orangebox_r4_so101`) | **0.908 / 0.845** | 0.908 / 0.803 | 0.908 / 0.791 | 0.907 / 0.781 | 0.904 / 0.757 |

Both twins only saw even episodes.

## Metric (as agreed, unchanged)

The metric is whole-arm silhouette IoU, per camera, on the real 640x480 pixel grid:

- The twin's arm mask is every geom of the arm subtree, base included. It is rendered as a padded pinhole image and
  warped onto the real grid with the fitted lens (`arm.Lens`). Pixels the render cannot fill count as background.
- It is compared with the unwarped SAM 2.1 mask of the real arm. The real masks are never resampled.
- Frames whose real mask covers less than 0.1 % of the image are excluded. There are none here.
- The fit only ever sees EVEN episodes (24 episodes, 192 frames). held_out = ODD episodes (16 episodes, 128 frames).
  all_frames = every visible frame.

`score.py <fit.json> --masks _c4,...` recomputes everything and writes per-frame IoU to `out/<fit>_score<tag>.json`.

## Model and joint mapping

- Arm: the official TheRobotStudio SO-101, `so101_new_calib.xml`, via the `arm.py` copy.
  - Base bottom at z = 0, pan axis = z, and pan 0 reaches +y.
  - Colours: orange body, black servos.
- Camera board. The real wrist camera sits in a printed square frame, about 45 mm, on the fixed jaw. The official
  `so101_new_calib_camera.xml` side bracket puts its module about 6 cm off the gripper axis, which is wrong for this
  arm. It is replaced by a box: orange frame with black PCB faces, body `cam_board` on the `gripper` body.
  - Pose: in the gripper frame, pos (-0.0004, 0.0600, -0.0373) m, quat (0.6815, -0.2200, -0.2168, -0.6635).
  - Size: 43 x 43 x 12 mm.
  - Pose and size come from a global search (`board_search.py`), then the polish.
- **Units: LeRobot NORMALISED** (`use_degrees=False`): -100..100 of each joint's calibrated travel, gripper 0..100.
  Evidence: `meta/stats.json` action min shoulder_lift = -100.0 and max elbow_flex = +100.0 exactly (the clamp of the
  normalised mode). The first state is [-3, -99, 99, 30, -55, 37].
- `q_deg = scale * state + offset` (sign +1 for all joints with the new calibration):

| joint         | scale (deg per unit) | offset (deg) |
| ------------- | -------------------- | ------------ |
| shoulder_pan  | 1.1954               | -4.82        |
| shoulder_lift | 1.0733               | -0.77        |
| elbow_flex    | 0.9690               | -6.13        |
| wrist_flex    | 1.0143               | +2.90        |
| wrist_roll    | 1.0 (pinned)         | -33.81       |

- The scales (0.97-1.20) are what normalised units predict (calibrated travel / 200), with no drift beyond that.
- Wrist roll: the state is constant at -55 +- 1.6, so its scale and offset are degenerate. The scale is pinned and
  the offset fitted, giving a roll angle of about -89 deg. Twin-free CMA runs started at offsets 0 and -30 both
  converged to -34..-39.
- Gripper: `q = -0.17453 + grip_off + grip_scale * g/100 * 1.91986 rad`, with grip_scale 1.128 and grip_off +0.0035
  rad.
- A gravity-compliance term (deg/Nm, `arm.py`) was tried. It fitted to 0 (no gain) and is inactive.
- Mesh comparison: the Menagerie SO-100 went through the same pipeline (joint seed, base anchor, board search) but was
  not polished. Held-out top / side was 0.829 / 0.791 on `_c4` and 0.822 / 0.714 on `_dino`
  (`out/bs5_so100_c2b.json`). The official SO-101 is kept.

## Cameras (world = arm frame) and lens terms

| camera | group  | pos (m)                 | quat (w,x,y,z)                   | fovy     | k1       | k2      | cx      | cy     | latency (frames) |
| ------ | ------ | ----------------------- | -------------------------------- | -------- | -------- | ------- | ------- | ------ | ---------------- |
| top    | all    | -0.0594, 0.3105, 0.6258 | -0.0608, -0.0021, 0.0642, 0.9961 | 36.07    | 0.0633   | -0.0316 | -0.0140 | 0.0044 | 0.0              |
| side   | e00-09 | 0.2642, 0.2109, 0.6116  | 0.8050, 0.1481, 0.0938, 0.5668   | 37.33    | -0.1450  | -0.0309 | -0.0100 | 0.0014 | -0.5             |
| side   | e10-48 | 0.3132, 0.2469, 0.5671  | 0.7880, 0.1594, 0.1581, 0.5732   | (shared) | (shared) |         |         |        |                  |

- MuJoCo camera convention: looks along -z, +y up.
- `cx` / `cy` are principal-point shifts as fractions of width / height.
- Guard rails: all |k1| <= 0.145 (limit 0.4) and principal-point shifts <= 1.4 % (limit 10 %). Before / after frames
  were checked anyway (`out/before_after.jpg`).
- Pixel aspect: a fx/fy scan peaks at 1.0, so pixels are square.
- Camera grouping: from phase correlation of per-episode median backgrounds (`static_check.py`).
  - top: 0 px shift in all 40 episodes, so one pose.
  - side: shifted (4, 4..6) px between episodes 0-9 and 10-48, 0 px within each, so one pose per group with fovy and
    lens shared. Both groups contain even episodes.
  - The two side poses differ more in 3D (about 6 cm) than the 5 px image shift implies. That is the depth/rotation
    ambiguity of a mostly flat workspace: it is right on pixels, but not a physical measurement of the move.
- Latency: the full-rate parquet (`fullrate.py`) is used for the grid and CMA. Top 0 frames, side -0.5 frames; it
  matters little here.

## Scene (RGB only; the metric never uses it)

- White table, beige floor.
- Black open box, about 12-13 cm square and 8.5 cm high, placed per episode. It is detected in the episode's first
  top frame (largest dark component, min-area rectangle) and ray-cast through the fitted top camera and lens
  (`objects.py`, `scene_orangebox.objects`).
- Yellow-green block (36 x 60 x 36 mm):
  - in the jaws (TCP) while the gripper is closed in grasp or release;
  - otherwise at its per-frame top-view detection (table, or the box floor when inside the box);
  - otherwise at the episode's first detection, or the box centre during release.
- Per-camera per-channel gain: real / sim mean colour over 16 frames.

## What was done

1. `static_check.py`: static cameras and side groups. Units from `meta/stats.json`. Joint signs checked: positive pan
   moves the arm to image-left from above, as the model predicts.
2. `hue_masks.py`: orange hue masks for seeding only, never scored. `seed_search.py`: twin-free camera search. Model
   surface samples (FK) are splatted on a coarse grid, then a random search over 60-80k poses, CMA on the splats,
   then CMA on renders.
3. `segment_dino.py`, SAM 2.1 hiera-small + Grounding-DINO tiny with "an orange robot arm.". Each candidate is chosen
   by orange-hue recall, minus white-table fraction, plus 0.25 x SAM score; consensus rounds add a twin silhouette
   (`--prompts`) and `--dark-neg`. Yellow-green block pixels are removed. Rounds:
   - `_dino`: twin-free;
   - `_c1`, `_c2`, `_c3`, `_c4`: consensus.
     Every run waited for the shared slot via `sam_wait.sh` (coordinator rule: `pgrep -f "so101_twin/.*segment"` empty
     for 12 s, ignoring shells whose text merely contains the pattern). Only one segmentation process of mine ran at a
     time. One of my early runs (before 07:51: `_dinotest`, `_dino` or `_c1`) overlapped with the other session's
     `fitkit/segment_arm.py` (reported by the coordinator),
     because my first wait pattern did not match that script; the waiter was switched to the coordinator's rule
     afterwards. At 08:19 another session's monitor shell (its command text contained the pattern) blocked both
     waiters while no segmentation was running; my waiter now ignores such shell wrappers, and only one python
     segmentation process was alive when `_c4` launched.
4. `fit.py` (copy of common): a first render fit on `_dino` stalled at about 0.62 / 0.60 fit-frame IoU.
   `joint_seed.py` (splat CMA over joint scales, offsets, gripper, board, both cameras and k1) found the
   normalised-unit joint map. Seeds from roll offsets 0 and -30 both converged; held-out on `_c1` was 0.855 / 0.730.
5. Gripper close-ups showed the official camera bracket was wrong, so the custom board was added. The splat sampler
   first skipped box geoms; after the fix, `board_search.py` (global, pose + size) reached 0.872 / 0.777 on `_c1`.
6. Static-base anchor: `static_parts.py`, `_c2b`, `js4` / `js5` and `bs5`, reaching 0.911 / 0.843 on `_c4`. Plus the
   matched non-anchored control (`ctl_*`).
7. Render polish with `fit.py` stages B, C, D on `_c4` (even episodes, 64 fit frames per camera, evals x0.8):
   `out/orangebox_r4_so101_fit.json`.
8. `render_set.py`: the set, `out/before_after.jpg` (8 odd-episode frames; per camera real | twin | IoU overlay;
   white = both, red = real only, blue = twin only) and `meta.json`.

Reproduce the final steps (repo root; MuJoCo venv `.venv`; SAM venv `neuroscope/molmoact2/.venv`, run via
`sam_wait.sh`):

```
.venv/bin/python so101_twin/orangebox/fit.py orangebox --mask-tag _c4 --split even --model so101 \
    --init-json so101_twin/orangebox/out/bs5_c2b.json --stages BCD --evals 0.8 --nfit 64 --out-tag _r4_so101
.venv/bin/python so101_twin/orangebox/score.py so101_twin/orangebox/out/orangebox_r4_so101_fit.json --masks _c4,_c2b,_c3,_c1,_dino
.venv/bin/python so101_twin/orangebox/render_set.py --fit out/orangebox_r4_so101_fit.json --score out/orangebox_r4_so101_fit_score_c4.json
```

## Limits

1. **Mask dependence of the side score.** It passes on base-including masks: `_c4` 0.845, and `_c2b` 0.803, whose
   prompts come from a different twin and whose base comes from a twin-free rule. It does not pass on masks that omit
   the base: `_c3` 0.791, `_c1` 0.781, and the twin-independent `_dino` 0.757.
   - `_dino` also has known errors: the clamp in 29 side frames (20 in episodes 0-9), and the box when the gripper is
     over it.
   - `_c4` was selected with help from a twin silhouette (the even-episode joint seed). That carries the small bias of
     every twin-prompted mask set (about +0.016 on lemon).
2. Side episodes 0-9 are weaker (held-out 0.789): fewer fit frames for that camera pose.
3. Not modelled: servo and camera cables (black, white-red loops), which SAM partly keeps. The camera board is a box,
   not the printed frame. Motion blur: 22 % of frames move more than 1 unit per frame.
4. The side group poses are not a physical measurement of the camera move (see Cameras).
5. Box, block and lighting are approximate (RGB only).

## Files

| file                                                                           | what                                                                                                                                                                |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene_orangebox.py`                                                           | scene: arm spec (camera board), cameras / groups, priors, objects, `frame_setup`                                                                                    |
| `arm.py`, `fit.py`, `fullrate.py`, `segment.py`, `render_set.py`               | copies of `so101_twin/common/` (`ORANGEBOX` edits: grip_off, camera board, compliance, pixel aspect, board / size / scale priors, `--cams`, masks / out paths here) |
| `static_check.py`, `hue_masks.py`, `objects.py`, `static_parts.py`             | static cameras, hue seed masks, object pixels, static base / clamp                                                                                                  |
| `seed_search.py`, `joint_seed.py`, `board_search.py`                           | twin-free seed, splat joint + camera CMA, global board search                                                                                                       |
| `segment_dino.py`, `sam_wait.sh`                                               | DINO + SAM segmentation (twin-free or consensus), queued on the shared slot                                                                                         |
| `score.py`, `tables.py`, `overlay.py`, `scan.py`, `scan_ar.py`, `scan_comp.py` | metric, tables, overlays, sensitivity scans                                                                                                                         |
| `masks/`                                                                       | `orangebox_<cam>_<tag>.npy` packed masks (`np.unpackbits(a, -1)[..., :640]`), `prompts_*` twin silhouettes                                                          |
| `out/`                                                                         | fits, score JSONs (per-frame IoU), logs, `before_after.jpg`, `base_anchor_check.png`, `masks_*_sheet.jpg`                                                           |
