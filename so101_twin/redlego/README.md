# so101_redlego_twin2: pixel-aligned MuJoCo twin of so101_redlego_real

Twin of `neuroscope/molmoact2/store/sets/so101_redlego_real` (320 frames = 40 episodes x 8; LeRobot v3.0 dataset
`data_so101/eruedas__so101_3cams_pickplace-red-lego_real`, robot so101_follower, task "Grab the red lego block and place
it in the brown box"; `top` = `observation.images.top`, `side` = `observation.images.left`; the wrist camera
`observation.images.gripper` is not used). Output set: `neuroscope/molmoact2/store/sets/so101_redlego_twin2/` (existing
twin layout). Everything for this scene is in `so101_twin/redlego/`; nothing in `so101_twin/common/`, `fitkit/`,
`jar/`, `pos5/` or any other scene was edited (the toolkit files were copied from `so101_twin/jar/`, itself the common
toolkit plus the ballcup / legocup2 / jar changes, and the copies edited; changes marked `REDLEGO`). The MolmoAct2
policy was never run.

## Result (pass = held-out median at least 0.80 on every camera)

**Pass is claimed on the strict reading A: held-out median top 0.826, side 0.833.** The real-occluder-excluded
reading B is reported too and also passes (top 0.831, side 0.839).

| reading                  | camera   | frames       | n   | median    | p10   | mean  |
| ------------------------ | -------- | ------------ | --- | --------- | ----- | ----- |
| **A strict arm-only**    | **top**  | **held_out** | 128 | **0.826** | 0.792 | 0.830 |
| A strict arm-only        | top      | all_frames   | 320 | 0.826     | 0.792 | 0.830 |
| B real-occluder-excluded | top      | held_out     | 128 | 0.831     | 0.803 | 0.837 |
| B real-occluder-excluded | top      | all_frames   | 320 | 0.832     | 0.802 | 0.837 |
| **A strict arm-only**    | **side** | **held_out** | 128 | **0.833** | 0.765 | 0.829 |
| A strict arm-only        | side     | all_frames   | 320 | 0.830     | 0.763 | 0.826 |
| B real-occluder-excluded | side     | held_out     | 128 | 0.839     | 0.783 | 0.839 |
| B real-occluder-excluded | side     | all_frames   | 320 | 0.836     | 0.782 | 0.837 |

Fraction of pixels removed by reading B per frame, |occluder & (twin | real)| / |twin | real|:

| camera | held-out mean | held-out median | all-frames mean | all-frames median | frames empty after exclusion |
| ------ | ------------- | --------------- | --------------- | ----------------- | ---------------------------- |
| top    | 0.0085        | 0.0000          | 0.0082          | 0.0000            | 0                            |
| side   | 0.0128        | 0.0058          | 0.0134          | 0.0057            | 0                            |

Frames excluded because the real arm is not in view (SAM arm mask below 0.1 % of the image): top 0 of 320 (0
held-out), side 0 of 320 (0 held-out).

- **A strict arm-only**: the twin's arm rendered alone (every geom of the arm subtree: base, links, gripper, plus the
  fitted wrist-camera bar and base cable loop described below; no scene object) vs the SAM mask of the real arm.
- **B real-occluder-excluded** (the agreed amendment): the same two masks with the pixels of REAL occluder masks removed
  from BOTH. Occluders = the red brick (covers the gripper while held, mostly towards the side camera) and the cardboard
  box (its front wall stands between the side camera and the gripper inside the box), segmented from the real frames
  only (`segment.py --mode dino`: Grounding DINO "a red lego brick." / "a cardboard box.", every box with score 0.3 or
  more, SAM 2.1 box prompt, kept by fixed colour / size rules; no twin input, no fitted parameter):
  `masks/redlego_<cam>_d1_occ.npy`. They were made before the first fit and never entered any fit.
- Twin-bias check (reference, not a claim): the same final twin against the purely twin-independent DINO+SAM masks
  `_d1k` scores held-out strict top 0.820 / side 0.830 (all frames 0.820 / 0.828; reading B 0.827 / 0.837),
  `out/score_final_indep.json`.

## Metric (as agreed, unchanged)

Whole-arm silhouette IoU between the twin's rendered ARM-ONLY mask and a SAM 2.1 mask of the real arm, per camera, on
the REAL 640x480 grid: the twin is rendered as a padded pinhole image and warped by the fitted lens onto the real grid
(`arm.Lens`, as in `so101_twin/common/arm.py`; pixels the padded render cannot fill count as background); the SAM mask
is never resampled. Fit only on EVEN episodes (24 episodes, 192 frames); held_out = ODD episodes (16 episodes, 128
frames); all_frames = all 320. `score.py` recomputes everything from the fit json (`out/score_final.json` carries the
per-frame IoUs of both readings).

## Masks (SAM 2.1 hiera-small; one SAM process at a time, the shared slot checked with pgrep before every run)

1. `_d1` twin-independent (`segment.py --mode dino`): Grounding DINO tiny "a black robot arm." (every box with score
   0.25 or more and at least half the best), SAM box prompt, union; the same run writes the brick / box masks (above).
   `clean_masks.py` gives `_d1k`: minus brick and box (dilated 2 px), opened with a 5x5 square (thin servo / USB
   cables), small far components dropped. `_d1k` seeded the cameras and drove the first fits.
2. `_t1` (`segment.py --mode twin`, prompts = the r1c twin silhouettes incl. bar and cable loop, `masks/prompts_r1/`):
   consensus prompting with two silhouettes, the twin's and `_d1k`: box from their union dilated 12 px, positives from
   their intersection, negatives far from the union plus hard negatives on dark pixels 20-90 px outside and on the brick
   / box masks; 3 prompt variants x 3 multimask outputs, kept = best mean 4-px-tolerant IoU with both silhouettes +
   0.25 x SAM score; 3x3-smoothed logits. Median agreement with the silhouettes: top 0.935, side 0.912.
3. `strip_objects.py` removes the real brick pixels (a held brick merged with the gripper is not the arm), giving
   `_t1o`; `open_masks.py` removes thin cables exactly as for `_d1k` (5x5 opening, small far components dropped),
   giving **`_t1ok`, the scored masks** (the twin has no cables; the opening is fixed, twin-independent
   post-processing).

## Model and joint mapping

Official SO-101 (TheRobotStudio SO-ARM100 Simulation/SO101, `so101_new_calib.xml`, so101_follower calibration), black.
The Menagerie SO-100 meshes were fitted through the same rounds and score lower (round r2, same masks and protocol:
held-out top 0.810 / side 0.804 vs 0.828 / 0.815 for SO-101 at that round); the final choice used only the even-episode
medians (`finalize.py`, `out/model_comparison.json`).

Units: LeRobot so101_follower joint positions in the dataset's units; the arm joints behave as LeRobot's NORMALISED
mode (-100..100 of each joint's calibrated travel, `use_degrees=False`): `meta/stats.json` reaches -98.7 (lift) and
+99.7 (elbow) without saturating exactly at +-100, so the units were settled by the fit: per-joint scales 0.96-1.18
deg/unit (pan 1.18 = about 236 deg of calibrated travel), not the 1.0 of degrees mode. Joint map
`q_rad = deg2rad(scale * state + offset)` (signs +1, nominal 0, as `arm.py` calib "new"):

| joint                        | scale (deg / unit)                                                         | offset (deg)       |
| ---------------------------- | -------------------------------------------------------------------------- | ------------------ |
| shoulder_pan                 | 1.182                                                                      | 2.83               |
| shoulder_lift                | 1.049                                                                      | -1.41              |
| elbow_flex                   | 0.957                                                                      | -3.01              |
| wrist_flex                   | 1.050                                                                      | -3.73              |
| wrist_roll                   | 1.002                                                                      | -17.23             |
| gripper (0..100, 0 = closed) | q = -0.1745 + grip_off + grip_scale * g/100 * 1.9199 rad, grip_scale 0.881 | grip_off 0.096 rad |

The wrist roll sits at -55 +- 4 units throughout; its zero was scanned over 360 deg (15 deg grid, `jseed.py
--roll-scan`) before the fit. A gravity-sag term (servo compliance, as jar / orangebox) was offered to the fit and
stayed at 0. Camera latency from the 30 fps parquet: top -1.0 frames, side -0.5 frames (the image shows the arm that
much before the recorded state).

Twin-specific arm additions (both are physically on the real robot and inside its SAM silhouette; their geometry is
fitted, not CAD; both are part of the arm-only mask in every reading):

- **wrist-camera bar**: the real wrist camera hangs at the end of a roughly 7 cm flat bar off the wrist (not the
  official bracket, which sits elsewhere on the gripper; checked against the frames, `out/wristmount_test.jpg`).
  Modelled as a 10 x 3 mm rod plus a 26 x 26 x 16 mm module box on the gripper body; pose and length from a global
  search (`bar_search.py`: 1500 random poses + CMA, even episodes, +0.015 mean IoU) refined in stage C. Params
  `[x, y, z, rotvec, length]` in the gripper frame: [-0.0085, 0.0134, 0.0164, 2.4127, 0.9494, -1.6657, 0.0667].
- **base cable loop**: a black cable loop / spool at the foot of the base, static in every side frame
  (`out/heat_base.jpg`, `out/blob_check.jpg`). Modelled as a fixed ellipsoid on the arm root (`blob_search.py`, even
  episodes, +0.017 mean IoU, hidden behind the base from the top camera); params `[x, y, z, rotvec, semi-axes]`:
  [0.0313, -0.0619, 0.0242, 0.0382, -0.9587, 0.3029, 0.0208, 0.0470, 0.0233].

## Cameras (single static pose each)

`static_check.py`: 0 px phase-correlation shift between all 40 episode medians for both cameras (only the exposure
changes between sessions), so one pose per camera, no per-session groups.

| camera | pos (m, world)          | quat (w,x,y,z)                 | fovy (deg) | k1      | k2      | cx      | cy      | latency (frames) |
| ------ | ----------------------- | ------------------------------ | ---------- | ------- | ------- | ------- | ------- | ---------------- |
| top    | -0.0237, 0.5736, 0.3944 | 0.0170, 0.0182, 0.5100, 0.8598 | 56.06      | -0.2011 | 0.0020  | -0.0089 | -0.0147 | -1.00            |
| side   | 0.3659, 0.5222, 0.1664  | 0.2133, 0.1656, 0.6699, 0.6916 | 46.91      | -0.2879 | -0.0092 | -0.0151 | -0.0272 | -0.50            |

World frame: table top z = 0, the shoulder-pan axis is the world z axis, the arm reaches +y at pan 0 (towards the top
camera), positive pan swings it towards +x (the box). MuJoCo camera convention (looks along -z, +y up). Lens:
x_d = x_u (1 + k1 r^2 + k2 r^4) (focal-length units), principal point offset (cx, cy) in fractions of (width, height),
applied as a warp of the padded pinhole render (masks and RGB).

Guard rails: |k1| = 0.20 (top) / 0.29 (side), both below 0.4, and principal-point shifts up to 2.7 % (below 10 %), so
no lens audit was required; the side value is a real barrel distortion (straight wall / door edges bow in the real side
frames, and the twin's table edge bows with them in `out/before_after.jpg`). Joint gains: 0.96-1.18 deg/unit are what
the normalised units explain; checked visually anyway at the minimum and maximum of every joint (`out/extremes.jpg`:
pan -54.7 / +74.3, lift -98 / +47, elbow -39 / +99, wrist 31 / 81 units). The twin follows the real arm at both ends of
every range: strict IoU top 0.81-0.87, side 0.74-0.90 (the 0.74 is the elbow minimum, frame 173, the arm reaching fast into the box: the twin gripper sits a few px low-left of the real one).

## Objects (RGB only; the metric never uses them)

`objects.py` (through the fitted cameras, from the real DINO+SAM brick / box masks): cardboard box = open box (floor +
4 walls, 4 mm), outer 0.255 x 0.311 m, 0.055 m high, shape fitted to the box masks of both views on the rest frame of
8 episodes (IoU 0.92), placed per episode (x, y, yaw; median per-episode IoU 0.93); red brick = 2x4 brick 63.7 x 31.7 x
19.2 mm with studs, per frame: on the table (151 frames, top-view mask centroid ray at half height, yaw from the
min-area box), held at the tool point (60), in the box (87), hidden when no view sees it (22). Table, back wall,
right wall and floor from background-median edges cast through the fitted top camera (`out/layout_px.json`);
per-camera per-channel colour gain.

## How it was built (rounds and scores, held-out strict medians top / side)

1. Download + `make_sets.py` (as specified); `sheet_real.py`, `static_check.py`, `ep_medians.py` (static cameras,
   exposure changes only); state ranges (no exact +-100 saturation).
2. Seed without SAM: `fg_masks.py` (dark, not brown), `seed.py` (random look-at cameras + CMA, 160x120; the side
   camera needed the any-azimuth search `--guess wide`).
3. SAM `_d1`, then `_d1k`; `jseed.py` (joint scales / offsets / gripper + both cameras, wrist-roll scan; both meshes):
   0.767 / 0.747 on `_d1k` (pinhole).
4. Fit r1 (`fit.py`, stages A-D: pose + fovy, + lens, latency grid + joints + all, polish; 10 % trimmed mean, 64 frames
   per camera, even episodes) on `_d1k`: SO-101 0.785 / 0.758, SO-100 0.778 / 0.748.
5. `bar_search.py` (+ wrist-camera bar): 0.813 / 0.780; r1b (stages C-D with the bar); `blob_search.py` (+ base
   cable loop): 0.824 / 0.803; r1c (stages C-D, both meshes) on `_d1k`.
6. Re-prompt and re-segment once: prompts from r1c (SO-101), SAM `_t1`, then `_t1o`, then `_t1ok`.
7. Fit r2 (stages B-D from r1c, both meshes, gravity sag offered) on `_t1ok`: SO-101 0.828 / 0.815, SO-100
   0.810 / 0.804.
8. r3: stage-D polish of r2 SO-101 with 2.5x the evaluations (per camera pose + fovy + lens): **0.826 / 0.833** (the
   side lens moved from k1 = -0.05 to -0.29; a variant started at k1 = -0.10, r3b, gave 0.826 / 0.813).
   `finalize.py` (candidates r2 SO-101, r3, r3b, r2 SO-100; rule = best mean of the EVEN-episode strict medians, so
   the held-out episodes never influenced the choice) wrote `out/final_fit.json` = r3 SO-101.
9. `score.py` (both readings, and against `_d1k`), `objects.py`, `meta_extra.py`, `render_set.py` (set +
   `out/before_after.jpg`), `extremes.py`.

Reproduce the last steps (repo root; MuJoCo venv `.venv`, SAM venv `neuroscope/molmoact2/.venv`):

```
.venv/bin/python so101_twin/redlego/fit.py redlego --mask-tag _t1ok --out-tag _r2_so101 --init-json \
    so101_twin/redlego/out/redlego_r1c_so101_fit.json --stages BCD --evals 0.8 --nfit 64 --obj-trim 0.1 --bar \
    --joint-params off,grip,gripoff,comp
.venv/bin/python so101_twin/redlego/fit.py redlego --mask-tag _t1ok --out-tag _r3_so101 --init-json \
    so101_twin/redlego/out/redlego_r2_so101_fit.json --stages D --evals 2.0 --nfit 64 --obj-trim 0.1
.venv/bin/python so101_twin/redlego/finalize.py --mask-tag _t1ok --cands out/redlego_r2_so101_fit.json,\
out/redlego_r3_so101_fit.json,out/redlego_r3b_so101_fit.json,out/redlego_r2_so100_fit.json
.venv/bin/python so101_twin/redlego/score.py --fit so101_twin/redlego/out/final_fit.json --mask-tag _t1ok \
    --occ-tag _d1_occ --out so101_twin/redlego/out/score_final.json
.venv/bin/python so101_twin/redlego/objects.py --fit so101_twin/redlego/out/final_fit.json \
    --layout so101_twin/redlego/out/layout_px.json
.venv/bin/python so101_twin/redlego/meta_extra.py
.venv/bin/python so101_twin/redlego/render_set.py --fit so101_twin/redlego/out/final_fit.json \
    --score so101_twin/redlego/out/score_final.json
```

## Limits

1. **Weakest frames**: the transport ("grasp" phase, half of the frames). The held brick covers the gripper towards
   the side camera (reading B removes it), the arm moves fast (motion blur; latency is fitted per camera, -1.0 / -0.5
   frames) and passes near the side image's right edge, where the lens model carries most weight. Side p10 0.765
   (strict, held-out). The side margin depended on the r3 polish (r2: 0.815): the side lens is weakly determined and
   CMA needed the larger budget to move k1 from -0.05 to -0.29.
2. **Twin-specific geometry**: the wrist-camera bar and the base cable loop are fitted primitives, not CAD; each adds
   about 0.015 mean IoU (round r1 without them: 0.785 / 0.758 held-out on `_d1k`). Other cables (servo leads, the USB
   lead along the arm) are not modelled and are removed from the masks by the 5x5 opening.
3. **Masks**: the scored masks are re-prompted from the twin (consensus with the twin-independent `_d1k`), so they carry
   some twin bias; against `_d1k` alone the final twin still passes (0.820 / 0.830 held-out).
4. **Units**: no joint saturates at exactly +-100 in `meta/stats.json`, so the normalised-units reading rests on the
   fitted scales (pan 1.18 deg/unit); the extremes check supports it.
5. **Lens / distance**: camera distance vs fovy is weakly constrained by arm silhouettes alone (the box gives an
   independent check: its fitted shape, 0.26 x 0.31 x 0.055 m, matches both views at IoU 0.92).
6. RGB: plain materials, one key light + headlight, no cables, per-channel gain; shadows are harder than the real ones.

## Files

| file                                                                                                         | what                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene_redlego.py`                                                                                           | scene description: arm spec, camera inits (from `out/seed_*.json`), joint priors, RGB objects / layout, `frame_setup`                                                   |
| `arm.py`, `fit.py`, `fullrate.py`                                                                            | toolkit copies (`fit.py`: + wrist-camera bar, base cable loop, `--bar`, fit-frame `mix`)                                                                                |
| `segment.py`, `sam_wait.sh`                                                                                  | SAM runs (dino / twin consensus modes); waiter for the shared SAM slot                                                                                                  |
| `fg_masks.py`, `seed.py`, `jseed.py`                                                                         | seed masks without SAM, camera seed search, joint + camera seed                                                                                                         |
| `clean_masks.py`, `strip_objects.py`, `open_masks.py`                                                        | mask post-processing (`_d1k`, `_t1o`, `_t1ok`)                                                                                                                          |
| `bar_search.py`, `blob_search.py`, `wscan.py`                                                                | global searches for the bar / cable loop; joint-offset scan                                                                                                             |
| `finalize.py`, `score.py`, `tables.py`, `meta_extra.py`                                                      | model choice (even episodes), scoring (both readings), README tables, meta fields                                                                                       |
| `objects.py`, `geom.py`, `render_set.py`                                                                     | object placement, camera rays, twin set writer + `out/before_after.jpg`                                                                                                 |
| `static_check.py`, `ep_medians.py`, `sheet_real.py`, `mask_sheet.py`, `overlay.py`, `heat.py`, `extremes.py` | diagnostics                                                                                                                                                             |
| `download.py`                                                                                                | dataset download (HF_HUB_DISABLE_XET=1, allow_patterns as specified)                                                                                                    |
| `masks/`                                                                                                     | packed mask sets: `_fg` (seed only), `_d1` / `_d1k` / `_d1_lego` / `_d1_box` / `_d1_occ`, `_t1` / `_t1o` / `_t1ok` (scored), `prompts_r1/`                              |
| `out/`                                                                                                       | fit jsons / logs (`redlego_r*_fit.*`), `final_fit.json`, `score_final*.json`, `model_comparison.json`, `objects.json`, `before_after.jpg`, `extremes.jpg`, check images |
