# so101_ballcup_twin2: pixel-aligned MuJoCo twin of so101_ballcup_real

Twin of `neuroscope/molmoact2/store/sets/so101_ballcup_real` (320 frames = 40 episodes x 8; LeRobot v2.1 dataset
`data_so101/ReubenLim__so101_ball_in_cup`, task "pick up the ball and place it in the cup"; `top` =
`observation.images.overhead`, `side` = `observation.images.front`). Output set:
`neuroscope/molmoact2/store/sets/so101_ballcup_twin2/` (existing twin layout). Everything for this scene is in
`so101_twin/ballcup/`; nothing in `so101_twin/common/`, `fitkit/`, `stack2/` or other scenes was edited (the toolkit
files were copied here and the copies edited, changes marked `BALLCUP` / `LEGOCUP2`).

## Result (pass = held-out median >= 0.80 on every camera)

**Pass is claimed on reading B (real-occluder-excluded) only: top 0.842, front 0.849 held-out median.** On the
strict reading A the top camera passes (0.838) and the front camera does not (0.798). Reading C is reference only
and is not used for any claim (it hides arm pixels behind the twin's own fitted cup / ball).

- **A strict arm-only**: the twin's arm rendered alone (no objects) vs the SAM arm mask, the reading the lemon /
  legocup fits used.
- **B real-occluder-excluded** (agreed with the parent and the other session): the same arm-only twin mask and the same
  SAM arm mask, with the pixels of REAL cup / ball occluder masks removed from BOTH. The occluder masks come from the
  real frames only (`masks/ballcup_<cam>_occ.npy`, `segment.py --mode occ`: Grounding DINO "a paper cup." /
  "a blue ball.", every box >= 0.3, SAM 2.1 box prompt, kept if bright (median max-channel >= 110) and < 20 % of the
  image; no twin input, no fitted parameter). They were computed after the fit and the fit was not changed afterwards.
- **C visible arm** (reference only): the twin's arm mask rendered with the twin's cup / ball in front of it.

| reading | camera / frames | n | median | p10 | mean |
|---|---|---|---|---|---|
| A strict arm-only | **top** held_out (odd episodes) | 128 | 0.838 | 0.785 | 0.834 |
| A strict arm-only | top all_frames | 320 | 0.842 | 0.781 | 0.833 |
| **B real-occluder-excluded** | **top** held_out (odd episodes) | 128 | 0.842 | 0.789 | 0.838 |
| B real-occluder-excluded | top all_frames | 320 | 0.846 | 0.786 | 0.836 |
| C visible arm (reference only) | top held_out | 128 | 0.838 | 0.785 | 0.835 |
| C visible arm (reference only) | top all_frames | 320 | 0.843 | 0.781 | 0.833 |
| check: C vs twin-independent masks _d1o | top held_out | 128 | 0.779 | 0.623 | 0.739 |
| A strict arm-only | **side** held_out (odd episodes) | 128 | 0.798 | 0.568 | 0.754 |
| A strict arm-only | side all_frames | 320 | 0.793 | 0.615 | 0.761 |
| **B real-occluder-excluded** | **side** held_out (odd episodes) | 128 | 0.849 | 0.792 | 0.833 |
| B real-occluder-excluded | side all_frames | 320 | 0.851 | 0.794 | 0.839 |
| C visible arm (reference only) | side held_out | 128 | 0.829 | 0.748 | 0.805 |
| C visible arm (reference only) | side all_frames | 320 | 0.829 | 0.759 | 0.813 |
| check: C vs twin-independent masks _d1o | side held_out | 128 | 0.791 | 0.686 | 0.779 |

Fraction of pixels excluded by reading B per frame, |occluder & (twin | real)| / |twin | real| (share of the strict
IoU union removed):

| camera | held-out mean | held-out median | all-frames mean | all-frames median | (of the twin arm mask: all-frames mean / median) |
|---|---|---|---|---|---|
| top | 0.0047 | 0.0000 | 0.0050 | 0.0000 | 0.0044 / 0.0000 |
| side | 0.0984 | 0.0442 | 0.0960 | 0.0484 | 0.1003 / 0.0507 |

Frames excluded because the real arm is not in view (SAM arm mask < 0.1 % of the image): top: 0 of 320 (0 held-out), side: 0 of 320 (0 held-out). No frame became
empty after the occluder exclusion.

Why the front camera needs B: the paper cup stands between the front camera and the arm base in about half of the
front frames, so a strict arm-only silhouette is charged for arm pixels SAM cannot see (the top camera is almost
unaffected: its excluded fraction is ~0).

## Metric (as agreed, unchanged)

Whole-arm silhouette IoU between the twin's rendered arm mask and a SAM 2.1 mask of the real arm, per camera, on the
REAL 640x480 pixel grid: the twin is rendered as a padded pinhole image and warped by the fitted lens onto the real
grid (`arm.Lens`; pixels the padded render cannot fill count as background, i.e. against the twin); the SAM mask is
never resampled. Fit only on EVEN episodes (24 episodes, 192 frames); held_out = ODD episodes (16 episodes, 128
frames); all_frames = all 320. `score.py` recomputes everything from the fit json (`out/score_final_*.json` carry
per-frame IoUs).

## Masks (SAM 2.1 hiera-small, one SAM process at a time, slot checked with pgrep before every run)

1. `_d1` twin-independent (`segment.py --mode dino`): Grounding DINO tiny boxes "a black robot arm." (every box >= 0.25
   and >= 0.5 x best), SAM box prompt, union; a second DINO pass "a blue ball. a paper cup." gives the ball / cup boxes
   and masks. Good arm masks, but they also take the black power plug and cable, the held ball and sometimes the cup.
   `fix_ball.py` drops "ball" masks that are dark (DINO boxing the black gripper as the ball while the real ball sits
   hidden in the cup: 84 front frames).
2. `_t1b` (`segment.py --mode twin`, prompts from the first fit): consensus prompting as `common/segment.py` with two
   silhouettes, the twin's (minus the real cup / ball masks) and the cleaned twin-independent mask `_d1c`; hard
   negatives on dark pixels 20-90 px outside (cable, plug) and on the ball / cup; candidate = best mean 4-px-tolerant
   IoU with both silhouettes + 0.25 x SAM score.
3. `_t2` = the same, re-prompted once from the better r2 twin (SO-100 mesh). `strip_objects.py` then removes the
   ball / cup mask pixels from the arm masks (a held ball SAM merged with the gripper is not the arm) -> **`_t2o`, the
   scored masks**. Sheets: `out/` check images; every mask set is packed bits in `masks/`.
   Twin-bias check: the final twin is also scored against `_d1o` (the purely twin-independent DINO+SAM masks with
   only the ball / cup removed; they still contain the plug / cable, so they understate every twin).

## Model and joint mapping

The dataset is LeRobot v2.1 OLD calibration in DEGREES (feature names `main_*`; shoulder_lift reaches 193 and elbow
179.5, impossible in the normalised -100..100 mode; no joint saturates at +-100 in the action stats), like the lemon
scene: q = deg2rad(sign * state + nominal + offset), no joint scales (fixed at 1). Gripper: 0 = closed (checked
against the images: the inverted map sticks the moving jaw out on closed frames).
Both meshes were fitted with the same masks / protocol; the SO-100 mesh (Menagerie `trs_so_arm100`, as the lemon
scene also preferred) scores higher on both cameras:

| model | top held-out median | side held-out median | fit json |
|---|---|---|---|
| so100 (r3_so100) | 0.834 | 0.824 | `so101_twin/ballcup/out/ballcup_r3_so100_fit.json` |
| so100 (r4_so100) | 0.834 | 0.820 | `so101_twin/ballcup/out/ballcup_r4_so100_fit.json` |
| so100 (r5_so100) | 0.838 | 0.829 | `so101_twin/ballcup/out/ballcup_r5_so100_fit.json` |
| so101 (r3_so101) | 0.820 | 0.805 | `so101_twin/ballcup/out/ballcup_r3_so101_fit.json` |
| so101 (r4_so101) | 0.820 | 0.804 | `so101_twin/ballcup/out/ballcup_r4_so101_fit.json` |

Chosen: see `meta.json` `arm_model`. Joint map of the chosen model:

| joint | sign | nominal (deg) | offset (deg) | scale |
|---|---|---|---|---|
| shoulder_pan | -1 | 0 | -1.34 | 1.000 (fixed) |
| shoulder_lift | -1 | 0 | -0.07 | 1.000 (fixed) |
| elbow_flex | +1 | 0 | -0.44 | 1.000 (fixed) |
| wrist_flex | +1 | 0 | -6.40 | 1.000 (fixed) |
| wrist_roll | -1 | 0 | -2.10 | 1.000 (fixed) |
| gripper | | | grip_scale 1.188 | |

## Cameras (single static pose each; `static_check.py`: top 0 px between episodes; front: only the cardboard box
backdrop shifts by <= 3 px between sessions, the table and the arm base do not)

| camera | pos (m, world) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |
|---|---|---|---|---|---|---|---|---|
| top | -0.0488, 0.1017, 0.5120 | 0.9838, 0.0817, 0.0481, 0.1523 | 67.81 | -0.1567 | 0.0047 | -0.0005 | -0.0282 | -1.07 |
| side | -0.1698, 0.5103, 0.2012 | -0.1392, -0.1112, 0.6073, 0.7742 | 40.33 | -0.0602 | 0.0509 | 0.0266 | 0.0121 | -1.87 |

World frame: table top z = 0, the shoulder-pan axis is the world z axis, the arm reaches +y at pan 0. Camera
convention MuJoCo (looks along -z, +y up). Lens: x_d = x_u (1 + k1 r^2 + k2 r^4) (focal-length units), principal
point offset (cx, cy) in fractions of (width, height), applied as a warp of the pinhole render (masks and RGB).
Guard rails: all |k1| < 0.4, principal-point shifts < 10 %, joint scales fixed at 1 (no drift), largest offset 6.4 deg
(wrist_flex), so no audit was required; the before/after figure and per-frame overlays were checked anyway.

## Objects (placed from detections in all episodes; cup shape fitted with the cameras on even episodes)

Cup: frustum (12 stacked cylinder slices), r_top 4.6 cm, r_bot 2.7 cm, height 12.1 cm (top-ray height fraction zc 0.91); per episode at the top-view cup-mask centroid ray (fitted top
camera) hitting z = zc * h. Ball: sphere r = 3.0 cm (ball-mask areas through the fitted cameras: front 2.9, top
3.1 cm; front IoU scan over 2.6-3.4 cm), per frame triangulated from the two views' ball-mask centroids (rays within
4 cm), else from the top view alone: in the cup if inside its rim, held at the gripper tool point if within 6 cm of
it, otherwise on the table; unseen balls are hidden. The cup shape and its
height fraction are fitted jointly with the cameras: the fit objective adds 0.5 x the cup IoU (twin cup vs SAM cup
masks, both views). This is a depth cue: from arm silhouettes alone the camera distance and fovy trade off (the first
fit put the front camera ~6 cm too close with fovy 47 deg, visible as a twin cup 30 % too wide).
RGB only: table, a dark mat around the base and a cardboard backdrop from `layout.py` (background-median edges cast
through the fitted cameras), per-camera per-channel colour gain.

## How it was built

1. `static_check.py` (static cameras), state ranges / `main_*` names (old calibration, degrees), gripper sense,
   `check.py` (twin outline on real frames).
2. Seed without SAM: `fg_masks.py` (dark + background-subtraction masks), `cam_search.py` (1200 random look-at
   cameras + CMA per camera, 160x120) -> IoU 0.49 (top) / 0.24 (front) against those rough masks.
3. SAM round `_d1` (twin-independent), `clean_masks.py` -> `_d1c`; fit r1 (`fit.py`, both meshes, even episodes,
   stages A-D: pose + fovy, + lens, latency grid + joint offsets + gripper scale, all, polish; 10 % trimmed mean).
4. `obj_pixels.py` (cup / ball centroids), cup term + dolly-zoom camera direction added to `fit.py`; SAM round `_t1b`
   from the r1 twin; fit r2 (both meshes) with the cup term (visible-arm masks).
5. SAM round `_t2` from the r2 SO-100 twin, `strip_objects.py` -> `_t2o`; ball radius 3.0 cm; fit r3 (both meshes,
   stages B-D from r2); r4 = stage-D polish with the cup weight lowered to 0.15 (no gain); r5 = stage-D polish of r3
   with the pure visible-arm objective (no cup term; the front arm IoU preferred a ~1 deg narrower fovy than the cup
   term allowed). `finalize.py`: within a mesh the round with the better EVEN-episode score (r5 for SO-100, r4 for
   SO-101), between meshes the better held-out score (SO-100) -> `out/final_fit.json` = r5 SO-100; `score.py`
   (visible / unoccluded / twin-independent masks). A top-only ball detection inside the cup rim is placed in the cup
   (fixes frames where the twin held a ball that was already released; scene construction, re-scored, no refit).
6. `layout.py`, `render_set.py` -> the set + `out/before_after.jpg`; `write_readme.py`.

Reproduce (repo root; MuJoCo venv `.venv`, SAM venv `neuroscope/molmoact2/.venv`):
```
.venv/bin/python so101_twin/ballcup/fit.py ballcup --mask-tag _t2o --split even --model so100 --out-tag _r3_so100 \\
    --evals 0.6 --nfit 56 --cup-w 0.5 --stages BCD --init-json so101_twin/ballcup/out/ballcup_r2_so100_fit.json
.venv/bin/python so101_twin/ballcup/fit.py ballcup --mask-tag _t2o --split even --model so100 --out-tag _r5_so100 \\
    --evals 1.2 --nfit 48 --occ --stages D --init-json so101_twin/ballcup/out/ballcup_r3_so100_fit.json
.venv/bin/python so101_twin/ballcup/finalize.py --mask-tag _t2o --cands out/ballcup_r3_so100_fit.json,\\
out/ballcup_r4_so100_fit.json,out/ballcup_r5_so100_fit.json,out/ballcup_r3_so101_fit.json,out/ballcup_r4_so101_fit.json
.venv/bin/python so101_twin/ballcup/score.py --fit so101_twin/ballcup/out/final_fit.json --mask-tag _t2o --occ
.venv/bin/python so101_twin/ballcup/layout.py --fit so101_twin/ballcup/out/final_fit.json
.venv/bin/python so101_twin/ballcup/render_set.py --fit so101_twin/ballcup/out/final_fit.json \\
    --score so101_twin/ballcup/out/score_final_occ.json --score-plain so101_twin/ballcup/out/score_final_plain.json \\
    --score-indep so101_twin/ballcup/out/score_final_indep.json
```

## Limits

1. **Front camera, strict reading A: 0.798 held-out, just under 0.80**; it passes only with the real occluder
   pixels excluded (B). B removes on average the front-view fraction given above (mostly the cup in front of the base
   and the held ball); where the real occluder masks are imperfect, B can also drop a few boundary pixels of the
   arm in both masks. The fit used the twin's own cup / ball (visible-arm objective and a cup IoU term on even
   episodes); B and A both score the arm-only twin mask, so neither depends on that object geometry.
2. Mesh: the SO-100 meshes beat the official SO-101 old_calib meshes by ~0.018 (top) / ~0.025 (front) held-out on
   the same masks, although the dataset is an SO-101 (the lemon scene showed the same). Neither has the servo cable
   loops or the perforated block under the pedestal; the SO-100 base sticks out to the right of the real pedestal in
   the front view (a constant blue rim in the overlays).
3. Masks: the scored masks were re-prompted from the twin (consensus with the twin-independent DINO+SAM mask), so they
   carry some twin bias. Against the purely twin-independent masks `_d1o` (which still contain the power plug / cable
   and some ball edges) the same twin scores top 0.779 / front 0.791 held-out.
4. Weakest frames: the held ball right in front of the front camera (transfer), where the ball (a plain 3.0 cm sphere
   at a triangulated position, biased when partly hidden) decides which gripper pixels are visible; and fast top
   frames with motion blur.
5. Lens: top k1 = -0.157 (barrel). The arm IoU prefers it (held-out top 0.810 with k1 = 0 and no refit), but the
   real mat edge is straight while the twin's bows by ~10 px at the image sides, so part of k1 may compensate model
   error. Front k1 = -0.06. All within the guard rails (|k1| < 0.4, principal point < 3 %, joint scales fixed at 1).
6. Camera distance vs fovy is weakly determined by the arm silhouettes alone; the cup term anchors it, but the cup is a
   plain frustum (front cup IoU ~0.66), so the front camera distance carries maybe +-2 cm of uncertainty.
7. Kinematics: largest offset wrist_flex -6.4 deg; camera latency -1.1 (top) / -1.9 (front) frames (the image shows
   the arm about one to two frames before the recorded state), interpolated from the 30 fps parquet.
8. RGB: plain materials and lighting, no cables or side clutter, per-channel gain; the backdrop box drifts <= 3 px
   between sessions in the real front view (static in the twin).

## Files

| file | what |
|---|---|
| `scene_ballcup.py` | scene description: arm spec, camera inits, objects (cup slices / ball, placement from the fitted cameras), RGB extras |
| `arm.py`, `fit.py`, `fullrate.py` | copies of the toolkit (`fit.py`: ballcup masks dir, cup term `--cup-w`, dolly-zoom camera parameter, visible-arm masks `--occ`, no old twin) |
| `segment.py`, `sam_wait.sh` | SAM runs (dino / twin consensus modes); polite waiter for the shared SAM slot |
| `fg_masks.py`, `cam_search.py` | seed masks without SAM and the camera seed search |
| `clean_masks.py`, `fix_ball.py`, `strip_objects.py`, `visible_prompts.py`, `export_prompts.py` | mask post-processing / prompts |
| `obj_pixels.py`, `geom.py`, `objects.py`, `layout.py` | object pixels, camera rays, first cup placement (superseded by the fitted cup), RGB layout |
| `score.py`, `score_occl.py`, `diag.py`, `check.py`, `sheet.py`, `static_check.py` | scoring (readings C / A, and B with `score_occl.py`, which also writes the meta.json key `held_out_real_occluder_excluded`) and diagnostics |
| `slot.sh`, `wait_log.sh` | shared-SAM-slot check / log wait kept in files so their command lines never match the slot pattern |
| `render_set.py`, `write_readme.py`, `readme_text.py` | twin set writer (+ before/after) and this README |
| `masks/` | all mask sets (packed bits), prompts. Superseded / debug: `_fg` (no-SAM seed), `_t1` (first consensus round, before `fix_ball.py`: lost the arm where a false 'ball' sat on the gripper), `_t0` + `prompts_test/` (3-frame smoke test), `_d1occ` (dry-run stand-in occluders from the dino run, not used for any reported number). Reading B's occluders: `ballcup_<cam>_occ.npy` (cup | ball), `_occ_cup.npy`, `_occ_ball.npy`, boxes in `ballcup_occ_occ.json` |
| `out/` | fit jsons / logs (`ballcup_r*_fit.*`), scores, `before_after.jpg`, `final_fit.json` |
