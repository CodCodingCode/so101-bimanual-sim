# so101_stack_twin2: refit of the SO-101 "stack" digital twin (does NOT pass)

This is the twin of `so101_stack_real`: 320 frames, 40 episodes x 8, from the LeRobot v3.0 dataset
`ichbinblau/so101_stack2cubes_dataset` (`so101_follower`, new calibration). It is rebuilt on the official SO-101 model
with refitted cameras, lens, camera latency and joint map. The set is at
`neuroscope/molmoact2/store/sets/so101_stack_twin2/` and uses the same layout as `so101_stack_twin`.

| camera | held-out (odd episodes) n / median / p10 / mean | all frames n / median / p10 / mean | old twin held-out median |
| ------ | ----------------------------------------------- | ---------------------------------- | ------------------------ |
| top    | 176 / **0.793** / 0.666 / 0.772                 | 320 / 0.785 / 0.662 / 0.772        | 0.299                    |
| side   | 158 / **0.513** / 0.023 / 0.451                 | 280 / 0.522 / 0.090 / 0.456        | 0.415                    |

The pass mark (held-out median >= 0.80 on every camera) is **not met**. The top camera misses it by 0.007. The side
camera is far below it. The section "What limits it" explains why.

## Metric (unchanged)

- What is scored: whole-arm silhouette IoU between the twin's arm mask and a SAM 2.1 mask of the real arm, per camera.
- The twin mask is a padded pinhole render, lens-warped onto the real 640x480 pixel grid (`arm.Lens`). Pixels the
  warp cannot fill count as background. The real masks are never resampled.
- Fitting uses EVEN episodes only. The score is on ODD episodes.
- Frames are excluded when the real arm mask covers less than 0.1 % of the image (arm not in view):
  - top: 0 frames excluded.
  - side: 40 of 320 frames excluded (18 of the 176 held-out frames). In these frames only the wrist-camera cable or
    nothing of the arm is in view. A background-subtraction check found moving pixels in 27 of them, all of them cable.
- Masks used: `so101_twin/masks/stack_<cam>_s5.npy`, packed bits (`np.unpackbits(a, -1)[..., :640].astype(bool)`).
  - top masks = round s4p: prompted part by part from the r2 twin.
  - side masks = round s3: prompted from the real image, without the twin.
- To choose between candidate fits, compare scores on EVEN frames only (`score.py`).

## Model and joint map

- Arm model: official TheRobotStudio `so101_new_calib.xml` plus the official wrist-camera mount
  (`so101_new_calib_camera.xml`), because the real arm carries a wrist camera. White body, black servos.
- Placement: base at the builder's pan axis (0, 0.0452, 0), reaching +y at pan = 0.
- Joint map: `q_rad[j] = deg2rad(scale[j] * state[j] + offset[j])`. All signs are +1 (official new_calib convention).
  - offset (deg) = [-0.41, -7.78, -0.99, +9.38, -0.45] for pan, lift, elbow, wrist_flex, wrist_roll.
  - scale = [0.994, 1.001, 0.996, 0.997, 0.990]. These are within 1 % of 1, so the guard rail is not triggered.
  - gripper: `q = -0.1745 + 1.121 * g/100 * (1.7453 + 0.1745)`.
- Units: the dataset is in LeRobot's normalised -100..100 units, not degrees. The action stats hit exactly -100 and
  +100, so `scale` is degrees per unit.
- How the map was checked:
  - The identity map with zero offsets is inconsistent with the data. It puts the held cube 1-3 cm below the table at
    the grasp, and no camera above the table explains the cube's image track.
  - The seed map came from the held-cube bundle adjustment (step 2 below). The silhouette fits then refined it.
  - Wrist-roll candidates 0 / 90 / 180 / -90 were tried; 0 wins.
- Camera latency (image time minus state time): top -1.28 frames, side -0.10 frames. The pose is interpolated from the
  full-rate parquet.

## Cameras

| camera | group       | pos (m, world frame)    | quat (w, x, y, z)                |
| ------ | ----------- | ----------------------- | -------------------------------- |
| top    | g0_e000-006 | 0.5366, 0.2471, 0.2175  | -0.1920, 0.5239, -0.0992, 0.8239 |
| top    | g1_e039-099 | 0.5101, 0.2485, 0.2144  | -0.2136, 0.5278, -0.1161, 0.8138 |
| top    | g2_e105-126 | 0.5160, 0.2557, 0.1882  | -0.1951, 0.5405, -0.0895, 0.8135 |
| top    | g3_e131-198 | 0.4866, 0.2367, 0.2398  | -0.2051, 0.4646, -0.1179, 0.8533 |
| side   | all         | 0.2662, 0.1267, -0.0178 | 0.6186, 0.6422, 0.3322, 0.3075   |

| camera | fovy (deg) | k1      | k2     | cx      | cy      |
| ------ | ---------- | ------- | ------ | ------- | ------- |
| top    | 38.99      | -0.0112 | 0.0041 | 0.0007  | -0.0073 |
| side   | 57.31      | -0.0343 | 0.0796 | -0.0116 | 0.0017  |

- The top camera has one pose per session and shares fovy and lens across sessions.
- The side camera has a single pose.
- All lens terms are small (|k1| < 0.4, |cx| and |cy| < 0.1), so the lens guard rail is not triggered.
- The top camera is not overhead. It is about 0.5 m out on the +x side and about 0.2 m up, and looks back at the
  arm base. The arm's base and shoulder fill the top-right corner of its image.
- The side camera sits at table height, about 13 cm from the green block and about 25 cm from the disc.

## How it was built

1. **Seed without any arm mask** (`cube_track.py`, `fullrate_cube.py`, `top_seed.py`, `bundle3.py`, `side_geo.py`).
   - The 4 cm wooden cube is the only wood-coloured object. While the gripper holds it, its image centroid is the
     projection of the gripper's tool-centre point.
   - Decoding the source videos at full rate gives about 1900 held-cube frames per camera from the even episodes. The
     script checks that decoded frames match the set exactly.
   - These frames give forward-kinematics-to-image correspondences. A bundle adjustment over them fits the top
     sessions, the joint offsets / scales and the latency. Top median reprojection: 5.6 px.
   - The side camera is seeded from where the resting cube appears on the disc and on the block, and how large it
     appears.
2. **Top camera sessions.**
   - The sessions come from phase correlation of the static background between episode medians: the table seam, disc
     and block. The arm is never used.
   - Shifts: episodes 0-6 at (0, 0) px, 39-99 at (0, -3), 105-126 at (+2, -1), 131-198 at (+45, +1).
   - Every session contains even (fit) episodes.
3. **SAM rounds.** Only one SAM process ran at a time; `pgrep -fl "[Pp]ython.*segment"` was checked before each start.
   - s1: whole-arm prompt from the seed twin, by consensus with background-subtraction masks. These masks were partial
     on the white-on-white top view (gripper only).
   - s2 / s4p: `segment_parts.py` prompts the whole arm and each part separately (base+shoulder, upper arm, forearm,
     wrist+gripper). A part's mask is kept only if it agrees with the part and stays inside the dilated twin
     silhouette. This gave full top masks, with a prompt-agreement median of 0.83.
   - s3 (side): `side_fgw_prompts.py` prompts from the real image itself: moving white pixels entering from the top or
     left edge. Twin-prompted side masks had followed the misplaced twin into the curtain.
4. **Fits.**
   - Fits used `common/fit.py` and the copy `stack2/fit.py`, with stages A, B, C (roll, latency, offsets + scales +
     gripper + latency, then everything) and D. The scene file is `common/scenes/stack.py`.
   - Fit sequence:
     - r1 on s1 masks.
     - r2 on s3 masks, with evals 0.7.
     - Three candidates on the final s5 masks: r2; r3relaxed (stages CD, joint-gain prior sd 0.12); r3b (stages BCD,
       default priors).
   - Selection: r3b, which has the best even-frame mean IoU (top 0.771, side 0.463).
5. **Render** (`render_twin2.py`).
   - Arm: official SO-101 with the wrist mount. The fitted lens is applied to the RGB exactly as to the masks.
   - Backdrops: shown only in the side view, as the builder did.
   - Colour gain: per channel, set from the real-to-sim mean colour over 16 frames.

## Objects

- Kept from the builder unchanged: object models, sizes, colours and the cube timeline (disc / jaws / block, by the
  gripper rule).
- **Changed: the object xy.**
  - The builder's xy disagree with the refit arm by 10-17 cm. The gripper would close on empty air, because the
    builder placed the objects by ray-casting through its own wrong cameras.
  - The disc and resting cube are moved to the twin's tool-centre point at the grasp. The green block is moved to it at
    the release. Both are medians over the 18 even episodes.
  - The table plane is set from the grasp height and kept 1.5 cm below the side camera.
- `render_twin2.py --builder-objects` restores the builder's xy.
- The placement is recorded in `meta.json` under `objects.placement`.
- Object appearance is still approximate. The builder's green block (5.2 cm) is larger than the real one (about 4 cm),
  and the disc is smaller than the real one. In the side view the block therefore renders much too large and too far
  left, because the side camera is close to it.

## What limits it

- **The joint map is wrong when the arm is extended.** Held-out median by pose:

  | camera | folded rest pose (n=90) | moving near the disc | moving near the block |
  | ------ | ----------------------- | -------------------- | --------------------- |
  | top    | 0.841                   | 0.702                | 0.718                 |
  | side   | 0.684                   | 0.289                | 0.144                 |
  - In the side view the real gripper near the block appears about 150 px to the right of the twin and about 1.5x
    larger. The real tool-centre point there is about 4 cm closer to the side camera than the twin's.
  - The top camera looks along nearly the same direction, so it hardly sees this error.
  - The dataset is in normalised -100..100 units, so each joint's zero and degrees-per-unit come from the user's
    calibration. They are not known.
  - The fits could not recover them:
    - Silhouette fits keep the gains within about 1-3 % of 1, even with a loose prior (r3relaxed).
    - The held-cube bundle with free gains (`bundle4.py`) always drifts to a telephoto side camera at its fovy bound
      (25-40 deg) with gains at their bounds. This is not physical: the cube is 2.5x larger on the block than on the
      disc, so the side camera is close.
  - So the extended-arm geometry of the real arm does not match the official SO-101 kinematics under any joint map
    tried. Likely causes are a different gripper or wrist-camera bracket, or a different base mounting (the fitted side camera
    sits 1.8 cm below the arm-base plane, so the table there must be lower than the base). It may also be that the side cube centroids are biased
    by the jaws occluding the cube.

- **The side camera sees almost nothing of the arm.** The arm takes about 4 % of the image (median) and is only the
  gripper, close-up. A 1 cm error there is about 50 px. Masks of the jaws alone are thin, so the IoU is unforgiving.
- **Top:** at rest it passes (0.84). The moving poses (0.70-0.72) pull the median down to 0.793. Session g3 (episodes
  131-198, n=88) is the weakest at 0.763; the others are 0.79-0.82.
- **Not modelled:** cables (red/black servo wires, the black wrist-camera USB cable) and the electronics under the
  base. SAM masks sometimes include short cable stubs.

## Files

| file                                                         | what                                                                                                                                                       |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cube_track.py`, `fullrate_cube.py`                          | wooden-cube track, per set frame and at full rate (even episodes, + parquet states)                                                                        |
| `bundle.py`, `seed_pnp.py`, `bundle2.py`                     | PnP helpers; first PnP attempts / joint-map scan (kept for the record)                                                                                     |
| `top_seed.py`, `bundle3.py`, `side_geo.py`                   | seed: top sessions (shared fovy), bundle with joint map, side from the resting cube                                                                        |
| `bundle4.py`                                                 | side + free joint gains from the held cube (diagnostic: converges to a non-physical telephoto side camera)                                                 |
| `cam_search.py`                                              | wide random + CMA search of one camera against masks                                                                                                       |
| `fg_masks.py`                                                | background-subtraction masks (seed / prompt consensus only, never scored)                                                                                  |
| `export_prompts.py`, `export_parts.py`                       | twin silhouettes / per-part label images for SAM prompting                                                                                                 |
| `segment_parts.py`                                           | SAM 2.1 prompted part by part (top masks)                                                                                                                  |
| `side_fgw_prompts.py`                                        | side prompts from the real image (moving white pixels from the top / left edge)                                                                            |
| `mask_stats.py`, `check_init.py`, `score.py`, `breakdown.py` | mask sanity, overlays, even / held-out scoring, per-pose breakdown                                                                                         |
| `render_twin2.py`                                            | writes the twin set + `out/before_after.jpg`                                                                                                               |
| `fit.py`, `arm.py`, `fullrate.py`, `segment.py`              | copies of `so101_twin/common/` (scene from `common/scenes/stack.py`, masks from `stack2/masks`; `--scale-sigma`; per-camera geom-group visibility for RGB) |
| `out/stack_r3b_fit.json`                                     | the chosen fit (params + stats); r1 / r2 fits are in `so101_twin/common/out/stack_r{1,2}_fit.*`                                                            |
| `out/before_after.jpg`                                       | 8 held-out frames, per camera: real, old twin, new twin, then SAM vs twin (white = both, red = real only, blue = twin only)                                |

To reproduce from the masks: run `.venv/bin/python so101_twin/stack2/fit.py stack --mask-tag _s5 --out-tag _r3b
--stages BCD --evals 0.5 --init-json so101_twin/common/out/stack_r2_fit.json`, then
`.venv/bin/python so101_twin/stack2/render_twin2.py --fit-json so101_twin/stack2/out/stack_r3b_fit.json --mask-tag _s5`.
