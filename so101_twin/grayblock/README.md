# so101_grayblock_twin2: pixel-aligned MuJoCo twin of so101_grayblock_real

This is the twin of `neuroscope/molmoact2/store/sets/so101_grayblock_real`: 320 frames, 40 episodes x 8. The source is the LeRobot v2.1 dataset `data_so101/Jeevesh2009__so101_gray_block_pickup_test` (Hub `Jeevesh2009/so101_gray_block_pickup_test`, robot `so101`, 86 episodes; episodes 0-43 downloaded, `make_sets.py` kept 40 of them: 15, 29, 31, 33 skipped). The task is "Pick the gray block and keep it in the box.": a black SO-101 on a white table against a white wall picks a small gray block and drops it into a blue cardboard box.

- Camera `top` = `observation.images.top`: static, in front of the arm, elevated (0.78 m in front, 0.47 m up); the base left of centre, the box on the right.
- Camera `side` = `observation.images.side`: static, elevated, from the arm's front-right (0.53 m, 0.50 m, 0.37 m up); a black chair fills the upper right of its background.
- The dataset has no wrist camera, and the arm carries no camera board.

Output set: `neuroscope/molmoact2/store/sets/so101_grayblock_twin2/`, in the existing twin layout (`images_top.npy`, `images_side.npy` uint8 [320, 480, 640, 3]; `state.npy` and `prompt.json` byte-identical to the real set; `labels.npz` = the real labels + real_or_sim 'sim', source 'grayblock_twin2', pair = arange(320); `meta.json`).

All code is in this folder. Nothing was edited in `so101_twin/common/`, `fitkit/`, other scene folders, `neuroscope/pi05` or `interp/`; the toolkit files were copied here from `so101_twin/jar/` (itself a copy chain from `common/`) and the copies edited (`GRAYBLOCK` marks). The MolmoAct2 policy was never run.

## Result (agreed metric; pass = held-out median >= 0.80 on every camera)

**PASS on both cameras on the STRICT reading** (arm-only twin mask vs the SAM arm mask, nothing excluded): **top 0.906, side 0.846** held-out median. The pass is claimed on the strict reading. The real-occluder-excluded reading is reported as agreed; nothing rests on it.

Final twin: `so101_twin/grayblock/out/grayblock_r3a_so101_fit.json` (copied to `out/final_fit.json`), model **so101** (official SO-101 mesh, old calibration). Scored masks: `masks/grayblock_<cam>_t2o.npy`. Per-frame IoUs: `out/score_final.json`.

| reading                | camera / frames                  | n   | median    | p10   | mean  |
| ---------------------- | -------------------------------- | --- | --------- | ----- | ----- |
| **strict arm-only**    | **top** held_out (odd episodes)  | 144 | **0.906** | 0.860 | 0.899 |
| **strict arm-only**    | top all_frames                   | 320 | 0.907     | 0.860 | 0.899 |
| real-occluder-excluded | top held_out (odd episodes)      | 144 | 0.906     | 0.860 | 0.899 |
| real-occluder-excluded | top all_frames                   | 320 | 0.907     | 0.860 | 0.899 |
| **strict arm-only**    | **side** held_out (odd episodes) | 144 | **0.846** | 0.763 | 0.834 |
| **strict arm-only**    | side all_frames                  | 320 | 0.847     | 0.764 | 0.836 |
| real-occluder-excluded | side held_out (odd episodes)     | 144 | 0.858     | 0.763 | 0.838 |
| real-occluder-excluded | side all_frames                  | 320 | 0.857     | 0.766 | 0.840 |

(fit split, even episodes, strict: top n 176, median 0.908, mean 0.900; side median 0.847, mean 0.838.)

- Frames excluded because the real arm is out of view (SAM arm mask < 0.1 % of the image): top 0 of 320 (0 held-out), side 0 of 320 (0 held-out). The arm is in view in every frame of both cameras; no frame became empty after the occluder exclusion.
- Real-occluder-excluded reading: the same arm-only twin mask and the same SAM arm mask, with the pixels of REAL occluder masks removed from BOTH (`masks/grayblock_<cam>_occx.npy`, from the real frames only, fixed constants, no twin input and no fitted parameter, never used by any fit objective): (a) the NEAR faces of the box (`occ_near.py`): connected components of the blue pixels of the real SAM box mask (Grounding DINO "a blue box." + SAM box prompt) under which the box silhouette ends, i.e. the outer faces of the walls that stand on the table (top view: the front wall; side view: the right-hand wall); the far walls' blue inner faces, the white floor and the side view's white near wall are not included; (b) the real gray-block masks (Grounding DINO "a gray block." + SAM, kept if small, not dark and grey: a held block can hide jaw pixels). Excluded fraction per frame = |occluder & (twin | real)| / |twin | real|:

| camera | held-out mean | held-out median | all-frames mean | all-frames median | of the twin mask (all-frames mean / median) |
| ------ | ------------- | --------------- | --------------- | ----------------- | ------------------------------------------- |
| top    | 0.0000        | 0.0000          | 0.0000          | 0.0000            | 0.0000 / 0.0000                             |
| side   | 0.0051        | 0.0022          | 0.0050          | 0.0016            | 0.0053 / 0.0015                             |

The box walls hardly ever stand in front of the arm here: in the top view the jaws stay above the front wall's rim (excluded fraction 0); in the side view the right-hand wall touches the jaw outline in some frames (0.5 % of the union on average). That is why the two readings are almost the same and why the strict reading carries the claim.

### The same final twin against the twin-INDEPENDENT masks

`_d1kc` = Grounding DINO "a black robot arm." boxes + SAM box prompts, union, minus the DINO+SAM block mask and the blue box pixels, 5x5 opening, main components, and in the side view minus the black chair's static region (no twin input at all). They drop the arm pixels in front of the chair by construction (DINO+SAM merged arm and chair whenever they touch), so they understate the side view:

| masks   | camera | held-out n | held-out median | p10   | mean  | all-frames median |
| ------- | ------ | ---------- | --------------- | ----- | ----- | ----------------- |
| `_d1kc` | top    | 144        | 0.905           | 0.860 | 0.898 | 0.905             |
| `_d1kc` | side   | 144        | 0.779           | 0.636 | 0.773 | 0.780             |

## Metric (as agreed, unchanged)

Whole-arm silhouette IoU, per camera, on the real 640x480 pixel grid. The twin's arm mask is every geom of the arm subtree (base included), rendered as a padded pinhole image and warped onto the real grid with the fitted lens (`arm.Lens`; pixels the padded render cannot fill count as background). It is compared with the SAM 2.1 mask of the real arm, which is never resampled. Frames whose real mask covers < 0.1 % of the image are excluded (none here). The fits only ever saw EVEN episodes (22 episodes, 176 frames); held_out = ODD episodes (18 episodes, 144 frames); all_frames = all 320. `score.py` recomputes everything from a fit json.

## Model and joint mapping

- Arm: **official TheRobotStudio SO-101, `so101_old_calib.xml`** via the `arm.py` copy (base bottom at z = 0 = table top, pan axis = world z, pan 0 reaches +y), no wrist camera. Colours: black body, jaws and servos (RGB only).
- **Units: DEGREES, LeRobot v2.1 OLD calibration** (feature names `main_*`; at the folded rest pose shoulder_lift reads up to 187.9 and elbow_flex 179.4, impossible in the normalised -100..100 mode; no joint saturates at exactly +-100 in `meta/episodes_stats.jsonl`; gripper 0..~50, 0 = closed). Like the lemon and ballcup scenes: `q_deg = sign * state + offset`, **joint scales fixed at 1** (degrees by construction, no gain fitted). The signs are the toolkit's old-calibration signs; verified by overlays (seed, r1: the arm reaches into the box on the right of the top view at negative pan, folds to the rest pose at lift ~187 / elbow ~177).

| joint         | sign | scale     | offset (deg) | gravity compliance (deg / N m) |
| ------------- | ---- | --------- | ------------ | ------------------------------ |
| shoulder_pan  | +1   | 1 (fixed) | +3.41        | 0 (not fitted)                 |
| shoulder_lift | -1   | 1 (fixed) | -8.27        | +2.12                          |
| elbow_flex    | +1   | 1 (fixed) | +3.82        | -1.33                          |
| wrist_flex    | +1   | 1 (fixed) | -16.92       | -0.08                          |
| wrist_roll    | -1   | 1 (fixed) | -3.41        | 0 (not fitted)                 |

- Gripper: `q = -0.17453 + grip_off + grip_scale * g/100 * 1.91986 rad`, grip_scale 1.073, grip_off +0.046 rad (open-jaw frames checked in the overlays).
- The offsets are the zero of this robot's old calibration relative to the model. The same map came out of the joint seed, every round and both meshes (wrist_flex -16.5 to -17.3, lift -7.3 to -9.8, elbow +3.0 to +4.6), so it is not a degeneracy of one fit; the pan offset is degenerate with a rotation of both cameras about the pan axis (only its prior holds it). Gravity compliance (`q += c * tau_gravity(q)`, as `jar/arm.py`) was added in r3 with a weak prior (sd 10); it stays small and changed the score by < 0.005.
- Rest pose: 109 of the 320 frames are at the folded rest pose (lift > 180); the fit frames are sampled uniformly at random from the even episodes (80 per camera), so they match the score's frame mix.

Mesh / round comparison (all scored on the final masks `_t2o` with the same protocol). The final fit is chosen by the EVEN-episode mean IoU, min over cameras; held-out shown only for information:

| fit                                                         | model | fitted on | k1 top / side   | even mean top / side | even median top / side | held-out median top / side | held-out p10 top / side |
| ----------------------------------------------------------- | ----- | --------- | --------------- | -------------------- | ---------------------- | -------------------------- | ----------------------- |
| `so101_twin/grayblock/out/grayblock_r2_so101_fit.json`      | so101 | `_t1o`    | -0.021 / +0.013 | 0.902 / 0.831        | 0.909 / 0.838          | 0.906 / 0.836              | 0.868 / 0.771           |
| `so101_twin/grayblock/out/grayblock_r2_so100_fit.json`      | so100 | `_t1o`    | -0.059 / -0.226 | 0.860 / 0.803        | 0.863 / 0.804          | 0.863 / 0.803              | 0.818 / 0.734           |
| **`so101_twin/grayblock/out/grayblock_r3a_so101_fit.json`** | so101 | `_t2o`    | -0.020 / -0.015 | 0.899 / 0.838        | 0.908 / 0.847          | 0.906 / 0.846              | 0.860 / 0.763           |
| `so101_twin/grayblock/out/grayblock_r3b_so101_fit.json`     | so101 | `_t2o`    | -0.041 / -0.012 | 0.898 / 0.834        | 0.901 / 0.839          | 0.902 / 0.840              | 0.873 / 0.772           |

The SO-100 (Menagerie) mesh trailed by 0.04 (top) and 0.03 (side) in r1 and r2 with the same masks and protocol, so r3 was run for the SO-101 mesh only.

## Cameras (world = arm frame) and lens terms

| camera | dataset feature         | pos (m)                 | quat (w,x,y,z)                  | fovy (deg) | k1      | k2      | cx      | cy      | latency (frames) |
| ------ | ----------------------- | ----------------------- | ------------------------------- | ---------- | ------- | ------- | ------- | ------- | ---------------- |
| top    | observation.images.top  | -0.0569, 0.7805, 0.4745 | 0.0078, -0.0264, 0.5299, 0.8476 | 35.85      | -0.0196 | -0.0292 | +0.0031 | -0.0037 | -0.71            |
| side   | observation.images.side | 0.5262, 0.4954, 0.3682  | 0.3381, 0.2315, 0.5070, 0.7584  | 54.78      | -0.0152 | +0.0346 | +0.0415 | +0.0082 | -1.00            |

- MuJoCo camera convention: looks along -z, +y up. Lens: x_d = x_u (1 + k1 r^2 + k2 r^4) (focal-length units), principal point offset (cx, cy) in fractions of (width, height), applied as a warp of the pinhole render (masks and RGB).
- **Guard rails: |k1| = top 0.020, side 0.015 (limit 0.4); principal-point shifts <= 4.2 % (side cx; limit 10 %). Joint gains: none fitted (degrees, scales fixed at 1); the only gain is the gripper's grip_scale 1.07 on its 0..100 map (checked on open-jaw overlays).** Within the rails, so no audit was required; the overlays (`out/before_after.jpg`, `out/ov_*.jpg`) were checked by eye anyway.
- Both cameras are static (`static_check.py`: 0 px phase-correlation shift between all 40 episode medians), so one pose per camera. The side camera's exposure drops from episode 23 on (and the box moves a little then); the camera does not. Latency (image time minus state time) from the full-rate 30 fps parquet (`fullrate.py`): a grid then CMA; the images show the arm 0.7 (top) and 1.0 (side) frames earlier than the recorded state.

## Scene (RGB only; the metric never uses it)

- White table (z = 0) and the white wall behind the arm: its table edge is read off the top background median (first dark -> bright transition per column, away from the arm and the box), ray-cast through the fitted top camera onto z = 0, RANSAC line (y = -0.060 x - 0.081 m); the wall ends at x = -0.48 m so the chair beyond the top view's right edge stays visible to the side camera. The side view's own table-wall edge, ray-cast the same way, gives y = 0.108 x - 0.084 (the same line near the base, ~10 deg apart: the side camera sees that edge at a grazing angle); only the top-view line is used.
- Blue box (blue outside, white inside), open top, 18.8 x 20.2 x 7.9 cm, placed per episode by ray-casting through the fitted top camera (`objects.py`): the front wall's lowest pixels (near blue face, majority of the episode's 8 frames) onto z = 0 give its bottom edge (yaw, width), its top pixels onto that vertical plane give the height, the far rim of the box mask at that height gives the depth; sizes = median over the 40 episodes (check: projected hull vs the SAM box masks, median IoU 0.74 over both views).
- Gray block (3 cm cube) moved by phase, read off the full-rate gripper signal (this dataset's gripper starts closed, opens around the block, holds it at ~15 and opens again over the box): before the hold on the table (138 frames), during it in the jaws at the tool point (68), after it on the box floor (114). Positions from the block-mask centroids (triangulated from both views when the rays meet within 3 cm, else the top-view ray at half its height), else the episode's median, else the box centre.
- The black chair seen only by the side camera: a dark slab facing the side camera, spanning the chair's pixel box, pushed back along the rays until it leaves the top view.
- Per camera and per session (episodes <= 22 / >= 23, the side camera's exposure change) per-channel colour gain (real / sim mean colour over 12 frames).

## Masks (SAM 2.1 hiera-small; one segmentation process at a time, the shared slot checked before every run by `sam_wait.sh`)

1. `_fg` (no SAM, `fg_masks.py`): dark pixels (max channel < 60), 5x5 opening (drops the power cable), the chair's static region removed in the side view. Only used to seed the cameras (`seed.py`, `jseed.py`).
2. `_d1` twin-independent (`segment.py --mode dino`): Grounding DINO tiny boxes "a black robot arm." (every box >= 0.25 and >= 0.5 x best), SAM box prompt, union; a second DINO pass "a blue box. a gray block." gives the box / block masks. `clean_masks.py` -> `_d1k` (minus the block mask unless > 2 % of the image, minus the blue box pixels, 5x5 opening, main components). In 41 side frames DINO+SAM merged the arm with the black chair, so `chair_cut.py` -> `_d1kc` removes the chair's static region (pixels dark in the side background median, right of x = 400) from the side masks; `_d1kc` is the consensus silhouette of the twin-prompted rounds and the twin-independent check.
3. `_occ` (`segment.py --mode occ`, real frames only): every DINO box >= 0.3 for "a gray block." (kept if 30 px .. 2 % of the image, not dark, grey) and "a blue box." (its blue pixels). The block masks are stripped from the arm masks and give SAM negatives; `occ_near.py` / `occ_union.py` build the occluder masks `_occn` / `_occx` described above.
4. `_t1` (`segment.py --mode twin`, prompts = the r1 SO-101 twin, `masks/prompts_r1`): consensus prompting as `common/segment.py` with two silhouettes, the twin's and `_d1kc` (dropped for a frame when it disagrees grossly with the twin: IoU < 0.4 or > 1.8 x its area; 1 side frame); box from their union dilated 12 px, positives from their intersection, negatives far from the union plus hard negatives on dark pixels 20-90 px outside (the chair, the cable) and on the real block / blue box masks; 3 prompt variants x 3 multimask outputs; kept = best mean 4-px-tolerant IoU with the silhouettes + 0.25 x SAM score. `strip_objects.py` removes the real block pixels -> `_t1o`.
5. `_t2` = the same, re-prompted ONCE from the official r2 SO-101 fit (`masks/prompts_r2`), block stripped -> **`_t2o`, the scored masks** (median prompt agreement top 0.974, side 0.931).

## How it was built (about 1 h 30 min)

1. Download (meta, data, the two cameras' videos of episodes 0-43), `make_sets.py` -> `so101_grayblock_real` (40 episodes; 22 even / 18 odd). `static_check.py` (both cameras static), units (old calibration, degrees).
2. `fg_masks.py` (no SAM) -> `seed.py` per camera (3000 random look-at cameras on a shell around the workspace, any azimuth, + CMA on the best 4; 160x120, joint map at the init) -> top 0.68 / side 0.60; `jseed.py` (CMA over the 5 joint offsets, gripper scale and both cameras' pose + fovy, 2500 evals) -> top 0.87 / side 0.68 against the rough masks.
3. SAM round `_d1`, `clean_masks.py` -> `_d1k`; r1 (`fit.py` stages A-D on `_d1k`, both meshes, 64 random fit frames per camera, 10 % trimmed mean): pose + fovy; + lens; latency grid, then CMA on joint offsets, gripper scale / offset and latencies; all together with the poses; polish. `chair_cut.py` -> `_d1kc`; SAM round `_t1` from the r1 SO-101 twin.
4. r2 (both meshes, `_t1o`, 80 random fit frames per camera, stages B-D from r1). SAM round `_t2` from the r2 SO-101 twin (the official fit), block stripped -> `_t2o`.
5. r3 (SO-101 only), `_t2o`, stages B-D from r2 plus gravity compliance: r3a with 80 random fit frames per camera, r3b with 96 frames, half farthest-point in joint space and half random (`--fit-frames mix`).
6. `final.sh`: `finalize.py` picks the final fit among r2 (both meshes), r3a, r3b by the even-episode mean IoU (min over cameras) on `_t2o` -> r3a; `score.py` (strict + occluder-excluded on `_t2o`; check on `_d1kc`); `make_meta_extra.py`; `objects.py`; `render_set.py` -> the set + `out/before_after.jpg` (8 odd-episode frames at different phases: real | twin | IoU overlay per camera; white = both, red = real arm only, blue = twin arm only).

Reproduce (repo root; MuJoCo venv `.venv`, SAM venv `neuroscope/molmoact2/.venv` via `sam_wait.sh`):

```
.venv/bin/python so101_twin/grayblock/static_check.py && .venv/bin/python so101_twin/grayblock/fg_masks.py   # also writes out/bg_median_<cam>.npy
zsh so101_twin/grayblock/sam_wait.sh d1 occ
.venv/bin/python so101_twin/grayblock/clean_masks.py --tag _d1 && .venv/bin/python so101_twin/grayblock/chair_cut.py
.venv/bin/python so101_twin/grayblock/occ_near.py && .venv/bin/python so101_twin/grayblock/occ_union.py
.venv/bin/python so101_twin/grayblock/seed.py top --n 3000 && .venv/bin/python so101_twin/grayblock/seed.py side --n 3000
.venv/bin/python so101_twin/grayblock/jseed.py --mask-tag _fg --evals 2500 --frames 24 --out so101_twin/grayblock/out/jseed_fg.json
F='--split even --obj-trim 0.1 --joint-params off,grip,gripoff'   # bash word-splits $F (zsh: use ${=F})
.venv/bin/python so101_twin/grayblock/fit.py grayblock $F --mask-tag _d1k --evals 0.6 --nfit 64 --fit-frames random --model so101 \
    --out-tag _r1_so101 --stages ABCD --init-json so101_twin/grayblock/out/jseed_fg.json --export-prompts prompts_r1
zsh so101_twin/grayblock/sam_wait.sh t1 && .venv/bin/python so101_twin/grayblock/strip_objects.py --tag _t1
.venv/bin/python so101_twin/grayblock/fit.py grayblock $F --mask-tag _t1o --evals 0.7 --nfit 80 --fit-frames random --model so101 \
    --out-tag _r2_so101 --stages BCD --init-json so101_twin/grayblock/out/grayblock_r1_so101_fit.json --export-prompts prompts_r2
zsh so101_twin/grayblock/sam_wait.sh t2 && .venv/bin/python so101_twin/grayblock/strip_objects.py --tag _t2
.venv/bin/python so101_twin/grayblock/fit.py grayblock --split even --obj-trim 0.1 --joint-params off,grip,gripoff,comp \
    --mask-tag _t2o --evals 0.7 --nfit 80 --fit-frames random --model so101 --out-tag _r3a_so101 --stages BCD \
    --init-json so101_twin/grayblock/out/grayblock_r2_so101_fit.json
zsh so101_twin/grayblock/final.sh
```

(The SO-100 runs are the same with `--model so100` and their own out-tags; r3b adds `--nfit 96 --fit-frames mix`.)

## Limits

1. Side view, black on black: whenever the gripper is in front of the black chair (182 side frames have arm-mask pixels over the chair's region), SAM loses part of the black gripper or takes a bit of the chair; median side IoU there 0.815 vs 0.884 elsewhere, and these frames set the side p10 (0.763; 77 side frames below 0.80, none in the top view). The twin's gripper there agrees with the top view; no manual mask edits were made.
2. Side view, pose dependence: rest pose 0.888 vs moving frames 0.821 (top 0.914 / 0.892). In the moving frames the twin's upper arm / forearm sit slightly low in the side view; gravity compliance did not remove it, so an unmodelled geometric difference (link lengths, base tilt) or the old calibration's per-joint zero is the likely cause.
3. Mask bias: the scored masks `_t2o` were prompted by the twin (in consensus with the twin-independent `_d1kc`). Against `_d1kc` alone the final twin scores top 0.905 / side 0.779 held-out (`_d1kc` drops the arm in front of the chair by construction, so it is not a better reference for the side view, only an independent one).
4. In some frames SAM leaves out the base (its red parts) in the side view, or keeps a held block that the block detector missed (red blob at the jaw tips); the red / white servo cable loops are not modelled (SAM keeps part of them).
5. Occluder-excluded reading: tiny excluded fractions (top 0, side 0.5 % on average); the near-face rule leaves out the side view's white near wall, and it also removes twin-overshoot pixels on a near face the arm is in front of, which is why no claim rests on it.
6. Joint map: degrees with offsets only; wrist_flex -16.9 deg is large but stable across meshes and rounds (this robot's old-calibration zero); the pan offset is degenerate with a common rotation of both cameras about the pan axis.
7. Motion blur in fast frames; latency is fitted per camera (-0.7 / -1.0 frames), not per frame.
8. RGB: plain materials, the box's inner faces all white (the real one is blue inside on one pair of walls), the chair as a slab, the wall as one plane (the side view's table-wall edge is ~10 deg off), no cables or shadows matched; the colour gain is per camera and session.

## Files

| file                                                                             | what                                                                                                                                          |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene_grayblock.py`                                                             | scene: arm spec (old calibration, degrees), camera inits, priors, RGB objects (`extras`, `frame_setup`)                                       |
| `arm.py`, `fit.py`, `fullrate.py`                                                | copies of the toolkit (via `jar/`): gravity compliance, random / mixed fit frames, `--fix-radial`, scene flags                                |
| `segment.py`, `sam_wait.sh`                                                      | SAM / DINO runs (dino, twin consensus with the disagreement guard, occ modes) and the shared-slot waiter                                      |
| `clean_masks.py`, `chair_cut.py`, `strip_objects.py`, `fg_masks.py`              | mask post-processing (`fg_masks.py`: the no-SAM seed masks)                                                                                   |
| `occ_near.py`, `occ_union.py`                                                    | real occluder masks (near blue box faces, union with the block) for the occluder-excluded reading                                             |
| `seed.py`, `jseed.py`                                                            | twin-free camera seed, joint seed (offsets + gripper + cameras; degrees, no scales)                                                           |
| `score.py`, `finalize.py`, `make_meta_extra.py`, `final.sh`, `readme_numbers.py` | metric (strict + occluder-excluded), final choice, meta extras, the final pipeline, README numbers                                            |
| `overlay.py`, `mask_sheet.py`, `static_check.py`, `sheet_real.py`, `try_rgb.py`  | diagnostics (IoU overlays, mask sheets, static cameras, real-frame sheets, RGB checks)                                                        |
| `objects.py`, `geom.py`, `render_set.py`                                         | object placement, camera rays, the twin set writer                                                                                            |
| `masks/`                                                                         | packed masks `grayblock_<cam>_<tag>.npy` (`np.unpackbits(a, -1)[..., :640]`), twin prompts `prompts_r1`, `prompts_r2` (+ `_so100`)            |
| `out/`                                                                           | seeds, fit jsons / logs (`grayblock_r*_<model>_fit.*`), scores, `final_fit.json`, `model_comparison.json`, `objects.json`, `before_after.jpg` |
