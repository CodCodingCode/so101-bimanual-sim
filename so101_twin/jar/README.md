# so101_jar_twin2: pixel-aligned MuJoCo twin of so101_jar_real

This is the twin of `neuroscope/molmoact2/store/sets/so101_jar_real`: 320 frames, 40 episodes x 8. The source is the LeRobot v3.0 dataset `data_so101/intheworld__so101_pick_and_place_2` (`so101_follower`, 60 episodes). The task is "Grab the magic cube, and put it into the jar": a black SO-101 with white jaws, on a light-wood table, puts a 3x3 magic cube into a white jar.

- Camera `top` = `observation.images.back`: static, overhead, looking down; the arm base is at the bottom edge.
- Camera `side` = `observation.images.top`: static, low, from the front-left at table height; the arm base is at the right edge.
- The dataset has no wrist camera.

Output set: `neuroscope/molmoact2/store/sets/so101_jar_twin2/`, in the existing twin layout.

All code is in this folder. Nothing was edited in `so101_twin/common/`, `fitkit/`, `pos5/`, other scene folders, `neuroscope/pi05` or `interp/`; the toolkit files were copied here (from `ballcup/`, itself a copy of `common/`) and the copies edited (`JAR` marks). The MolmoAct2 policy was never run.

## Result (agreed metric; pass = held-out median >= 0.80 on every camera)

**PASS on both cameras on the STRICT reading** (arm-only twin mask vs the SAM arm mask, nothing excluded): top 0.869, side 0.871 held-out median. The pass is claimed on the strict reading. The real-occluder-excluded reading is reported as well, as agreed, but nothing rests on it.

Final twin: `so101_twin/jar/out/jar_r3_so101_fit.json` (copied to `out/final_fit.json`), model **so101**. Scored masks: `masks/jar_<cam>_t2o.npy`. Per-frame IoUs: `out/score_final.json`.

| reading | camera / frames | n | median | p10 | mean |
|---|---|---|---|---|---|
| **strict arm-only** | **top** held_out (odd episodes) | 128 | **0.869** | 0.791 | 0.851 |
| **strict arm-only** | top all_frames | 320 | 0.868 | 0.774 | 0.845 |
| real-occluder-excluded | top held_out (odd episodes) | 128 | 0.869 | 0.818 | 0.854 |
| real-occluder-excluded | top all_frames | 320 | 0.867 | 0.795 | 0.849 |
| **strict arm-only** | **side** held_out (odd episodes) | 128 | **0.871** | 0.793 | 0.847 |
| **strict arm-only** | side all_frames | 320 | 0.870 | 0.756 | 0.844 |
| real-occluder-excluded | side held_out (odd episodes) | 128 | 0.898 | 0.802 | 0.865 |
| real-occluder-excluded | side all_frames | 320 | 0.893 | 0.766 | 0.861 |

- Frames excluded because the real arm is out of view (SAM arm mask < 0.1 % of the image): top 0 of 320 (0 held-out), side 0 of 320 (0 held-out). The arm is in view in every frame of both cameras.
- Real-occluder-excluded reading: the same arm-only twin mask and the same SAM arm mask, with the pixels of the REAL magic-cube and jar masks removed from BOTH (`masks/jar_<cam>_occ.npy`, `segment.py --mode occ`: Grounding DINO "a rubik's cube." / "a white jar." on the real frames, every box >= 0.3, SAM 2.1 box prompt; cube kept if < 8 % of the image, not dark and colourful; jar: only its bright pixels (max channel >= 110, the white wall / rim, not the dark inside); no twin input, no fitted parameter; computed before the r2-r5 fits; no fit objective excludes them; their cube part is also stripped from the SAM arm masks, and both serve as SAM negatives). Excluded fraction per frame = |occluder & (twin | real)| / |twin | real|:

| camera | held-out mean | held-out median | all-frames mean | all-frames median | of the twin mask (all-frames mean / median) |
|---|---|---|---|---|---|
| top | 0.0125 | 0.0000 | 0.0175 | 0.0000 | 0.0162 / 0.0000 |
| side | 0.0427 | 0.0542 | 0.0387 | 0.0360 | 0.0370 / 0.0321 |

  Caveat on that reading: in the side view 99 % of the excluded pixels come from the jar mask and 80 % from rest frames, where the white jaws are IN FRONT of the white jar. The jar's bright mask there contains the white jaws themselves (44 % of the excluded pixels are in both the twin and the real arm mask) and the jar next to them where the twin's jaw tips overhang (46 % twin-only, 11 % real-only). So this reading also removes visible arm pixels, not only arm pixels hidden by the jar. In the top view 95 % come from the jar (the gripper at or in the jar during the release; 49 % both, 22 % twin-only, 29 % real-only). That is why the pass is claimed on the strict reading only.

### The same final twin against the twin-INDEPENDENT masks

`_d1k` = Grounding DINO "a black robot arm." boxes + SAM box prompts, union, minus the DINO+SAM cube mask, 5x5 opening, main components (no twin input at all). These masks miss the white jaws in some side frames and drop the thin cable loops:

| masks | camera | held-out n | held-out median | p10 | mean | all-frames median |
|---|---|---|---|---|---|---|
| `_d1k` | top | 128 | 0.878 | 0.722 | 0.841 | 0.877 |
| `_d1k` | side | 128 | 0.839 | 0.730 | 0.818 | 0.839 |

## Metric (as agreed, unchanged)

Whole-arm silhouette IoU, per camera, on the real 640x480 pixel grid. The twin's arm mask is every geom of the arm subtree (base included), rendered as a padded pinhole image and warped onto the real grid with the fitted lens (`arm.Lens`; pixels the padded render cannot fill count as background). It is compared with the SAM 2.1 mask of the real arm, which is never resampled. Frames whose real mask covers < 0.1 % of the image are excluded (none here). The fits only ever saw EVEN episodes (24 episodes, 192 frames); held_out = ODD episodes (16 episodes, 128 frames); all_frames = all 320. `score.py` recomputes everything from a fit json.

## Model and joint mapping

- Arm: **official TheRobotStudio SO-101, `so101_new_calib.xml`** via the `arm.py` copy (base bottom at z = 0 = table top, pan axis = world z, pan 0 reaches +y), no wrist camera (the dataset has none). Colours: black body and servos, white jaws (RGB only).
- **Units: LeRobot NORMALISED** (`use_degrees=False`): -100..100 of each joint's calibrated travel, gripper 0..100 (0 = closed). Evidence: `meta/stats.json` action max elbow_flex = +100.0 and wrist_flex = +100.0 exactly, observation.state max wrist_flex = +100.0 (the clamp of the normalised mode). The rest state is about [-2, -84, 99.5, 2, -4, 2].
- `q_deg = sign * scale * state + nominal + offset` (+1 for all joints (new calibration)); the scales and offsets are fitted (even episodes):

| joint | scale (deg per unit) | offset (deg) | compliance (deg / N m) |
|---|---|---|---|
| shoulder_pan | 0.9047 | +9.29 | +0.00 |
| shoulder_lift | 0.9538 | -14.08 | +0.02 |
| elbow_flex | 0.9127 | +1.17 | -0.19 |
| wrist_flex | 1.0488 | -2.81 | -1.08 |
| wrist_roll | 0.9839 | +4.49 | +0.00 |

- Gripper: `q = -0.17453 + grip_off + grip_scale * g/100 * 1.91986 rad`, grip_scale 0.830, grip_off -0.0230 rad.
- The scales (0.90-1.05 deg/unit) are what normalised units predict (calibrated travel of about 181-210 deg over 200 units); the joint seeds, every round and both meshes found the same map (pan 0.90-0.93, lift 0.94-0.97, elbow 0.89-0.97, wrist flex 1.01-1.06), so no extra audit was needed; overlays were checked anyway. The pan offset is degenerate with a rotation of both cameras about the pan axis (only its prior holds it); the lift offset is the zero of the user's calibration relative to the model.
- Gravity compliance (as `orangebox/arm.py`): `q += compliance * (tau_gravity(q) - 0)` on lift / elbow / wrist flex, fitted in r3-r5 (weak prior, sd 10 deg / N m); it stays near 0 (no measurable gain).
- Rest pose matters: 198 of the 320 frames are at or near the folded rest pose (lift < -75, elbow > 90), so the fits sample their frames uniformly at random from the even episodes (64 per camera), not by farthest-point sampling in joint space, which under-weighted the rest pose (r1: side rest frames 0.75 vs moving 0.82).

Mesh / round comparison (all scored on the final masks with the same protocol). The final fit is chosen by the EVEN-episode mean IoU, min over cameras (set before the r4 / r5 results were seen; the provisional pick among the r2 fits used the even median; r3 wins under either rule); held-out shown only for information:

| fit | model | fitted on | k1 top / side | even mean top / side | even median top / side | held-out median top / side | held-out p10 top / side |
|---|---|---|---|---|---|---|---|
| `so101_twin/jar/out/jar_r2_so101_fit.json` | so101 | `_t1o` | -0.046 / +0.212 | 0.835 / 0.825 | 0.856 / 0.859 | 0.857 / 0.863 | 0.799 / 0.774 |
| `so101_twin/jar/out/jar_r2_so100_fit.json` | so100 | `_t1o` | +0.006 / +0.216 | 0.792 / 0.850 | 0.806 / 0.874 | 0.806 / 0.876 | 0.777 / 0.816 |
| `so101_twin/jar/out/jar_r3_so101_fit.json` | so101 | `_t2o` | -0.046 / +0.166 | 0.840 / 0.843 | 0.866 / 0.869 | 0.869 / 0.871 | 0.791 / 0.793 |
| `so101_twin/jar/out/jar_r4_so101_fit.json` | so101 | `_t2o` | -0.067 / +0.346 | 0.836 / 0.843 | 0.855 / 0.859 | 0.859 / 0.864 | 0.806 / 0.810 |
| `so101_twin/jar/out/jar_r5_so101_fit.json` | so101 | `_t2o` | -0.057 / +0.000 | 0.848 / 0.835 | 0.867 / 0.854 | 0.869 / 0.856 | 0.816 / 0.793 |

(all scored on `_t2o`)

## Cameras (world = arm frame) and lens terms

| camera | dataset feature | pos (m) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |
|---|---|---|---|---|---|---|---|---|---|
| top | observation.images.back (overhead) | 0.0839, 0.0121, 0.5145 | 0.9792, 0.1936, 0.0597, 0.0137 | 51.78 | -0.0458 | -0.0723 | +0.0038 | -0.0098 | +0.28 |
| side | observation.images.top (low side view) | -0.4055, -0.1233, 0.2255 | 0.6922, 0.5862, -0.2530, -0.3365 | 46.93 | +0.1659 | -0.0133 | -0.0488 | -0.0177 | +0.48 |

- MuJoCo camera convention: looks along -z, +y up. Lens: x_d = x_u (1 + k1 r^2 + k2 r^4) (focal-length units), principal point offset (cx, cy) in fractions of (width, height), applied as a warp of the pinhole render (masks and RGB).
- Guard rails: |k1| = top 0.046, side 0.166 (limit 0.4); principal-point shifts <= 4.9 % (limit 10 %). Within the rails, so no audit was required. The side k1 is positive (pincushion) in every round and for both meshes. A straight-line check was tried and is inconclusive: the long straight edges of the side view (door frame, table-wall edge) run almost through the image centre, where radial terms cannot bend them (measured bow <= 0.1 px over 160-260 px; the fitted lenses predict <= 0.25 px). Overlays were checked by eye (`out/before_after.jpg`).
- Both cameras are static (`static_check.py`: 0 px phase-correlation shift between all 40 episode medians), so one pose per camera. Latency from the full-rate 30 fps parquet (`fullrate.py`): a grid then CMA; it is below half a frame for both cameras.

## Scene (RGB only; the metric never uses it)

- Light-wood table (z = 0), a white wall and a door-coloured backdrop for the side view (the table-wall edge read off the side background median and ray-cast through the fitted side camera).
- Jar: white cylinder, R 9.5 cm, H 19.9 cm (fitted to the DINO+SAM jar masks of both views on 8 episodes, median), dark inside and a light rim; placed per episode by a Nelder-Mead fit of its silhouette in both views (median per-episode jar IoU 0.86).
- Magic cube (5.7 cm, sticker faces) by phase: on the table at its top-view detection (ray-cast at half its height, yaw from the min-area box) 90 frames; in the jaws at the tool point while the gripper is around it 105; inside the jar after the release 124; hidden 1.
- Per-camera per-channel colour gain (real / sim mean colour over 16 frames).

## Masks (SAM 2.1 hiera-small; one segmentation process at a time, shared slot checked before every run)

1. `_d1` twin-independent (`segment.py --mode dino`): Grounding DINO tiny boxes "a black robot arm." (every box >= 0.25 and >= 0.5 x best), SAM box prompt, union; a second DINO pass "a white jar. a rubik's cube." gives the jar / cube boxes and SAM masks. `clean_masks.py` -> `_d1k`: minus the cube mask (rejected when > 3 % of the image: DINO took the arm for the cube in 130 side frames where the cube sat in the jar; the first version `_d1c` lacked that check and emptied 86 side frames) and minus the bright part of the jar mask, 5x5 opening, main components. `_d1c` seeded the cameras / joint map and the first fit (r1); `_d1k` is the twin-independent consensus silhouette of every SAM round and the independent check above.
2. `_occ` (`segment.py --mode occ`): the REAL occluder masks (cube, jar) described above; no fit objective excludes them (their cube part is stripped from the SAM arm masks, and both give SAM negatives).
3. `_t1` (`segment.py --mode twin`, prompts = the r1 SO-101 twin, `masks/prompts_r1`): consensus prompting as `common/segment.py` with two silhouettes, the twin's and `_d1k`; box from their union dilated 12 px, positives from their intersection, negatives far from the union plus hard negatives on dark pixels 20-90 px outside and on the real cube / jar masks (`_occ`); 3 prompt variants x 3 multimask outputs; kept = best mean 4-px-tolerant IoU with the silhouettes + 0.25 x SAM score. `strip_objects.py` removes the real cube pixels (`_occ_cube`, a held cube merged with the gripper is not the arm) -> `_t1o`.
4. `_t2` = the same, re-prompted ONCE from the official r2 SO-101 fit (`masks/prompts_r2`), cube stripped -> **`_t2o`, the scored masks** (median prompt agreement top 0.963, side 0.928).

## How it was built

1. Download (meta, data, the two cameras' videos), `make_sets.py` -> `so101_jar_real` (episodes 0..58, 24 even / 16 odd). `static_check.py` (both cameras static), units from `meta/stats.json` (normalised).
2. `_d1` / `_d1k` masks; `seed.py` (twin-free random look-at search + CMA per camera, 160x120, joint map at the init) -> top 0.59 / side 0.59; `jseed.py` (joint CMA over 4 joint scales, 5 offsets, gripper scale and both camera poses + fovy, 3000 evals, two seeds converged to the same map) -> 0.77 (trimmed IoU at 160x120). The same for the SO-100 mesh.
3. r1 (`fit.py` stages B-D on `_d1c`, both meshes, farthest-point fit frames). Overlays showed the folded rest pose (62 % of the frames) fitted worst in the side view. SAM round `_t1` from the r1 SO-101 twin.
4. r2 (both meshes, `_t1o`, 64 RANDOM fit frames per camera, 10 % trimmed mean, stages B-D: pose + fovy + lens; latency grid + CMA on joint offsets, scales, gripper scale / offset and latency; all together with the poses; polish). SAM round `_t2` from the r2 SO-101 twin (the official fit).
5. r3 (`_t2o`, stages B-D from r2 plus gravity compliance, which fitted to ~0: no gain). The SO-100 r3 was stopped to free the second fit slot: its r2 already trailed by 0.05 on the top camera (table above). The r3 side view was worst when the arm swings toward the side camera (pan < -40 units, 12 frames of episodes 46-58: median 0.60; the twin arm too large there), so r4 = r2 refitted with 80 fit frames per camera, half farthest-point in joint space and half random (`--fit-frames mix`), and r5 = the same with the side camera's radial terms held at 0 (`--fix-radial side`, stages A-D) to test whether the positive side k1 is real.
6. `finalize.py` picks the final fit among r2 (both meshes), r3, r4, r5 by the even-episode mean IoU (min over cameras); `score.py` (strict + occluder-excluded on `_t2o`; check on `_d1k`).
7. `objects.py` (jar / cube / wall placement through the final cameras), `render_set.py` -> the set + `out/before_after.jpg`; `write_readme.py` -> this file.

Reproduce the final steps (repo root; MuJoCo venv `.venv`, SAM venv `neuroscope/molmoact2/.venv` via `sam_wait.sh`):

```
zsh so101_twin/jar/sam_wait.sh t2 && .venv/bin/python so101_twin/jar/strip_objects.py --tag _t2
F='--mask-tag _t2o --split even --model so101 --evals 0.7 --obj-trim 0.1 --joint-params off,grip,gripoff,comp'
.venv/bin/python so101_twin/jar/fit.py jar $F --out-tag _r3_so101 --nfit 64 --stages BCD --fit-frames random \
    --init-json so101_twin/jar/out/jar_r2_so101_fit.json
.venv/bin/python so101_twin/jar/fit.py jar $F --out-tag _r4_so101 --nfit 80 --stages BCD --fit-frames mix \
    --init-json so101_twin/jar/out/jar_r2_so101_fit.json
.venv/bin/python so101_twin/jar/fit.py jar $F --out-tag _r5_so101 --nfit 80 --stages ABCD --fit-frames mix \
    --fix-radial side --init-json so101_twin/jar/out/jar_r2_so101_fit.json
(cd so101_twin/jar && ../../.venv/bin/python finalize.py --mask-tag _t2o --cands out/jar_r2_so101_fit.json,out/jar_r2_so100_fit.json,out/jar_r3_so101_fit.json,out/jar_r4_so101_fit.json,out/jar_r5_so101_fit.json)
.venv/bin/python so101_twin/jar/score.py --fit so101_twin/jar/out/final_fit.json --mask-tag _t2o --occ-tag _occ \
    --out so101_twin/jar/out/score_final.json
.venv/bin/python so101_twin/jar/score.py --fit so101_twin/jar/out/final_fit.json --mask-tag _d1k \
    --out so101_twin/jar/out/score_final_d1k.json && .venv/bin/python so101_twin/jar/make_meta_extra.py
.venv/bin/python so101_twin/jar/objects.py --fit so101_twin/jar/out/final_fit.json
.venv/bin/python so101_twin/jar/render_set.py --fit so101_twin/jar/out/final_fit.json \
    --score so101_twin/jar/out/score_final.json
```

## Limits

1. Mask bias: the scored masks `_t2o` were prompted by the twin (in consensus with the twin-independent `_d1k`). Against `_d1k` alone the final twin scores top 0.878 / side 0.839 held-out (`_d1k` misses white-jaw pixels in some side frames and drops the cable loops, so it is not a better reference, only an independent one).
2. Not modelled: the red / white servo cable loops beside the arm (SAM keeps most of them: constant red fringes in the overlays), and the red table clamp / black plate in front of the base in the side view.
3. Side view, rest pose (62 % of the frames): the white jaws sit in front of the white jar, where SAM separates jaw from jar least well; the twin's jaw tips overhang there. The occluder-excluded reading removes those pixels (and visible jaw pixels too), which is why it reads higher in the side view and why no claim rests on it.
4. Strict reading charges the twin for arm pixels hidden by the real jar (gripper inside the jar) or the held cube; both are small (see the excluded fractions).
5. Joint map: linear normalised-unit map per joint (+ gravity compliance). The pan offset is degenerate with a common rotation of both cameras about the pan axis; the wrist-roll state is nearly constant (-3 +- 4 units), so its scale is pinned near 1 by its prior.
6. Lens: the side camera's k1 is positive (pincushion) in every round and for both meshes (final +0.166). Within the guard rails; part of it may compensate model error. r5 holds the side radial terms at 0 and still passes (held-out side 0.856, top 0.869) but scores lower on the even episodes; r4 (half farthest-point frames) pushed k1 to +0.35 without fixing the frames below.
7. Arm swung toward the side camera (pan < -40 units, 12 frames, episodes 46-58, cube at the front left): the side twin is too large / too high there (side IoU ~0.6 in every round; the top view of the same frames is ~0.84). Neither the camera distance (a dolly-zoom scan), the joint map, gravity compliance nor more weight on these poses (r4) fixed it; an unmodelled geometric difference of the real arm (link lengths, base tilt) is the likely cause. These frames are 4 % of the set and do not move the medians.
8. Masks: in a few frames a person places the cube and SAM's side arm mask includes the human arm (e.g. frame 26, held-out side IoU 0.35); kept as is (no manual mask edits).
9. Motion blur in fast frames.
10. RGB: plain materials, the jar as a straight cylinder (the real jar is bulged with feet), the cube's stickers approximate, no cables; the backdrop is a wall plane plus a door-coloured plane.

## Files

| file | what |
|---|---|
| `scene_jar.py` | scene: arm spec, camera inits, priors, RGB objects (`extras`, `frame_setup`) |
| `arm.py`, `fit.py`, `fullrate.py` | copies of the toolkit (via `ballcup/`); `JAR` edits: gravity compliance, random / mixed fit frames, `--fix-radial`, scene flags |
| `segment.py`, `sam_wait.sh` | SAM / DINO runs (dino, twin consensus, occ modes) and the shared-slot waiter |
| `clean_masks.py`, `strip_objects.py`, `fg_masks.py` | mask post-processing (`fg_masks.py`: a no-SAM seed tried and not used) |
| `seed.py`, `jseed.py` | twin-free camera seed, joint seed (scales / offsets + cameras) |
| `score.py`, `finalize.py`, `make_meta_extra.py`, `final.sh` | metric (strict + occluder-excluded), final choice, meta extras, the final pipeline |
| `overlay.py`, `mask_sheet.py`, `static_check.py`, `sheet_real.py` | diagnostics (IoU overlays, mask sheets, static cameras, real-frame sheets) |
| `objects.py`, `geom.py`, `render_set.py`, `write_readme.py` | object placement, camera rays, the twin set writer, this README |
| `masks/` | packed masks `jar_<cam>_<tag>.npy` (`np.unpackbits(a, -1)[..., :640]`), twin prompts `prompts_r1`, `prompts_r2` |
| `out/` | seeds, fit jsons / logs (`jar_r*_<model>_fit.*`), scores, `final_fit.json`, `model_comparison.json`, `objects.json`, `before_after.jpg` |
