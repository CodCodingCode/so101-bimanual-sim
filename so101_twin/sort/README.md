# so101_sort_twin — MuJoCo digital twin of `so101_sort_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_sort_real`
(LeRobot v3.0 dataset `data_so101/hubnemo__so101_sort_cubes`, task "Grab the purple cube and put it into the
cardboard box, ignore the other objects"). Each twin frame shares the real frame's joint state, camera pose (fitted /
hand-placed, see below) and an approximate scene layout, and differs only in rendering.

Output: `neuroscope/molmoact2/store/sets/so101_sort_twin/` — same layout as the real set (`meta.json`,
`images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json` byte-identical to the real
set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='sort_twin'` and `pair[i] = i`), plus
`sim_qpos.npy` (the qpos actually rendered). `meta.json` has `twin_of: "so101_sort_real"`, `cameras: ["top","side"]`,
the joint mapping (`joint_map`), the camera parameters (`cameras_fit`), the per-camera IoU (`camera_iou`) and gains.

Built from a copy of `../lemon/` (same script set; joint mapping taken from `../pickplace/`, the other v3.0 twin).

## Files

| file                 | what                                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`     | single SO-101 arm scene -> `so101_sort.xml` (table z=0, base at origin reaching +y). Cameras / objects at the top.      |
| `twin.py`            | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation masks, runtime camera override), real-set loader |
| `check_joints.py`    | real vs sim side-by-side for a few frames -> `out/joints_check*.png`                                                    |
| `fit_cameras.py`     | silhouette-IoU camera fit (`--cam top/top_b/side`), joint-offset refinement (`--offsets`), `--apply` -> build_scene.py  |
| `place_objects.py`   | per-episode cube layout from the purple mask of each episode's first frame, ray-cast through the session camera         |
| `brightness_gain.py` | per-camera gain so the twin's mean intensity matches the real exposure                                                  |
| `replay.py`          | replays all 320 real states and writes the twin set                                                                     |
| `make_sheet.py`      | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                     |
| `out/`               | `pairs_sheet.jpg`, `fit_*.png` (real / sim / mask overlay), `camera_fit_*.json`, `cube_layouts.json`, logs, checks      |

Run order: `build_scene.py` -> `fit_cameras.py --cam top`, `--cam top_b`, `--cam side` -> `--apply` ->
`place_objects.py` -> `brightness_gain.py` (paste into replay.GAIN) -> `replay.py` -> `make_sheet.py`. All with `.venv/bin/python`.

## Joint mapping (verified visually and by the top-camera silhouette fit)

The dataset is LeRobot v3.0 (`so101_follower`, degrees mode, zero = calibration "middle" pose; gripper 0..100, observed
2.4 = closed .. 65 = open). Menagerie `so_arm100` joint order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (rad):

```
sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j]),   j = pan, lift, elbow, wrist_flex, wrist_roll
SIGNS       = [-1, +1, +1, +1, +1]
OFFSETS_DEG = [ 0, -90, +90,  0, 180]
Jaw         = -0.174 + gripper/100 * 1.924
```

Same signs as `../pickplace` and the derived (un-refined) offsets. A Nelder-Mead refinement of offsets 1..4 on the
silhouette IoU moved wrist_flex by +53 deg and roll by +22 deg for +0.01 IoU — judged to be the noisy real mask
being exploited (the gripper orientation already matched in `out/fit_top_b.png`), so it was **not** kept.
Checked on `out/joints_check_final.png` and the fit overlays: the forearm hole-plate, the gripper's black motor on top
of the blue housing, pan toward the box (+x = image right from the top camera) and the elbow fold all match the real
frames. The elbow reaches 190 deg at rest (Menagerie limit 180; `mj_forward` does not clamp), as in pickplace.

## Cameras

World frame: table top z = 0, arm base at the origin, arm reaches +y. The real "top" camera sits ~30 cm above and
just behind the base, looking along +y and down. **It was moved between two recording sessions** (visible in the
backgrounds of `out/episodes_top.jpg`): session A = episodes 4..55 (frames 0..175), session B = episodes 66..101
(frames 176..319). The twin has two static cameras, `top` and `top_b`, and `replay.py` picks one per episode
(`build_scene.top_cam_for_episode`).

The real "side" (`observation.images.front`) camera is a wrist camera: the right finger tip stays put while the left
one sweeps with the gripper, so it is modelled as camera `side` on the `Fixed_Jaw` body (fingers along -y, jaw opens
along x in that frame). It looks essentially straight along the fingers (checked by mapping jaw-frame directions to
world at several frames: frame 7 looks forward at the laptop, 311 down at the table, 203 at the box — only the -y
direction matches all) and is offset ~4.5 cm to the +z side of the finger plane, which puts the finger bases at the
top edge of the image and the tips pointing down into it, as in the real frames.

| cam               | frame     | pos (m)                 | yaw / pitch / roll (deg) | fovy (deg) | mean IoU 640x480 (18 frames)    |
| ----------------- | --------- | ----------------------- | ------------------------ | ---------- | ------------------------------- |
| top (session A)   | world     | (-0.030, -0.026, 0.303) | 83.8 / -27.3 / -3.1      | 57.0       | **0.35** (per frame 0.03..0.69) |
| top_b (session B) | world     | (-0.016, -0.025, 0.330) | 85.5 / -36.7 / -0.3      | 50.5       | **0.21** (per frame 0.00..0.63) |
| side (wrist)      | Fixed_Jaw | (-0.010, 0.000, 0.045)  | -90 / -10 / 180          | 60.0       | **0.35** (per frame 0.09..0.62) |

Fit (top cams): real arm mask = HSV threshold, light-blue wrist/gripper OR the white arm body, which reads lilac
(session A, bluish light) / lavender (session B, greyer light) against the lower-saturation white table; sim mask =
segmentation render of the arm's non-motor visual geoms; 18 frames per session by farthest-point sampling in joint
space; 250-400 random samples + Nelder-Mead from the best 4 at 320x240, IoU reported at 640x480.

**Why the top IoUs are low**: a white arm on a white table. The colour mask only catches the blue gripper and the
parts of the arm that are tinted by the room light; the lower forearm and the base read exactly like the table and
are missing from the real mask (red = real-only, blue = sim-only in `out/fit_top*.png`), and in the frames where the
arm is at rest it is almost out of view, so the IoU is dominated by mask noise (those frames score 0.0-0.1; frames
with the arm extended score 0.5-0.7). Geometrically the overlays line up (arm position, size, gripper orientation) in
both sessions; a shape-based or learned arm segmentation would be needed to push the number up.

**Wrist camera**: the silhouette objective (real = dark pixels = black finger pads, sim = every robot geom) is not
discriminative for this camera: every random/NM run converged to poses that look sideways into the jaw housing and
match the two tips with the housing's outline (IoU ~0.42 but visually wrong). Two other traps were found on the way:
MuJoCo's default near clip plane (1% of the model extent, ~7 cm here) sliced the fingers 3 cm from the camera
(`<map znear="0.0005">` fixes it), and a jaw-only sim mask let the wrist piece fill the view for free. The pose above
was therefore chosen from a grid in the physically consistent family (along the fingers, +z side) by IoU and visual
check; its IoU of 0.35 is evaluated, not optimised. Residual: the sim tips sit nearer the image corners and are
shorter than the real ones (the Menagerie SO-100 finger shape differs from the real SO-101 pads), and the real lens is
wide with visible distortion.

Exposure: per-camera gain top 0.72, side 0.69 (real cameras are darker than the render); no colour matching.

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes recoloured: body white-lilac (`0.80 0.80 0.90`), motors black, wrist + jaws light blue.
  The real finger pads are black; in the twin the whole jaw is blue.
- Table: white box (far edge at y = 0.85); dark back wall; a black PC-tower box and a grey laptop slab far behind the
  arm, placed by eye. No cables, books, mug, hands, or the black pouch.
- Cardboard box: open-top brown box 17 x 13 x 15 cm centred at (0.40, 0.50), from ray-casting its bottom-front edge
  through both session cameras (the two sessions disagree by ~10 cm in y, the mean was used).
- Cubes: up to 4 purple 4.2 cm cubes per episode at the positions detected in the episode's first frame
  (`out/cube_layouts.json`, `out/place_check.png`): 30/40 episodes have detections (1-4 cubes each; dark-shadowed or
  occluded cubes are missed), 10 episodes use the default 4-cube layout. The **target** cube is the one nearest the
  gripper (site `tcp`) at the episode's first `grasp` frame; during replay it sits on the table for `approach`,
  between the jaws for `grasp` / `transfer_open`, and inside the box for `release` (phase label is gripper-threshold
  based, so the hand-off is a few frames early/late in some pairs). Other cubes never move; the real non-target cubes
  are sometimes nudged.
- Pinhole cameras, no distortion, no motion blur (several real frames are heavily blurred), no shadows from the real
  room lights.

## Not done / could be improved

- A proper arm segmentation for the top cameras (the IoU is mask-limited, not pose-limited).
- A better wrist-camera fit: a finger-tip-specific objective (e.g. only the lowest dark blobs in the top third) or a
  calibration from the real cube's apparent size would pin down the offset and fovy.
- Per-session sub-clusters: session B episodes 66..68 and 72..75 may have a slightly different camera than 81..101.
- The cardboard box position/size per session, the black book/pouch, cables.
