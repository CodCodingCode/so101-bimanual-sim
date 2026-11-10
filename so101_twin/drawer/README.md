# so101_drawer_twin — MuJoCo digital twin of `so101_drawer_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_drawer_real`
(LeRobot v3.0 dataset `data_so101/0mrrabbit0__so101_place_block_in_drawer`, task "Put the left block into the
drawer"). Each twin frame shares the real frame's joint state, camera pose (fitted) and scene layout (approximate)
and differs only in rendering. Built from the `so101_twin/lemon` template (same file roles).

Output: `neuroscope/molmoact2/store/sets/so101_drawer_twin/` — same layout as the real set (`meta.json`,
`images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json` byte-identical to the
real set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='drawer_twin'` and `pair[i] = i`).
`meta.json` has `twin_of: "so101_drawer_real"`, `cameras: ["top","side"]`, the joint mapping (`joint_map`), the
fitted camera parameters (`cameras_fit`, with the frame each one is expressed in) and the brightness gains.

**The real "top" camera is not static.** Despite the brief, `observation.images.front` ("top") is the SO-101
wrist camera: the white jaws fill the bottom of every frame and the view moves with the gripper. "side"
(`observation.images.head`) is a static overhead camera. The twin therefore has the "top" camera inside the
`Fixed_Jaw` body (parameters in that body's frame) and fits it by the jaw silhouette.

## Files

| file                 | what                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`     | scene -> `so101_drawer.xml` (+ `breadboard.png` hole texture). Cameras, drawer, blocks, laptop, keyboard are parameters at the top |
| `twin.py`            | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation masks: all / nomotor / body / jaws, camera override)        |
| `check_joints.py`    | real vs sim side-by-side for a few frames -> `out/joints_check.png`                                                                |
| `fit_cameras.py`     | silhouette-IoU camera fit (`--cam side` / `--cam top`), joint-offset refinement (`--offsets`), `--apply` writes CAMS               |
| `place_objects.py`   | object (x, y) on the table from hand-picked overhead pixel centroids, ray-cast through the fitted "side" camera                    |
| `brightness_gain.py` | per-camera gain so the twin's mean intensity matches the real exposure                                                             |
| `replay.py`          | replays all 320 real states (block by phase, drawer tray by an image-derived open/closed flag) and writes the twin set             |
| `make_sheet.py`      | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                                |
| `out/`               | `pairs_sheet.jpg`, `fit_side.png`, `fit_top.png` (real / sim / mask overlay), `camera_fit_*.json`, `offsets_fit.json`, `*.log`     |

Run order: `build_scene.py` -> `fit_cameras.py --cam side`, `--cam top` -> `--offsets` -> `--apply` ->
`place_objects.py` (paste into OBJECTS / DRAWER_UNIT) -> `brightness_gain.py` (paste into replay.GAIN) -> `replay.py`
-> `make_sheet.py`. All with the repo-root `.venv/bin/python` (MuJoCo 3.3.0). No GPU model was run.

## Joint mapping

LeRobot v3.0 `so_follower` state: five joints in degrees relative to the calibration "middle" pose, gripper 0..100
(observed 2 = closed, up to 34 = open). Menagerie `so_arm100` order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw:

```
sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j]),   j = pan, lift, elbow, wrist_flex, wrist_roll
SIGNS       = [-1,    +1,     +1,     +1,     +1 ]
OFFSETS_DEG = [-1.94, -94.13, +88.55, -22.47, 180]     (0..3 fitted from the first guess [0, -90, +90, 0]; roll by eye)
Jaw         = -0.174 + gripper/100 * 1.924
```

Derivation: the dataset's most folded pose (lift -97, elbow +98, wrist +73; e.g. frames 7, 240) must land on the
Menagerie `rest` keyframe (Pitch -190, Elbow +178, Wrist_Pitch +68 deg) -> Pitch = lift - 90, Elbow = elbow + 90,
Wrist_Pitch = wrist. The brief's "unusual rest pose" [7, -25, 37, 42, 6, 2] is frame 0, which is mid-episode
(t = 60, phase transfer_open), not a rest pose. Positive real pan swings the arm toward the laptop (image left, -x)
while positive Menagerie Rotation goes to +x, so pan is flipped. Checked visually on 7 frames (`out/joints_check.png`:
folded rest, full extension, pan -27 deg, over the drawer) and by the overhead silhouette fit on 18 frames spanning
the joint space. Offsets 0..3 were then refined by Nelder-Mead on the overhead IoU (0.581 -> 0.613); the
wrist_flex offset (-22 deg) is the least constrained of the four by a top-down silhouette.

Wrist roll: the real roll stays within -3.5..+28 deg, so the overhead IoU cannot tell the sign or offset apart
(all of {+-1} x {0, 160, 180, 200} within 0.003). Offset 180 was chosen from the wrist-camera view
(`out/roll_check.png`): with 0 the wrist camera looks at the laptop and sees the blocks bottom-left; with 180 it sees
the drawer upper-left and the blocks ahead/right like the real frames. Sign +1 (as in the pickplace twin) is a
convention here, not a measurement.

## Cameras (fitted by silhouette IoU, 640x480, pinhole)

World frame: breadboard surface z = 0, Menagerie Base body origin at (0, 0, 0) rotated 180 deg so the arm reaches
+y (the pan axis is at (0, 0.045)). Overhead image up = +y, right = +x.

| cam  | frame     | pos (m)               | yaw / pitch / roll (deg) | fovy (deg) | mean IoU 640x480 (18 frames)     |
| ---- | --------- | --------------------- | ------------------------ | ---------- | -------------------------------- |
| side | world     | (0.090, 0.029, 0.387) | 92.4 / -78.5 / 3.2       | 70.1       | **0.613** (per frame 0.46..0.72) |
| top  | Fixed_Jaw | (0.042, 0.018, 0.062) | -111.6 / -16.5 / 23.1    | 41.7       | **0.542** (per frame 0.21..0.76) |

Angles are the `build_scene.cam_xyaxes` convention (yaw about z, pitch = elevation, roll about the view axis),
for "top" applied in the Fixed_Jaw body frame (fingers along -y, jaw opening along x, jaw axis z; the camera sits
on the +z side of the jaw plane, which is the only side that gives fixed jaw left / moving jaw right with the
fingertips pointing up as in the real frames).

Overhead fit: the real green arm is overexposed to near-white in the lit episodes, so the real mask is V > 140
(the board's glare stays below ~130) or shaded green / orange by hue, minus the purple / cyan blocks, with fixed
pixel zones zeroed in both masks (drawer unit, the orange leader-arm parts beside the base, the clamp strip, the
laptop area, x > 345 where the arm never goes). Sim mask = segmentation of all printed arm parts (black motors are
holes in both). 18 frames by farthest-point sampling over the five joints among the brightly lit episodes.
500 random samples + Nelder-Mead from the best 4 at 320x240, IoU reported at full resolution.

Wrist fit: real mask = white pixels (S < 55, V > 165) in rows >= 200; sim mask = segmentation of the two jaw meshes,
same row cut. Frames: those whose upper part has no white (no drawer / laptop in view), spread over gripper
opening and wrist angle (18 frames, 600 random samples + Nelder-Mead). The jaw silhouette only pins the camera
relative to the jaw; where the camera looks in the scene is additionally cross-checked by eye (see wrist roll above).

A table-plane anchor (pan axis pixel + hole pitch) was tried as a penalty and rejected: the pixel I took for the
pan axis is the elbow motor of the folded arm (the pan axis is hidden under the arm in the real view), and the
anchored fit dropped to IoU 0.50 with a wrong camera. The code is still in `fit_cameras.py` with `ANCHOR_W = 0`.
`out/camera_fit_side_unanchored.json` is the same as `camera_fit_side.json` (kept for the record).

Why the IoU is not higher: the real SO-101 prints (shoulder housing, wrist-camera mount, the orange forearm end)
differ from the Menagerie SO-100 meshes; the real arm is washed out so its mask has holes where the sim has none;
the overhead camera has visible wide-angle perspective (fitted fovy 70 deg) and probably lens distortion; in the
wrist view the real jaws are blurred and the drawer's white walls leak into the "white" mask in some frames.

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes, green body, black motors, white jaws (the wrist-camera mount is not modelled), plus
  an orange box at the wrist end of the forearm (the real forearm has an orange section with a hole grid there).
- Table: a large dark box with a procedural optical-breadboard texture (25 mm hole pitch; the grid phase relative
  to the arm is not matched). Grey wall far behind.
- Drawer unit at (-0.010, 0.335), 0.113 x 0.130 x 0.055 m (x/y from ray-casting the top plate's edges through the
  fitted overhead camera, height from the front face's pixel height), blue-grey top with a thin white rim, white
  tray with a grey floor that slides -y by 0.07 m when open.
- Drawer state: there is no label for it, so `replay.drawer_open()` reads it off the real overhead frame (fraction of
  bright pixels in the strip in front of the unit, normalised by the unit's top plate; 229/320 frames open). It is
  noisy in the five dark-lit episodes (17-31) and some episodes start with the drawer already open.
- Blocks: purple (the "left block", moved) at (0.032, 0.184), cyan at (0.108, 0.191), from ray-casting the pixel
  centroids of episode 1's last frame. **The real block layout varies by episode** (some episodes have an orange
  block that is the one being carried, the blocks are on the robot's left in others; frame 0 of episode 17 shows
  them elsewhere). The twin uses this one layout, and always carries the purple block.
- Purple block during replay: on the table for `phase == approach`, at the `tcp` site between the jaws for `grasp` /
  `transfer_open`, on the tray floor for `release` (the tray may be closed then, hiding it). The phase label is
  gripper-threshold based, so the block can jump to the jaws a few frames early.
- Laptop (dark base left of the board, light screen further left, visible in the wrist view at rest) and keyboard
  (right) are plain boxes placed by ray-casting; cables, the leader-arm parts beside the base, the clamp and the
  operator's hand are not modelled.
- Lighting: two lights + headlight; per-camera brightness gain top 1.06, side 1.66 (the real overhead has a large
  glare patch that dominates its mean, so the twin board is lifted more than the arm would need). No noise, blur,
  white balance or lens distortion. The five dark-lit episodes are rendered with the same lighting as the rest.

## Not done / could be improved

- Per-episode object layout and block colour (needs segmenting each episode's first frame; the ray-casting helper
  is there), and the real block carried in each episode.
- A proper SO-101 mesh set (the wrist-camera mount, shoulder housing) would raise both IoUs; a distortion model
  for the wide overhead camera.
- The wrist camera's view direction relative to the jaw is only weakly constrained by the jaw silhouette; a fit
  on scene features seen by the wrist camera (drawer edges, hole grid) would pin it.
- Drawer open/closed from the images is a heuristic; the drawer travel (0.07 m) is a guess.
