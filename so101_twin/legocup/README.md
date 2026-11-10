# so101_legocup_twin — MuJoCo digital twin of `so101_legocup_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_legocup_real`
(LeRobot v2.1 dataset `data_so101/Rayenghali__so101_follower_main_put_the_red_lego_block_in_the_black_cup_eps100_fps30_480x640`,
task "Put the red lego block in the black cup"). Each twin frame shares the real frame's joint state, camera pose
(fitted) and scene layout (approximate, per episode) and differs only in rendering.

Output: `neuroscope/molmoact2/store/sets/so101_legocup_twin/` — same layout as the real set (`meta.json`,
`images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json` byte-identical to the
real set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='legocup_twin'` and `pair[i] = i` =
index into the real set). `meta.json` has `twin_of: "so101_legocup_real"`, `cameras: ["top", "side"]`, the joint
mapping, the fitted camera parameters (with each camera's parent body) and the brightness gains.

Copied and adapted from `so101_twin/lemon/` (same file roles); all sim work with the root `.venv/bin/python`
(MuJoCo 3.3.0).

## The one thing that is different from the other twins: "side" is a wrist camera

Despite the set's camera names, the real `side` stream (`observation.images.secondary_0`) is a camera mounted on
the gripper: the two jaw prongs hang from the top edge of every frame and the view swings with the arm. The
twin therefore attaches `side` to the `Fixed_Jaw` body and fits its pose in that body's frame (jaw tips are
along -y there). `top` (`observation.images.main`) is a fixed overhead camera in front of the robot looking back
and steeply down; the arm base and upper arm are above the top edge of the image, only the forearm/wrist/gripper
enter from the top.

## Files

| file                 | what                                                                                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`     | single black SO-101 scene -> `so101_legocup.xml` (table z=0, base at origin reaching +y, grey mat, black mug, red 4x4 brick). CAMS, mat and object parameters at the top.                                                     |
| `twin.py`            | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation arm mask, runtime camera override in the parent frame), real-set loader; loads `out/offsets_fit.json` and the jaw gain from `out/camera_fit_side.json` |
| `check_joints.py`    | real vs sim side-by-side for a few frames -> `out/joints_check.png`                                                                                                                                                           |
| `fit_cameras.py`     | `--roi` (mat ROI), `--cam top/side` silhouette-IoU fit, `--offsets` joint-offset refinement (top camera), `--apply` writes CAMS                                                                                               |
| `place_objects.py`   | per-episode block / cup (x,y) from colour segmentation of the real top frame + ray cast -> `out/objects.json`; `--mat` mat corners                                                                                            |
| `brightness_gain.py` | per-camera gain so the twin's mean intensity matches the real exposure                                                                                                                                                        |
| `replay.py`          | replays all 320 real states (objects per episode, block by phase) and writes the twin set                                                                                                                                     |
| `make_sheet.py`      | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                                                                                                                           |
| `out/`               | `pairs_sheet.jpg`, `fit_top.png`, `fit_side.png` (real / sim / mask overlay), `camera_fit_*.json`, `offsets_fit.json`, `objects.json`, `objects_check.png`, logs                                                              |

Run order: `build_scene.py` -> `fit_cameras.py --roi` -> `--cam top` -> `--offsets` -> `--cam side --p0=...` ->
`--apply` -> `place_objects.py --mat` (paste MAT_* into build_scene.py, rebuild) -> `place_objects.py` ->
`brightness_gain.py` (paste into replay.GAIN) -> `replay.py` -> `make_sheet.py`.

## Joint mapping (verified visually and by the silhouette fit)

The dataset is v2.1 but was recorded with the new `so101_follower` degrees calibration (zero = arm straight up,
rest pose ~[-10, -98, 98, 50, -3], gripper 0..100 with ~1 = closed, ~33 = holding the 32 mm brick, max 53).
Menagerie `so_arm100` joint order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (radians):

```
sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j]),   j = pan, lift, elbow, wrist_flex, wrist_roll
SIGNS       = [-1, +1, +1, +1, +1]
OFFSETS_DEG = [+0.04, -89.72, +88.80, +2.74, +180]     (first guess [0, -90, +90, 0, 180]; lift/elbow/wrist refined by --offsets)
Jaw         = -0.174 + JAW_GAIN * gripper/100 * (1.75 + 0.174),   JAW_GAIN = 2.13 (fitted with the wrist camera)
```

Pan sign checked on the extreme frames (real pan -33 = arm on the image right / +18 = image left, i.e. real +pan
swings the arm to world +x = image left for a camera looking -y). The 180 deg wrist-roll offset is the
pickplace-style choice; with it the wrist camera fit lands on the correct side of the gripper (the moving jaw is
the left prong in the wrist view and opens to the left, as in the real frames) and the block/cup layout seen by
the wrist camera is not mirrored. The jaw gain is what the wrist-camera silhouettes want: at 33 % the real
moving jaw already swings out of the wrist view, which the Menagerie jaw at gain 1 does not; at the data maximum
(53 %) the sim jaw reaches 1.99 rad, slightly past the Menagerie joint limit (not clamped by `mj_forward`).

## Cameras (fitted by arm-silhouette IoU)

Real arm mask: the arm is black, so `max(RGB) < 90` inside the grey-mat ROI (per-pixel 60th percentile of
brightness over the set > 95), connected components touching the top border (the arm enters from the top), minus
a 95 px disc around the episode's cup (the cup is black too). Wrist mask: `max(RGB) < 60` components touching
the top border (the prongs). Sim mask: segmentation render of the arm's visual geoms (restricted to the same
ROI). 18 frames per camera by farthest-point sampling in joint space (top: frames with the arm covering > 2 % of
the image; wrist: frames with the jaws over the mat, real mask < 12 %, spread over gripper openings). Random
search + Nelder-Mead at 320x240, IoU reported at 640x480.

| cam  | parent    | pos (m)                | yaw / pitch / roll (deg) | fovy (deg) | IoU 18 fit frames (min..max) | IoU all eligible frames               |
| ---- | --------- | ---------------------- | ------------------------ | ---------- | ---------------------------- | ------------------------------------- |
| top  | world     | (-0.042, 0.466, 0.469) | -85.4 / -68.5 / 6.0      | 35.0       | **0.516** (0.25..0.77)       | 0.538 mean, 0.550 median (166 frames) |
| side | Fixed_Jaw | (0.006, -0.012, 0.078) | -92.1 / -25.4 / 183.9    | 59.9       | **0.729** (0.49..0.90)       | 0.755 mean, 0.792 median (232 frames) |

(`side` pos/angles are in the Fixed_Jaw frame: 7.8 cm above the jaw plane, looking along the jaws and 25 deg away
from them, rolled 180 so the prongs hang from the top of the image.) Overlays: `out/fit_top.png`, `out/fit_side.png`.

Why the top IoU is not higher: the real arm has a large square wrist-camera mount plate hanging below the gripper
and visible wires that are not in the Menagerie mesh; the real SO-101 wrist/gripper housings are bulkier; the
real jaws are black on a dark table so the mask also catches the arm's shadow edge; frames where the cup disc
cuts into the arm lose pixels; pinhole model without distortion. The wrist camera fits well because its target
(the prongs) is rigidly attached to it.

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes, every link / motor / jaw black (the real SO-101 print). Link lengths are the
  SO-100 ones. No camera-mount plate, wires or stickers.
- Table: dark grey box; grey mat 0.33 x 0.31 m centred at (-0.035, 0.251), yaw 10 deg (corners ray-cast from
  real frame 0 through the fitted top camera); dark wall behind. The keyboard and the table edge at the left of
  the real top view are not modelled.
- Objects: black mug = 32 thin boxes in a ring, r = 4.0 cm, h = 8 cm, floor disc, capsule handle on -x; red
  4x4 lego brick 32 x 32 x 19 mm box + 16 studs. **Per-episode positions** from the first approach frame of
  each episode (`out/objects.json`, all 40 episodes detected, see `out/objects_check.png`); ~1-2 cm accuracy
  (the cup centroid includes the handle and the rim is seen from above).
- Block during replay: on the table for `phase == approach`, between the jaws (site `tcp` on Fixed_Jaw, oriented
  with the jaw) for `grasp` / `transfer_open`, inside the cup for `release`. The phase label is gripper-threshold
  based so the hand-off is a few frames early/late in some pairs.
- Exposure: per-camera brightness gain top 1.03, side 0.92; no colour matching, noise, blur or lens distortion.
  The real wrist camera has strong vignetting / auto-exposure swings (dark frames over the cup) that are not
  modelled.

## Not done / could be improved

- The wrist camera's view of the table objects is only roughly right: small pose errors in the jaw frame and the
  1-2 cm object errors move the block/cup several tens of pixels in that view (the jaw silhouettes themselves
  match). A fit of the wrist pose on the block position in the wrist view would tighten this.
- Wrist-camera mount plate on the fixed jaw (would add top-view IoU), SO-101 housings, wires, lens distortion.
- Only one round of offset / camera alternation; the pan offset and wrist-roll offset were not fitted (fixed at
  0 and 180). Jaw gain 2.13 is plausible for this calibration but was fitted only from the wrist silhouettes.
