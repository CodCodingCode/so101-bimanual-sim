# so101_lemon_twin — MuJoCo digital twin of `so101_lemon_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_lemon_real`
(LeRobot v2.1 dataset `data_so101/Beegbrain__pick_lemon_and_drop_in_bowl`). Each twin frame shares the real
frame's joint state, camera pose (fitted) and scene layout (approximate) and differs only in rendering.

Output: `neuroscope/molmoact2/store/sets/so101_lemon_twin/` — same layout as the real set
(`meta.json`, `images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json`
byte-identical to the real set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='lemon_twin'`
and `pair[i] = i` = index into the real set). `meta.json` has `twin_of: "so101_lemon_real"` plus the joint mapping,
fitted camera parameters and brightness gains used.

## Files

| file                 | what                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `build_scene.py`     | single SO-101 arm scene -> `so101_lemon.xml` (table z=0, base at origin reaching +y). Camera/object parameters at the top.     |
| `twin.py`            | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation arm mask, runtime camera override), real-set loader     |
| `check_joints.py`    | real vs sim side-by-side for a few frames -> `out/joints_check.png`                                                            |
| `fit_cameras.py`     | silhouette-IoU camera fit (`--cam top/side`), joint-offset refinement (`--offsets`), `--apply` writes CAMS into build_scene.py |
| `place_objects.py`   | object (x,y) on the table from hand-picked pixel centroids in real frame 0, ray-cast through both fitted cameras               |
| `brightness_gain.py` | per-camera gain so the twin's mean intensity matches the real exposure                                                         |
| `replay.py`          | replays all 320 real states and writes the twin set                                                                            |
| `make_sheet.py`      | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                            |
| `out/`               | `pairs_sheet.jpg`, `fit_top.png`, `fit_side.png` (real / sim / mask overlay), `camera_fit_*.json`, `offsets_fit.json`, logs    |

Run order: `build_scene.py` -> `fit_cameras.py --cam top`, `--cam side` -> `--offsets` -> `--apply` -> `place_objects.py`
(paste into OBJECTS/BOWL) -> `brightness_gain.py` (paste into replay.GAIN) -> `replay.py` -> `make_sheet.py`. All with `.venv/bin/python`.

## Joint mapping (verified)

The dataset is old-LeRobot (v2.1, `main_*` names, `robot_type: so100`): the five revolute joints are in degrees in the
old SO-100 calibration, the gripper is linear 0..100 (observed 1 = closed at rest, ~51 = open, ~36 = holding the lemon).
Menagerie `so_arm100` joint order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (radians):

```
sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j]),   j = pan, lift, elbow, wrist_flex, wrist_roll
SIGNS       = [-1, -1, +1, +1, -1]
OFFSETS_DEG = [ 0, +1.17, -1.84, +1.72, 0]      (fitted; 0 everywhere gives IoU only ~0.015 lower)
Jaw         = -0.174 + gripper/100 * (1.75 + 0.174)
```

i.e. the old-LeRobot zero pose coincides with the Menagerie zero pose; the real rest pose (lift 189, elbow 180,
wrist 62) maps onto the Menagerie `rest` keyframe (-3.32, 3.11, 1.18 rad). Checked visually on 6 frames
(`out/joints_check.png`) and by the silhouette fit on 18 frames spanning the joint space (`out/fit_*.png`): pan
toward the bowl (-x = robot's left) and wrist roll are the right way round. The wrist-roll sign is the least
constrained (the roll is small, -14..+6 deg, in most frames); the offset of pan/roll was not fitted (kept 0).

## Cameras (fitted, world frame: table top z=0, base at origin, arm reaches +y)

Despite the dataset's names, the real "top" camera is a low view from the robot's right (+x) looking across the
table, and the "side" camera is an overhead view from in front of the robot (+y) looking back and down ~58 deg.

| cam  | pos (m)                | yaw / pitch / roll (deg) | fovy (deg) | mean IoU 640x480 (18 frames)     |
| ---- | ---------------------- | ------------------------ | ---------- | -------------------------------- |
| top  | (0.449, 0.262, 0.145)  | 184.3 / -3.7 / -1.2      | 42.8       | **0.709** (per frame 0.61..0.77) |
| side | (-0.031, 0.491, 0.368) | -88.7 / -58.0 / 3.2      | 46.6       | **0.615** (per frame 0.39..0.71) |

Fit: real arm mask = HSV threshold (orange body OR blue jaws); sim mask = segmentation render of the arm's orange+blue
visual geoms; 400-sample random box search + Nelder-Mead from the best 4 at 320x240, IoU reported at full
resolution. Cross-check: casting the two fitted cameras' rays through the same object pixels onto the table plane
agrees within ~2 cm for apple / lemon / strawberry / peach (bowl ~9 cm, rim centre is ambiguous).

Why the IoU is not higher: the real SO-101 base is bigger than the Menagerie SO-100 base (extra orange pixels below
the shoulder in the top view), the real jaws/wires are not modelled exactly, the real "side" view is dark so parts of
the blue jaws drop out of the real mask, and the colour mask picks up red fruit edges. Pinhole model, no lens
distortion (RealSense colour has little).

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes recoloured (orange body, black motors, blue jaws). Link lengths are the SO-100 ones.
- Table: wood-coloured box; wall plane behind the scene (x = -0.62); floor far below. Real wall/table textures are not modelled.
- Objects: lemon (ellipsoid), apple, strawberry, peach (sphere/ellipsoids) and a red bowl built from 40 flared thin
  boxes + a floor disc. Positions from real frame 0 (episode 0): lemon (-0.006, 0.280), apple (0.120, 0.214),
  strawberry (0.101, 0.345), peach (-0.042, 0.411), bowl (-0.203, 0.270). **The real objects move between episodes**
  (the lemon start position and the fruit layout vary by several cm); the twin uses this one layout for all 40 episodes.
- Lemon during replay: on the table for `phase == approach`, between the jaws (site `tcp` on Fixed_Jaw) for
  `grasp` / `transfer_open`, in the bowl for `release`. The phase label is gripper-threshold based, so near the grasp
  the lemon may already be "in the jaws" a few frames early (visible in some pairs), and the drop height is not modelled.
- Exposure: per-camera brightness gain top 0.92, side 0.67 applied to the render (the real overhead camera is much
  darker); no colour-temperature matching, no noise/blur.

## Not done / could be improved

- Per-episode object layout (would need segmenting each episode's first frame; the ray-casting helper is there).
- The SO-101 base geometry, wires, and the real jaw shape; lens distortion; joint offsets for pan/roll.
- Stage 2 of the fit (offsets + camera refit) was run for one round only; another round would move things by < 0.01 IoU.
