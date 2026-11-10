# so101_ringpole_twin — MuJoCo digital twin of `so101_ringpole_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_ringpole_real`
(LeRobot v3.0 dataset `data_so101/nbirukov__so101_pick_stack_ring_pole`, task "Pick up a ring and put it on the
stacking pole"). Each twin frame shares the real frame's joint state, camera pose (fitted) and scene layout
(per-episode, approximate) and differs only in rendering.

Output: `neuroscope/molmoact2/store/sets/so101_ringpole_twin/` — same layout as the real set (`meta.json`,
`images_top.npy`, `images_side.npy` uint8 [320,480,640,3], `state.npy` and `prompt.json` byte-identical to the
real set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='ringpole_twin'` and `pair[i] = i`,
plus `sim_qpos.npy` = the rendered qpos). `meta.json` has `twin_of: "so101_ringpole_real"`, `cameras ["top","side"]`,
the joint mapping, the fitted camera parameters (`cameras_fit`), the fit IoUs and the brightness gains.

Built from `so101_twin/lemon/` (same scripts, adapted); all sim work with the root venv `.venv/bin/python`
(MuJoCo 3.3.0).

## Files

| file               | what                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`   | single SO-101 scene -> `so101_ringpole.xml`: white arm, dark wood table, orange backdrop, pole + base + ring (mocap), cameras                     |
| `twin.py`          | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation masks, runtime camera override), real-set loader                           |
| `check_joints.py`  | real vs sim side-by-side for a few frames -> `out/joints_check.png`                                                                               |
| `fit_cameras.py`   | white-silhouette IoU camera fit (`--cam top/side`), joint-offset refinement (`--offsets`), `--apply` writes CAMS                                  |
| `place_objects.py` | per-episode pole / ring detection in the first real top frame, ray-cast through the fitted top camera -> `objects.json`                           |
| `replay.py`        | replays all 320 real states (ring follows the phase label) and writes the twin set                                                                |
| `make_sheet.py`    | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                                               |
| `objects.json`     | per dataset episode: pole (x, y), ring (x, y), pole kind (bare rod / disc base), detection pixels                                                 |
| `out/`             | `pairs_sheet.jpg`, `fit_top.png`, `fit_side.png` (real / sim / mask overlay), `camera_fit_*.json`, `offsets_fit.json`, `objects_detect.jpg`, logs |

Run order: `build_scene.py` -> `fit_cameras.py --cam top` -> `--offsets` -> `--apply` -> `place_objects.py` ->
`fit_cameras.py --cam side` -> `--apply` -> `replay.py` -> `make_sheet.py`.

## World frame

Table top z = 0, arm base origin at (0, 0, 0), the arm reaches along **+x** at pan = 0 (Menagerie base rotated
+90 deg about z; +Rotation swings the arm toward +y = counter-clockwise from above). In the real overhead view the
base is at the left and the arm reaches to the right; +y is image-up, the camera tripod / near table edge is at -y.

## Joint mapping (v3.0 `so101_follower` calibration)

`state = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] deg, gripper 0..100`;
sim joints `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]` rad:

```
sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])
SIGNS       = [-1,  +1,    +1,    +1,   +1   ]
OFFSETS_DEG = [ 0, -86.6, +84.1, +8.8, 153.6]
Jaw         = -0.174 + gripper/100 * 1.924
```

Started from the pickplace twin's fitted v3.0 mapping (`[-14.8, -92.4, +81.9, +6.0, +153.6]`); lift / elbow /
wrist offsets refined by Nelder-Mead on the top-camera silhouette IoU (18 frames; `out/offsets_fit.json`).
The pan offset is kept at 0: it is fully degenerate with the top camera's yaw and the shoulder camera's mount yaw
(both fitted), so any real pan offset is absorbed there. The wrist-roll offset (153.6) was not refitted (the roll is
nearly constant, -59..-43 deg, in this data; the fit is insensitive to it). Checked visually on 7 frames
(`out/joints_check.png`) and in the pairs sheet: rest pose folds back like the real one, pan toward the near table
edge (+65 deg) is the right way round, reaches to the pole / ring match.

## Cameras (fitted by white-silhouette IoU, 640x480; pinhole, no distortion)

Real "white" mask = HSV `S < 110 & V > 150` (top; the white balance swings between episodes, so loose) /
`S < 70 & V > 150` (side); sim mask = segmentation render of the arm's white visual geoms + pole (+ pole base
disc for the top view). 400-sample random box search + Nelder-Mead at 320x240, IoU reported at 640x480,
frames by farthest-point sampling in joint space among usable frames.

| cam  | frame                                    | pos (m)                | yaw / pitch / roll (deg) | fovy (deg) | mean IoU (18 frames)             |
| ---- | ---------------------------------------- | ---------------------- | ------------------------ | ---------- | -------------------------------- |
| top  | world (static overhead)                  | (0.179, -0.157, 0.727) | 69.1 / -79.7 / 22.8      | 34.1       | **0.454** (per frame 0.37..0.53) |
| side | `pan_frame` (rotates with the pan joint) | (0.051, -0.097, 0.058) | 21.5 / -5.7 / -3.7       | 21.2       | **0.319** (per frame 0.00..0.91) |

`pan_frame` is a body under `Rotation_Pitch` with its origin at the pan axis, x = arm forward, z = up; the
side camera therefore sits ~5 cm forward, ~10 cm to the robot's right and ~6 cm above the pan axis, looking
~22 deg inward (toward the arm's centre line) and slightly down with a narrow lens (fovy 21 deg, i.e. the real
"sholder" camera is a telephoto-ish module: the 2 cm pole fills 80 px at ~30 cm). The yaw / lateral offset pair is
partly degenerate at these distances, so the position is less certain than the view direction.

Why the IoUs are modest: the top mask is limited by the real arm's wires and bulkier SO-101 housings (the twin uses
the Menagerie SO-100 meshes), glossy table reflections that read as white in some episodes (frames with > 20 % white
were excluded from the fit), and the orange-tinted episodes. The side view is a thin pole most of the time (an IoU
knife edge: the fit uses masks dilated by 6 px and only pole-dominated untinted frames; several fit frames score 0
because the real pole is just out of view or the per-episode pole position is a few cm off), and the arm appears
only as blurred over-exposed close-ups that threshold badly. Visually (`out/fit_side.png`, `out/pairs_sheet.jpg`) the
pole, ring and gripper land where the real ones are.

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes, white body, black motors (same link lengths as the SO-101, different housings).
- Table: dark wood box (lighter near edge strip at -y, beige wall strip beyond +y), orange emissive backdrop wall at
  x = 0.75 (the real backlit curtain), warm fill light. The camera tripod, cables, and the desk / audio interface
  seen by the shoulder camera at pan > 45 deg are not modelled.
- Pole: white cylinder r = 1.1 cm, h = 19 cm, on a purple disc base (r 4 cm, 1.2 cm) — the base is hidden in the
  first 7 episodes of the set (dataset episodes 4..13), where the real pole stands bare.
- Ring: orange flat torus (24 thin boxes, r_out 3.2 cm, r_in 1.2 cm, h 2.2 cm). All 40 episodes' first frames show an
  orange ring, so the twin uses orange throughout (`RING_COLORS` has pink / yellow if ever needed).
- **Per-episode layout**: pole and ring (x, y) detected automatically in each episode's first real top frame
  (`place_objects.py`: ring = compact bright orange blob; pole = lavender / white disc or bare rod away from the
  arm) and ray-cast onto the table through the fitted top camera. Detection succeeded for all 40 episodes
  (`out/objects_detect.jpg`); accuracy is a few cm (the rod's base end is the point nearest the image centre, the
  disc centroid is biased by the pole's projection).
- Ring during replay: on the table at its start position for `phase == approach`, 1.2 cm below the `tcp` site of
  the fixed jaw for `grasp` / `transfer_open`, on the pole base for `release`. The phase label is
  gripper-threshold based, so the ring can be "in the jaws" a little early / late relative to the real frames.
- Exposure: per-camera brightness gain top 1.09, side 1.12 (twin mean intensity ~117 / 122 vs real 118 / 128);
  no white-balance / tint matching (the real set alternates between neutral and strongly orange-tinted episodes),
  no blur / noise.

## Not done / could be improved

- Per-frame ring placement in the shoulder view (the ring is inferred from the phase label, not tracked).
- The SO-101 housings, wires, the camera mount on the shoulder, lens distortion of the shoulder camera.
- A joint offsets + both-camera alternating refit (only the top camera took part in the offsets round); another
  side-camera round with better pole positions (e.g. from the side view itself) would raise its IoU.
- Tint-matched rendering per episode (the tinted episodes would pair better with a warmer render).
