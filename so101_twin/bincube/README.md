# so101_bincube_twin — MuJoCo digital twin of `so101_bincube_real`

Rendered twin of the 320 frames (40 episodes x 8) of `neuroscope/molmoact2/store/sets/so101_bincube_real`
(LeRobot v2.1 dataset `data_so101/MCeut__so101_lab_openBin_putCube`, task "Open bin and put cube"). Each twin frame
shares the real frame's joint state, the fitted camera poses and an approximate scene layout and differs only in
rendering. Built from the `so101_twin/lemon/` template (same scripts, adapted).

Output: `neuroscope/molmoact2/store/sets/so101_bincube_twin/` — exactly the real set's layout: `meta.json`,
`images_top.npy`, `images_side.npy` (uint8 [320,480,640,3]), `state.npy` and `prompt.json` byte-identical to the
real set, `labels.npz` = the real labels with `real_or_sim='sim'`, `source='bincube_twin'` and `pair = arange(320)`
(index into the real set). `meta.json` has `twin_of: "so101_bincube_real"`, `cameras: ["top", "side"]`, the joint
mapping, the fitted camera parameters (`cameras_fit`), per-camera IoU (`camera_fit_iou`), gains and the lid/cube rule.

## IMPORTANT: the real "top" camera is a wrist camera

Despite the set's camera names, `observation.images.front` ("top") is a camera mounted on the gripper: both jaws
are visible at the bottom of every frame and the view moves with the arm. `observation.images.left` ("side") is the
static camera (a view from in front-left of the robot looking back and down at the arm, bin and cube). The twin
therefore attaches the sim "top" camera to the `Fixed_Jaw` body (its parameters are in that body's frame) and fits
it on the jaw silhouettes; "side" is a world camera.

## Files

| file                 | what                                                                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_scene.py`     | single black SO-101 arm scene -> `so101_bincube.xml` (board z=0, base at origin reaching +y), CAMS / BIN / LID / CUBE parameters at the top               |
| `twin.py`            | `real_to_sim()` joint mapping, `Sim` (offscreen renderer, segmentation arm mask, runtime camera override in the parent-body frame)                        |
| `check_joints.py`    | real vs sim side-by-side for a few frames -> `out/joints_check.png`                                                                                       |
| `fit_cameras.py`     | silhouette-IoU camera fit (`--cam top/side`, `--local`), `--offsets` (not run), `--apply` writes CAMS into build_scene.py                                 |
| `seed_side_pnp.py`   | side-camera seed from 2D-3D correspondences (cube-in-jaws pixel vs sim TCP, base pixel) -> `out/side_pnp.json`                                            |
| `place_objects.py`   | bin (x,y) from a hand-picked rim pixel and per-episode cube (x,y) from the yellow centroid, ray-cast through the fitted side camera -> `out/cube_xy.json` |
| `check_roll.py`      | wrist-roll sign/offset grid by projecting the table cube into the wrist view -> `out/roll_check.json`                                                     |
| `brightness_gain.py` | per-camera gain so the twin's mean intensity matches the real exposure                                                                                    |
| `replay.py`          | replays all 320 real states (lid + cube by a gripper state machine) and writes the twin set; `out/twin_states.npz` = lid/cube state per frame             |
| `make_sheet.py`      | `out/pairs_sheet.jpg` (8 random pairs: real top / twin top / real side / twin side)                                                                       |
| `out/`               | `pairs_sheet.jpg`, `fit_top.png`, `fit_side.png` (real / sim / mask overlay), `camera_fit_*.json`, `cube_xy.json`, logs                                   |

Run order: `build_scene.py` -> `fit_cameras.py --cam top` -> `seed_side_pnp.py` (paste into CAMS["side"]) ->
`fit_cameras.py --cam side --local` -> `--apply` -> `place_objects.py` (paste BIN xy) -> `check_roll.py` (paste the
roll offset into twin.OFFSETS_DEG) -> `brightness_gain.py` (paste into replay.GAIN) -> `replay.py` -> `make_sheet.py`.
All with the root `.venv/bin/python` (MuJoCo 3.3.0).

## Joint mapping

`state = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] deg, gripper 0..100` (LeRobot v2.1 recorded
with the new `so101_follower` calibration; rest pose about [-4, -94, 75, 72, -48, 29]). Menagerie `so_arm100` joints
`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` (rad):

```
sim_rad[j] = deg2rad(SIGNS[j] * state[j] + OFFSETS_DEG[j])
SIGNS       = [-1,  +1,  +1,  +1,   +1 ]
OFFSETS_DEG = [ 0, -90, +90,   0, +105 ]
Jaw         = -0.174 + gripper/100 * (1.75 + 0.174)
```

This is the pickplace-style mapping (same calibration) with the derived first-guess offsets: the rest pose lands on
the Menagerie `rest` keyframe (Pitch -184 vs -190 deg, Elbow 165 vs 178, Wrist_Pitch 72 vs 63). Verified visually
(`out/joints_check.png`, `out/pairs_sheet.jpg`: folding, reaching to the cube left/right of the bin, lifting over the
bin all match the real side view) and by the side silhouette fit. The pan/lift/elbow/wrist offsets were NOT refined
(`--offsets` not run, time box); the pickplace values (-14.8, -92.4, +81.9, +6.0) are of the same order.
Wrist roll is the uncertain joint: the wrist camera rides on the Fixed_Jaw body so the jaw silhouette cannot see the
roll, and the side silhouette is insensitive to it (IoU spread 0.004 over all sign/offset candidates). The offset
+105 (sign +1) is the best of a 15-degree grid by projecting each episode's table cube into the wrist view
(`check_roll.py`: median error 198 px vs 262 px for the pickplace-like +180; all candidates are poor, so treat the
world's orientation in the wrist view as approximate).

## Cameras (fitted by arm-silhouette IoU, 640x480 pinhole, no distortion)

| cam  | frame            | pos (m)                   | yaw / pitch / roll (deg) | fovy (deg) | mean IoU 640x480 (18 frames)     |
| ---- | ---------------- | ------------------------- | ------------------------ | ---------- | -------------------------------- |
| top  | `Fixed_Jaw` body | (0.0073, 0.0098, -0.0607) | -106.6 / 16.2 / 173.4    | 45.0       | **0.384** (per frame 0.03..0.74) |
| side | world            | (-0.176, 0.518, 0.383)    | -66.3 / -42.6 / -8.2     | 36.3       | **0.332** (per frame 0.22..0.54) |

Masks (the arm is black): real = `max(RGB) < 70`; side: minus the truly static dark background (pixels dark in

> = 97% of frames: breadboard, pegboard, base/controller, bin interior), both masks evaluated outside that region;
> top: dark connected components touching the bottom image edge (the two jaws), gated to the region the jaws sweep.
> Sim = segmentation render of the arm's visual geoms (side: without the static Base body). 18 frames by farthest-point
> sampling in joint space; box random search at 160x120 / 320x240 then Nelder-Mead from the best 4-5, IoU at full
> resolution. The side fit was seeded by `seed_side_pnp.py` (81 correspondences, base pixel (345,172)) after the
> box search kept locking onto a close-up of the gripper.

Why the IoU is below 0.5: (top) the jaws are thin wedges, so a few pixels of misalignment cost a lot, the real jaw
mask is a sliver when the gripper is closed on the cube (IoU 0.03-0.2 there), and the real camera has visible
barrel distortion; (side) the real mask also contains the black lid, the bin interior when the lid moves, cables and
the board-edge lines, which no arm mask can match, and the real SO-101 base/shoulder housing is bulkier than the
Menagerie SO-100 mesh. The side camera that the fit converges to (0.65 m away, 43 deg down, 36 deg fovy) places the
arm, board and breadboard edges where the real ones are (`out/fit_side.png`).

## Scene / rendering (approximate)

- Arm: Menagerie SO-100 meshes, all black (printed parts and motors; the real jaws are black too). Base at the origin
  on the board (the real arm sits on an aluminium rail clamp and has a controller box next to it: not modelled).
- Table: light-wood board (0.60 x 0.90 m, by eye) on a black optical breadboard; brown wall/box behind the robot.
  The real bags, cables and pegboard are not modelled.
- Bin: grey open box 9.4 x 7.4 x 12 cm at (-0.013, 0.299) (ray-cast from the rim pixel in frame 0; one position for
  all episodes although the real bin moves by a few cm between episodes). Lid: flat dark box 10.4 x 8.4 x 1.4 cm.
  **Closed** = on top of the bin; **open** = flipped upside down onto the table beyond the bin's edge in direction
  `replay.LID_OPEN_DIR` = (0.97, -0.26) (image-left of the bin in the side view, where it lies in most real
  episodes; in a few episodes, e.g. 33, the real lid lies to the right of the bin instead).
- Cube: yellow 4 cm box. Start position per episode from the yellow centroid of the first table frame in the side view
  (`out/cube_xy.json`; 38/40 episodes, mean (0.00, 0.16), x in [-0.14, +0.14]); episodes without a usable frame use
  (0.00, 0.163).
- Lid / cube states per frame (`replay.episode_states`): the phase label is gripper-threshold based and mixes the lid
  pinch with the cube grasp, so a small state machine on the gripper reading is used instead: gripper < 12 = pinching
  the thin lid handle (first pinch run: lid closed for its first two frames = grab + lift, open afterwards; episodes
  with no pinch start open; a second pinch run late in the episode = closing the lid again), gripper entering 15..26.5
  from outside while the lid is open = cube between the jaws (site `tcp` on Fixed_Jaw), gripper opening again = cube in
  the bin. 100 closed-lid frames, cube table/jaws/bin = 165/83/72. Frames are ~2.3 s apart, so the transitions are
  only right to within a frame (e.g. the cube may be shown on the table while the real jaws are already around it).
- Exposure: per-camera gain top 1.035, side 0.761; no colour matching, noise or blur.

## Not done / could be improved

- Joint-offset refinement (`fit_cameras.py --offsets`) and a better wrist-roll determination (fit the wrist camera
  pose and the roll jointly on the cube/bin positions in the wrist view).
- Per-episode bin position; the lid's real hinge geometry (it swings about the rim and partly leans on the table).
- Lens distortion of both real cameras (the wrist camera is clearly wide-angle with barrel distortion; the twin uses
  a 45 deg pinhole fitted on the jaws only, so the world looks narrower than in the real wrist view).
- The real side view has the arm's cables, the rail and the controller box in the silhouette; none are modelled.
