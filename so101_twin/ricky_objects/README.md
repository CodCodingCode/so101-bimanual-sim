# ricky_objects: scene objects for so101_ricky_twin2

The twin set `neuroscope/molmoact2/store/sets/so101_ricky_twin2` had the arm fitted (fitkit, frozen) but a placeholder
scene (black table, wall behind the camera, no objects). This folder adds the scene and rewrites only
`images_top.npy` / `images_side.npy`. Arm, cameras, lens, lag and joint map are the frozen fit
`so101_twin/fitkit/out/ricky_so101_sess.json` (per-session views top/top_b/top_c, side/side_b chosen by episode exactly as
twin2.py does).

## What is modelled (world = twin arm base frame, base bottom at z = 0)

- Wooden table: box, top z = 0, centre (0.176, 0.213), half (0.513, 0.268) m, yaw 5.5 deg. Rectangle IoU fit to
  wood|mat colour masks of the five per-session median frames (rays cast onto z = 0): IoU 0.86-0.88 top, 0.95 side.
- Dark mat: 2 mm box, top z = 0.0005, centre (0.016, 0.265), half (0.248, 0.200) m, yaw 5.5 deg (IoU 0.83 top, 0.94 side).
- Walls: back wall through the table's far edge, side wall through the room corner seen by the top camera; both stop at
  z = 1.0 so the placeholder key light still lights the arm exactly as before; nearly unlit material (no spot-cone ellipse).
- Chair back (top view only, behind the side camera): box from its image box cast onto y = -0.25.
- Bowl ("plate"): truncated-cone paper bowl, r_bottom 4.9 cm, r_top 6.2 cm, height 4.7 cm, 2.5 mm wall, navy outside,
  white inside/rim. Static per episode.
- Block ("toy"): box 2.7 x 2.9 x 2.2 cm, red-brown (per-camera colour).
- Colours per camera (`work/colours.json`): real target colour / rendered colour of a white material per region.

## How objects are placed

- Grounding-DINO ("a white bowl." / "a small brown block.") + SAM 2.1 on each episode's first frame, both cameras
  (`segment_objects.py`, MolmoAct2 venv, never the policy).
- `fit_objects.py`: convex silhouettes of the models projected through the frozen cameras (`camgeo.py` = twin2's exact
  pinhole + radial lens model, verified to 1 px against the twin arm mask), IoU to the SAM masks in both views jointly;
  shapes and one support-plane height shared by all episodes, (x, y[, yaw]) per episode.
- Block by phase (`block_state.py`): block detections in all 320 frames x 2 cameras vs three hypotheses (table pose /
  twin jaw tips / inside the bowl), monotone table -> held -> plate per episode, open gripper cannot hold.
  Result: 180 table, 68 held, 72 plate. Held = between the twin jaw tips (midpoint of fixed and moving jaw tips, 1.4 cm
  back), plate = on the bowl floor at the side-view detection.

## Residual placement error (episode first frames, model silhouette vs SAM mask)

| object | camera | IoU median (min) | centroid px median (max) | cm median |
| ------ | ------ | ---------------- | ------------------------ | --------- |
| bowl   | top    | 0.833 (0.815)    | 11.2 (13.6)              | 0.44      |
| bowl   | side   | 0.861 (0.825)    | 4.9 (6.4)                | 0.40      |
| block  | top    | 0.710 (0.635)    | 6.7 (13.9)               | 0.26      |
| block  | side   | 0.830 (0.758)    | 1.2 (5.0)                | 0.09      |

Single-camera fits disagree by 2.0 cm (bowl) / 1.9 cm (block) median: the frozen top camera sits 10 cm above the
table and sees it at a grazing angle; its object silhouettes come out taller than the real ones (bowl and block look
slightly too tall in the top view). The two views agree best with the objects standing on z = 6.4 mm (they float
6 mm above the mat); the table/mat stay at z = 0 so the arm base keeps its fitted pixels. Not fixable without
refitting the cameras.

## Arm identity check (`check_identity.py`, 20 frames x 2 cameras, `out/arm_identity_check.json`)

- Arm-only silhouette re-rendered with the frozen fit equals the stored `arm_mask_*` in 38/40; the other 2 differ only
  by jaw-tip pixels below z = 0 that the placeholder table already hid (stored is a subset of arm-only, 0 px outside).
- Arm pixels in the new images are a subset of the arm-only silhouette (0 px outside). Arm pixels visible before but
  hidden now: 11278 px of 2.61 M (0.43 %): 10683 by the block in front of the jaws, 552 by the 0.5 mm mat edge at
  the base / jaw tips on the table, 43 by arm-arm draw-order ties.
- RGB on arm pixels visible in both (eroded 2 px): >= 99.93 % identical per frame (max diff 66 on a few pixels: shadows
  of the new objects). The re-render equals the written images in 40/40.
- state.npy, prompt.json, labels.npz, arm_mask_*.npy byte-identical (md5 in backup/checksums_before.txt); meta.json
  differs only in "objects" (arm_iou, views, joint blocks byte-identical).

## Files

- `backup/images_{top,side}.npy`, `backup/meta.json.orig`: the set before this change.
- `out/before_after.jpg`: 8 frames from different episodes, real | old twin | new twin per camera.
- `build_ricky.py` (calib | render | preview), `finalize_ricky.py` (install | meta), `objtwin.py` (Twin2 + scene hook,
  statistic pinned), `fitkit_copy/` (frozen copies of fitkit code; reproduces the old twin byte-identically, see
  `check_repro.py`). `work/`: medians, fits, masks, states, colours.

## Still off

- Top view: bowl and block slightly too tall (camera inconsistency above); no printed text on the bowl, no mat logo.
- Side view: the arm's power cable and wrist camera are not modelled; the floor/skirting at the far left is wood.
- Block state is decided per frame from detections; a few frames around grasp/release can be one phase off.
