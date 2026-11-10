# pickplace_objects: re-placed objects for so101_pickplace_twin2

`neuroscope/molmoact2/store/sets/so101_pickplace_twin2` showed the builder's box and brick placed through the
builder's OLD cameras (visibly off). This folder re-places the same object models (so101_twin/pickplace/build_scene.py:
transparent box with an orange rim, 2x4 pink brick) through the NEW fitted cameras
(`so101_twin/fitkit/out/pickplace_so101_sam1.json`, frozen) and rewrites only `images_top.npy` / `images_side.npy`.
The builder scene (table, lights, skybox) is otherwise kept; `fitkit_copy/` reproduces the old twin byte-identically
(`check_repro.py`).

## What is modelled (world = twin arm base frame, base bottom at z = 0)

- Box: builder model, outer size fitted to 15.1 x 10.9 cm (builder guess 13 x 11), height 6 cm (builder), rim 6 mm.
- Brick: builder model (3.2 x 1.6 x 1.1 cm + studs).
- Table: builder table kept (top z = 0), recoloured per camera.
- Back wall moved from y = 1.0 to the table's far edge y = 0.10 (the real top view shows the wall right behind the
  box); top at z = 1.0 so no light is blocked; nearly unlit material.
- Shipping boxes (side camera only, geom group 4): the brown boxes at the top right of the side view, placed from
  the side-view median (front face x = 0.27, bottom on z = 0). Through the frozen top camera such boxes would show at
  the top right of the top view, where the real top frames have none (the two frozen cameras disagree this far from
  the arm), so they are drawn for the side camera only.
- Per-camera colours (`work/colours.json`); thin rim and brick colours sampled from the real orange/pink pixels.

## How objects are placed (`fit_objects.py`, `block_state.py`)

- Box: orange colour mask (twin arm silhouette removed) on each episode's first frame in both cameras; (x, y, yaw) per
  episode and the shared size by a one-sided chamfer (every detected rim pixel to the model rim) in both views.
- Brick on the table: pink mask centroids of both cameras triangulated (two-ray least squares) -> (x, y, centre z)
  per episode; yaw from the top-view IoU; top-view ray at the median height where the side camera misses it (9 eps).
- Brick by phase: Grounding-DINO "a small pink lego brick." + SAM on all 300 frames x 2 cameras vs the three
  hypotheses (table pose / twin jaw tips / inside the box), monotone table -> held -> plate per episode, open gripper
  cannot hold: 129 table, 81 held, 90 in the box. Held = between the twin jaw tips; in box = on the box floor.

## Residual placement error (episode first frames)

| object  | camera | metric                 | median              | max                   |
| ------- | ------ | ---------------------- | ------------------- | --------------------- |
| box rim | top    | chamfer px (cm)        | 0.52 (0.05)         | 0.82                  |
| box rim | side   | chamfer px (cm)        | 1.08 (0.08)         | 1.56                  |
| brick   | top    | IoU / centroid px (cm) | 0.754 / 1.96 (0.18) | min IoU 0.53 / 6.7 px |
| brick   | side   | IoU / centroid px (cm) | 0.752 / 4.29 (0.31) | min IoU 0.0 / 21.8 px |

Two-view brick rays pass 2.9 mm (median, max 7.9 mm) from the triangulated centre. The triangulated centres sit about
1 cm higher than a brick lying on z = 0; the box and bricks stand on z = 1.05 cm (bricks at their own triangulated
height, clipped to >= half height) so both views match; the table stays at z = 0 so the arm base keeps its pixels.

## Arm identity check (`check_identity.py`, 20 frames x 2 cameras, `out/arm_identity_check.json`)

- Arm-only silhouette re-rendered with the frozen fit: stored `arm_mask_*` is a subset of it in 40/40 (0 px outside);
  equal in 19/40, the rest differ by arm pixels the builder's old box/brick covered.
- Arm pixels in the new images are a subset of the arm-only silhouette (0 px outside). Visible before but hidden now:
  33562 px of 1.27 M, all by the new box glass walls / brick in front of the arm (plus 6 px arm-arm ties); 96 px visible
  now that the old box covered.
- RGB on arm pixels visible in both (eroded 2 px): >= 99.86 % identical per frame (max diff 25: glass in front, shadows).
  Re-render equals the written images in 39/40; the 40th (frame 0 top, the first render of a fresh renderer) differs
  by 1 grey level on 3 pixels (GPU first-frame effect, also present in the original pipeline).
- state.npy, prompt.json, labels.npz, arm_mask_*.npy byte-identical; meta.json differs only in "objects".

## Files

- `backup/`: images and meta.json before this change, md5s. `out/before_after.jpg`: real | old twin | new twin.
- `build_pickplace.py` (calib | render | preview), `finalize_pickplace.py` (install | meta), `objtwin.py`.

## Still off

- The builder's glass (alpha 0.14, specular) is more visible than the real box walls.
- Brick side-view centroid error has a tail (max 21.8 px) on a few episodes where the side detection is partial.
- Table far edge on the right of the top view and the wall/cardboard depth are approximate (low contrast, the two
  cameras disagree far from the arm). No LeRobot sticker / wiring detail.
