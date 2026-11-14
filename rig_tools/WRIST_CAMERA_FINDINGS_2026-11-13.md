# Wrist cameras in the sim don't match the real ones (2026-11-13)

From inference-testing-04 (MolmoAct2 bimanual sim-to-real). The user asked for every sim session to switch training
to the corrected wrist cameras once a validated rig file exists. **No validated file exists yet**; this note says what
is known and what is still being checked.

## What is settled

- **All three real cameras are the same model:** InnoMaker U20CAM-1080P-S1 (user). The datasheet gives 130° diagonal
  and 103–106° horizontal, but that is the full 16:9 sensor at 1920×1080.
- **The real rig captures at 640×480.** Each camera was opened on dzkchen tonight the way the runner opens it (dshow,
  MJPG): all three return 640×480.
- **640×480 is the centre 4:3 crop of the sensor, not a squeeze.** In the right wrist camera at 640×480, 1280×720 and
  1920×1080, scaled to the same height, the gripper jaws are identical in size and position; the wider modes only add
  picture at the sides. Pixels are square, so fx = fy.
- **The current sim wrist lenses are wrong.** In `so101_camera_rig.json` (and `calibrated_full`) the wrists are the
  11-10 jaw-only fit, labelled Arducam IMX708: fx 466/502, k1 = +0.19/+0.22 (pincushion), 66°/61° horizontal. The
  real lens is a wide barrel lens. The overhead (same model, fitted from real photos) is fx 612.6, k1 −0.93,
  k2 +0.94: **67° horizontal, 50° vertical, 78° diagonal at 640×480.**
- **This is the main reason the real policy misses the cube.** We replayed the 13-11 real runs through the sim twin
  (same joint readings, cube where the overhead saw it):
  - The overhead matches to about 2 px.
  - The wrist view does not: the sim cube is about 2× the real cube's area and offset 30–100 px (on the 448 px image).
  - The policy aims 3.6 cm from the cube on real frames, versus 1.9 cm on the twin frames.
  - The adapter (per-token) can't fix a geometric difference.

## Being checked now (H100, ~/bimanual/rigfit/wristfit_spec.py)

- For each candidate lens, refit each wrist camera's mount pose (3 position + 3 rotation) to the real gripper-jaw SAM
  masks of the 11-10 calibration runs, rendering **through pi_sim's own lens path** (render_plan + remap).
- Candidates:
  - InnoMaker-style equidistant lens, 4:3 crop, at 90/95/100/103/106° full-sensor horizontal (67–79° at 640×480).
  - `oh`: the overhead's measured lens copied to both wrists, which is the most direct choice since it's the same
    model.
- First results (left wrist, held-out jaw IoU):

  | Lens           | Held-out jaw IoU |
  | -------------- | ---------------- |
  | Current rig    | 0.82             |
  | 100° candidate | **0.84**         |
  | 103° candidate | **0.85**         |

  The camera moved about 1 cm and 2–4°.

- The jaw IoU barely discriminates between lenses, because the jaws are near the lens at the image bottom. **The
  deciding test is the cube:** render the 13-11 real poses with the candidate and require the sim cube in the wrist
  view to come out about 1.0× the real area and within a few px.
- Note: `rig_tools/data/so101_camera_rig.innomaker_fit.json` (squeeze/fit_wrist_innomaker.py) reports jaw IoU
  0.52–0.56 with k1 −1.17/−1.30 and right fx 754. That's well below the 0.72/0.82 of the current rig, so please don't
  train on it without a cube-level check.

## When validated

I'll write `rig_tools/data/so101_camera_rig.innomaker_validated.json` (and the H100 copy
~/bimanual/sim/pi_sim/assets/so101_camera_rig.innomaker_validated.json) and update this note with the numbers. Then:

- Switch training to it: re-render any image data.
- Re-check any real↔sim pairs or adapters built with the old wrists: they were trained on the wrong wrist geometry.

I'm not editing anyone's existing rig files or training configs.
