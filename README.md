# sim

Real-to-sim twin of the bimanual SO-101 bench. Code moved here from `../inference-testing/so101_twin` on 2026-11-10.

- `so101_twin/` — twin code: `common/arm.py` (official SO-101 MJCF builder), `common/fit.py` (camera/lens/latency/joint
  fit to arm masks), `common/segment.py` (SAM masks), per-dataset twins (`pos5/`, `lemon/`, ...), `replay_real.py`.
  Fitted parameters (`*/out/*.json`) are included; renders, masks and npz outputs are not.
- `SIM2REAL_DIGITAL_TWIN_PLAN.md` — the plan this serves.

Setup: the code resolves data relative to this folder, so link the big folders once:

    for d in neuroscope data_so101 data_real yam_sim .venv; do ln -sfn ../inference-testing/$d $d; done

Live real rig: `ssh ... -L 8104:127.0.0.1:8104 dzkchen` exposes the read-only SO-101 monitor at
`http://localhost:8104` (`/api/state`: joints in degrees, both arms; `/cam/...`: overhead, left_wrist, right_wrist).

Known leftover: some per-scene `sam_wait.sh` / `final.sh` scripts hard-code `../inference-testing/...` paths.

## Two-arm clamp: sim, RL, real rig (added 2026-11-11)

- `pi_sim_so101/` — MuJoCo sim of the two SO-101 arms matched to the real bench (232 mm spacing, calibrated cameras,
  fitted servo dynamics, CoACD gripper collision, wood table).
  - `squeeze/` — two-arm tip clamp with no jaws: both grippers press the cube's sides, lift, carry and drop it into
    the box. Scripted teacher (`clamp_teacher.py`), RL env (`clamp_rl.py`), PPO (`train.py`), recovery by DAgger
    (`dagger.py`), camera-policy distillation (`distill.py`). Final models with scores are in `squeeze/models/`.
- `rig_tools/` — real rig: station web UI (moved to `../so101/ui/rig_ui.py` with the portal, 2026-11-14), scripted real clamp runner (`real_clamp.py`), camera and
  joint calibration, dynamics fitting.
- `laptop_ik/` — IK helpers for the robot laptop.
