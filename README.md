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
