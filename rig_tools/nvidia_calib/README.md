# NVIDIA SO-101 calibration check (vendored)

`so101_check_calibration.py` and `calibration_stats.json` are copied unmodified from
https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop (`docker/real/scripts/`, commit ce807d9, 2026-07-20,
Apache-2.0, (c) NVIDIA). The procedure is described at
https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html

Do not run their script directly on this rig. Run `rig_tools/nvidia_calib_check.py`, which differs from it in two ways:

- Our arms keep their calibration in servo EEPROM, not in a lerobot JSON. The wrapper rebuilds the lerobot
  calibration file from the EEPROM.
- Their live check connects through lerobot's `connect()`, and that writes registers (including torque off). The
  wrapper feeds their `live_check` from a read-only bus instead.

Their `static_check`, `live_check` and `print_report` run unchanged.
