# Final models (copied out of the untracked runs_* folders)

All numbers are sim results from the last line of each run's log, not real-rig results.

| File | Source run | What it is | Last sim result |
|---|---|---|---|
| clamp_base.zip | runs_clamp/ppo | First BC→PPO clamp teacher (state) | — |
| clamp_recovery.zip | runs_clamp_recovery | Recovery teacher (nudges, slips, pushes) | — |
| camera_student.pt | runs_clamp_recovery/student | Camera student distilled from clamp_recovery | — |
| camera_student_v3.pt | runs_clamp_v3/student | Camera student, v3 physics | — |
| tri_policy.pt | runs_clamp_v3/tri_policy | Detector-input policy | 0.97 success / 0.66 clean |
| grip_detector.pt | runs_clamp_v3/grip_detector | Detector-input grip policy | 0.92 success |
| chunk_policy_v2.pt | runs_clamp_v3/chunk_policy_v2 | Action-chunk policy | 0.69 delivered / 0.62 clean |
| clamp_wide_guard4.zip (+ _obs_stats.json, _vecnorm.pkl) | runs_clamp_wide_guard4/best | Wide-range curriculum PPO teacher | reached stage 5 (cube ±2.6 cm) |
| chunk_student_rotation_guard4.pt | runs_chunk/rotation_guard4_best | Chunk student, rotation task; best candidate for the real rig | 0.70 success; 38/45 clean with the rig controller + real-servo error |

Not included: the SmolVLA checkpoints (runs_vla/smolvla_*, about 1 GB each) and the 71 MB grip_teacher.
Status of the ±12 cm push-then-clamp task: not working yet. The scripted push is 3/83 clean in
runs/push_clamp_map_log.txt, and runs_chunk/pushclamp_guard4 stopped after round 0.
