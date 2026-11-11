# Trained models: two-arm tip clamp into the box (SO-101, MuJoCo)

| file | what | inputs | sim result |
|---|---|---|---|
| `clamp_base.zip` + `clamp_base_obs_stats.npz` | PPO (BC warm start), no disturbances | sim state (81-D) | 98% success, plain task |
| `clamp_recovery.zip` + `clamp_recovery_obs_stats.npz` | DAgger from the base, with nudges / slips / pushes | sim state (81-D) | 95% success, 82% contact-free, with disturbances |
| `camera_student.pt` | vision policy distilled from the scripted teacher (DAgger) | overhead + 2 wrist cams 64x64, joints, attempt clock | 93% success, 72% contact-free, with disturbances |

Load the state policies with `train.BCPPO.load(path)` and normalise observations with the stats file:
`clip((obs - mean) / sqrt(var + 1e-8), -10, 10)`. Load the camera student with `distill.build_student()` plus
`load_state_dict(torch.load(path))`.

Retrain: `train.py` (demos / bc / ppo), `dagger.py` (recovery), `distill.py` (camera student). Watch: `watch_clamp_task.py`, `record_student.py`.
