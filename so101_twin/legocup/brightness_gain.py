"""Per-camera brightness gain so the twin's mean intensity matches the real camera exposure (printed; paste into replay.GAIN)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import replay  # noqa: E402
from twin import CAMS, Sim, real_set  # noqa: E402

fs = real_set()
sim = Sim()
idx = np.linspace(0, fs.n - 1, 12).astype(int)
for c in CAMS:
    real = np.mean([fs.obs(int(i))["images"][c].mean() for i in idx])
    replay.GAIN[c] = 1.0
    simm = np.mean([replay.render_frame(sim, fs.state[i], str(fs.labels["phase"][i]), int(fs.labels["episode"][i]))[c].mean() for i in idx])
    print(f"{c}: real mean {real:.1f}, sim mean {simm:.1f}, gain {real / simm:.3f}")
sim.close()
