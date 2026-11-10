"""GRAYBLOCK copy of so101_twin/jar/fullrate.py (unchanged except scene / set / mask names: grayblock, so101_grayblock_real). Full-rate (30 fps) LeRobot states around each real-set frame, for camera-latency compensation.

    from fullrate import FullRate
    fr = FullRate(fs)                       # fs = the real FrameSet; None-safe: fr.ok is False if data is missing
    q = fr.state_at(i, dt_frames=-2.0)      # LeRobot state of real frame i shifted by dt frames (linear interp.)

The real sets keep 8 frames per episode with labels episode / t (frame index); their state equals the dataset's
observation.state at that frame (checked on lemon). The parquet files are read from <repo>/<meta.dataset>/data/
(LeRobot v2.x layout data/chunk-XXX/episode_XXXXXX.parquet; v3.x file layout is tried too).
"""
from __future__ import annotations

import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


class FullRate:
    def __init__(self, fs):
        self.fs = fs
        self.ep = np.asarray(fs.labels["episode"]).astype(int)
        self.t = np.asarray(fs.labels["t"]).astype(int)
        self.seq: dict[int, np.ndarray] = {}
        self.ok = False
        ds = fs.meta.get("dataset")
        if not ds:
            return
        base = ROOT / ds / "data"
        if not base.exists():
            return
        import pyarrow.parquet as pq

        files = sorted(base.rglob("*.parquet"))
        need = set(self.ep.tolist())
        for f in files:
            tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
            epi = np.asarray(tb.column("episode_index").to_numpy())
            fi = np.asarray(tb.column("frame_index").to_numpy())
            st = np.asarray(tb.column("observation.state").to_pylist(), np.float32)
            for e in np.unique(epi):
                if int(e) in need:
                    m = epi == e
                    order = np.argsort(fi[m])
                    self.seq[int(e)] = st[m][order]
        self.ok = all(e in self.seq for e in need)
        if self.ok:   # sanity: the real set's state must equal the parquet state at (episode, t)
            err = max(float(np.abs(self.seq[e][t] - s).max()) for e, t, s in zip(self.ep, self.t, fs.state))
            self.ok = err < 1e-3

    def state_at(self, i: int, dt_frames: float = 0.0) -> np.ndarray:
        if not self.ok or dt_frames == 0.0:
            return np.asarray(self.fs.state[i], float)
        s = self.seq[int(self.ep[i])]
        x = np.clip(self.t[i] + dt_frames, 0, len(s) - 1)
        a = int(np.floor(x))
        b = min(a + 1, len(s) - 1)
        w = x - a
        return (1 - w) * s[a].astype(float) + w * s[b].astype(float)

    def speed(self, i: int) -> float:
        """joint speed |dq/dt| (deg per frame, 5 revolute joints, central difference)."""
        if not self.ok:
            return float("nan")
        s = self.seq[int(self.ep[i])]
        t = self.t[i]
        a, b = max(t - 1, 0), min(t + 1, len(s) - 1)
        return float(np.linalg.norm((s[b, :5] - s[a, :5]) / max(b - a, 1)))
