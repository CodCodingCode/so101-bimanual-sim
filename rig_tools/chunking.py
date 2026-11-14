"""Action chunking, ACT-style: the policy predicts the next K tool-point targets per arm every tick; every tick is driven
by a weighted average of all chunks that cover it (temporal ensembling), which smooths the motion. Shared by the sim
trainer (squeeze/chunk_policy.py) and the real runner (real_policy.py), and the same format as the sim-74 wide-clamp
student:

    chunk: (K=8, 2 arms, 3) tool-point targets RELATIVE to the commanded tool point at prediction time, metres x SCALE
    execution: average the targets of every chunk covering this tick, weights exp(-M * i) with i = 0 for the OLDEST
               chunk (ACT), then step toward that target with the training's per-tick cap.
"""
from __future__ import annotations

from collections import deque

import numpy as np

K = 8
SCALE = 20.0
M = 0.1


class ChunkEnsembler:
    def __init__(self, k=K, m=M, scale=SCALE):
        self.k, self.m, self.scale = k, m, scale
        self.chunks = deque()          # (t0, ee_at_t0 (2, 3), offsets (k, 2, 3) in metres)

    def reset(self):
        self.chunks.clear()

    def add(self, t, ee_now, chunk_scaled):
        self.chunks.append((t, np.asarray(ee_now, float).copy(), np.asarray(chunk_scaled, float).reshape(self.k, 2, 3) / self.scale))
        while self.chunks and self.chunks[0][0] + self.k <= t:
            self.chunks.popleft()

    def target(self, t):
        """Ensembled absolute tool-point target (2, 3) for tick t (None if no chunk covers it)."""
        ts, ws = [], []
        for i, (t0, ee0, off) in enumerate(c for c in self.chunks if c[0] <= t < c[0] + self.k):
            ts.append(ee0 + off[t - t0])
            ws.append(np.exp(-self.m * i))
        if not ts:
            return None
        w = np.asarray(ws) / np.sum(ws)
        return np.tensordot(w, np.asarray(ts), axes=1)


def chunk_labels(actions, step_m, k=K, scale=SCALE):
    """Chunk labels from a sequence of per-tick actions (T, 6) in [-1, 1] that move each tool point by action * step_m:
    label[t] = cumulative tool-point offsets of the next k actions (held at the end of the episode), x scale."""
    A = np.asarray(actions, float).reshape(len(actions), 2, 3) * step_m
    T = len(A)
    out = np.zeros((T, k, 2, 3))
    for t in range(T):
        seq = A[t:t + k]
        if len(seq) < k:
            seq = np.concatenate([seq, np.zeros((k - len(seq), 2, 3))])
        out[t] = np.cumsum(seq, axis=0)
    return (out * scale).reshape(T, -1).astype(np.float32)


def action_toward(target, ee_now, step_m):
    """Per-tick action in [-1, 1] (as the policy's training) that moves ee_now toward the ensembled target."""
    return np.clip((np.asarray(target) - np.asarray(ee_now)) / step_m, -1, 1).reshape(-1).astype(np.float32)
