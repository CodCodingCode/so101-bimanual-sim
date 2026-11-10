"""pos5 helpers: real set, mask I/O, twin silhouettes for a params dict, IoU statistics (agreed metric).

Masks live in pos5/masks/pos5_<cam><tag>.npy as np.packbits(bool [320, 480, 640], axis=-1).
Twin silhouettes: ARM-ONLY segmentation render of the official SO-101, padded pinhole + lens warp onto the real
480 x 640 grid (arm.Lens), the arm posed with the latency-shifted full-rate state (fit.frame_state).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

CAMS = ("top", "side")
MASKS = HERE / "masks"
OUT = HERE / "out"


def scene():
    return F.load_scene("pos5")


def unpack(a):
    return np.unpackbits(a, axis=-1)[..., :640].astype(bool)


def load(cam: str, tag: str) -> np.ndarray:
    return unpack(np.load(MASKS / f"pos5_{cam}{tag}.npy"))


def save(cam: str, tag: str, m: np.ndarray) -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    np.save(MASKS / f"pos5_{cam}{tag}.npy", np.packbits(m.astype(bool), axis=-1))


def load_params(path) -> dict:
    j = json.loads(pathlib.Path(path).read_text())
    return j["params"] if "params" in j else j


def twin_masks(P: dict, cams=CAMS, frames=None, sc=None) -> dict:
    sc = sc or scene()
    P = F.reconcile(sc, P)
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    idx = range(fs.n) if frames is None else frames
    out = {}
    for c in cams:
        grp = F.frame_groups(sc, c)
        out[c] = np.stack([rig.mask(c, grp[i], F.frame_state(sc, P, c, i), 640, 480) for i in idx])
    return out


def iou(sim: np.ndarray, real: np.ndarray, excl: np.ndarray | None = None) -> np.ndarray:
    """per-frame IoU over [n, H, W]; with excl, those pixels are removed from both masks."""
    if excl is not None:
        sim, real = sim & ~excl, real & ~excl
    inter = (sim & real).reshape(len(sim), -1).sum(1)
    union = (sim | real).reshape(len(sim), -1).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def stat(v) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=float("nan"), p10=float("nan"), mean=float("nan"))
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                mean=round(float(np.mean(v)), 4))


def visible(real: np.ndarray) -> np.ndarray:
    return real.reshape(len(real), -1).mean(1) >= F.VISIBLE


def odd() -> np.ndarray:
    fs = F.real_set(scene())
    return np.asarray(fs.labels["episode"]).astype(int) % 2 == 1


def report(v: np.ndarray, vis: np.ndarray) -> dict:
    o = odd()
    return dict(held_out=stat(np.where(vis & o, v, np.nan)), all_frames=stat(np.where(vis, v, np.nan)))
