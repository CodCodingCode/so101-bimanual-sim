"""Wooden-cube track at full rate (every frame of the source videos) for the EVEN episodes of the real set.

    .venv/bin/python so101_twin/stack2/fullrate_cube.py      # -> out/fullrate_cube.npz

Per even episode and camera: cube centroid / bbox / area per frame (cube_track.wood), plus the parquet state per frame.
Used only to seed the cameras (seed_pnp.py) and to test joint-map hypotheses; odd episodes are never read.
Also checks that the decoded frame at (episode, t) equals the real set's image (timestamp mapping).
"""
import pathlib
import sys

import av
import numpy as np
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
from cube_track import wood  # noqa: E402
from sets import Store  # noqa: E402

DS = ROOT / "data_so101" / "ichbinblau__so101_stack2cubes_dataset"


def episodes_meta():
    rows = {}
    for f in sorted((DS / "meta" / "episodes").rglob("*.parquet")):
        d = pq.read_table(f).to_pydict()
        for k in range(len(d["episode_index"])):
            rows[int(d["episode_index"][k])] = {c: d[c][k] for c in d if not c.startswith("stats/")}
    return rows


def decode(cam: str, meta: dict, step: int = 1):
    vp = DS / "videos" / f"observation.images.{cam}" / f"chunk-{meta[f'videos/observation.images.{cam}/chunk_index']:03d}" / \
        f"file-{meta[f'videos/observation.images.{cam}/file_index']:03d}.mp4"
    t0 = float(meta[f"videos/observation.images.{cam}/from_timestamp"])
    n = int(meta["length"])
    c = av.open(str(vp))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    c.seek(int(max(t0 - 1.0, 0) / s.time_base), stream=s, backward=True, any_frame=False)
    out = {}
    for fr in c.decode(s):
        ts = float(fr.pts * s.time_base)
        k = int(round((ts - t0) * 30))
        if k < 0:
            continue
        if k >= n:
            break
        if k % step == 0:
            out[k] = fr.to_ndarray(format="rgb24")
    c.close()
    return out


def main():
    fs = Store().frames("so101_stack_real")
    ep = np.asarray(fs.labels["episode"]).astype(int)
    tt = np.asarray(fs.labels["t"]).astype(int)
    meta = episodes_meta()
    states = {}
    for f in sorted((DS / "data").rglob("*.parquet")):
        tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
        e = np.asarray(tb.column("episode_index").to_numpy())
        fi = np.asarray(tb.column("frame_index").to_numpy())
        st = np.asarray(tb.column("observation.state").to_pylist(), np.float32)
        for x in np.unique(e):
            m = e == x
            states[int(x)] = st[m][np.argsort(fi[m])]
    out = {}
    evens = sorted(set(int(e) for e in ep if e % 2 == 0))
    for e in evens:
        for cam in ("top", "side"):
            frames = decode(cam, meta[e])
            n = len(states[e])
            arr = np.full((n, 7), np.nan, np.float32)
            for k, img in frames.items():
                w = wood(img, cam)
                if w is not None:
                    arr[k, :3] = w[:3]
                    arr[k, 3:] = w[3]
            out[f"{cam}_{e}"] = arr
            # timestamp check against the real set
            for i in np.nonzero(ep == e)[0][:2]:
                if int(tt[i]) in frames:
                    diff = np.abs(frames[int(tt[i])].astype(int) - np.asarray(fs.obs(int(i))["images"][cam]).astype(int)).mean()
                    print(f"ep {e} {cam} t={tt[i]}: mean |decoded - set| = {diff:.2f}", flush=True)
            if cam == "top":
                bg = np.median(np.stack(list(frames.values())[::4]), axis=0).astype(np.uint8)
                out[f"bg_top_{e}"] = bg
        out[f"state_{e}"] = states[e]
        print(f"ep {e}: {len(states[e])} frames, cube seen top {np.isfinite(out[f'top_{e}'][:, 0]).mean():.2f} "
              f"side {np.isfinite(out[f'side_{e}'][:, 0]).mean():.2f}", flush=True)
    np.savez_compressed(HERE / "out" / "fullrate_cube.npz", **out)


if __name__ == "__main__":
    main()
