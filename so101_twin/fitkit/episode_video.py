"""Full-rate real-vs-twin video of one episode, both cameras:

    .venv/bin/python so101_twin/fitkit/episode_video.py ricky 13
    .venv/bin/python so101_twin/fitkit/episode_video.py pickplace 7

Layout per frame: [real top | twin top] over [real side | twin side], 30 fps H.264, written to
so101_twin/fitkit/out/video_<scene>_ep<episode>.mp4.

Twin: the frozen arm fit (cameras, lens, joint map, camera lag) at EVERY recorded timestep, plus the objects of
so101_twin/<scene>_objects/build_<scene>.py (session 72). Those objects are defined per keyframe (8 per episode); here
the block's state between two keyframes switches at the moment the gripper closes (table -> held) or opens
(held -> bowl/box) fastest, so it is picked up and dropped when the real gripper acts. Real: decoded from the
dataset's own video at the same timestamps.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import armfit  # noqa: E402

ROOT = armfit.ROOT


def video_source(ds: pathlib.Path, ep: int, key: str):
    """(mp4 path, start time in that file) for LeRobot v2.1 (one file per episode) or v3.0 (concatenated files)."""
    import pyarrow.parquet as pq

    info = json.loads((ds / "meta" / "info.json").read_text())
    if str(info["codebase_version"]).startswith("v3"):
        for f in sorted((ds / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            t = pq.read_table(f).to_pydict()
            if ep in t["episode_index"]:
                k = t["episode_index"].index(ep)
                ci, fi = t[f"videos/{key}/chunk_index"][k], t[f"videos/{key}/file_index"][k]
                return ds / "videos" / key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4", float(t[f"videos/{key}/from_timestamp"][k])
        raise KeyError(ep)
    cs = int(info["chunks_size"])
    return ds / "videos" / f"chunk-{ep // cs:03d}" / key / f"episode_{ep:06d}.mp4", 0.0


def decode_run(path: pathlib.Path, t0: float, n: int, fps: float) -> list:
    """n consecutive frames starting at time t0."""
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0.0) / s.time_base), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            if fr.time is None or fr.time < t0 - 0.5 / fps:
                continue
            out.append(fr.to_ndarray(format="rgb24"))
            if len(out) == n:
                break
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 9 * len(text) + 8, 20], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(255, 230, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["ricky", "pickplace"])
    ap.add_argument("episode", type=int)
    ap.add_argument("--tile-w", type=int, default=480)
    a = ap.parse_args()
    B = armfit.load_module(ROOT / "so101_twin" / f"{a.scene}_objects" / f"build_{a.scene}.py", f"build_{a.scene}_video")
    T = B.make()
    vo = B.view_of(T)
    F = T.F
    e = a.episode
    S = F.raw[e]
    n = len(S)
    fps = float(F.fs.meta.get("fps", 30.0))
    j = F.joint

    def q_at(t: int) -> np.ndarray:
        x = float(np.clip(t + F.lag, 0, n - 1))
        lo = int(np.floor(x))
        hi = min(lo + 1, n - 1)
        st = S[lo] * (1 - (x - lo)) + S[hi] * (x - lo)
        qb = np.asarray(F.cfg["real_to_sim"](st), float)
        return j["sign"] * ((1 + j["gain"]) * (qb - F.q_mean) + F.q_mean) + j["off"]

    # block state timeline from the keyframes' states, switch points at the fastest gripper close / open
    keys = np.flatnonzero(F.ep == e)
    keys = keys[np.argsort(F.t[keys])]
    kst = [B.STATE.get(str(int(i)), {"state": "table"})["state"] for i in keys]
    g = S[:, 5].astype(float)
    dg = np.diff(g)
    rep = np.empty(n, int)
    rep[:] = keys[0]
    switch = []
    for a_, b_ in zip(range(len(keys) - 1), range(1, len(keys))):
        ta, tb = int(F.t[keys[a_]]), int(F.t[keys[b_]])
        if kst[a_] == kst[b_] or tb <= ta:
            switch.append((tb, b_))
            continue
        seg = dg[ta:tb]
        ts = ta + 1 + (int(np.argmin(seg)) if kst[b_] == "held" else int(np.argmax(seg)))
        switch.append((ts, b_))
    cur = 0
    for t in range(n):
        while cur < len(switch) and t >= switch[cur][0]:
            cur += 1
        rep[t] = keys[switch[cur - 1][1]] if cur else keys[0]

    # real frames
    ds = pathlib.Path(F.fs.meta["dataset"])
    ds = ds if ds.is_absolute() else ROOT / ds
    real = {}
    for cam, key in F.fs.meta["feature_keys"].items():
        path, t0 = video_source(ds, e, key)
        real[cam] = decode_run(path, t0, n, fps)
    n = min(n, *(len(v) for v in real.values()))

    tw = a.tile_w
    th = int(round(tw * T.H / T.W / 2)) * 2
    out = ROOT / "so101_twin" / "fitkit" / "out" / f"video_{a.scene}_ep{e}.mp4"
    with av.open(str(out), "w") as c:
        st = c.add_stream("libx264", rate=int(round(fps)))
        st.width, st.height, st.pix_fmt = 2 * tw, 2 * th, "yuv420p"
        st.options = {"crf": "20", "preset": "medium"}
        for t in range(n):
            q = q_at(t)
            for adr, x in zip(T.qadr, q):
                T.d.qpos[adr] = x
            mujoco.mj_forward(T.m, T.d)
            B.objects_fn(T, int(rep[t]))
            mujoco.mj_forward(T.m, T.d)
            rows = []
            for cam in ("top", "side"):
                img, _ = T.render(vo[cam][e])
                r = np.asarray(Image.fromarray(real[cam][t]).resize((tw, th)))
                s_ = np.asarray(Image.fromarray(img).resize((tw, th)))
                rows.append(np.concatenate([label(r, f"REAL {cam}  t={t}"), label(s_, f"TWIN {cam}  t={t}")], 1))
            frame = av.VideoFrame.from_ndarray(np.concatenate(rows, 0), format="rgb24")
            for pkt in st.encode(frame):
                c.mux(pkt)
            if (t + 1) % 60 == 0:
                print(f"{t + 1}/{n}", flush=True)
        for pkt in st.encode():
            c.mux(pkt)
    print("wrote", out, n, "frames,", round(n / fps, 1), "s; block switches at",
          [int(s) for s, _ in switch], "states", kst)


if __name__ == "__main__":
    main()
