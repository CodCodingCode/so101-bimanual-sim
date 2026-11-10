"""Per-episode pole and ring positions on the table, detected in the first real frame of each episode (top view)
and ray-cast through the fitted top camera onto the table plane.

    .venv/bin/python so101_twin/ringpole/place_objects.py        # -> objects.json, out/objects_detect.jpg

Detection (overhead frame, 640x480):
  ring  = largest compact blob of bright orange/yellow pixels (the ring is orange in every episode; its highlights
          read as yellow under both white balances); centre = blob centroid at height RING.h/2.
  pole  = compact white / lavender component away from the arm (the pole base disc is lavender in the untinted
          episodes and nearly white in the orange-tinted ones; the first 7 episodes have a bare white pole with no
          disc). Disc -> centroid; bare rod -> the end closest to the image centre (the base of a vertical pole
          projects toward the nadir). Pole base at z = 0.
Episodes where a detection fails fall back to the previous episode's (or the default) layout and are listed.
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

OUT = HERE / "objects.json"
CX, CY = 320, 240


def masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.asarray(rgb).astype(int)
    hsv = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).convert("HSV")).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    white = (hsv[..., 1] < 110) & (hsv[..., 2] > 150) & (R - B < 60)      # R-B: drop the ring's highlights
    lav = (B - R > 15) & (hsv[..., 2] > 170)
    pink = (R > 170) & (R - G > 60) & (B > G + 10) & ~lav
    yellow = (R > 200) & (G > 170) & (B < 130) & (hsv[..., 1] > 90)
    for m in (white, lav, pink, yellow):
        m[:25] = False                      # beige wall strip
    return dict(white=white, lav=lav, pink=pink, yellow=yellow)


def components(m: np.ndarray):
    lab, n = ndimage.label(m)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        area = len(xs)
        bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        out.append(dict(area=area, fill=area / bbox, cx=xs.mean(), cy=ys.mean(), xs=xs, ys=ys))
    return out


def detect(rgb: np.ndarray) -> dict:
    m = masks(rgb)
    res = {}
    # ring: biggest compact yellow/orange-highlight blob
    cands = [c for c in components(ndimage.binary_dilation(m["yellow"], iterations=2))
             if 150 < c["area"] < 8000 and c["fill"] > 0.35 and c["cx"] > 150]
    if cands:
        c = max(cands, key=lambda c: c["area"])
        res["ring"] = (float(c["cx"]), float(c["cy"]))
    # pole: compact white/lavender component that is not the arm (largest component) and sits to the right
    allw = m["white"] | m["lav"] | m["pink"]
    allw = ndimage.binary_opening(allw, iterations=1)
    comps = components(allw)
    if comps:
        def extent(c):
            return max(c["xs"].max() - c["xs"].min(), c["ys"].max() - c["ys"].min())
        # the arm (fragmented into several white pieces) lives at x < 250 in the first frames; the pole base disc
        # is a compact blob, a bare rod a small elongated one far to the right
        def is_disc(c):
            return 1500 < c["area"] < 14000 and c["fill"] > 0.5 and c["cx"] > 180 and extent(c) < 160
        # untinted episodes: the lavender disc alone (the white mask merges it with table reflections / the arm)
        discs = [c for c in components(ndimage.binary_opening(m["lav"], iterations=2)) if is_disc(c)]
        discs = discs or [c for c in comps if is_disc(c) and (c["cx"] > 250 or c["cy"] > 300)]   # arm pieces: x < 250, upper half
        if not discs:                       # tinted frames where the disc merged with the arm: its pink shading
            discs = [c for c in components(ndimage.binary_opening(m["pink"], iterations=1))
                     if 600 < c["area"] < 6000 and c["cx"] > 250 and extent(c) < 120]
        rods = [c for c in comps if 100 < c["area"] < 2500 and c["fill"] > 0.2 and c["cx"] > 400 and 25 < extent(c) < 120]
        cands = discs or rods
        if cands:
            c = max(cands, key=lambda c: c["area"])
            if c in discs:                  # disc base
                res["pole"] = (float(c["cx"]), float(c["cy"]))
                res["pole_kind"] = "disc"
            else:                           # bare rod: base end = closest point to the image centre
                d = (c["xs"] - CX) ** 2 + (c["ys"] - CY) ** 2
                k = np.argsort(d)[:15]
                res["pole"] = (float(c["xs"][k].mean()), float(c["ys"][k].mean()))
                res["pole_kind"] = "rod"
    return res


def ray(sim: Sim, cam: str, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    cid = sim.cam_id[cam]
    pos = sim.d.cam_xpos[cid].copy()
    R = sim.d.cam_xmat[cid].reshape(3, 3)      # columns: camera x (right), y (up), z (backward)
    fovy = np.radians(sim.m.cam_fovy[cid])
    fy = (480 / 2) / np.tan(fovy / 2)
    d_cam = np.array([(u - CX) / fy, -(v - CY) / fy, -1.0])
    d = R @ d_cam
    return pos, d / np.linalg.norm(d)


def cast(sim: Sim, uv, z: float) -> list[float]:
    o, d = ray(sim, "top", *uv)
    t = (z - o[2]) / d[2]
    hit = o + t * d
    return [round(float(hit[0]), 4), round(float(hit[1]), 4)]


def main():
    fs = real_set()
    sim = Sim()
    mujoco.mj_forward(sim.m, sim.d)
    episodes = fs.meta["episodes"]
    objects, tiles, failed = {}, [], []
    prev = dict(pole=list(build_scene.POLE["xy"]), ring=list(build_scene.RING["xy"]))
    for e_i, ep in enumerate(episodes):
        i = int(np.nonzero(fs.labels["episode"] == ep)[0][np.argmin(fs.labels["t"][fs.labels["episode"] == ep])])
        rgb = fs.obs(i)["images"]["top"]
        det = detect(rgb)
        lay = dict(color="orange", frame=i)
        for k in ("pole", "ring"):
            if k in det:
                lay[k] = cast(sim, det[k], 0.0 if k == "pole" else build_scene.RING["h"] / 2)
                lay[k + "_px"] = [round(x, 1) for x in det[k]]
            else:
                lay[k] = prev[k]
                failed.append((ep, k))
        lay["pole_kind"] = det.get("pole_kind", "fallback")
        objects[str(ep)] = lay
        prev = lay
        im = Image.fromarray(np.asarray(rgb))
        dr = ImageDraw.Draw(im)
        for k, col in (("pole", (120, 80, 255)), ("ring", (255, 255, 0))):
            if k in det:
                u, v = det[k]
                dr.ellipse((u - 14, v - 14, u + 14, v + 14), outline=col, width=4)
        dr.text((5, 5), f"ep{ep} #{i} pole={lay['pole']} ring={lay['ring']} {lay['pole_kind']}", fill=(255, 255, 255))
        tiles.append(np.asarray(im.resize((320, 240))))
        print(f"ep {ep:3d} frame {i:3d}: pole {lay['pole']} ({lay['pole_kind']}) ring {lay['ring']}")
    sim.close()
    OUT.write_text(json.dumps(objects, indent=1))
    rows = [np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(len(tiles) // 4)]
    (HERE / "out").mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(HERE / "out" / "objects_detect.jpg", quality=88)
    print("wrote", OUT, "failed detections:", failed)


if __name__ == "__main__":
    main()
