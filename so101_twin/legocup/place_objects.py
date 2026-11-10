"""Per-episode cup / block positions on the table from the real top camera (fitted), by colour segmentation + ray cast.

    .venv/bin/python so101_twin/legocup/place_objects.py          # -> out/objects.json, out/objects_check.png
    .venv/bin/python so101_twin/legocup/place_objects.py --mat    # prints the mat centre / half size / yaw from PIX_MAT

For every episode the first 'approach' frame (the block still on the table) is segmented: block = red pixels inside
the mat ROI (largest component), cup = dark pixels inside the ROI whose component does not touch the top border
(that is the arm), largest component. The ray through each centroid is intersected with the horizontal plane at
the object's visual-centre height (block: half height; cup: ~half the cup height, since the centroid of the rim +
body blob sits about there). Only the top camera is used (the "side" camera is the wrist camera).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
from fit_cameras import DARK_THR, TOP_ROWS, roi  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

OUT = HERE / "out"
Z_BLOCK = build_scene.BLOCK["half"][2]
Z_CUP = 0.5 * build_scene.CUP["height"]
# mat corners in real frame 0 (640x480), read by eye: top-left, top-right, bottom-right, bottom-left
PIX_MAT = [(133, 10), (556, 10), (608, 445), (48, 428)]


def ray(sim: Sim, cam: str, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    cid = sim.cam_id[cam]
    pos = sim.d.cam_xpos[cid].copy()
    R = sim.d.cam_xmat[cid].reshape(3, 3)      # columns: camera x (right), y (up), z (backward)
    fovy = np.radians(sim.m.cam_fovy[cid])
    fy = (480 / 2) / np.tan(fovy / 2)
    d_cam = np.array([(u - 320) / fy, -(v - 240) / fy, -1.0])
    d = R @ d_cam
    return pos, d / np.linalg.norm(d)


def hit_plane(sim: Sim, u: float, v: float, z: float) -> np.ndarray:
    o, d = ray(sim, "top", u, v)
    t = (z - o[2]) / d[2]
    return (o + t * d)[:2]


def largest(mask: np.ndarray, min_area: int):
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes))
    if sizes[k] < min_area:
        return None
    return lab == (k + 1)


def segment(img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    red = (r > 140) & (r - g > 60) & (r - b > 60) & roi()
    dark = (img.max(-1) < DARK_THR["top"]) & roi()
    lab, _ = ndimage.label(dark)
    touch = set(np.unique(lab[:TOP_ROWS, :])) - {0}
    cup = dark & ~np.isin(lab, list(touch))
    cup = ndimage.binary_opening(cup, iterations=2)
    return largest(red, 60), largest(cup, 1500)


def centroid(m: np.ndarray) -> tuple[float, float]:
    v, u = np.nonzero(m)
    return float(u.mean()), float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", action="store_true")
    a = ap.parse_args()
    fs = real_set()
    sim = Sim()
    mujoco.mj_forward(sim.m, sim.d)
    if a.mat:
        c = np.stack([hit_plane(sim, u, v, 0.0) for u, v in PIX_MAT])
        ctr = c.mean(0)
        e1 = 0.5 * ((c[1] - c[0]) + (c[2] - c[3]))     # along the top edge
        e2 = 0.5 * ((c[3] - c[0]) + (c[2] - c[1]))
        yaw = np.degrees(np.arctan2(e1[1], e1[0]))
        print("mat corners (x, y):", np.round(c, 3).tolist())
        print(f"MAT_CENTER = ({ctr[0]:.3f}, {ctr[1]:.3f}); MAT_HALF = ({np.linalg.norm(e1) / 2:.3f}, "
              f"{np.linalg.norm(e2) / 2:.3f}, 0.001); MAT_YAW_DEG = {yaw:.1f}")
        return
    episodes = sorted(set(int(e) for e in fs.labels["episode"]))
    out, tiles = {}, []
    last = None
    for e in episodes:
        idx = [int(i) for i in np.nonzero(fs.labels["episode"] == e)[0]]
        res = {}
        for i in idx:
            if str(fs.labels["phase"][i]) != "approach":
                continue
            img = fs.obs(i)["images"]["top"]
            blk, cup = segment(img)
            if "block" not in res and blk is not None:
                u, v = centroid(blk)
                res["block"] = [round(float(x), 4) for x in hit_plane(sim, u, v, Z_BLOCK)]
                res["block_px"] = [round(u), round(v)]
                res["frame"] = i
            if "cup" not in res and cup is not None:
                u, v = centroid(cup)
                res["cup"] = [round(float(x), 4) for x in hit_plane(sim, u, v, Z_CUP)]
                res["cup_px"] = [round(u), round(v)]
            if "block" in res and "cup" in res:
                break
        for k in ("block", "cup"):
            if k not in res:
                res[k] = last[k] if last else list(build_scene.BLOCK["xy"] if k == "block" else build_scene.CUP["xy"])
                res[k + "_missing"] = True
        out[str(e)] = res
        last = res
        print(e, res, flush=True)
        if e % 5 == 0:
            i = res.get("frame", idx[0])
            im = Image.fromarray(fs.obs(i)["images"]["top"].copy())
            dr = ImageDraw.Draw(im)
            for k, col in (("block_px", (0, 255, 0)), ("cup_px", (0, 200, 255))):
                if k in res:
                    u, v = res[k]
                    dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=col, width=3)
            dr.text((4, 4), f"ep{e} #{i} block {res['block']} cup {res['cup']}", fill=(255, 255, 0))
            tiles.append(np.asarray(im))
    OUT.mkdir(exist_ok=True)
    (OUT / "objects.json").write_text(json.dumps(out, indent=1))
    Image.fromarray(np.concatenate(tiles, 0)).save(OUT / "objects_check.png")
    print("wrote", OUT / "objects.json")
    sim.close()


if __name__ == "__main__":
    main()
