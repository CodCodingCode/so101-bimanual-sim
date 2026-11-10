"""GRAYBLOCK: REAL occluder masks for the occluder-excluded reading = the NEAR faces of the box (the walls that can stand
in front of the gripper), from the real frames only (no twin input, no fitted parameter, fixed constants).

    .venv/bin/python so101_twin/grayblock/occ_near.py   # -> masks/grayblock_<cam>_occn.npy (+ out/sheet_occn.jpg)

Input per frame / camera: the SAM 2.1 box mask of the real frame (masks/grayblock_<cam>_d1_box.npy: Grounding DINO
"a blue box." box -> SAM box prompt, segment.py --mode dino) and the real image. Faces = connected components (>= 150 px)
of the box mask's BLUE pixels (b >= 70, b - r >= 45, b - g >= 20; the box is blue outside and partly blue inside), holes
filled (the printed logo). A face is NEAR when the box silhouette ends right below it (the pixel 6 px below its lowest
pixel lies outside the box mask) in >= 60 % of its columns: the outer faces of the near walls stand on the table, while
the far walls' blue inner faces have the white floor below them. Occluder = union of the near blue faces (top view: the
front wall; side view: the right-hand wall). Both cameras look down into the open box, so a gripper inside it can only
be hidden by near walls. Conservative: the side view's WHITE near wall (the one with the printed marker) is not
included (an edge-based face split could not separate it from the white floor reliably: tried, not used).
"""
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def near_faces(box, rgb):
    if box.sum() < 300:
        return np.zeros_like(box)
    x = rgb.astype(np.int16)
    blue = (x[..., 2] >= 70) & (x[..., 2] - x[..., 0] >= 45) & (x[..., 2] - x[..., 1] >= 20)
    lab, n = ndi.label(ndi.binary_opening(box & blue, iterations=1))
    out = np.zeros_like(box)
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        f = lab[sl] == k
        if f.sum() < 150:
            continue
        ff = np.zeros_like(box)
        ff[sl] = f
        ys, xs = np.nonzero(ff)
        low = {}
        for x, y in zip(xs, ys):
            if y > low.get(x, -1):
                low[x] = y
        cols = np.array(list(low.keys()))
        yl = np.array([low[x] for x in cols])
        below = np.clip(yl + 6, 0, box.shape[0] - 1)
        touch = ~box[below, cols]
        if touch.mean() >= 0.6:
            out |= ndi.binary_fill_holes(ff)
    return out


def main():
    from sets import Store

    fs = Store().frames("so101_grayblock_real")
    sheet = []
    for c in ("top", "side"):
        imgs = np.load(fs.path / f"images_{c}.npy", mmap_mode="r")
        box = load(M / f"grayblock_{c}_d1_box.npy")
        out = np.stack([near_faces(box[i], np.asarray(imgs[i])) for i in range(fs.n)])
        np.save(M / f"grayblock_{c}_occn.npy", np.packbits(out, axis=-1))
        a = out.reshape(fs.n, -1).mean(1)
        print(c, "near-face occluder: frames with pixels", int((a > 0).sum()), "mean area", round(float(a.mean()), 4))
        for i in (0, 13, 100, 165, 245, 317):
            im = np.asarray(imgs[i]).copy()
            im[out[i]] = (0.5 * im[out[i]] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            sheet.append(im)
    rows = [np.concatenate(sheet[j:j + 6], 1) for j in (0, 6)]
    Image.fromarray(np.concatenate(rows, 0)).resize((1920, 480)).save(HERE / "out" / "sheet_occn.jpg", quality=85)


if __name__ == "__main__":
    main()
