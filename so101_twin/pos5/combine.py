"""Final real-arm masks: the twin-free chroma-key mask, with the white-on-white parts restored from SAM.

    .venv/bin/python so101_twin/pos5/combine.py --ck _ck --sam _s1 --tag _c1 [--guard _tw1]

Per frame:  M = (ck & ~static & ~wall)  |  (SAM & static & ~wall & touching & (box | changed))
            minus the real cube pixels and the base clamp, holes < 300 px filled.
  ck      chroma-key arm mask (ck_masks.py): crisp boundary against the green screen, decides OUTSIDE the static pixels
  static  pixels that are not green in >= 97 % of the frames (white box, pale wall corner, mat marks), outside the base
          box: there the chroma key cannot tell the white arm from the white box / wall or from the arm's shadow on
          them, so SAM decides (SAM 2.1 prompted by the twin with chroma consensus and hard negatives, segment.py)
  wall    the pale magenta wall corner (side view, also behind the base): R - G >= 10 and B - G >= 8 and min(RGB)
          >= 140, inside static | base box (the white arm is neutral or green-tinted: B - G median -5); twin-free
  changed outside the box, SAM-static pixels must differ from the median background (max |I - Bg| > 12): the arm in
          front of the wall corner / mat marks changes them, a SAM leak onto the unchanged wall does not
  clamp   side view only: the black table clamp + orange pad at the base's lower left (dark max(RGB) < 90 or orange
          R - G > 60 in the median background, x < 45, 220 <= y < 290; dilated 1 px); static, not part of the arm
  touching  SAM-static blobs count only if they touch the chroma arm (8-connected after a 2-px dilation), so a SAM
          leak into the box that is not attached to the arm cannot enter; --guard TAG (off by default) additionally
          limits them to within 12 px of that twin silhouette
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage as ndi

import ck_masks
import fit as F
import objects
import p5


def clamp_mask(cam: str) -> np.ndarray:
    m = np.zeros((480, 640), bool)
    if cam != "side":
        return m
    x = np.load(p5.MASKS / "bg_side.npy").astype(np.int16)
    m = (x.max(-1) < 90) | (x[..., 0] - x[..., 1] > 60)
    keep = np.zeros_like(m)
    keep[220:290, :45] = True
    return ndi.binary_dilation(m & keep, iterations=1)


def wall_px(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.int16)
    return (x[..., 0] - x[..., 1] >= 10) & (x[..., 2] - x[..., 1] >= 8) & (x.min(-1) >= 140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="_ck")
    ap.add_argument("--sam", default="_s1")
    ap.add_argument("--tag", default="_c1")
    ap.add_argument("--guard", default="")
    ap.add_argument("--chg", type=int, default=12)
    a = ap.parse_args()
    fs = F.real_set(p5.scene())
    boxes = objects.box_masks()
    for cam in p5.CAMS:
        imgs = np.asarray(fs._img[cam])
        bg = np.load(p5.MASKS / f"bg_{cam}.npy").astype(np.int16)
        region = p5.load(cam, "_static")[0] | ck_masks.base_mask(cam)
        clamp = clamp_mask(cam)
        ck = p5.load(cam, a.ck)
        sam = p5.load(cam, a.sam)
        static = p5.load(cam, "_static")[0]
        cube = p5.load(cam, "_cube")
        guard = p5.load(cam, a.guard) if a.guard else None
        out = np.zeros_like(ck)
        added = []
        for i in range(len(ck)):
            wall = wall_px(imgs[i]) & region
            chg = np.abs(imgs[i].astype(np.int16) - bg).max(-1) > a.chg
            base = ck[i] & ~static & ~wall
            add = sam[i] & static & ~cube[i] & ~wall & (boxes[cam] | chg)
            if guard is not None:
                add &= ndi.binary_dilation(guard[i], iterations=12)
            if add.any():   # keep only SAM-static blobs that touch the chroma arm
                lab, n = ndi.label(add, structure=np.ones((3, 3)))
                touch = np.unique(lab[ndi.binary_dilation(base, iterations=2) & (lab > 0)])
                add = np.isin(lab, touch[touch > 0])
            m = (base | add) & ~cube[i] & ~clamp
            holes, nh = ndi.label(~m)
            if nh:
                hs = ndi.sum(~m, holes, range(1, nh + 1))
                m |= np.isin(holes, 1 + np.flatnonzero(hs < 300)) & ~cube[i] & ~clamp
            out[i] = m
            added.append(int(add.sum()))
        p5.save(cam, a.tag, out)
        print(cam, "mean area", round(float(out.mean()), 4), "SAM-static px added per frame: mean",
              int(np.mean(added)), "frames with any", int(np.sum(np.array(added) > 0)), flush=True)


if __name__ == "__main__":
    main()
