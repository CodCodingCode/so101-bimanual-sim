"""Real-occluder masks for the occluder-excluded reading (agreed amendment), from the REAL frames only (no twin):

  cube  the cyan cube's pixels (PIL-HSV hue 120-150, sat >= 60, value >= 50; opened 1 px, dilated 2 px) - when the
        cube is held it hides part of the gripper;
  box   the white box's pixels where the box is visible: the box region of the median background image (static
        not-green component through the box seed, holes filled) where the frame is unchanged from that background
        (max |I - Bg| over RGB <= --chg); when the gripper is lowered into the box, the box walls hide it.
  clamp (side) the black table clamp + orange pad in front of the base's lower left (combine.clamp_mask; static).
O = cube | box | clamp  ->  masks/pos5_<cam>_occ.npy.  Used ONLY for scoring with a frozen fit (score.py --excl _occ).

    .venv/bin/python so101_twin/pos5/occ_masks.py [--chg 20]
"""
from __future__ import annotations

import argparse

import numpy as np

import combine
import fit as F
import objects
import p5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chg", type=int, default=20)
    a = ap.parse_args()
    fs = F.real_set(p5.scene())
    boxes = objects.box_masks()
    for cam in p5.CAMS:
        imgs = np.asarray(fs._img[cam])
        bg = np.load(p5.MASKS / f"bg_{cam}.npy").astype(np.int16)
        cube = p5.load(cam, "_cube")
        occ = np.zeros_like(cube)
        clamp = combine.clamp_mask(cam)
        for i in range(fs.n):
            same = np.abs(imgs[i].astype(np.int16) - bg).max(-1) <= a.chg
            occ[i] = cube[i] | (boxes[cam] & same) | clamp
        p5.save(cam, "_occ", occ)
        print(cam, "box px", int(boxes[cam].sum()), "occ px/frame mean", int(occ.reshape(fs.n, -1).sum(1).mean()))


if __name__ == "__main__":
    main()
