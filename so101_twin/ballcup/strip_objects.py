"""Final cleanup of a SAM arm-mask set: remove the pixels of the SAM ball / cup masks of the dino run (twin-independent,
ball masks filtered by fix_ball.py) from the SAM arm masks: a held ball that SAM merged with the gripper is not the
arm. Only removes pixels (no resampling, no twin input). -> masks/ballcup_<cam><tag>o.npy

    .venv/bin/python so101_twin/ballcup/strip_objects.py --tag _t1b
"""
import argparse
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--obj-tag", default="_d1")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm = load(M / f"ballcup_{c}{a.tag}.npy")
        obj = load(M / f"ballcup_{c}{a.obj_tag}_ball.npy") | load(M / f"ballcup_{c}{a.obj_tag}_cup.npy")
        out = arm & ~obj
        lost = (arm & obj).reshape(len(arm), -1).sum(1) / np.maximum(arm.reshape(len(arm), -1).sum(1), 1)
        np.save(M / f"ballcup_{c}{a.tag}o.npy", np.packbits(out, axis=-1))
        print(c, f"frames changed {(lost > 0).sum()}, > 2 % of the mask removed: {(lost > 0.02).sum()}, max {lost.max():.3f}")


if __name__ == "__main__":
    main()
