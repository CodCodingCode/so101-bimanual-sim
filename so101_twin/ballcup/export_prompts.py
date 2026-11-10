"""Twin arm silhouettes (lens-warped, 640x480, packed) for every real frame from a fit json -> masks/<dir>/ballcup_<cam>.npy

    .venv/bin/python so101_twin/ballcup/export_prompts.py --fit out/<run>_fit.json --dir prompts_r1 [--limit N]
"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--occ", action="store_true", help="visible arm: the twin's cup / ball (objects.json) occlude it")
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    sc.MASK_OCCLUDERS = a.occ
    if a.fit:
        fj = json.loads(pathlib.Path(a.fit).read_text())
        sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
        P = F.reconcile(sc, fj["params"])
    else:
        P = F.init_params(sc)
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    n = a.limit or fs.n
    d = HERE / "masks" / a.dir
    d.mkdir(parents=True, exist_ok=True)
    for c in sc.CAMERAS:
        ms = np.zeros((fs.n, 480, 80), np.uint8)
        for i in range(n):
            ms[i] = np.packbits(rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480, i=i), axis=-1)
        np.save(d / f"ballcup_{c}.npy", ms)
    print("wrote", d)


if __name__ == "__main__":
    main()
