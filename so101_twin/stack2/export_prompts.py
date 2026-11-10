"""Twin arm silhouettes for every real frame (prompts for segment.py), from the scene init or a fit json.

    .venv/bin/python so101_twin/stack2/export_prompts.py OUT_DIR [--fit-json F] [--dt -1.5] [--fg-top]

--fg-top additionally writes OUT_DIR_fg/: top = the background-subtraction mask (masks/stack_top_fg.npy), side = the
twin silhouette again, so that `segment.py --prompts OUT_DIR,OUT_DIR_fg` prompts the top camera by consensus of the
twin and the moving-pixel mask and the side camera by the twin alone.
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fit-json", default="")
    ap.add_argument("--dt", type=float, default=-1.5)
    ap.add_argument("--fg-top", action="store_true")
    a = ap.parse_args()
    sc = F.load_scene("stack")
    if a.fit_json:
        P = F.reconcile(sc, json.loads(pathlib.Path(a.fit_json).read_text())["params"])
    else:
        P = F.init_params(sc)
        for c in P["cams"]:
            P["cams"][c]["dt_frames"] = a.dt
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in sc.CAMERAS:
        grp = F.frame_groups(sc, c)
        ms = np.stack([np.packbits(rig.mask(c, grp[i], F.frame_state(sc, P, c, i), 640, 480), axis=-1)
                       for i in range(fs.n)])
        np.save(out / f"stack_{c}.npy", ms)
        area = np.unpackbits(ms, axis=-1).reshape(fs.n, -1).mean(1)
        print(f"{c}: twin arm area mean {area.mean():.3f}, empty {int((area == 0).sum())} frames", flush=True)
    if a.fg_top:
        fg = pathlib.Path(str(out) + "_fg")
        fg.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "masks" / "stack_top_fg.npy", fg / "stack_top.npy")
        shutil.copyfile(out / "stack_side.npy", fg / "stack_side.npy")
        print("wrote", fg)


if __name__ == "__main__":
    main()
