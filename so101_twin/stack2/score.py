"""Score fit jsons on a mask tag (full-res real grid, exactly fit.evaluate): EVEN (fit split, used to choose between
candidate fits), held-out ODD and all frames.   .venv/bin/python so101_twin/stack2/score.py _s5 A.json [B.json ...]"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

tag = sys.argv[1]
sc = F.load_scene("stack")
fs = F.real_set(sc)
ep = np.asarray(fs.labels["episode"]).astype(int)
real = {c: np.unpackbits(np.load(F.TWIN / "masks" / f"stack_{c}{tag}.npy"), axis=-1)[..., :640].astype(bool)
        for c in sc.CAMERAS}
for path in sys.argv[2:]:
    P = F.reconcile(sc, json.loads(pathlib.Path(path).read_text())["params"])
    ev = F.evaluate(F.Rig(sc, P), sc, real, None)
    out = []
    for c in sc.CAMERAS:
        v = ev[c]["iou_new"]
        even = np.where(ep % 2 == 0, v, np.nan)
        out.append(f"{c}: even med {np.nanmedian(even):.3f} mean {np.nanmean(even):.3f} | held-out med "
                   f"{ev[c]['new_heldout']['median']:.3f} p10 {ev[c]['new_heldout']['p10']:.3f} | all med {ev[c]['new']['median']:.3f}")
    print(pathlib.Path(path).name, " || ".join(out), flush=True)
