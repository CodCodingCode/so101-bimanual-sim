"""Held-out IoU breakdown of a fit json on a mask tag: per top session, rest vs moving, and by where the gripper is
(near the disc: pan < 20; near the green block: pan > 30).   .venv/bin/python so101_twin/stack2/breakdown.py FIT _s5"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

sc = F.load_scene("stack")
P = F.reconcile(sc, json.loads(pathlib.Path(sys.argv[1]).read_text())["params"])
tag = sys.argv[2]
fs = F.real_set(sc)
real = {c: np.unpackbits(np.load(F.TWIN / "masks" / f"stack_{c}{tag}.npy"), axis=-1)[..., :640].astype(bool)
        for c in sc.CAMERAS}
ev = F.evaluate(F.Rig(sc, P), sc, real, None)
ep = np.asarray(fs.labels["episode"]).astype(int)
odd = ep % 2 == 1
st = fs.state
rest = st[:, 1] < -90
disc = (st[:, 0] < 20) & ~rest
block = (st[:, 0] > 30) & ~rest
for c in sc.CAMERAS:
    v = ev[c]["iou_new"]
    def m(k):
        x = v[k & odd]
        x = x[np.isfinite(x)]
        return f"{np.median(x):.3f} (n={len(x)})" if len(x) else "-"
    print(f"{c}: held-out median rest {m(rest)} | moving near disc {m(disc)} | moving near block {m(block)}")
    if c == "top":
        grp = F.frame_groups(sc, c)
        print("   sessions:", {g: m(grp == g) for g in sorted(set(grp))})
