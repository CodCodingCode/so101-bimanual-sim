"""GRAYBLOCK diagnostic: real | twin RGB for a few frames (both cameras). .venv python try_rgb.py FIT OUT [frames]"""
import sys, json, pathlib
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit as F
sc = F.load_scene("grayblock")
fj = json.load(open(sys.argv[1]))
import dataclasses
sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", "so101"))
P = F.reconcile(sc, fj["params"])
rig = F.Rig(sc, P, full=True)
fs = F.real_set(sc)
idx = [int(x) for x in sys.argv[3:]] or [0, 100, 245, 250]
rows = []
for i in idx:
    t = []
    for c in ("top", "side"):
        rig.pose(c, "all", F.frame_state(sc, P, c, i)); sc.frame_setup(rig, fs, i)
        t += [np.asarray(fs.obs(i)["images"][c]), rig.arm.render(c, 640, 480, rig.lens[c])]
    rows.append(np.concatenate(t, 1))
Image.fromarray(np.concatenate(rows, 0)).resize((1280, 240 * len(idx))).save(sys.argv[2], quality=85)
