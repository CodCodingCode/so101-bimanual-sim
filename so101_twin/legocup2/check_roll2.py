"""Wrist-roll offset scan at the init cameras: IoU vs SAM (top / side) per roll offset, plus top renders."""
import json
import sys
import pathlib
import dataclasses

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

sc = F.load_scene("legocup")
fs = F.real_set(sc)
P0 = F.init_params(sc)
real = {c: F.load_masks("legocup", c) for c in sc.CAMERAS}
sel = np.linspace(0, fs.n - 1, 40).astype(int)
sc2 = F.load_scene("legocup")
sc2.ARM = dataclasses.replace(sc.ARM, colors={"body": (0.75, 0.75, 0.75, 1), "jaw": (0.2, 0.5, 1.0, 1),
                                              "motor": (0.3, 0.3, 0.3, 1), "cam_mount": (1.0, 0.45, 0.1, 1),
                                              "camera": (0.1, 0.9, 0.2, 1)})
rig = F.Rig(sc2, P0)
rows = []
for r in (-90.0, 0.0, 90.0, 180.0):
    P = json.loads(json.dumps(P0))
    P["offsets_deg"][4] = r
    rig.set_params(P)
    out = []
    for c in sc.CAMERAS:
        sims = np.stack([rig.mask(c, "all", fs.state[i], 320, 240) for i in sel])
        v = F.iou_frames(sims, F.down2(real[c][sel]))
        out.append(f"{c} {np.nanmedian(v):.3f}")
    print(f"roll {r:+.0f}: " + ", ".join(out), flush=True)
    rows.append(np.concatenate([rig.rgb("top", "all", fs.state[i], 320, 240) for i in (5, 85, 13)]
                               + [np.asarray(fs.obs(i)["images"]["top"])[::2, ::2] for i in (5,)], 1))
Image.fromarray(np.concatenate(rows, 0)).save(HERE / "out" / "check_roll2.jpg", quality=85)
print("states", {i: np.round(fs.state[i], 1).tolist() for i in (5, 85, 13)})
