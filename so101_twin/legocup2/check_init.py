"""Visual + numeric check of the joint mapping / camera init: twin silhouettes over real frames (SAM outline green,
twin blue fill) for wrist-roll offset candidates. -> out/check_init.jpg, prints IoU vs SAM per candidate."""
import json
import sys
import time
import pathlib

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

sc = F.load_scene("legocup")
fs = F.real_set(sc)
P0 = F.init_params(sc)
if len(sys.argv) > 1:
    P0 = F.reconcile(sc, json.loads(pathlib.Path(sys.argv[1]).read_text())["params"])
real = {c: F.load_masks("legocup", c) for c in sc.CAMERAS}
rig = F.Rig(sc, P0, full=True)
ep = fs.labels["episode"].astype(int)
idx = [5, 13, 85, 97, 3, 230]
rolls = [0.0, 180.0] if len(sys.argv) == 1 else [P0["offsets_deg"][4]]
rows = []
for r in rolls:
    P = json.loads(json.dumps(P0))
    P["offsets_deg"][4] = r
    rig.set_params(P)
    # numeric: 48 spread frames
    sel = np.linspace(0, fs.n - 1, 48).astype(int)
    t0 = time.time()
    for c in sc.CAMERAS:
        sims = np.stack([rig.mask(c, "all", fs.state[i], 640, 480) for i in sel])
        v = F.iou_frames(sims, real[c][sel])
        print(f"roll {r:+.0f} {c}: IoU median {np.nanmedian(v):.3f} mean {np.nanmean(v):.3f} ({time.time() - t0:.1f}s)")
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            im = np.asarray(fs.obs(i)["images"][c]).copy()
            s = rig.mask(c, "all", fs.state[i], 640, 480)
            rgb = rig.rgb(c, "all", fs.state[i], 640, 480)
            ov = im.astype(np.float32)
            ov[s] = 0.5 * ov[s] + 0.5 * np.array([40, 90, 255])
            o = real[c][i] & ~F.A.np.roll(real[c][i], 1, 0) | real[c][i] & ~np.roll(real[c][i], 1, 1)
            ov[o] = (40, 255, 40)
            tiles += [ov.astype(np.uint8), rgb]
        rows.append(np.concatenate(tiles, 1))
im = Image.fromarray(np.concatenate(rows, 0))
im = im.resize((im.width // 2, im.height // 2))
im.save(HERE / "out" / "check_init.jpg", quality=85)
print("rows: per roll candidate", rolls, "frames", idx)
