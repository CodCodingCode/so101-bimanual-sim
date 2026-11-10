"""before_after.jpg: 8 frames from different episodes; per row real | old twin | new twin for camera top, then side."""
import pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
realdir, olddir, newdir, out = map(pathlib.Path, sys.argv[1:5])
z = np.load(realdir / "labels.npz")
ep = z["episode"]
eps = np.unique(ep)
sel = eps[np.linspace(0, len(eps) - 1, 8).astype(int)]
idx = [int(np.flatnonzero(ep == e)[k % int((ep == e).sum())]) for k, e in enumerate(sel)]
tw = 320
rows = []
for i in idx:
    tiles = []
    for c in ("top", "side"):
        for d, lab in ((realdir, "real"), (olddir, "old twin"), (newdir, "new twin")):
            a = np.load(d / f"images_{c}.npy", mmap_mode="r")[i]
            im = Image.fromarray(np.asarray(a))
            im = im.resize((tw, int(tw * im.size[1] / im.size[0])))
            ImageDraw.Draw(im).text((5, 5), f"{lab} {c} f{i} ep{int(ep[i])} {z['phase'][i]}", fill=(255, 255, 0))
            tiles.append(np.asarray(im))
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).save(out, quality=88)
print(out, idx)
