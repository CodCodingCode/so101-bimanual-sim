"""Contact sheet of segment_objects.py results: real frame with mask tint + box per prompt."""
import pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope/molmoact2"))
from sets import Store
setname, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
frames = [int(x) for x in sys.argv[4].split(",")]
z = np.load(npz, allow_pickle=True)
fs = Store().frames(setname)
H, W = z["shape"]
cols = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
prompts = list(dict.fromkeys(z["prompt"].tolist()))
tiles = []
for i in frames:
    row = []
    for c in ("top", "side"):
        rgb = np.asarray(fs._img[c][i]).copy().astype(float)
        sel = np.flatnonzero((z["frame"] == i) & (z["cam"] == c))
        im = None
        boxes = []
        for k in sel:
            m = np.unpackbits(z["bits"][k], axis=-1)[..., :W].astype(bool)
            col = np.array(cols[prompts.index(z["prompt"][k])], float)
            rgb[m] = 0.5 * rgb[m] + 0.5 * col
            boxes.append((z["box"][k], cols[prompts.index(z["prompt"][k])], z["score"][k]))
        im = Image.fromarray(rgb.astype(np.uint8)); d = ImageDraw.Draw(im)
        for b, col, s in boxes:
            if np.isfinite(b).all():
                d.rectangle(list(map(float, b)), outline=col, width=3); d.text((b[0], b[1] - 12), f"{s:.2f}", fill=col)
        row.append(np.asarray(im.resize((W // 3, H // 3))))
    tiles.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(tiles, 0)).save(out, quality=85)
