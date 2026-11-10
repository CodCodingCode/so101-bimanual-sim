"""Per-session median background of the real frames (the arm and the moving objects wash out)."""
import json, pathlib, sys
import numpy as np
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope/molmoact2"))
from sets import Store
setname, fitj = sys.argv[1], sys.argv[2]
fs = Store().frames(setname)
st = json.loads(pathlib.Path(fitj).read_text())
ep = np.asarray(fs.labels["episode"])
(HERE / "work").mkdir(exist_ok=True)
for v, d in st["views"].items():
    eps = d.get("episodes") or sorted(set(ep.tolist()))
    idx = np.flatnonzero(np.isin(ep, eps))
    sel = idx[np.linspace(0, len(idx) - 1, min(40, len(idx))).astype(int)]
    stack = np.stack([np.asarray(fs._img[d["cam"]][i]) for i in sel])
    med = np.median(stack, axis=0).astype(np.uint8)
    np.save(HERE / "work" / f"median_{v}.npy", med)
    Image.fromarray(med).save(HERE / "work" / f"median_{v}.jpg", quality=92)
    print(v, len(sel), flush=True)
