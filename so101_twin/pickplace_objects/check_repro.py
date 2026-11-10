"""Reproduce the existing twin2 arm render with the copied (frozen) fitkit code and compare to the stored set.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/check_repro.py ricky so101_twin/fitkit/out/ricky_so101_sess.json so101_ricky_twin2 0,9,100
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fitkit_copy"))
import twin2  # noqa: E402

scene, fit, setname, idx = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
t0 = time.time()
T = twin2.Twin2(scene, pathlib.Path(fit))
print("built", round(time.time() - t0, 1), "s", "extent", T.m.stat.extent, "center", T.m.stat.center, "znear", T.m.vis.map.znear,
      "zfar", T.m.vis.map.zfar, "ngeom", T.m.ngeom, flush=True)
F = T.F
view_of = {}
for v, vc in F.views.items():
    for e in (vc.get("episodes") or np.unique(F.ep).tolist()):
        view_of.setdefault(vc["cam"], {})[int(e)] = v
sd = twin2.armfit.ROOT / "neuroscope/molmoact2/store/sets" / setname
imgs = {c: np.load(sd / f"images_{c}.npy", mmap_mode="r") for c in F.fs.cameras}
masks = {c: np.load(sd / f"arm_mask_{c}.npy", mmap_mode="r") for c in F.fs.cameras}
for i in idx:
    T.pose(i)
    for c in F.fs.cameras:
        v = view_of[c][int(F.ep[i])]
        t1 = time.time()
        img, m = T.render(v)
        old = np.asarray(imgs[c][i])
        om = np.unpackbits(np.asarray(masks[c][i]), axis=-1)[..., : T.W].astype(bool)
        d = np.abs(img.astype(int) - old.astype(int))
        print(f"frame {i} cam {c} view {v}: render {time.time() - t1:.2f}s  max|d| {d.max()}  n(d>0) {(d.max(-1) > 0).sum()}  "
              f"mask equal {np.array_equal(m, om)} (n arm {om.sum()}, xor {(m ^ om).sum()})", flush=True)
