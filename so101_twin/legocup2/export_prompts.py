"""Twin arm silhouettes for every real frame (640x480, lens warp, camera latency) from a params json ->
so101_twin/masks/<DIR>/legocup_<cam>.npy (packed bits uint8 [n, 480, 80]), the prompts for segment(_lc2).py.

    .venv/bin/python so101_twin/legocup2/export_prompts.py PARAMS.json [DIR=prompts_twin2]
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

sc = F.load_scene("legocup")
pj = pathlib.Path(sys.argv[1])
P = F.reconcile(sc, json.loads((pj if pj.is_absolute() else HERE / pj).read_text())["params"])
d = F.MASKS / (sys.argv[2] if len(sys.argv) > 2 else "prompts_twin2")
d.mkdir(parents=True, exist_ok=True)
fs = F.real_set(sc)
rig = F.Rig(sc, P)
cams = sys.argv[3].split(",") if len(sys.argv) > 3 else list(sc.CAMERAS)
for c in cams:
    ms = np.stack([np.packbits(rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480), axis=-1) for i in range(fs.n)])
    np.save(d / f"legocup_{c}.npy", ms)
    area = np.unpackbits(ms, axis=-1).reshape(fs.n, -1).mean(1)
    print(f"{c}: wrote {d / f'legocup_{c}.npy'} area mean {area.mean():.4f} empty {(area == 0).sum()}", flush=True)
