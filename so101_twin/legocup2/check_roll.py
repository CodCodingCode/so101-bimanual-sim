"""Top-camera close-up for the wrist-roll candidates: real | twin roll 0 | twin roll 180 (camera parts coloured:
mount orange, module green, jaws blue). -> out/check_roll.jpg"""
import json
import sys
import pathlib

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
import dataclasses

sc = F.load_scene("legocup")
sc.ARM = dataclasses.replace(sc.ARM, colors={"body": (0.75, 0.75, 0.75, 1), "jaw": (0.2, 0.5, 1.0, 1),
                                             "motor": (0.3, 0.3, 0.3, 1), "cam_mount": (1.0, 0.45, 0.1, 1),
                                             "camera": (0.1, 0.9, 0.2, 1)})
fs = F.real_set(sc)
P0 = F.init_params(sc)
if len(sys.argv) > 1:
    P0 = F.reconcile(sc, json.loads(pathlib.Path(sys.argv[1]).read_text())["params"])
rig = F.Rig(sc, P0, full=False)
rows = []
for i in (5, 85, 13, 120):
    tiles = [np.asarray(fs.obs(i)["images"]["top"])]
    for r in (0.0, 180.0):
        P = json.loads(json.dumps(P0))
        P["offsets_deg"][4] = r
        rig.set_params(P)
        tiles.append(rig.rgb("top", "all", fs.state[i], 640, 480))
    rows.append(np.concatenate(tiles, 1))
im = Image.fromarray(np.concatenate(rows, 0))
im = im.resize((im.width * 2 // 3, im.height * 2 // 3))
im.save(HERE / "out" / "check_roll.jpg", quality=85)
