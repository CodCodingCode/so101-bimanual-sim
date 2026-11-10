"""Per-part twin silhouettes (0 bg, 1 base+shoulder, 2 upper arm, 3 forearm, 4 wrist+gripper) for segment_parts.py.

    .venv/bin/python so101_twin/stack2/export_parts.py OUT_DIR --fit-json F
"""
import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

PART_OF = {"base": 1, "shoulder": 1, "upper_arm": 2, "lower_arm": 3, "wrist": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fit-json", required=True)
    a = ap.parse_args()
    sc = F.load_scene("stack")
    P = F.reconcile(sc, json.loads(pathlib.Path(a.fit_json).read_text())["params"])
    rig = F.Rig(sc, P)
    m = rig.arm.m
    lut = np.zeros(m.ngeom + 1, np.uint8)
    for g in rig.arm.arm_geoms:
        b = m.geom_bodyid[g]
        part = 4
        x = b
        while x > 0:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, x)
            if nm in PART_OF:
                part = PART_OF[nm]
                break
            x = m.body_parentid[x]
        lut[g + 1] = part
    fs = F.real_set(sc)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in sc.CAMERAS:
        grp = F.frame_groups(sc, c)
        L = np.zeros((fs.n, 480, 640), np.uint8)
        for i in range(fs.n):
            rig.pose(c, grp[i], F.frame_state(sc, P, c, i))
            seg = rig.lens[c].render(rig.arm, c, 640, 480, seg=True)
            ok = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            L[i] = np.where(ok, lut[np.clip(seg[..., 0] + 1, 0, m.ngeom)], 0)
        np.savez_compressed(out / f"stack_{c}_parts.npz", labels=L)
        print(c, "part areas", [round(float((L == k).mean()), 4) for k in range(1, 5)], flush=True)


if __name__ == "__main__":
    main()
