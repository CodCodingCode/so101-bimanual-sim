"""Pickplace twin with re-placed objects on top of the frozen twin2 arm fit (fitkit/out/pickplace_so101_sam1.json).
The builder scene (so101_twin/pickplace/so101_pickplace.xml: table, walls, lights, skybox) is kept; its static box
and brick are replaced by mocap copies of the same models (build_scene.py: transparent box with an orange rim, 2x4
pink brick) placed per episode from detections through the NEW cameras; the table/background are adjusted
(work/static_fit.json) and coloured per camera (work/colours.json).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/pickplace_objects/build_pickplace.py calib|render|preview <frames> <out.jpg>
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import objtwin  # noqa: E402

FIT = ROOT / "so101_twin/fitkit/out/pickplace_so101_sam1.json"
SET = ROOT / "neuroscope/molmoact2/store/sets/so101_pickplace_twin2"
REAL = ROOT / "neuroscope/molmoact2/store/sets/so101_pickplace_real"
WORK = HERE / "work"
OBJ = json.loads((WORK / "objects_fit.json").read_text())
STATE = json.loads((WORK / "block_state.json").read_text()) if (WORK / "block_state.json").exists() else {}
COL = json.loads((WORK / "colours.json").read_text()) if (WORK / "colours.json").exists() else {}
STATIC = json.loads((WORK / "static_fit.json").read_text()) if (WORK / "static_fit.json").exists() else {}
BOX_H = 0.06            # builder BOX_SIZE z (a transparent wall height is not measurable from the masks)
BOX_WALL = 0.002
STATIC_MATS = ["worktop", "wall", "floor", "s_cardboard"]
OBJ_MATS = ["box_glass", "box_rim", "brick"]
LAB = np.load(REAL / "labels.npz")
EP = LAB["episode"]
HELD_OFFSET = [0.0, 0.0, 0.0]


def scene_fn(root):
    wb = root.find("worldbody")
    asset = root.find("asset")
    for b in list(wb):
        if b.tag == "body" and b.get("name") in ("box", "brick"):
            wb.remove(b)
    ET.SubElement(asset, "material", name="s_cardboard", rgba="0.6 0.45 0.3 1", specular="0", shininess="0.1")
    for g in wb.findall("geom"):
        if g.get("name") == "table_top" and STATIC.get("table"):
            t = STATIC["table"]
            g.set("size", f"{t['half'][0]} {t['half'][1]} 0.01")
            g.set("pos", f"{t['center'][0]} {t['center'][1]} -0.01")
        if g.get("name") == "wall_back" and STATIC.get("wall_back_y") is not None:
            g.set("pos", f"0 {STATIC['wall_back_y'] + 0.01} 0.45")
            g.set("size", "2 0.01 0.55")          # top at z = 1.0: never between the key light and the arm
            g.set("pos", f"0 {STATIC['wall_back_y'] + 0.01} 0.45")
    for cb in STATIC.get("cardboard", []):
        ET.SubElement(wb, "geom", name=cb["name"], type="box", size=" ".join(map(str, cb["half"])), pos=" ".join(map(str, cb["center"])),
                      quat=f"{math.cos(cb['yaw'] / 2)} 0 0 {math.sin(cb['yaw'] / 2)}", material="s_cardboard", contype="0", conaffinity="0",
                      group="4")   # drawn for the side camera only (see static_fit.json cardboard_how)
    sx, sy = OBJ["shapes"]["rim"]
    sz, r = BOX_H, OBJ["rim"] / 2
    box = ET.SubElement(wb, "body", name="obj_box", mocap="true", pos="0 0 -5")
    ET.SubElement(box, "geom", name="obj_box_floor", type="box", size=f"{sx / 2} {sy / 2} {BOX_WALL}", pos=f"0 0 {BOX_WALL}",
                  material="box_glass", contype="0", conaffinity="0")
    for nm, pos, size in (("obj_box_nx", (-sx / 2 + BOX_WALL, 0, sz / 2), (BOX_WALL, sy / 2, sz / 2)),
                          ("obj_box_px", (sx / 2 - BOX_WALL, 0, sz / 2), (BOX_WALL, sy / 2, sz / 2)),
                          ("obj_box_ny", (0, -sy / 2 + BOX_WALL, sz / 2), (sx / 2, BOX_WALL, sz / 2)),
                          ("obj_box_py", (0, sy / 2 - BOX_WALL, sz / 2), (sx / 2, BOX_WALL, sz / 2))):
        ET.SubElement(box, "geom", name=nm, type="box", size=" ".join(map(str, size)), pos=" ".join(map(str, pos)),
                      material="box_glass", contype="0", conaffinity="0")
    for nm, pos, size in (("obj_rim_nx", (-sx / 2 + r, 0, r), (r, sy / 2, r)), ("obj_rim_px", (sx / 2 - r, 0, r), (r, sy / 2, r)),
                          ("obj_rim_ny", (0, -sy / 2 + r, r), (sx / 2, r, r)), ("obj_rim_py", (0, sy / 2 - r, r), (sx / 2, r, r))):
        ET.SubElement(box, "geom", name=nm, type="box", size=" ".join(map(str, size)), pos=" ".join(map(str, pos)),
                      material="box_rim", contype="0", conaffinity="0")
    hx, hy, hz = OBJ["shapes"]["brick"]
    br = ET.SubElement(wb, "body", name="obj_brick", mocap="true", pos="0 0 -5")
    ET.SubElement(br, "geom", name="obj_brick", type="box", size=f"{hx} {hy} {hz}", material="brick", contype="0", conaffinity="0")
    for i in range(4):
        for j in range(2):
            ET.SubElement(br, "geom", name=f"obj_stud_{i}{j}", type="cylinder", size="0.0024 0.0009",
                          pos=f"{(i - 1.5) * hx / 2} {(j - 0.5) * hy} {hz + 0.0009}", material="brick", contype="0", conaffinity="0")


def yawq(y):
    return (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))


def objects_fn(T, i):
    e = str(int(EP[i]))
    zo = OBJ["z_objects"]
    bx, by, byaw = OBJ["episodes"][e]["rim"]["pose"]
    T.mocap("obj_box", (bx, by, zo), yawq(byaw))
    kx, ky, kyaw, kz = OBJ["episodes"][e]["brick"]["pose"]      # kz: triangulated centre height of this episode
    hx, hy, hz = OBJ["shapes"]["brick"]
    s = STATE.get(str(i), {"state": "table"})
    if s["state"] == "table":
        T.mocap("obj_brick", (kx, ky, kz), yawq(kyaw))
    elif s["state"] == "held":
        sid = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        R = T.d.site_xmat[sid].reshape(3, 3)
        pos = T.d.site_xpos[sid] + R @ np.asarray(s.get("offset", HELD_OFFSET))
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, R.ravel())
        T.mocap("obj_brick", pos, qq)
    else:   # in the box, on its floor
        px, py = s.get("xy", (bx, by))
        T.mocap("obj_brick", (px, py, zo + 2 * BOX_WALL + hz), yawq(s.get("yaw", kyaw)))


def make(colours=None):
    T = objtwin.ObjTwin("pickplace", FIT, scene_fn=scene_fn, colours=colours if colours is not None else COL, objects_fn=objects_fn)
    T.cam_groups = {"side": (4,)}
    return T


def view_of(T):
    vo = {}
    for v, vc in T.F.views.items():
        for e in (vc.get("episodes") or np.unique(T.F.ep).tolist()):
            vo.setdefault(vc["cam"], {})[int(e)] = v
    return vo


def calib():
    T = make(colours={})
    for mat in STATIC_MATS + OBJ_MATS:       # white probe (keep the glass alpha)
        mid = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
        a = T.m.mat_rgba[mid][3]
        T.m.mat_rgba[mid] = (1, 1, 1, a)
        T.m.mat_emission[mid] = 0
    vo = view_of(T)
    real = dict(top=np.load(REAL / "images_top.npy", mmap_mode="r"), side=np.load(REAL / "images_side.npy", mmap_mode="r"))
    gid = lambda n: mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_GEOM, n)
    groups = {"worktop": ["table_top"], "wall": ["wall_back", "wall_right"], "s_cardboard": [c["name"] for c in STATIC.get("cardboard", [])],
              "box_rim": ["obj_rim_nx", "obj_rim_px", "obj_rim_ny", "obj_rim_py"], "brick": ["obj_brick"]}
    acc = {c: {k: ([], []) for k in groups} for c in ("top", "side")}
    first = [int(np.flatnonzero(EP == e)[0]) for e in np.unique(EP)]
    for i in first[::4]:
        T.pose(i)
        for c in ("top", "side"):
            v = vo[c][int(EP[i])]
            med = np.load(WORK / f"median_{v}.npy")
            rgb, _ = T.render(v)
            seg = T.render_seg(v)
            R = np.asarray(real[c][i])
            near_arm = ndi.binary_dilation(np.isin(seg, T.arm_geoms), iterations=10)
            for k, names in groups.items():
                ids = [gid(n) for n in names if gid(n) >= 0]
                if not ids:
                    continue
                reg = np.isin(seg, ids)
                reg = ndi.binary_erosion(reg, iterations=1 if k in ("box_rim", "brick") else 4) & ~near_arm
                if reg.sum() < 10:
                    continue
                if k in ("box_rim", "brick"):
                    # thin / small objects: target = the real pixels of that colour near the model (1-2 px of
                    # placement error would otherwise sample the table)
                    from PIL import Image as _I
                    hsv = np.asarray(_I.fromarray(R).convert("HSV")).astype(int)
                    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
                    cm = ((h_ >= 8) & (h_ <= 35) & (s_ >= 90) & (v_ >= 100)) if k == "box_rim" else (((h_ >= 235) | (h_ <= 10)) & (s_ >= 100) & (v_ >= 90))
                    rr = ndi.binary_dilation(np.isin(seg, ids), iterations=4) & cm & ~near_arm
                    if rr.sum() < 10:
                        continue
                    acc[c][k][0].append(R[rr].astype(float))
                    acc[c][k][1].append(rgb[reg].astype(float))
                    continue
                acc[c][k][0].append(med[reg].astype(float))
                acc[c][k][1].append(rgb[reg].astype(float))
    out = {}
    for c in acc:
        out[c] = {}
        for k, (tg, wh) in acc[c].items():
            if not tg:
                continue
            tg, wh = np.concatenate(tg), np.concatenate(wh)
            target, white = np.median(tg, 0), np.maximum(np.median(wh, 0), 1.0)
            ratio = target / white
            kmax = float(ratio.max())
            L = float(white.mean() / 255.0)
            if k == "wall":       # nearly unlit (emission dominates): no spot-cone pattern on the wall
                E = 4.0
                rgba, em = np.clip(target / 255.0 / (E + L), 0, 1), E
            else:
                rgba, em = (ratio / kmax, (kmax - 1) * L) if kmax > 1 else (ratio, 0.0)
            out[c][k] = [[float(x) for x in rgba] + [1.0], float(em)]
            print(c, k, "target", np.round(target), "white", np.round(white), "rgba", np.round(rgba, 3), "em", round(float(em), 3))
        # the glass keeps the builder's tint and alpha; the floor is never seen
    (WORK / "colours.json").write_text(json.dumps(out, indent=1))


def render_all(out_dir: pathlib.Path):
    T = make()
    vo = view_of(T)
    n = T.F.fs.n
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = {c: np.lib.format.open_memmap(out_dir / f"images_{c}.npy", mode="w+", dtype=np.uint8, shape=(n, T.H, T.W, 3)) for c in ("top", "side")}
    t0 = time.time()
    for i in range(n):
        T.pose(i)
        for c in ("top", "side"):
            img, _ = T.render(vo[c][int(EP[i])])
            imgs[c][i] = img
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    for c in imgs:
        imgs[c].flush()


def preview(idx, path):
    T = make()
    vo = view_of(T)
    real = dict(top=np.load(REAL / "images_top.npy", mmap_mode="r"), side=np.load(REAL / "images_side.npy", mmap_mode="r"))
    rows = []
    for i in idx:
        T.pose(i)
        row = []
        for c in ("top", "side"):
            img, _ = T.render(vo[c][int(EP[i])])
            row += [np.asarray(real[c][i]), img]
        rows.append(np.concatenate([np.asarray(Image.fromarray(x).resize((400, 300))) for x in row], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path, quality=88)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calib":
        calib()
    elif cmd == "preview":
        preview([int(x) for x in sys.argv[2].split(",")], sys.argv[3])
    elif cmd == "render":
        render_all(WORK / "new")
