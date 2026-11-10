"""Ricky twin with objects: fitted table/mat/walls + per-episode bowl and block on top of the frozen twin2 arm fit.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/build_ricky.py calib      # colour calibration
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/build_ricky.py render     # work/images_<cam>.npy
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/build_ricky.py preview 0,40,..   # quick sheet
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
import camgeo  # noqa: E402
import objtwin  # noqa: E402

FIT = ROOT / "so101_twin/fitkit/out/ricky_so101_sess.json"
SET = ROOT / "neuroscope/molmoact2/store/sets/so101_ricky_twin2"
REAL = ROOT / "neuroscope/molmoact2/store/sets/so101_ricky_real"
WORK = HERE / "work"
STATIC = json.loads((WORK / "static_fit.json").read_text())
OBJ = json.loads((WORK / "objects_fit.json").read_text())
STATE = json.loads((WORK / "block_state.json").read_text()) if (WORK / "block_state.json").exists() else {}
COL = json.loads((WORK / "colours.json").read_text()) if (WORK / "colours.json").exists() else {}
BOWL_WALL = 0.0025     # paper bowl wall / floor thickness
STATIC_MATS = ["s_table", "s_mat", "s_wall_back", "s_wall_left", "s_floor", "s_chair"]
# chair back left of the arm in the top view: its image box (cols 0-200, rows 108-439, same in all three top sessions)
# cast onto the plane y = -0.25 (depth is not observable from one camera; it is behind the side camera)
CHAIR = dict(center=(0.715, -0.25, 0.14), half=(0.235, 0.01, 0.19))
OBJ_MATS = ["o_bowl_out", "o_bowl_in", "o_block"]


def rot2(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def walls():
    """Back wall at the table's far edge (table-local x = -hx); side wall through the room corner seen by the top
    camera (pixel column 940 of the top_c median, cast onto the back wall plane)."""
    t = STATIC["table"]
    yaw = math.radians(t["yaw_deg"])
    cx, cy = t["center"]
    hx, hy = t["half"]
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])            # table-local +x in world
    back_pt = np.array([cx, cy, 0.0]) + n * (-hx)
    st = json.loads(FIT.read_text())
    K = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
    p = [st["views"]["top_c"][k] for k in K]
    X = camgeo.cast_plane(p, [940.0], [300.0], 720, 1280, n, float(n @ back_pt))[0]
    ly = -math.sin(yaw) * (X[0] - cx) + math.cos(yaw) * (X[1] - cy)   # table-local y of the corner
    return dict(yaw=yaw, back_local_x=-hx, side_local_y=float(ly), corner=X.tolist())


def scene_fn(root):
    wb = root.find("worldbody")
    asset = root.find("asset")
    for g in list(wb):
        if g.tag == "geom" and g.get("name") in ("table_top", "wall"):
            wb.remove(g)
    for g in wb.iter("geom"):
        if g.get("name") == "floor":
            g.set("material", "s_floor")
    for name in STATIC_MATS + OBJ_MATS:
        ET.SubElement(asset, "material", name=name, rgba="1 1 1 1", specular="0", shininess="0.1", reflectance="0")
    o = OBJ["shapes"]
    rb, rt, h = o["bowl"]
    asset.append(objtwin.mesh_frustum_shell("bowl_out", rb, rt, h, outward=True))
    k = (rt - rb) / h
    asset.append(objtwin.mesh_frustum_shell("bowl_in", rb - BOWL_WALL, rt - BOWL_WALL - k * 0.0, h - BOWL_WALL, outward=False,
                                            bottom=True, z0=BOWL_WALL))
    asset.append(objtwin.mesh_annulus("bowl_rim", rt - BOWL_WALL, rt, h))
    t, m = STATIC["table"], STATIC["mat"]
    yaw_t, yaw_m = math.radians(t["yaw_deg"]), math.radians(m["yaw_deg"])
    q = lambda y: f"{math.cos(y / 2):.8f} 0 0 {math.sin(y / 2):.8f}"
    ET.SubElement(wb, "geom", name="s_table", type="box", size=f"{t['half'][0]} {t['half'][1]} 0.015",
                  pos=f"{t['center'][0]} {t['center'][1]} -0.015", quat=q(yaw_t), material="s_table", contype="0", conaffinity="0")
    ET.SubElement(wb, "geom", name="s_mat", type="box", size=f"{m['half'][0]} {m['half'][1]} 0.001",
                  pos=f"{m['center'][0]} {m['center'][1]} -0.0005", quat=q(yaw_m), material="s_mat", contype="0", conaffinity="0")   # top 0.5 mm above the table: hides at most 0.5 mm of the base
    w = walls()
    cx, cy = t["center"]
    # walls stop at z = 1.0: the placeholder key light (0.3, -0.4, 1.5) must keep lighting the arm exactly as before
    bx, by = rot2(yaw_t, w["back_local_x"] - 0.01, 0.0)
    ET.SubElement(wb, "geom", name="s_wall_back", type="box", size="0.01 3 0.875", pos=f"{cx + bx} {cy + by} 0.125",
                  quat=q(yaw_t), material="s_wall_back", contype="0", conaffinity="0")
    sx, sy = rot2(yaw_t, 0.0, w["side_local_y"] - 0.01)
    ET.SubElement(wb, "geom", name="s_wall_left", type="box", size="3 0.01 0.875", pos=f"{cx + sx} {cy + sy} 0.125",
                  quat=q(yaw_t), material="s_wall_left", contype="0", conaffinity="0")
    ET.SubElement(wb, "geom", name="s_chair", type="box", size=" ".join(map(str, CHAIR["half"])), pos=" ".join(map(str, CHAIR["center"])),
                  material="s_chair", contype="0", conaffinity="0")
    b = ET.SubElement(wb, "body", name="obj_bowl", mocap="true", pos="0 0 -5")
    for gname, mesh, mat in (("obj_bowl_out", "bowl_out", "o_bowl_out"), ("obj_bowl_in", "bowl_in", "o_bowl_in"), ("obj_bowl_rim", "bowl_rim", "o_bowl_in")):
        ET.SubElement(b, "geom", name=gname, type="mesh", mesh=mesh, material=mat, contype="0", conaffinity="0", mass="0.01")
    bk = ET.SubElement(wb, "body", name="obj_block", mocap="true", pos="0 0 -5")
    hx, hy, hz = OBJ["shapes"]["block"]
    ET.SubElement(bk, "geom", name="obj_block", type="box", size=f"{hx} {hy} {hz}", material="o_block", contype="0", conaffinity="0", mass="0.01")


LAB = np.load(REAL / "labels.npz")
EP = LAB["episode"]


def yawq(y):
    return (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))


def objects_fn(T, i):
    e = str(int(EP[i]))
    zo = OBJ["z_objects"]
    bx, by = OBJ["episodes"][e]["bowl"]["pose"]
    T.mocap("obj_bowl", (bx, by, zo))
    kx, ky, kyaw = OBJ["episodes"][e]["block"]["pose"]
    hx, hy, hz = OBJ["shapes"]["block"]
    s = STATE.get(str(i), {"state": "table"})
    if s["state"] == "table":
        T.mocap("obj_block", (kx, ky, zo + hz), yawq(kyaw))
    elif s["state"] == "held":
        sid = mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        R = T.d.site_xmat[sid].reshape(3, 3)
        pos = T.d.site_xpos[sid] + R @ np.asarray(s.get("offset", HELD_OFFSET))
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, R.ravel())
        T.mocap("obj_block", pos, qq)
    else:   # in the bowl: on its floor at the detected xy (bowl centre if unseen)
        px, py = s.get("xy", (bx, by))
        T.mocap("obj_block", (px, py, zo + BOWL_WALL + hz), yawq(kyaw))


HELD_OFFSET = [0.0, 0.0, 0.0]


def make(colours=None):
    return objtwin.ObjTwin("ricky", FIT, scene_fn=scene_fn, colours=colours if colours is not None else COL, objects_fn=objects_fn)


def view_of(T):
    vo = {}
    for v, vc in T.F.views.items():
        for e in (vc.get("episodes") or np.unique(T.F.ep).tolist()):
            vo.setdefault(vc["cam"], {})[int(e)] = v
    return vo


def calib():
    """Per-camera material colours: real target colour / rendered colour of a white material, per geom region."""
    T = make(colours={})
    vo = view_of(T)
    real = np.load(REAL / "images_top.npy", mmap_mode="r"), np.load(REAL / "images_side.npy", mmap_mode="r")
    real = dict(top=real[0], side=real[1])
    gid = lambda n: mujoco.mj_name2id(T.m, mujoco.mjtObj.mjOBJ_GEOM, n)
    groups = {"s_table": ["s_table"], "s_mat": ["s_mat"], "s_wall_back": ["s_wall_back"], "s_wall_left": ["s_wall_left"],
              "s_floor": ["floor"], "s_chair": ["s_chair"], "o_bowl_out": ["obj_bowl_out"], "o_bowl_in": ["obj_bowl_in", "obj_bowl_rim"], "o_block": ["obj_block"]}
    acc = {c: {k: ([], []) for k in groups} for c in ("top", "side")}
    first = [int(np.flatnonzero(EP == e)[0]) for e in np.unique(EP)]
    for i in first[::3]:
        T.pose(i)
        for c in ("top", "side"):
            v = vo[c][int(EP[i])]
            med = np.load(WORK / f"median_{v}.npy")
            rgb, _ = T.render(v)               # every material white
            seg = T.render_seg(v)
            R = np.asarray(real[c][i])
            arm = np.isin(seg, T.arm_geoms)
            near_arm = ndi.binary_dilation(arm, iterations=12)
            for k, names in groups.items():
                reg = np.isin(seg, [gid(n) for n in names])
                reg = ndi.binary_erosion(reg, iterations=4) & ~near_arm
                if reg.sum() < 30:
                    continue
                src = R if k.startswith("o_") else med      # objects: this frame; static scene: session median
                acc[c][k][0].append(src[reg].astype(float))
                acc[c][k][1].append(rgb[reg].astype(float))
    out = {}
    for c in acc:
        out[c] = {}
        for k, (tg, wh) in acc[c].items():
            if not tg:
                continue
            tg, wh = np.concatenate(tg), np.concatenate(wh)
            target = np.median(tg, 0)
            white = np.maximum(np.median(wh, 0), 1.0)
            ratio = target / white
            kmax = float(ratio.max())
            L = float(white.mean() / 255.0)
            if k in ("s_wall_back", "s_wall_left"):
                # nearly unlit walls: emission E dominates, so the placeholder spot light's cone (which must stay as it
                # is for the arm) leaves no bright ellipse on them: colour ~ rgba * (E + light)
                E = 4.0
                rgba, em = np.clip(target / 255.0 / (E + L), 0, 1), E
            elif kmax > 1:
                rgba, em = ratio / kmax, (kmax - 1) * L
            else:
                rgba, em = ratio, 0.0
            out[c][k] = [[float(x) for x in rgba] + [1.0], float(em)]
            print(c, k, "target", np.round(target), "white", np.round(white), "rgba", np.round(rgba, 3), "em", round(float(em), 3))
    (WORK / "colours.json").write_text(json.dumps(out, indent=1))


def render_all(out_dir: pathlib.Path, idx=None):
    T = make()
    vo = view_of(T)
    n = T.F.fs.n
    idx = list(range(n)) if idx is None else idx
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = {c: np.lib.format.open_memmap(out_dir / f"images_{c}.npy", mode="w+", dtype=np.uint8, shape=(n, T.H, T.W, 3)) for c in ("top", "side")}
    t0 = time.time()
    for k, i in enumerate(idx):
        T.pose(i)
        for c in ("top", "side"):
            img, _ = T.render(vo[c][int(EP[i])])
            imgs[c][i] = img
        if (k + 1) % 40 == 0:
            print(f"{k + 1}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    for c in imgs:
        imgs[c].flush()
    return T


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
        rows.append(np.concatenate([np.asarray(Image.fromarray(x).resize((480, 270))) for x in row], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path, quality=88)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calib":
        calib()
    elif cmd == "preview":
        preview([int(x) for x in sys.argv[2].split(",")], sys.argv[3])
    elif cmd == "render":
        render_all(WORK / "new")
    elif cmd == "walls":
        print(walls())
