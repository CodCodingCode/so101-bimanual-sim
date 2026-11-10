"""Write a refitted twin set: the builder's scene (table, objects, lights) with the fitted arm model, cameras, lens,
camera lag and joint map from an armfit fit, scored against the SAM masks.

    .venv/bin/python so101_twin/fitkit/twin2.py <scene> <fit.json> [--overwrite]

Output: neuroscope/molmoact2/store/sets/so101_<scene>_twin2/ in the twin layout (images_<cam>.npy uint8
[n,H,W,3], state.npy / prompt.json identical to the real set, labels.npz = real labels with real_or_sim='sim',
source='<scene>_twin2', pair = arange(n)), plus arm_mask_<cam>.npy (packed bits, the twin arm silhouette per frame)
and meta.json with the fitted parameters and the arm IoU against the SAM masks: held_out {median, p10, mean} on the
fit's held-out frames (odd episodes) and all_frames for reference.

Rendering: each view is rendered as a pinhole image with a 25% wider field of view and the same focal length, then
warped through the fitted radial lens (k1, k2) and principal-point offset onto the real pixel grid. Objects stay at
the builder's scene positions unless the scene config gives an objects(model, data, frame) hook.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import armfit  # noqa: E402

PAD = 1.25
SO100_MESHES = None


def _strip_so100(root: ET.Element) -> tuple[ET.Element, list[ET.Element]]:
    """Remove the Menagerie arm from a builder scene. Returns (root, builder extras that hung off arm bodies)."""
    wb = root.find("worldbody")
    arm = next(b for b in wb.findall("body") if b.find('.//joint[@name="Rotation"]') is not None)
    extras = []
    for b in arm.iter("body"):
        for c in list(b):
            if c.tag in ("site",) and c.get("name") == "tcp":
                extras.append((b.get("name"), c))
    wb.remove(arm)
    arm_ref = ET.parse(armfit.MODELS["so100"]["xml"]).getroot()
    arm_meshes = {m.get("name") for m in arm_ref.find("asset").findall("mesh")}
    asset = root.find("asset")
    for m in list(asset):
        if m.tag == "mesh" and m.get("name") in arm_meshes:
            asset.remove(m)
    for tag in ("actuator", "contact", "keyframe", "equality", "tendon", "sensor"):
        for e in root.findall(tag):
            root.remove(e)
    for d in root.findall("default"):
        for sub in list(d):
            if sub.get("class") == "so_arm100":
                d.remove(sub)
        if d.get("class") == "so_arm100":
            root.remove(d)
    return root, extras


def _add_arm(root: ET.Element, kind: str, base_pos, base_quat, look: dict, wrist_cam: bool = False,
             board=None, board_size=None) -> None:
    arm = armfit._arm_root(kind, wrist_cam)
    if board is not None and kind == "so101":
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.radians(np.asarray(board[3:6], float)), "xyz")
        grip = next(b for b in arm.iter("body") if b.get("name") == "gripper")
        ET.SubElement(grip, "geom", name="wrist_board", type="box", group="2", contype="0", conaffinity="0",
                      material="arm_look_body", size=" ".join(map(str, board_size or (0.0215, 0.0215, 0.006))),
                      pos=" ".join(f"{x:.5f}" for x in board[:3]), quat=" ".join(f"{x:.6f}" for x in q))
    spec = armfit.MODELS[kind]
    meshdir = pathlib.Path(spec["meshdir"])
    for d in arm.findall("default"):
        root.insert(1, copy.deepcopy(d))
    asset = root.find("asset")
    for a in arm.find("asset"):
        a = copy.deepcopy(a)
        if a.tag == "mesh":
            a.set("name", "arm_" + (a.get("name") or pathlib.Path(a.get("file")).stem.split(".")[0]))
            a.set("file", str(meshdir / a.get("file")))
        if a.tag == "material":
            a.set("name", "arm_" + a.get("name"))
        asset.append(a)
    for name, rgba in (("arm_look_body", look.get("body", "0.9 0.9 0.9 1")), ("arm_look_servo", look.get("servo", "0.08 0.08 0.08 1")),
                       ("arm_look_jaw", look.get("jaw", look.get("body", "0.9 0.9 0.9 1")))):
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular="0.2", shininess="0.3")
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    shift_xy, shift_z = armfit._base_alignment(kind)
    bq = np.array([float(x) for x in base_quat.split()]) if isinstance(base_quat, str) else np.asarray(base_quat, float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, bq)
    off = R.reshape(3, 3) @ np.array([shift_xy[0], shift_xy[1], shift_z])
    base.set("pos", " ".join(f"{v:.6f}" for v in np.asarray(base_pos, float) + off))
    base.set("quat", " ".join(f"{v:.8f}" for v in bq))
    jaw_meshes = {"so100": {"Fixed_Jaw", "Moving_Jaw"}, "so101": {"wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"}}[kind]
    for g in base.iter("geom"):
        mesh = g.get("mesh")
        if mesh is None:
            continue
        g.set("mesh", "arm_" + mesh)
        is_servo = mesh.startswith("sts3215") or "Motor" in mesh
        g.set("material", "arm_look_servo" if is_servo else "arm_look_jaw" if mesh in jaw_meshes else "arm_look_body")
    for j in base.iter("joint"):
        j.set("name", "arm_" + j.get("name"))
    for b in base.iter("body"):
        b.set("name", "arm_" + b.get("name"))
    root.find("worldbody").append(base)


def minimal_scene(sc: dict) -> str:
    """Bare scene for replacement datasets with no builder: table top at z=0, a wall behind, soft lights. Objects are
    not modelled yet (listed in meta.json as missing)."""
    t = sc.get("table_rgba", "0.15 0.15 0.15 1")
    w = sc.get("wall_rgba", "0.85 0.85 0.85 1")
    tc = sc.get("table_center", (0.0, 0.25))
    th = sc.get("table_half", (0.6, 0.45))
    wall = sc.get("wall", (0.0, 0.9))
    return f"""<mujoco model="twin2_minimal"><compiler angle="radian" autolimits="true"/>
<visual><global offwidth="1280" offheight="720"/><headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/><quality shadowsize="4096"/></visual>
<asset><material name="table" rgba="{t}" specular="0.1"/><material name="wall" rgba="{w}"/><material name="floor" rgba="0.5 0.5 0.5 1"/></asset>
<worldbody><light name="key" pos="0.3 -0.4 1.5" dir="-0.2 0.3 -1" diffuse="0.35 0.35 0.35" castshadow="true"/>
<geom name="table_top" type="box" size="{th[0]} {th[1]} 0.01" pos="{tc[0]} {tc[1]} -0.01" material="table"/>
<geom name="floor" type="plane" size="0 0 0.1" pos="0 0 -0.75" material="floor"/>
<geom name="wall" type="box" size="2 0.01 1.2" pos="{wall[0]} {wall[1]} 0.45" material="wall"/>
<body name="Base"><joint name="Rotation" type="hinge"/><geom type="sphere" size="0.001" contype="0" conaffinity="0"/></body>
</worldbody></mujoco>"""


class Twin2:
    def __init__(self, scene: str, fit_json: pathlib.Path):
        self.scene = scene
        self.cfg = armfit.scene_config(scene)
        self.st = json.loads(pathlib.Path(fit_json).read_text())
        self.kind = self.st["model"]
        self.F = armfit.Fit(dict(self.cfg, real_mask="sam", parts=dict(servos=True)), self.kind)
        self.F.load_state(fit_json)
        self.H, self.W = self.F.H, self.F.W
        bd = armfit.TWIN / scene
        found = list(bd.glob(f"so101_{scene}.xml")) if bd.exists() else []
        root = ET.parse(found[0]).getroot() if found else ET.fromstring(minimal_scene(self.cfg.get("scene", {})))
        comp = root.find("compiler")
        if comp is not None and comp.get("meshdir"):
            md = pathlib.Path(comp.get("meshdir"))
            md = md if md.is_absolute() else (bd / md)
            for m in root.find("asset").findall("mesh"):
                m.set("file", str(md / m.get("file")))
            comp.attrib.pop("meshdir")
        for t in root.iter("texture"):
            if t.get("file") and not pathlib.Path(t.get("file")).is_absolute():
                t.set("file", str(bd / t.get("file")))
        for c in list(root.find("worldbody").iter("camera")):
            pass
        root, _ = _strip_so100(root)
        # drop every builder camera; the fitted views are added below
        for parent in root.iter():
            for c in list(parent):
                if c.tag == "camera":
                    parent.remove(c)
        wb = self.cfg.get("wrist_board")
        _add_arm(root, self.kind, self.cfg["base_pos"], self.cfg["base_quat"], self.cfg.get("look", {}),
                 self.cfg.get("wrist_cam", False), self.st.get("board") if wb else None,
                 (wb or {}).get("size"))
        mount_body = {"jaw": "arm_" + armfit.MODELS[self.kind]["mount"]["jaw"],
                      "shoulder": "arm_" + armfit.MODELS[self.kind]["mount"]["shoulder"]}
        wb = root.find("worldbody")
        for v in self.F.views:
            m = self.F.mounts[v]
            parent = wb if m == "world" else next(b for b in wb.iter("body") if b.get("name") == mount_body[m])
            ET.SubElement(parent, "camera", name="v_" + v, pos="0 0 1", xyaxes="1 0 0 0 1 0", fovy="45")
        vis = root.find("visual")
        g = vis.find("global") if vis is not None and vis.find("global") is not None else None
        rw, rh = int(math.ceil(self.W * PAD)), int(math.ceil(self.H * PAD))
        if g is not None:
            g.set("offwidth", str(max(rw, 640)))
            g.set("offheight", str(max(rh, 480)))
        self.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        # near clip is relative to the model extent; the full scene is much larger than the fitter's arm-only model, so a
        # wrist camera would clip the jaws. Use the fitter's absolute near distance.
        fm = self.F._full().m
        self.m.vis.map.znear = fm.vis.map.znear * fm.stat.extent / self.m.stat.extent
        self.d = mujoco.MjData(self.m)
        self.rw, self.rh = rw, rh
        self.r = mujoco.Renderer(self.m, rh, rw)
        self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "arm_" + n)]
                     for n in self._joint_names()]
        self.arm_geoms = np.array([gi for gi in range(self.m.ngeom)
                                   if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[gi]) or "").startswith("arm_")
                                   and self.m.geom_group[gi] == 2])
        self.maps = {v: self._lens_map(v) for v in self.F.views}

    def _joint_names(self):
        a = ET.parse(armfit.MODELS[self.kind]["xml"]).getroot()
        return [j.get("name") for j in a.find("worldbody").iter("joint") if j.get("name")][:6]

    def _lens_map(self, v):
        """Real pixel -> padded pinhole pixel (inverse of armfit.resample_real's pinhole -> real map)."""
        p = self.F.p[v]
        h, w = self.H, self.W
        f = (h / 2) / math.tan(math.radians(p[6]) / 2)
        vv, uu = np.mgrid[0:h, 0:w].astype(float)
        xd = (uu + 0.5 - w / 2 - p[9]) / f
        yd = (vv + 0.5 - h / 2 - p[10]) / f
        xn, yn = xd.copy(), yd.copy()
        for _ in range(30):   # fixed point for x_d = x_n (1 + k1 r^2 + k2 r^4)
            r2 = xn * xn + yn * yn
            dd = 1 + p[7] * r2 + p[8] * r2 * r2
            xn, yn = xd / dd, yd / dd
        return np.stack([f * yn + self.rh / 2 - 0.5, f * xn + self.rw / 2 - 0.5])

    def set_view(self, v):
        p = self.F.p[v]
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "v_" + v)
        self.m.cam_pos[cid] = p[:3]
        self.m.cam_quat[cid] = armfit.cam_quat(p[3], p[4], p[5])
        # same focal length on the padded canvas
        self.m.cam_fovy[cid] = 2 * math.degrees(math.atan(math.tan(math.radians(p[6]) / 2) * self.rh / self.H))
        mujoco.mj_camlight(self.m, self.d)   # recompute camera poses (a mounted camera moves with its body)

    def pose(self, i):
        q = self.F.q(i)
        for a, x in zip(self.qadr, q):
            self.d.qpos[a] = x
        hook = self.cfg.get("objects")
        if hook:
            hook(self.m, self.d, i)
        mujoco.mj_forward(self.m, self.d)

    def render(self, v):
        self.set_view(v)
        self.r.update_scene(self.d, camera="v_" + v)
        img = self.r.render()
        mp = self.maps[v]
        out = np.stack([ndi.map_coordinates(img[..., c].astype(float), mp, order=1, mode="nearest") for c in range(3)], -1)
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera="v_" + v)
        seg = self.r.render()[..., 0]
        self.r.disable_segmentation_rendering()
        arm = np.isin(seg, self.arm_geoms).astype(float)
        mask = ndi.map_coordinates(arm, mp, order=1, mode="constant") > 0.5
        return np.clip(out, 0, 255).astype(np.uint8), mask


def iou_stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), median=float(np.median(x)), p10=float(np.percentile(x, 10)), mean=float(x.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("fit")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    T = Twin2(a.scene, pathlib.Path(a.fit))
    F, fs = T.F, T.F.fs
    name = f"so101_{a.scene}_twin2"
    out = fs.path.parent / name
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} exists (use --overwrite)")
    cams = fs.cameras
    view_of = {}   # dataset camera -> {episode: view}
    for v, vc in F.views.items():
        for e in (vc.get("episodes") or np.unique(F.ep).tolist()):
            view_of.setdefault(vc["cam"], {})[int(e)] = v
    imgs = {c: np.zeros((fs.n, T.H, T.W, 3), np.uint8) for c in cams}
    masks = {c: np.zeros((fs.n, T.H, T.W), bool) for c in cams}
    ious = {v: {} for v in F.views}
    t0 = time.time()
    for i in range(fs.n):
        T.pose(i)
        for c in cams:
            v = view_of.get(c, {}).get(int(F.ep[i]))
            if v is None:
                continue
            img, m = T.render(v)
            imgs[c][i], masks[c][i] = img, m
            rm = F.real(i, v, 1) > 0.5
            u = (rm | m).sum()
            ious[v][i] = float((rm & m).sum() / u) if u > 50 else None
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{fs.n} {time.time() - t0:.0f}s", flush=True)
    ev = {v: set(map(int, F.ev_idx[v])) for v in F.views}
    held = {v: iou_stats([ious[v][i] for i in ious[v] if i in ev[v]]) for v in F.views}
    held_all = {v: iou_stats([ious[v][i] for i in ious[v] if int(F.ep[i]) % 2 == 1]) for v in F.views}
    allf = {v: iou_stats(list(ious[v].values())) for v in F.views}
    by_cam = {}
    for v in F.views:
        c = F.views[v]["cam"]
        by_cam.setdefault(c, []).extend([ious[v][i] for i in ious[v] if i in ev[v]])
    held_cam = {c: iou_stats(x) for c, x in by_cam.items()}
    out.mkdir(parents=True, exist_ok=True)
    for c in cams:
        np.save(out / f"images_{c}.npy", imgs[c])
        np.save(out / f"arm_mask_{c}.npy", np.packbits(masks[c], axis=-1))
    np.save(out / "state.npy", fs.state.astype(np.float32))
    shutil.copyfile(fs.path / "prompt.json", out / "prompt.json")
    labels = {k: np.asarray(v) for k, v in fs.labels.items()}
    labels["real_or_sim"] = np.array(["sim"] * fs.n)
    labels["source"] = np.array([f"{a.scene}_twin2"] * fs.n)
    labels["pair"] = np.arange(fs.n, dtype=np.int64)
    np.savez(out / "labels.npz", **labels)
    meta = dict(name=name, n=int(fs.n), cameras=list(cams), real_or_sim="sim", source=f"{a.scene}_twin2",
                twin_of=fs.meta["name"], image_shape=[T.H, T.W, 3], dataset=fs.meta.get("dataset"),
                episodes=fs.meta.get("episodes"), per_episode=fs.meta.get("per_episode"),
                built=time.strftime("%Y-%m-%d %H:%M:%S"), builder="fitkit/twin2.py (session 57ffc516)",
                arm_model=T.kind, fit=str(pathlib.Path(a.fit)), views=T.st["views"], lag_frames=T.st.get("lag_frames"),
                joint=dict(sign=T.st["joint_sign"], off_rad=T.st["joint_off_rad"], gain=T.st.get("joint_gain"),
                           q_mean=T.st.get("q_mean"), base_map="scene twin.real_to_sim -> SO-100 radians -> sign*((1+gain)(q-q_mean)+q_mean)+off"),
                iou_metric="whole-arm silhouette IoU vs twin-prompted SAM 2.1 masks; held_out = the fit's held-out frames "
                           "(odd episodes, never used in the fit); held_out_all_odd = every odd-episode frame; all_frames for reference",
                arm_iou=dict(held_out_per_camera=held_cam, held_out=held, held_out_all_odd=held_all, all_frames=allf),
                objects=("builder scene positions" + (" + scene hook" if T.cfg.get("objects") else " (static)"))
                if (armfit.TWIN / a.scene).exists() else "NOT MODELLED: minimal scene (table, wall); objects missing")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["arm_iou"], indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
