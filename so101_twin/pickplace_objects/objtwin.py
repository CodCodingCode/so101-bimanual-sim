"""Twin2 with modelled scene objects. The arm, cameras, lens, lag and joint map are the frozen fit, built by the
copied fitkit code exactly as twin2.Twin2 builds them; a scene hook then replaces the placeholder table/wall and adds
objects (mocap bodies posed per frame), and <statistic> is pinned to the placeholder model's values so the near/far
clip and the shadow-map extent (both scaled by the model extent in MuJoCo) are unchanged, so the arm renders
byte-identically wherever nothing new covers it.

Per-camera material colours: `colours[cam][material] = (rgba, emission)` are applied before each render (the two
real cameras have different white balance / exposure; the arm's materials are never touched)."""
from __future__ import annotations

import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE / "fitkit_copy") not in sys.path:
    sys.path.insert(0, str(HERE / "fitkit_copy"))
import armfit  # noqa: E402
import twin2  # noqa: E402


class ObjTwin(twin2.Twin2):
    def __init__(self, scene: str, fit_json, scene_fn=None, colours=None, objects_fn=None):
        import json
        self.scene = scene
        self.cfg = armfit.scene_config(scene)
        self.st = json.loads(pathlib.Path(fit_json).read_text())
        self.kind = self.st["model"]
        self.F = armfit.Fit(dict(self.cfg, real_mask="sam", parts=dict(servos=True)), self.kind)
        self.F.load_state(fit_json)
        self.H, self.W = self.F.H, self.F.W
        bd = armfit.TWIN / scene
        found = list(bd.glob(f"so101_{scene}.xml")) if bd.exists() else []
        root = ET.parse(found[0]).getroot() if found else ET.fromstring(twin2.minimal_scene(self.cfg.get("scene", {})))
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
        root, _ = twin2._strip_so100(root)
        for parent in root.iter():
            for c in list(parent):
                if c.tag == "camera":
                    parent.remove(c)
        twin2._add_arm(root, self.kind, self.cfg["base_pos"], self.cfg["base_quat"], self.cfg.get("look", {}),
                       self.cfg.get("wrist_cam", False))
        mount_body = {"jaw": "arm_" + armfit.MODELS[self.kind]["mount"]["jaw"],
                      "shoulder": "arm_" + armfit.MODELS[self.kind]["mount"]["shoulder"]}
        wb = root.find("worldbody")
        for v in self.F.views:
            m = self.F.mounts[v]
            parent = wb if m == "world" else next(b for b in wb.iter("body") if b.get("name") == mount_body[m])
            ET.SubElement(parent, "camera", name="v_" + v, pos="0 0 1", xyaxes="1 0 0 0 1 0", fovy="45")
        vis = root.find("visual")
        g = vis.find("global") if vis is not None and vis.find("global") is not None else None
        rw, rh = int(math.ceil(self.W * twin2.PAD)), int(math.ceil(self.H * twin2.PAD))
        if g is not None:
            g.set("offwidth", str(max(rw, 640)))
            g.set("offheight", str(max(rh, 480)))
        # placeholder model statistics (what twin2 compiled): pinned below
        m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.stat0 = dict(extent=float(m0.stat.extent), center=m0.stat.center.copy(), meansize=float(m0.stat.meansize),
                          meanmass=float(m0.stat.meanmass), meaninertia=float(m0.stat.meaninertia))
        del m0
        if scene_fn is not None:
            scene_fn(root)
        for s in root.findall("statistic"):
            root.remove(s)
        ET.SubElement(root, "statistic", extent=repr(self.stat0["extent"]), center=" ".join(repr(float(x)) for x in self.stat0["center"]),
                      meansize=repr(self.stat0["meansize"]), meanmass=repr(self.stat0["meanmass"]),
                      meaninertia=repr(self.stat0["meaninertia"]))
        self.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        assert abs(self.m.stat.extent - self.stat0["extent"]) < 1e-12
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
        self.obj_geoms = np.array([gi for gi in range(self.m.ngeom)
                                   if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[gi]) or "").startswith("obj_")])
        self.maps = {v: self._lens_map(v) for v in self.F.views}
        self.colours = colours or {}
        self.objects_fn = objects_fn
        self.cam_of = {v: vc["cam"] for v, vc in self.F.views.items()}

    def mocap(self, body: str, pos, quat=(1, 0, 0, 0)):
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        k = self.m.body_mocapid[bid]
        self.d.mocap_pos[k] = pos
        self.d.mocap_quat[k] = quat

    def pose(self, i):
        q = self.F.q(i)
        for a, x in zip(self.qadr, q):
            self.d.qpos[a] = x
        mujoco.mj_forward(self.m, self.d)          # arm kinematics first: a held object follows the jaws
        if self.objects_fn is not None:
            self.objects_fn(self, i)
        mujoco.mj_forward(self.m, self.d)

    def set_colours(self, cam):
        for mat, (rgba, em) in self.colours.get(cam, {}).items():
            mid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
            self.m.mat_rgba[mid] = rgba
            self.m.mat_emission[mid] = em

    def _opt(self, v):
        """Default visual option (groups 0-2) plus the extra geom groups this camera sees (self.cam_groups)."""
        opt = mujoco.MjvOption()
        for g in getattr(self, "cam_groups", {}).get(self.cam_of[v], ()):
            opt.geomgroup[g] = 1
        return opt

    def render(self, v):
        """twin2.Twin2.render with a per-camera scene option (identical to it when the camera has no extra groups)."""
        from scipy import ndimage as ndi
        self.set_colours(self.cam_of[v])
        self.set_view(v)
        opt = self._opt(v)
        self.r.update_scene(self.d, camera="v_" + v, scene_option=opt)
        img = self.r.render()
        mp = self.maps[v]
        out = np.stack([ndi.map_coordinates(img[..., c].astype(float), mp, order=1, mode="nearest") for c in range(3)], -1)
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera="v_" + v, scene_option=opt)
        seg = self.r.render()[..., 0]
        self.r.disable_segmentation_rendering()
        arm = np.isin(seg, self.arm_geoms).astype(float)
        mask = ndi.map_coordinates(arm, mp, order=1, mode="constant") > 0.5
        return np.clip(out, 0, 255).astype(np.uint8), mask

    def render_seg(self, v):
        """Geom-id segmentation on the real grid (nearest), -1 = background."""
        from scipy import ndimage as ndi
        self.set_view(v)
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera="v_" + v, scene_option=self._opt(v))
        seg = self.r.render()[..., 0].astype(np.int32)
        self.r.disable_segmentation_rendering()
        return ndi.map_coordinates(seg, self.maps[v], order=0, mode="nearest")


def mesh_frustum_shell(name: str, rb: float, rt: float, h: float, outward: bool, n: int = 64, bottom: bool = False, z0: float = 0.0):
    """Side surface of a truncated cone (open), faces oriented outward (outer skin) or inward (inner skin); with
    bottom=True also a disk at z0 facing up (the inside floor). Returns an ET <mesh> element (inertia='shell')."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vb = np.stack([rb * np.cos(a), rb * np.sin(a), np.full(n, z0)], 1)
    vt = np.stack([rt * np.cos(a), rt * np.sin(a), np.full(n, z0 + h)], 1)
    V = np.concatenate([vb, vt])
    Fc = []
    for k in range(n):
        k2 = (k + 1) % n
        tri = [(k, k2, n + k2), (k, n + k2, n + k)]
        for t in tri:
            Fc.append(t if outward else t[::-1])
    if bottom:
        V = np.concatenate([V, [[0, 0, z0]]])
        c = len(V) - 1
        for k in range(n):
            Fc.append((c, k, (k + 1) % n) if True else None)
    return ET.Element("mesh", name=name, vertex=" ".join(f"{x:.6f}" for x in V.ravel()),
                      face=" ".join(str(int(x)) for x in np.array(Fc).ravel()), inertia="shell")


def mesh_annulus(name: str, r_in: float, r_out: float, z: float, n: int = 64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.concatenate([np.stack([r_in * np.cos(a), r_in * np.sin(a), np.full(n, z)], 1),
                        np.stack([r_out * np.cos(a), r_out * np.sin(a), np.full(n, z)], 1)])
    Fc = []
    for k in range(n):
        k2 = (k + 1) % n
        Fc += [(k, n + k, n + k2), (k, n + k2, k2)]
    return ET.Element("mesh", name=name, vertex=" ".join(f"{x:.6f}" for x in V.ravel()),
                      face=" ".join(str(int(x)) for x in np.array(Fc).ravel()), inertia="shell")
