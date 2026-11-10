"""Arm-identity check for an objects twin (generic: pass the build module name).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/check_identity.py build_ricky <set_dir> <backup_dir> [n=20]

For n frames spread over the episodes, per camera:
  armonly : arm-only silhouette re-rendered with the frozen fit (only geom group 2 = the arm meshes drawn)
  stored  : the set's arm_mask_<cam>.npy (written by twin2; kept byte-identical)
  visible : arm pixels in the NEW scene (segmentation of the full new scene)
Reports: armonly vs stored (xor px; for a builder scene the stored mask already lacks the arm pixels the builder's
objects covered), visible subset of armonly, armonly minus visible = pixels where a new scene geom is in front (by
geom), and RGB of the new image vs the backed-up old image on arm pixels visible in both (eroded 2 px)."""
import importlib, json, pathlib, sys
import mujoco
import numpy as np
from scipy import ndimage as ndi
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
B = importlib.import_module(sys.argv[1])
setdir, bkdir = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
T = B.make()
vo = B.view_of(T)
n = T.F.fs.n
eps = np.unique(B.EP)
pick = [int(np.flatnonzero(B.EP == e)[k % int((B.EP == e).sum())]) for k, e in enumerate(eps[np.linspace(0, len(eps) - 1, N).astype(int)])]
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[2] = 1
from scipy import ndimage as _ndi  # noqa
res = []
for c in ("top", "side"):
    stored = np.load(setdir / f"arm_mask_{c}.npy", mmap_mode="r")
    new = np.load(setdir / f"images_{c}.npy", mmap_mode="r")
    old = np.load(bkdir / f"images_{c}.npy", mmap_mode="r")
    for i in pick:
        v = vo[c][int(B.EP[i])]
        T.pose(i)
        T.set_view(v)
        T.r.enable_segmentation_rendering()
        T.r.update_scene(T.d, camera="v_" + v, scene_option=opt)
        s_arm = T.r.render()[..., 0].astype(float)
        T.r.disable_segmentation_rendering()
        armonly = _ndi.map_coordinates(np.isin(s_arm, T.arm_geoms).astype(float), T.maps[v], order=1, mode="constant") > 0.5
        img, vis = T.render(v)          # same call that wrote the set: vis = arm pixels visible in the new scene
        same_img = bool(np.array_equal(img, np.asarray(new[i])))
        st = np.unpackbits(np.asarray(stored[i]), axis=-1)[..., : T.W].astype(bool)
        seg = T.render_seg(v)
        hidden = armonly & ~vis
        by = {}
        for g in np.unique(seg[hidden]):
            nm = mujoco.mj_id2name(T.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) if g >= 0 else "background"
            by[nm or str(int(g))] = int((seg[hidden] == g).sum())
        core = ndi.binary_erosion(vis, iterations=2) & ndi.binary_erosion(st, iterations=2)
        d = np.abs(np.asarray(new[i]).astype(int) - np.asarray(old[i]).astype(int)).max(-1)[core]
        r = dict(frame=int(i), cam=c, view=v, armonly_px=int(armonly.sum()), stored_px=int(st.sum()),
                 armonly_xor_stored=int((armonly ^ st).sum()), stored_minus_armonly=int((st & ~armonly).sum()),
                 visible_px=int(vis.sum()), visible_outside_armonly=int((vis & ~armonly).sum()), hidden_px=int(hidden.sum()),
                 hidden_by=by, rerender_equals_written=same_img,
                 old_visible_now_hidden=int((st & ~vis).sum()),
                 old_visible_now_hidden_by={(mujoco.mj_id2name(T.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(int(g))) if g >= 0 else "background": int((seg[st & ~vis] == g).sum()) for g in np.unique(seg[st & ~vis])},
                 new_visible_not_old=int((vis & ~st).sum()),
                 rgb_core_px=int(core.sum()), rgb_core_identical_frac=float((d == 0).mean()) if len(d) else None,
                 rgb_core_max_absdiff=int(d.max()) if len(d) else None, rgb_core_mean_absdiff=float(d.mean()) if len(d) else None)
        res.append(r)
        print(json.dumps(r), flush=True)
summ = dict(frames=pick, n_checks=len(res),
            armonly_equals_stored=sum(r["armonly_xor_stored"] == 0 for r in res),
            stored_outside_armonly_px_max=max(r["stored_minus_armonly"] for r in res),
            visible_outside_armonly_px_max=max(r["visible_outside_armonly"] for r in res),
            hidden_px_total=sum(r["hidden_px"] for r in res),
            hidden_by_total={k: sum(r["hidden_by"].get(k, 0) for r in res) for k in sorted({k for r in res for k in r["hidden_by"]})},
            rerender_equals_written=sum(r["rerender_equals_written"] for r in res),
            old_visible_now_hidden_total=sum(r["old_visible_now_hidden"] for r in res),
            old_visible_now_hidden_by_total={k: sum(r["old_visible_now_hidden_by"].get(k, 0) for r in res) for k in sorted({k for r in res for k in r["old_visible_now_hidden_by"]})},
            new_visible_not_old_total=sum(r["new_visible_not_old"] for r in res),
            arm_px_total=sum(r["armonly_px"] for r in res),
            rgb_core_identical_frac_min=min(r["rgb_core_identical_frac"] for r in res if r["rgb_core_identical_frac"] is not None),
            rgb_core_max_absdiff=max(r["rgb_core_max_absdiff"] for r in res if r["rgb_core_max_absdiff"] is not None))
print(json.dumps(summ, indent=1))
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "arm_identity_check.json").write_text(json.dumps(dict(summary=summ, per_frame=res), indent=1))
