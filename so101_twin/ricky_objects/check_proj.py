"""Check camgeo.project against the twin2 arm mask: project every arm mesh vertex and compare extents."""
import pathlib, sys
import mujoco
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fitkit_copy")); sys.path.insert(0, str(HERE))
import twin2, camgeo
scene, fit = sys.argv[1], sys.argv[2]
T = twin2.Twin2(scene, pathlib.Path(fit))
F = T.F
view_of = {}
for v, vc in F.views.items():
    for e in (vc.get("episodes") or np.unique(F.ep).tolist()):
        view_of.setdefault(vc["cam"], {})[int(e)] = v
for i in [int(x) for x in sys.argv[3].split(",")]:
    T.pose(i)
    pts = []
    for g in T.arm_geoms:
        if T.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = T.m.geom_dataid[g]
        vv = T.m.mesh_vert[T.m.mesh_vertadr[mid]:T.m.mesh_vertadr[mid] + T.m.mesh_vertnum[mid]]
        pts.append(vv @ T.d.geom_xmat[g].reshape(3, 3).T + T.d.geom_xpos[g])
    P = np.concatenate(pts)
    for c in F.fs.cameras:
        v = view_of[c][int(F.ep[i])]
        img, m = T.render(v)
        u, w_, z = camgeo.project(F.p[v], P, T.H, T.W)
        ok = (u >= 0) & (u < T.W - 1) & (w_ >= 0) & (w_ < T.H - 1) & (z > 0)
        ui, vi = np.round(u[ok]).astype(int), np.round(w_[ok]).astype(int)
        inside = m[vi, ui].mean()
        ys, xs = np.nonzero(m)
        print(f"frame {i} {c}/{v}: verts in mask {inside:.3f}; mask bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]  "
              f"proj bbox x[{u[ok].min():.1f},{u[ok].max():.1f}] y[{w_[ok].min():.1f},{w_[ok].max():.1f}]")
