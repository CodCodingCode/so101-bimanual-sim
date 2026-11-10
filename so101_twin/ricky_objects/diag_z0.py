"""Diagnostic: is the two-view disagreement a table-height (plane z0) offset? Joint fit of z0 + bowl shape +
per-episode bowl xy on a subset of episodes, both views."""
import json, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import camgeo, objfit
o = json.load(open('work/objects_fit.json'))
st = json.load(open('../fitkit/out/ricky_so101_sess.json'))
K = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
P = {v: [d[k] for k in K] for v, d in st['views'].items()}
z = np.load('work/seg_first.npz', allow_pickle=True); H, W = map(int, z['shape'])
lab = np.load('../../neuroscope/molmoact2/store/sets/so101_ricky_real/labels.npz'); ep = lab['episode']
vo = {}
for v, d in st['views'].items():
    for e in d['episodes']: vo.setdefault(d['cam'], {})[e] = v
D = {}
for k in range(len(z['frame'])):
    i, c, pr = int(z['frame'][k]), str(z['cam'][k]), str(z['prompt'][k]); e = int(ep[i])
    if 'bowl' not in pr or e % 4: continue
    m = np.unpackbits(z['bits'][k], axis=-1)[..., :W].astype(bool)
    crop = objfit.crop_of(m, 60, H, W)
    D.setdefault(e, {})[c] = (m[crop[0]:crop[1], crop[2]:crop[3]], crop)
eps = sorted(D)
def L(e, xy, z0, sh, cams=('top', 'side')):
    out = []
    for c in cams:
        m, crop = D[e][c]
        out.append(1 - objfit.iou(objfit.hull_mask(P[vo[c][e]], objfit.frustum_pts(xy[0], xy[1], z0, *sh), H, W, crop), m))
    return np.mean(out)
for z0 in (-0.03, -0.015, 0.0, 0.015, 0.03):
    sh = np.array(o['shapes']['bowl'])
    pos = {e: np.array(o['episodes'][str(e)]['bowl']['pose']) for e in eps}
    for rnd in range(2):
        for e in eps:
            r = minimize(lambda x: L(e, x, z0, sh), pos[e], method='Nelder-Mead', options=dict(initial_simplex=np.vstack([pos[e], pos[e] + [0.01, 0], pos[e] + [0, 0.01]]), maxfev=200))
            pos[e] = r.x
        r = minimize(lambda s: np.mean([L(e, pos[e], z0, s) for e in eps]) if (s > 0.01).all() else 1, sh, method='Nelder-Mead',
                     options=dict(initial_simplex=np.vstack([sh] + [sh + np.eye(3)[j] * 0.005 for j in range(3)]), maxfev=200))
        sh = r.x
    iou_t = 1 - np.mean([L(e, pos[e], z0, sh, ('top',)) for e in eps]); iou_s = 1 - np.mean([L(e, pos[e], z0, sh, ('side',)) for e in eps])
    print(f"z0 {z0:+.3f}: shape {np.round(sh, 4)} IoU top {iou_t:.3f} side {iou_s:.3f}", flush=True)
