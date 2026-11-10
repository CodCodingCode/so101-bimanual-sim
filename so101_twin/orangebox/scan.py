"""1-D sensitivity scans of single parameters (render IoU, even episodes, 320x240) around a fit json.
    .venv/bin/python so101_twin/orangebox/scan.py <fit.json> <masks tag>"""
import json, sys, copy
import numpy as np
sys.path.insert(0, '.')
import fit as F
sc = F.load_scene('orangebox')
P0 = F.reconcile(sc, json.load(open(sys.argv[1]))['params'])
tag = sys.argv[2]
rig = F.Rig(sc, P0)
fs = F.real_set(sc)
ep = np.asarray(fs.labels['episode']).astype(int)
idx = np.nonzero(ep % 2 == 0)[0][::3]
real = {c: F.down2(F.load_masks('orangebox', c, tag)[idx]) for c in sc.CAMERAS}
grp = {c: F.frame_groups(sc, c)[idx] for c in sc.CAMERAS}


def score(P):
    rig.set_params(P)
    out = {}
    for c in sc.CAMERAS:
        v = []
        for k, i in enumerate(idx):
            s = rig.mask(c, grp[c][k], F.frame_state(sc, P, c, i), 320, 240)
            r = real[c][k]
            v.append((s & r).sum() / max((s | r).sum(), 1))
        out[c] = round(float(np.median(v)), 3), round(float(np.mean(v)), 3)
    return out


print('base', score(P0))
scans = {
    'roll_off': (lambda P, d: P['offsets_deg'].__setitem__(4, P['offsets_deg'][4] + d), (-30, -20, -10, 10, 20, 30)),
    'wflex_off': (lambda P, d: P['offsets_deg'].__setitem__(3, P['offsets_deg'][3] + d), (-8, -4, 4, 8)),
    'elbow_off': (lambda P, d: P['offsets_deg'].__setitem__(2, P['offsets_deg'][2] + d), (-6, -3, 3, 6)),
    'lift_off': (lambda P, d: P['offsets_deg'].__setitem__(1, P['offsets_deg'][1] + d), (-6, -3, 3, 6)),
    'grip_off': (lambda P, d: P.__setitem__('grip_off', P.get('grip_off', 0) + d), (-0.2, -0.1, 0.1, 0.2)),
    'board_size': (lambda P, d: P.__setitem__('board_size', P.get('board_size', 1) + d), (-0.3, 0.3, 0.6)),
}
for name, (fn, ds) in scans.items():
    res = []
    for d in ds:
        P = copy.deepcopy(P0)
        fn(P, d)
        res.append((d, score(P)))
    print(name, res, flush=True)
