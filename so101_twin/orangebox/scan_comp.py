import json, sys, copy
import numpy as np
sys.path.insert(0, '.')
import fit as F
exec(open('scan.py').read().split("print('base'")[0].split("import fit as F\n", 1)[1])
print('base', score(P0), flush=True)
for j, name in ((1, 'lift'), (2, 'elbow'), (3, 'wflex')):
    res = []
    for kk in (-10, -5, 5, 10):
        P = copy.deepcopy(P0)
        c = [0.0] * 5
        c[j] = kk
        P['compliance'] = c
        res.append((kk, score(P)))
    print(name, res, flush=True)
