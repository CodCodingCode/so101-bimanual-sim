import json, sys, copy
import numpy as np
sys.path.insert(0, '.')
exec(open('scan.py').read().split("print('base'")[0])
print('base', score(P0), flush=True)
for ar in (0.85, 0.92, 0.96, 1.04, 1.08, 1.15, 1.25, 1.33):
    P = copy.deepcopy(P0)
    P['cams']['side']['lens']['ar'] = ar
    P['cams']['top']['lens']['ar'] = ar
    print(ar, score(P), flush=True)
