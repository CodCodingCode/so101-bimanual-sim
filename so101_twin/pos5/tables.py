"""README tables from a score json (score.py): per camera and reading, held_out / all_frames {n, median, p10, mean},
held-out medians by phase, fraction of held-out frames >= 0.80, excluded fractions.

    .venv/bin/python so101_twin/pos5/tables.py out/pos5_r4_fit_score_c3_ck_s2_lsam_occ.json
"""
import json
import sys

import numpy as np

import fit as F
import p5

j = json.loads(open(sys.argv[1]).read())
fs = F.real_set(p5.scene())
phase = np.asarray(fs.labels["phase"])
odd = p5.odd()
for tag, rr in j["results"].items():
    print(f"\n### masks {tag}")
    for c, r in rr.items():
        for reading, key, pf in (("strict", "strict", "per_frame"), ("occluder-excluded", "excluded", "per_frame_excluded")):
            if key not in r:
                continue
            s = r[key]
            v = np.array([np.nan if x is None else x for x in r[pf]])
            ge = np.mean(v[odd & np.isfinite(v)] >= 0.8)
            ph = ", ".join(f"{p} {np.nanmedian(v[odd & (phase == p)]):.3f}" for p in ("approach", "grasp", "release"))
            print(f"| {c} | {reading} | held_out n={s['held_out']['n']} median **{s['held_out']['median']:.3f}** "
                  f"p10 {s['held_out']['p10']:.3f} mean {s['held_out']['mean']:.3f} | all_frames n={s['all_frames']['n']} "
                  f"median {s['all_frames']['median']:.3f} p10 {s['all_frames']['p10']:.3f} mean "
                  f"{s['all_frames']['mean']:.3f} | held-out >= 0.80: {ge:.0%} | by phase (held-out median): {ph} |")
        if "excluded_fraction" in r:
            e = r["excluded_fraction"]
            print(f"| {c} | excluded fraction of the union | held_out mean {e['held_out']['mean']:.3f} median "
                  f"{e['held_out']['median']:.3f} | all_frames mean {e['all_frames']['mean']:.3f} median "
                  f"{e['all_frames']['median']:.3f} |")
