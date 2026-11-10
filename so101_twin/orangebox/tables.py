"""Markdown tables from score.py outputs.  .venv/bin/python so101_twin/orangebox/tables.py out/<a>_score<tag>.json ..."""
import json
import sys

for f in sys.argv[1:]:
    o = json.load(open(f))
    print(f"\n`{f.split('/')[-1]}` (model {o['model']}, masks `{o['masks']}`)\n")
    print("| camera / frames | n | median | p10 | mean | excluded (arm not in view) |")
    print("|---|---|---|---|---|---|")
    for c, v in o["cams"].items():
        for k, lab in (("held_out", "held_out (odd episodes)"), ("all_frames", "all_frames")):
            s = v[k]
            ex = v["excluded_held_out"] if k == "held_out" else v["excluded_not_in_view"]
            print(f"| {c} {lab} | {s['n']} | {s['median']:.3f} | {s['p10']:.3f} | {s['mean']:.3f} | {ex} |")
