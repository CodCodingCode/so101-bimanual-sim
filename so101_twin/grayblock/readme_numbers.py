"""GRAYBLOCK: print the numbers for README.md (markdown rows) from out/score_final.json, out/final_fit.json,
out/model_comparison.json, out/score_final_d1kc.json, out/objects.json and the twin set's meta.json."""
import json
import pathlib

import numpy as np

H = pathlib.Path(__file__).resolve().parent
O = H / "out"
sj = json.loads((O / "score_final.json").read_text())
fj = json.loads((O / "final_fit.json").read_text())
cm = json.loads((O / "model_comparison.json").read_text())
dk = json.loads((O / "score_final_d1kc.json").read_text())
st = sj["stats"]
P = fj["params"]
print("## readings")
for c in ("top", "side"):
    for rd, key in (("strict arm-only", "strict_arm_only"), ("real-occluder-excluded", "occluder_excluded")):
        for sp in ("held_out", "all_frames"):
            v = st[c][key][sp]
            print(f"| {rd} | {c} {sp} | {v['n']} | {v['median']:.3f} | {v['p10']:.3f} | {v['mean']:.3f} |")
    print(c, "excluded not in view", st[c]["excluded_not_in_view"], "empty after excl",
          st[c].get("frames_empty_after_exclusion"))
    fu, ft = st[c]["excluded_fraction_of_union"], st[c]["excluded_fraction_of_twin_mask"]
    print(f"| {c} | {fu['held_out']['mean']:.4f} | {fu['held_out']['median']:.4f} | {fu['all_frames']['mean']:.4f} | "
          f"{fu['all_frames']['median']:.4f} | {ft['all_frames']['mean']:.4f} / {ft['all_frames']['median']:.4f} |")
    print(c, "even", st[c]["fit_split_even"])
print("## d1kc check")
for c in ("top", "side"):
    v = dk["stats"][c]["strict_arm_only"]
    print(f"| `_d1kc` | {c} | {v['held_out']['n']} | {v['held_out']['median']:.3f} | {v['held_out']['p10']:.3f} | "
          f"{v['held_out']['mean']:.3f} | {v['all_frames']['median']:.3f} |")
print("## joint map")
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
S = (1, -1, 1, 1, -1)
cp = P.get("compliance", [0.0] * 5)
for j, s, o, k in zip(J, S, P["offsets_deg"], cp):
    print(f"| {j} | {s:+d} | 1 (fixed) | {o:+.2f} | {k:+.2f} |")
print("grip_scale", round(P["grip_scale"], 4), "grip_off", round(P.get("grip_off", 0.0), 4))
print("## cameras")
for c in ("top", "side"):
    p = P["cams"][c]
    g = p["groups"]["all"]
    L = p["lens"]
    print(f"| {c} | " + ", ".join(f"{x:.4f}" for x in g["pos"]) + " | " + ", ".join(f"{x:.4f}" for x in
          np.asarray(g["quat"]) / np.linalg.norm(g["quat"])) + f" | {p['fovy']:.2f} | {L['k1']:+.4f} | {L['k2']:+.4f} | "
          f"{L['cx']:+.4f} | {L['cy']:+.4f} | {p.get('dt_frames', 0.0):+.2f} |")
print("## model comparison")
for r in cm["rows"]:
    print(f"| `{r['fit_json']}` | {r['model']} | `{r['mask_tag_fitted']}` | {r['k1']['top']:+.3f} / {r['k1']['side']:+.3f} | "
          f"{r['even_mean']['top']:.3f} / {r['even_mean']['side']:.3f} | {r['even_median']['top']:.3f} / "
          f"{r['even_median']['side']:.3f} | {r['held_out_median']['top']:.3f} / {r['held_out_median']['side']:.3f} | "
          f"{r['held_out_p10']['top']:.3f} / {r['held_out_p10']['side']:.3f} |")
print("chosen", cm["chosen_fit"])
ob = json.loads((O / "objects.json").read_text())
print("## objects", ob["box_size"], ob["box_iou"], ob["block_counts"], list(ob["layout"].keys()))
