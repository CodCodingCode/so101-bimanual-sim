"""Fill README.md tables from the final fit / score jsons (so the numbers in the README are the computed ones).

    .venv/bin/python so101_twin/ballcup/write_readme.py --fit out/final_fit.json
"""
import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


TEMPLATE = open(HERE / "readme_text.py").read().split('"""')[1] if (HERE / "readme_text.py").exists() else ""


def row(name, s):
    return f"| {name} | {s['n']} | {s['median']:.3f} | {s['p10']:.3f} | {s['mean']:.3f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    a = ap.parse_args()
    fj = json.loads((HERE / a.fit).read_text())
    P = fj["params"]
    S = {k: json.loads((HERE / "out" / f"score_final_{k}.json").read_text())["stats"] for k in ("occ", "plain", "indep")}
    cmp_ = json.loads((HERE / "out" / "model_comparison.json").read_text())
    t = TEMPLATE
    RO = json.loads((HERE / "out" / "score_final_realocc.json").read_text())["stats"]
    res = ["| reading | camera / frames | n | median | p10 | mean |", "|---|---|---|---|---|---|"]
    for c in ("top", "side"):
        r = RO[c]
        res.append("| A strict arm-only " + row(f"**{c}** held_out (odd episodes)", r["strict_arm_only"]["held_out"]))
        res.append("| A strict arm-only " + row(f"{c} all_frames", r["strict_arm_only"]["all_frames"]))
        res.append("| **B real-occluder-excluded** " + row(f"**{c}** held_out (odd episodes)",
                                                          r["occluder_excluded"]["held_out"]))
        res.append("| B real-occluder-excluded " + row(f"{c} all_frames", r["occluder_excluded"]["all_frames"]))
        res.append("| C visible arm (reference only) " + row(f"{c} held_out", S["occ"][c]["held_out"]))
        res.append("| C visible arm (reference only) " + row(f"{c} all_frames", S["occ"][c]["all_frames"]))
        res.append("| check: C vs twin-independent masks _d1o " + row(f"{c} held_out", S["indep"][c]["held_out"]))
    fr = ["| camera | held-out mean | held-out median | all-frames mean | all-frames median | (of the twin arm mask: all-frames mean / median) |",
          "|---|---|---|---|---|---|"]
    for c in ("top", "side"):
        u, t_ = RO[c]["excluded_fraction_of_union"], RO[c]["excluded_fraction_of_twin_mask"]
        fr.append(f"| {c} | {u['held_out']['mean']:.4f} | {u['held_out']['median']:.4f} | {u['all_frames']['mean']:.4f} | "
                  f"{u['all_frames']['median']:.4f} | {t_['all_frames']['mean']:.4f} / {t_['all_frames']['median']:.4f} |")
    exc = ", ".join(f"{c}: {S['occ'][c]['excluded_not_in_view']['all']} of 320 ({S['occ'][c]['excluded_not_in_view']['held_out']} held-out)"
                    for c in ("top", "side"))
    cams = ["| camera | pos (m, world) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in ("top", "side"):
        p = P["cams"][c]
        g = p["groups"]["all"]
        L = p["lens"]
        cams.append(f"| {c} | {', '.join(f'{v:.4f}' for v in g['pos'])} | {', '.join(f'{v:.4f}' for v in g['quat'])} | "
                    f"{p['fovy']:.2f} | {L['k1']:.4f} | {L['k2']:.4f} | {L['cx']:.4f} | {L['cy']:.4f} | "
                    f"{p.get('dt_frames', 0.0):.2f} |")
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    sign = cmp_["joint_sign"]
    joints = ["| joint | sign | nominal (deg) | offset (deg) | scale |", "|---|---|---|---|---|"]
    for k, nm in enumerate(names):
        joints.append(f"| {nm} | {sign[k]:+.0f} | {cmp_['joint_nominal'][k]:.0f} | {P['offsets_deg'][k]:+.2f} | "
                      f"{P.get('scales', [1.0] * 5)[k]:.3f} (fixed) |")
    joints.append(f"| gripper | | | grip_scale {P['grip_scale']:.3f} | |")
    cm = ["| model | top held-out median | side held-out median | fit json |", "|---|---|---|---|"]
    for r in cmp_["rows"]:
        cm.append(f"| {r['model']} ({r['round']}) | {r['top']:.3f} | {r['side']:.3f} | `{r['fit']}` |")
    cup = P["cup"]
    t = t.replace("RESULT_TABLE", "\n".join(res)).replace("EXCLUDED", exc).replace("FRACTION_TABLE", "\n".join(fr))
    for c in ("top", "side"):
        t = t.replace(f"B_{c.upper()}", f"{RO[c]['occluder_excluded']['held_out']['median']:.3f}")
        t = t.replace(f"A_{c.upper()}", f"{RO[c]['strict_arm_only']['held_out']['median']:.3f}").replace("CAMERA_TABLE", "\n".join(cams))
    t = t.replace("JOINT_TABLE", "\n".join(joints)).replace("MODEL_TABLE", "\n".join(cm))
    t = t.replace("CUP_SHAPE", f"r_top {cup['r_top'] * 100:.1f} cm, r_bot {cup['r_bot'] * 100:.1f} cm, height "
                               f"{cup['h'] * 100:.1f} cm (top-ray height fraction zc {cup['zc']:.2f})")
    (HERE / "README.md").write_text(t)
    print("wrote README.md")


if __name__ == "__main__":
    main()
