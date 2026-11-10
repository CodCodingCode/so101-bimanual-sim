"""REDLEGO: print the README tables (markdown) from the final fit / score jsons.
    .venv/bin/python so101_twin/redlego/tables.py --fit out/final_fit.json --score out/score_final.json [--cmp a.json,b.json]"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def row(reading, cam, split, s):
    return f"| {reading} | {cam} | {split} | {s['n']} | {s['median']:.3f} | {s['p10']:.3f} | {s['mean']:.3f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--cmp", default="")
    a = ap.parse_args()
    fj = json.loads((HERE / a.fit).read_text())
    sj = json.loads((HERE / a.score).read_text())
    P, st = fj["params"], sj["stats"]
    print("| reading | camera | frames | n | median | p10 | mean |\n|---|---|---|---|---|---|---|")
    for c in ("top", "side"):
        for k in ("held_out", "all_frames"):
            print(row("A strict arm-only", c, k, st[c]["strict_arm_only"][k]))
        if "occluder_excluded" in st[c]:
            for k in ("held_out", "all_frames"):
                print(row("B real-occluder-excluded", c, k, st[c]["occluder_excluded"][k]))
    print()
    if "occluder_excluded" in st["top"]:
        print("| camera | held-out mean | held-out median | all-frames mean | all-frames median | frames empty after exclusion |")
        print("|---|---|---|---|---|---|")
        for c in ("top", "side"):
            u = st[c]["excluded_fraction_of_union"]
            print(f"| {c} | {u['held_out']['mean']:.4f} | {u['held_out']['median']:.4f} | {u['all_frames']['mean']:.4f} | "
                  f"{u['all_frames']['median']:.4f} | {st[c]['frames_empty_after_exclusion']} |")
        print()
    print("excluded (real arm < 0.1 % of the image):", {c: st[c]["excluded_not_in_view"] for c in ("top", "side")})
    print("\n| camera | pos (m, world) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in ("top", "side"):
        p = P["cams"][c]
        g = p["groups"]["all"]
        L = p["lens"]
        print(f"| {c} | {', '.join(f'{v:.4f}' for v in g['pos'])} | {', '.join(f'{v:.4f}' for v in g['quat'])} | "
              f"{p['fovy']:.2f} | {L['k1']:.4f} | {L['k2']:.4f} | {L['cx']:.4f} | {L['cy']:.4f} | {p.get('dt_frames', 0):.2f} |")
    print("\n| joint | scale (deg / unit) | offset (deg) |\n|---|---|---|")
    for j, n in enumerate(NAMES):
        print(f"| {n} | {P['scales'][j]:.3f} | {P['offsets_deg'][j]:.2f} |")
    print(f"| gripper | grip_scale {P['grip_scale']:.3f} | grip_off {P.get('grip_off', 0.0):.3f} rad |")
    if "bar" in P:
        print("bar", [round(v, 4) for v in P["bar"]])
    if "blob" in P:
        print("blob", [round(v, 4) for v in P["blob"]])
    for f in [x for x in a.cmp.split(",") if x]:
        j = json.loads((HERE / f).read_text())
        s = j["stats"]
        print(f"{f}: model {j.get('model')} " + " ".join(
            f"{c} held-out {s[c]['new_heldout']['median']:.3f} all {s[c]['new']['median']:.3f}" for c in ("top", "side")))


if __name__ == "__main__":
    main()
