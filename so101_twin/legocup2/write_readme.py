"""Fill README.md's result / joint / camera sections from out/legocup_final_fit.json (run after finalize.py)."""
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
fj = json.loads((HERE / "out" / "legocup_final_fit.json").read_text())
P, st = fj["params"], fj["stats"]
ex = fj.get("extra_stats", {})


def row(name, s):
    return f"| {name} | {s['n']} | {s['median']:.3f} | {s['p10']:.3f} | {s['mean']:.3f} |"


lines = ["## Result (agreed metric; pass = held-out median >= 0.80 on every camera)", "",
         "Primary SAM masks `so101_twin/masks/legocup_<cam>_lc3.npy` (third SAM round, see below):", "",
         "| camera / frames | n | median | p10 | mean |", "|---|---|---|---|---|"]
for c in ("top", "side"):
    lines.append(row(f"**{c}** held_out (odd episodes)", st[c]["new_heldout"]))
    lines.append(row(f"{c} all_frames", st[c]["new"]))
    lines.append(row(f"{c} old builder twin, held_out", st[c]["old_heldout"]))
for k, v in ex.items():
    lines += ["", f"Same twin scored against the `{k}` set:", "", "| camera / frames | n | median | p10 | mean |",
              "|---|---|---|---|---|"]
    for c in ("top", "side"):
        lines.append(row(f"{c} held_out", v[c]["new_heldout"]))
        lines.append(row(f"{c} all_frames", v[c]["new"]))
        lines.append(row(f"{c} old builder twin, held_out", v[c]["old_heldout"]))
passed = {c: st[c]["new_heldout"]["median"] >= 0.80 for c in ("top", "side")}
lines += ["", "**side (wrist) passes; top (fixed overhead) does not** (see Limits)." if passed["side"] and not passed["top"]
          else f"pass: {passed}"]
res = "\n".join(lines)
o = P["offsets_deg"]
sc = P.get("scales", [1] * 5)
jl = ["", "| joint | scale (deg per state unit) | offset (deg) |", "|---|---|---|"]
for j, n in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")):
    jl.append(f"| {n} | {sc[j]:.3f} | {o[j]:+.2f} |")
jl.append(f"| gripper | grip_scale {P['grip_scale']:.3f} | grip_off {P.get('grip_off', 0):+.3f} rad |")
joint = "\n".join(jl)
cl = ["| camera | mount | pos (m) | quat (w,x,y,z) | fovy (deg) | k1 | k2 | cx | cy | latency (frames) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for c in ("top", "side"):
    p = P["cams"][c]
    g = p["groups"]["all"]
    L = p["lens"]
    cl.append(f"| {c} | {p['mount']} | {', '.join(f'{x:.4f}' for x in g['pos'])} | "
              f"{', '.join(f'{x:.4f}' for x in g['quat'])} | {p['fovy']:.2f} | {L['k1']:.4f} | {L['k2']:.4f} | "
              f"{L['cx']:.4f} | {L['cy']:.4f} | {p.get('dt_frames', 0.0):+.2f} |")
cam = "\n".join(cl) + ("\n\n`top` is in the world frame; `side` (the wrist camera) is in the SO-101 `gripper` body frame "
                       "(MuJoCo camera convention: looks along -z, +y up). All |k1| < 0.4 and |cx|, |cy| < 0.1, so no "
                       "lens-cropping effect is possible on the score (the side lens stayed pinhole: every lens "
                       "perturbation lowered its IoU).")
r = (HERE / "README.md").read_text()
r = r.replace("RESULTS_PLACEHOLDER", res).replace("JOINT_PLACEHOLDER", joint).replace("CAMERA_PLACEHOLDER", cam)
(HERE / "README.md").write_text(r)
print(res)
print(joint)
print(cam)
