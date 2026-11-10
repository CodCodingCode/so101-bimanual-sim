"""Merge the per-camera fits into out/legocup_final_fit.json and score it with the agreed metric on both SAM mask sets
(primary: the re-segmented _lc2 masks; also the original builder-prompted masks).

    .venv/bin/python so101_twin/legocup2/finalize.py --top out/legocup_r2top_fit.json --side out/legocup_r2side_fit.json
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--top", required=True)
ap.add_argument("--side", required=True)
ap.add_argument("--primary", default="_lc3")
ap.add_argument("--tags", default=",_lc2,_lc3")
a = ap.parse_args()
T = json.loads((HERE / a.top).read_text())["params"]
S = json.loads((HERE / a.side).read_text())["params"]
P = dict(cams=dict(top=T["cams"]["top"], side=S["cams"]["side"]), offsets_deg=T["offsets_deg"],
         scales=T.get("scales", [1.0] * 5), grip_scale=S["grip_scale"], grip_off=S.get("grip_off", 0.0))
out = HERE / "out" / "legocup_final_fit.json"
out.write_text(json.dumps(dict(params=P, merged_from=[a.top, a.side]), indent=1))
py = str(HERE.parents[1] / ".venv" / "bin" / "python")
subprocess.run([py, str(HERE / "score_lc2.py"), str(out), "--tags", a.tags, "--json", "out/score_final.json"], check=True)
sc = json.loads((HERE / "out" / "score_final.json").read_text())
prim = sc[a.primary if a.primary else "orig"]
strip = lambda d: {c: {k: v for k, v in d[c].items() if k != "per_frame_new"} for c in d}   # noqa: E731
fj = dict(params=P, merged_from=[a.top, a.side], stats=strip(prim),
          extra_stats={(k if k != "orig" else "orig_masks") + ("" if k == "orig" else "_masks"): strip(v)
                       for k, v in sc.items() if k != (a.primary or "orig")},
          mask_files={c: f"so101_twin/masks/legocup_{c}{a.primary}.npy" for c in ("top", "side")},
          protocol="whole-arm silhouette IoU between the twin's arm mask (640x480, lens-warped padded pinhole render, "
                   "unfilled pixels = background) and the SAM 2.1 mask of the real arm, per camera, frames with the real "
                   "arm visible (>= 0.1% of the image); every fit stage used EVEN episodes only; held_out = ODD episodes. "
                   "Primary masks: so101_twin/masks/legocup_<cam>" + a.primary + ".npy (segment_lc2.py: consensus prompts "
                   "builder twin + this twin, hard negatives on the black cup / red brick, brick pixels removed); "
                   "extra_stats: orig_masks = the original builder-prompted so101_twin/masks/legocup_<cam>.npy, "
                   "_lc2_masks = the previous SAM round.")
out.write_text(json.dumps(fj, indent=1))
for c in ("top", "side"):
    h, al = fj["stats"][c]["new_heldout"], fj["stats"][c]["new"]
    print(f"FINAL {c}: held_out n={h['n']} median {h['median']:.3f} p10 {h['p10']:.3f} mean {h['mean']:.3f} | all_frames "
          f"n={al['n']} median {al['median']:.3f} p10 {al['p10']:.3f} mean {al['mean']:.3f}")
print("wrote", out)
