"""Pick the final twin and write its scores (all with score.py, same masks / protocol).

    .venv/bin/python so101_twin/ballcup/finalize.py --cands out/ballcup_r4_so100_fit.json,out/ballcup_r4_so101_fit.json \
        [--extra out/ballcup_r3_so100_fit.json,...] --mask-tag _t2o

Candidates are scored with the visible-arm masks on the held-out (odd) episodes; the kept one maximises the smaller
of the two cameras' held-out medians (the pass criterion). Writes out/final_fit.json (copy), out/score_final_occ.json
(primary), out/score_final_plain.json (unoccluded silhouette), out/score_final_indep.json (vs the twin-independent
_d1o masks, visible arm), out/model_comparison.json.
"""
import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from score import score  # noqa: E402


def dump(path, fit_json, tag, occ, out, per):
    path.write_text(json.dumps(dict(fit=fit_json, mask_tag=tag, occluders=occ, stats=out,
                                    per_frame={c: [None if v != v else round(float(v), 4) for v in per[c]] for c in per}),
                               indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--mask-tag", default="_t2o")
    a = ap.parse_args()
    # selection: within a mesh, the round with the better EVEN-episode (fit split) score; between meshes, the better
    # held-out score (the prescribed protocol: fit both meshes on even episodes, keep the better held-out)
    rows = []
    for f in a.cands.split(","):
        fj = json.loads((HERE / f).read_text())
        out, _ = score(str(HERE / f), a.mask_tag, True)
        r = dict(model=fj["model"], round=f.split("ballcup_")[-1].split("_fit")[0], fit=f"so101_twin/ballcup/{f}",
                 top=out["top"]["held_out"]["median"], side=out["side"]["held_out"]["median"],
                 top_even=out["top"]["fit_split_even"]["median"], side_even=out["side"]["fit_split_even"]["median"],
                 file=f)
        rows.append(r)
        print({k: v for k, v in r.items() if k != "file"}, flush=True)
    per_model = {}
    for r in rows:
        b = per_model.get(r["model"])
        if b is None or min(r["top_even"], r["side_even"]) > min(b["top_even"], b["side_even"]):
            per_model[r["model"]] = r
    best = max(per_model.values(), key=lambda r: min(r["top"], r["side"]))
    print("per mesh (by even split):", {m: r["round"] for m, r in per_model.items()}, "-> chosen", best["round"])
    f = best["file"]
    for r in rows:
        r.pop("file")
    shutil.copyfile(HERE / f, HERE / "out" / "final_fit.json")
    fj = json.loads((HERE / f).read_text())
    model = fj["model"]
    for name, tag, occ in (("occ", a.mask_tag, True), ("plain", a.mask_tag, False), ("indep", "_d1o", True)):
        out, per = score(str(HERE / f), tag, occ)
        dump(HERE / "out" / f"score_final_{name}.json", f"so101_twin/ballcup/{f}", tag, occ, out, per)
        print(name, {c: (v["held_out"]["median"], v["all_frames"]["median"]) for c, v in out.items()}, flush=True)
    A = F.A
    calib = "old"
    (HERE / "out" / "model_comparison.json").write_text(json.dumps(dict(
        chosen=model, chosen_fit=f"so101_twin/ballcup/{f}", mask_tag=a.mask_tag, metric="held-out median, visible arm",
        rows=rows, joint_sign=(A.SIGNS_SO100[calib] if model == "so100" else A.SIGNS[calib]).tolist(),
        joint_nominal=(A.NOMINAL_SO100[calib] if model == "so100" else [0.0] * 5).tolist() if model == "so100" else [0.0] * 5),
        indent=1))
    print("final:", f)


if __name__ == "__main__":
    main()
