"""REDLEGO: choose the final fit and write out/final_fit.json (+ out/model_comparison.json).

    .venv/bin/python so101_twin/redlego/finalize.py --mask-tag _t1ok --cands out/a_fit.json,out/b_fit.json

Within the candidates, the choice uses only the EVEN-episode (fit split) median of the strict reading on the scored
masks, so the held-out episodes never influence it; every candidate's held-out numbers are reported."""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mask-tag", required=True)
ap.add_argument("--occ-tag", default="_d1_occ")
ap.add_argument("--cands", required=True)
a = ap.parse_args()
rows = []
for f in a.cands.split(","):
    p = HERE / f
    out, _ = S.score(str(p), a.mask_tag, a.occ_tag)
    fj = json.loads(p.read_text())
    even = float(np.mean([out[c]["fit_split_even"]["median"] for c in out]))
    rows.append(dict(fit=f, model=fj.get("model"), even_mean_median=round(even, 4),
                     **{f"{c}_{k}": out[c]["strict_arm_only"][k]["median"] for c in out for k in ("held_out", "all_frames")},
                     **{f"{c}_excl_held_out": out[c]["occluder_excluded"]["held_out"]["median"] for c in out}))
    print(rows[-1], flush=True)
best = max(rows, key=lambda r: r["even_mean_median"])
fj = json.loads((HERE / best["fit"]).read_text())
fj["chosen_from"] = best["fit"]
(HERE / "out" / "final_fit.json").write_text(json.dumps(fj, indent=1))
(HERE / "out" / "model_comparison.json").write_text(json.dumps(dict(rule="max mean over cameras of the EVEN-episode "
                                                                    "strict median", rows=rows, chosen=best["fit"]), indent=1))
print("chosen", best["fit"])
