"""JAR: pick the final twin among candidate fits by the EVEN-episode (fit split) strict score on the final masks
(the held-out odd episodes are not used for the choice), write out/final_fit.json and out/model_comparison.json.

    .venv/bin/python so101_twin/jar/finalize.py --mask-tag _t2o --cands out/a_fit.json,out/b_fit.json
Rule (fixed before the r4 results were seen): maximise min over cameras of the even-episode MEAN IoU (bulk and tail;
the even median alone is dominated by the folded rest pose, 62 % of the frames); report held-out for all.
"""
import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-tag", default="_t2o")
    ap.add_argument("--cands", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.cands.split(","):
        p = HERE / f if not pathlib.Path(f).is_absolute() else pathlib.Path(f)
        fj = json.loads(p.read_text())
        out, _ = score(str(p), a.mask_tag)
        row = dict(fit_json=f"so101_twin/jar/{p.relative_to(HERE)}", model=fj.get("model"), mask_tag_fitted=fj.get("mask_tag"),
                   even_median={c: v["fit_split_even"]["median"] for c, v in out.items()},
                   even_mean={c: v["fit_split_even"]["mean"] for c, v in out.items()},
                   even_p10={c: v["fit_split_even"]["p10"] for c, v in out.items()},
                   held_out_median={c: v["strict_arm_only"]["held_out"]["median"] for c, v in out.items()},
                   held_out_p10={c: v["strict_arm_only"]["held_out"]["p10"] for c, v in out.items()})
        row["even_min"] = min(row["even_mean"].values())
        row["k1"] = {c: round(fj["params"]["cams"][c]["lens"]["k1"], 4) for c in ("top", "side")}
        rows.append(row)
        print(row, flush=True)
    best = max(rows, key=lambda r: r["even_min"])
    src = HERE.parents[1] / best["fit_json"]
    shutil.copyfile(src, HERE / "out" / "final_fit.json")
    (HERE / "out" / "model_comparison.json").write_text(json.dumps(dict(
        metric="strict arm-only IoU on masks " + a.mask_tag, rule="max over candidates of min over cameras of the "
        "EVEN-episode mean IoU (fit split); held-out (odd) shown, not used for the choice", rows=rows,
        chosen_fit=best["fit_json"]), indent=1))
    print("chosen", best["fit_json"])


if __name__ == "__main__":
    main()
