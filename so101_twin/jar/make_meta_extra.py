"""JAR: out/meta_extra.json for render_set.py (merged into the set's meta.json): which reading the pass is claimed on,
the twin-independent mask check, the mesh / round comparison and the fit source.
    .venv/bin/python so101_twin/jar/make_meta_extra.py"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
O = HERE / "out"
sj = json.loads((O / "score_final.json").read_text())
dk = json.loads((O / "score_final_d1k.json").read_text())
cm = json.loads((O / "model_comparison.json").read_text())
cams = ("top", "side")
st = sj["stats"]
strict_pass = all(st[c]["strict_arm_only"]["held_out"]["median"] >= 0.80 for c in cams)
excl_pass = all(st[c]["occluder_excluded"]["held_out"]["median"] >= 0.80 for c in cams)
extra = dict(
    pass_criterion="held-out (odd episodes) median >= 0.80 on every camera",
    pass_strict=strict_pass, pass_occluder_excluded=excl_pass,
    pass_claimed_on="strict arm-only reading (held_out / all_frames above)" if strict_pass else
    ("real-occluder-excluded reading" if excl_pass else "none (fails)"),
    metric_reading="held_out / all_frames = STRICT reading: arm-only twin mask vs the SAM arm mask, nothing excluded",
    real_occluder_masks=("so101_twin/jar/masks/jar_<cam>_occ.npy: Grounding DINO 'a rubik's cube.' / 'a white jar.' on "
                         "the real frames only, every box >= 0.3, SAM 2.1 box prompt; cube kept if < 8 % of the image, "
                         "not dark and colourful; jar: only its bright pixels (max channel >= 110); no twin input, no "
                         "fitted parameter, never used by any fit objective; removed from BOTH masks for the "
                         "*_real_occluder_excluded reading"),
    twin_independent_mask_check=dict(
        note="same final twin vs the purely twin-independent DINO+SAM masks _d1k (cube removed; they miss white-jaw "
             "pixels in some side frames and drop the cable loops)", mask_tag=dk["mask_tag"],
        held_out={c: dk["stats"][c]["strict_arm_only"]["held_out"] for c in cams},
        all_frames={c: dk["stats"][c]["strict_arm_only"]["all_frames"] for c in cams}),
    arm_model_comparison=dict(metric=cm["metric"], rule=cm["rule"], rows=cm["rows"]),
    fit_source=cm["chosen_fit"])
(O / "meta_extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps({k: extra[k] for k in ("pass_strict", "pass_occluder_excluded", "pass_claimed_on", "fit_source")}))
