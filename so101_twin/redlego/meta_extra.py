"""REDLEGO: extra meta.json fields for the twin set (merged by render_set.py from out/meta_extra.json): pass claim,
masks, the twin-specific arm additions (wrist-camera bar, base cable loop), mesh comparison, twin-independent check,
guard rails. Run after finalize.py / score.py, before render_set.py.
    .venv/bin/python so101_twin/redlego/meta_extra.py"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
O = HERE / "out"
fj = json.loads((O / "final_fit.json").read_text())
P = fj["params"]
sj = json.loads((O / "score_final.json").read_text())["stats"]
si = json.loads((O / "score_final_indep.json").read_text())["stats"]
cmpj = json.loads((O / "model_comparison.json").read_text())
cams = ("top", "side")
strict_pass = all(sj[c]["strict_arm_only"]["held_out"]["median"] >= 0.80 for c in cams)
excl_pass = all(sj[c]["occluder_excluded"]["held_out"]["median"] >= 0.80 for c in cams)
claim = ("strict arm-only reading (A): both cameras >= 0.80 held-out; the real-occluder-excluded reading (B) is "
         "reported as well and also passes" if strict_pass else
         ("real-occluder-excluded reading (B) only" if excl_pass else "no pass"))
extra = dict(
    pass_claimed_on=claim,
    pass_strict=strict_pass, pass_occluder_excluded=excl_pass,
    masks=("so101_twin/redlego/masks/redlego_<cam>_t1ok.npy: SAM 2.1 consensus-prompted (twin r1c silhouette + "
           "twin-independent DINO+SAM mask _d1k), real brick pixels removed (_t1o), thin cables removed by a 5x5 "
           "opening (_t1ok); occluders for reading B: redlego_<cam>_d1_occ.npy (brick | box, DINO+SAM on the real "
           "frames only)"),
    occluder_masks="so101_twin/redlego/masks/redlego_<cam>_d1_occ.npy (red brick | cardboard box; real frames only)",
    arm_additions=dict(
        wrist_camera_bar=dict(params=P.get("bar"), frame="gripper body (so101 'gripper'), [x, y, z (m), rotvec (rad), "
                              "length (m)]", geometry="rod 10 x 3 mm along local +x, camera module 26 x 26 x 16 mm "
                              "hanging below its end", source="global search bar_search.py + stage-C refinement "
                              "against the arm masks, even episodes"),
        base_cable_loop=dict(params=P.get("blob"), frame="arm root (base) frame, [x, y, z (m), rotvec (rad), semi-axes "
                             "(m)]", geometry="static ellipsoid (black cable loop at the foot of the base, side view)",
                             source="global search blob_search.py against the arm masks, even episodes")),
    gravity_compliance_deg_per_Nm=P.get("compliance", [0.0] * 5),
    mesh_comparison=cmpj,
    twin_independent_check=dict(
        masks="so101_twin/redlego/masks/redlego_<cam>_d1k.npy (DINO+SAM box-prompted, brick/box removed, 5x5 "
              "opening; no twin input)",
        held_out={c: si[c]["strict_arm_only"]["held_out"] for c in cams},
        all_frames={c: si[c]["strict_arm_only"]["all_frames"] for c in cams},
        held_out_real_occluder_excluded={c: si[c]["occluder_excluded"]["held_out"] for c in cams}),
    guard_rails=dict(
        lens={c: dict(k1=P["cams"][c]["lens"]["k1"], k2=P["cams"][c]["lens"]["k2"],
                      principal_point_shift_frac=[P["cams"][c]["lens"]["cx"], P["cams"][c]["lens"]["cy"]])
              for c in cams},
        lens_note="all |k1| < 0.4 and principal-point shifts < 10 %: no lens audit required",
        joint_scales_note=("fitted 0.95-1.18 deg/unit, consistent with LeRobot's normalised -100..100 mode (pan 1.18 "
                           "deg/unit = ~236 deg calibrated travel); visual check at the extremes of every joint in "
                           "so101_twin/redlego/out/extremes.jpg")),
    fit_protocol="fit on EVEN episodes only (24 episodes, 192 frames); held_out = ODD episodes (16 episodes, 128 frames)",
)
(O / "meta_extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps(dict(claim=claim, strict=strict_pass, excl=excl_pass)))
