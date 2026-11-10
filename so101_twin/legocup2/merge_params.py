"""Merge per-camera fit outputs into one params json: cameras from their own runs, joint offsets from the top run,
gripper (scale, offset) from the wrist run (the jaws dominate the wrist view).

    python merge_params.py out/merged.json --top out/legocup_r1top_fit.json --side out/legocup_r1side_fit.json
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--top", required=True)
ap.add_argument("--side", required=True)
ap.add_argument("--grip-from", default="side", choices=("side", "top"))
a = ap.parse_args()
T = json.loads(pathlib.Path(a.top).read_text())["params"]
S = json.loads(pathlib.Path(a.side).read_text())["params"]
G = S if a.grip_from == "side" else T
P = dict(cams=dict(top=T["cams"]["top"], side=S["cams"]["side"]), offsets_deg=T["offsets_deg"],
         grip_scale=G["grip_scale"], grip_off=G.get("grip_off", 0.0), scales=T.get("scales", [1.0] * 5))
pathlib.Path(a.out).write_text(json.dumps(dict(params=P, merged_from=[a.top, a.side]), indent=1))
print(json.dumps(P, indent=1))
