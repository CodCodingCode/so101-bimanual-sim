"""RGB-only scene layout from the fitted cameras (no effect on any arm mask): the dark mat around the base and the
cardboard backdrop, by casting background-median edge pixels onto the table plane z = 0.

    .venv/bin/python so101_twin/ballcup/layout.py --fit out/<run>_fit.json     # -> out/layout.json

top view: the mat's front edge is the white -> dark transition at row ~348 (columns 250-560 of the median background);
front view: the same edge is the dark -> white transition at row ~335, the box's bottom edge the tan -> dark
transition at row ~290 (both at the image centre column).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geom import cam_ray, hit_z  # noqa: E402


def edge_row(col, start, thr, below=True):
    r = np.nonzero(col < thr)[0] if below else np.nonzero(col > thr)[0]
    r = r[r > start]
    return int(r[0]) if len(r) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    a = ap.parse_args()
    P = json.loads(pathlib.Path(a.fit).read_text())["params"]
    bt = np.load(HERE / "out" / "bg_median_top.npy").astype(float).mean(-1)
    bs = np.load(HERE / "out" / "bg_median_side.npy").astype(float).mean(-1)
    ys_top = []
    for x in (260, 320, 400, 480, 540):
        r = edge_row(bt[:, x], 200, 80)
        if r is not None:
            ys_top.append(hit_z(*cam_ray(P, "top", x, r), 0.0)[1])
    xs_l = hit_z(*cam_ray(P, "top", 75, 400), 0.0)[0]
    xs_r = hit_z(*cam_ray(P, "top", 600, 400), 0.0)[0]
    r_white = edge_row(bs[:, 320], 300, 150, below=False)
    r_box = edge_row(bs[:, 320], 200, 85)
    y_mat_side = hit_z(*cam_ray(P, "side", 320, r_white), 0.0)[1] if r_white else None
    y_box = hit_z(*cam_ray(P, "side", 320, r_box), 0.0)[1] if r_box else None
    y_front = float(np.median(ys_top))
    # the box's bottom edge ray is nearly parallel to the table (it may stand behind / below the table edge): use it
    # only if it lands within 0.5 m of the base, else put the backdrop 22 cm behind the pan axis
    y_back = float(y_box) if (y_box is not None and -0.5 < y_box < -0.05) else -0.22
    # the mat edge is straight in the top image (row ~348): its direction on the table gives the mat / table yaw
    pl = hit_z(*cam_ray(P, "top", 260, edge_row(bt[:, 260], 200, 80)), 0.0)
    pr = hit_z(*cam_ray(P, "top", 540, edge_row(bt[:, 540], 200, 80)), 0.0)
    yaw = float(np.degrees(np.arctan2(pr[1] - pl[1], pr[0] - pl[0])))
    mid = 0.5 * (pl + pr)
    # backdrop: perpendicular to the front camera's viewing direction projected on the table
    from geom import mat_from_quat
    g = P["cams"]["side"]["groups"]["all"]
    fwd = -mat_from_quat(g["quat"])[:, 2]
    byaw = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 90.0
    out = dict(mat_yaw_deg=yaw, back_yaw_deg=byaw, mat_edge_mid=[float(mid[0]), float(mid[1])],
               mat_front_y_top=y_front, mat_front_y_side=None if y_mat_side is None else float(y_mat_side),
               backdrop_y=y_back, table_x=[float(xs_l), float(xs_r)],
               MAT_CENTER=[float(mid[0]), float(mid[1]) - 0.3], MAT_HALF=[1.0, 0.3, 0.0005],
               BACK_CENTER=[0.0, y_back - 0.01, 0.5])
    (HERE / "out" / "layout.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
