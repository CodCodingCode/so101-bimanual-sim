"""Install the rendered images into so101_pickplace_twin2 and add the meta.json "objects" section.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/pickplace_objects/finalize_pickplace.py install   # copy images (backup must exist)
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/pickplace_objects/finalize_pickplace.py meta      # write the objects section

Only images_top.npy / images_side.npy and the "objects" key of meta.json change; everything else is verified
byte-identical against backup/checksums_before.txt and backup/meta.json.orig."""
import hashlib, json, pathlib, shutil, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SET = ROOT / "neuroscope/molmoact2/store/sets/so101_pickplace_twin2"
BK = HERE / "backup"
WORK = HERE / "work"
KEEP = ["state.npy", "prompt.json", "labels.npz", "arm_mask_top.npy", "arm_mask_side.npy"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def before():
    out = {}
    for line in (BK / "checksums_before.txt").read_text().splitlines():
        name = line.split("(")[1].split(")")[0]
        out[name] = line.split("= ")[1].strip()
    return out


def check_keep():
    b = before()
    bad = [k for k in KEEP if md5(SET / k) != b[k]]
    assert not bad, f"changed: {bad}"
    return {k: b[k] for k in KEEP}


def install():
    b = before()
    for c in ("top", "side"):
        assert md5(BK / f"images_{c}.npy") == b[f"images_{c}.npy"], "backup does not match the original"
        new = np.load(WORK / "new" / f"images_{c}.npy", mmap_mode="r")
        old = np.load(BK / f"images_{c}.npy", mmap_mode="r")
        assert new.shape == old.shape and new.dtype == old.dtype
        assert new.reshape(new.shape[0], -1).max(1).min() > 0, "a frame was never rendered"
        shutil.copyfile(WORK / "new" / f"images_{c}.npy", SET / f"images_{c}.npy")
    print("installed; kept files unchanged:", check_keep())


def summary(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    return dict(n=int(len(a)), median=round(float(np.median(a)), 3), max=round(float(a.max()), 3), min=round(float(a.min()), 3)) if len(a) else dict(n=0)


def meta():
    orig_txt = (BK / "meta.json.orig").read_text()
    m = json.loads(orig_txt)
    cur = json.loads((SET / "meta.json").read_text())
    assert {k: v for k, v in cur.items() if k != "objects"} == {k: v for k, v in m.items() if k != "objects"}, "meta changed outside objects"
    st = json.loads((WORK / "static_fit.json").read_text())
    ob = json.loads((WORK / "objects_fit.json").read_text())
    bs = json.loads((WORK / "block_state.json").read_text())
    col = json.loads((WORK / "colours.json").read_text())
    eps = ob["episodes"]
    resid = {"brick": {}, "box_rim": {}}
    for c in ("top", "side"):
        rr = [eps[e]["brick"][c] for e in eps if eps[e]["brick"][c]]
        resid["brick"][c] = dict(n=len(rr), iou=summary([x["iou"] for x in rr]), centroid_px=summary([x["centroid_px"] for x in rr]),
                                 centroid_cm=summary([x["centroid_cm"] for x in rr]))
        rr = [eps[e]["rim"][c] for e in eps if eps[e]["rim"][c]]
        resid["box_rim"][c] = dict(n=len(rr), chamfer_px=summary([x["chamfer_px"] for x in rr]), chamfer_cm=summary([x["chamfer_cm"] for x in rr]))
    gaps = [g for e in eps for g in eps[e]["brick"].get("ray_gap_cm", [])]
    resid["brick"]["two_view_ray_gap_cm"] = summary(gaps)
    states = {s_: sum(v["state"] == s_ for v in bs.values()) for s_ in ("table", "held", "plate")}
    ident = json.loads((HERE / "out" / "arm_identity_check.json").read_text())["summary"] if (HERE / "out" / "arm_identity_check.json").exists() else None
    objects = dict(
        status="RE-PLACED (so101_twin/pickplace_objects, %s): the builder's box and brick re-placed per episode through the NEW fitted cameras, brick follows the task phase; back wall moved to the table edge, shipping boxes added (side camera), per-camera colours; arm, cameras, lens, lag and joint map untouched (frozen fit %s)" % (time.strftime("%Y-%m-%d %H:%M"), m.get("fit")),
        previous=m.get("objects"),
        renderer="so101_twin/pickplace_objects/build_pickplace.py (objtwin.ObjTwin = fitkit twin2.Twin2 build of so101_twin/pickplace/so101_pickplace.xml + scene hook; the builder's static box/brick bodies replaced by mocap copies of the same models; <statistic> pinned to the builder model; lights unchanged; per-camera material colours, arm materials never touched)",
        frame="world = twin arm base frame (base at the origin, base bottom z = 0)",
        static=dict(table="builder table kept (top z = 0, x -0.45..0.45, y -0.60..0.10), recoloured per camera",
                    wall_back=dict(y=st["wall_back_y"], top_z=1.0, note=st["wall_note"] + "; nearly unlit material (emission 4)"),
                    shipping_boxes=dict(boxes=st["cardboard"], how=st["cardboard_how"]),
                    not_modelled=["LeRobot sticker / wiring on the real arm", "box wall reflections"]),
        box=dict(kind="builder model: transparent box (glass alpha 0.14) with an orange rim (6 mm square section) along the bottom edges",
                 outer_size_m=[ob["shapes"]["rim"][0], ob["shapes"]["rim"][1], 0.06], height_note="0.06 m = builder BOX_SIZE z (transparent walls not measurable)",
                 placement="per episode: orange colour mask (twin arm silhouette removed) on the episode's first frame in both cameras; (x, y, yaw) and the shared outer size by a one-sided chamfer (every detected rim pixel to the model rim silhouette) through the frozen cameras, both views jointly",
                 per_episode_pose={e: eps[e]["rim"]["pose"] for e in eps}),
        brick=dict(kind="builder model: 2x4 brick box + 8 studs", half_size_m=ob["shapes"]["brick"],
                   placement="table pose per episode: pink colour mask centroids in both cameras triangulated (least-squares ray intersection) -> (x, y, centre z), z clipped to [hz, 3.5 cm] (top-view ray on the median height when the side camera misses it), yaw from the top-view IoU; per frame state table -> held -> plate from Grounding-DINO 'a small pink lego brick.' + SAM detections in every frame of both cameras vs the three hypotheses projected through the frozen cameras (monotone dynamic programme per episode; gripper reading < 3 % = open cannot hold); held = between the jaw tips of the twin gripper (midpoint of the fixed-jaw tip site and the moving-jaw tip, moved back by the brick half width, gripper orientation); plate = on the box floor at the detected position (box centre if unseen)",
                   state_counts=states, per_frame_state={k: v["state"] for k, v in bs.items()},
                   per_episode_table_pose_xyyawz={e: eps[e]["brick"]["pose"] for e in eps}),
        support_plane=dict(z_objects_m=ob["z_objects"], note="the brick centres triangulate %.1f mm higher than a brick lying on z = 0 would (median); the box rim and the brick stand on z = %.4f, the per-episode brick height is the triangulated one. The table stays at z = 0 so the arm base keeps its fitted pixels; a camera-fit inconsistency away from the arm, not refitted" % (1000 * (ob["z_objects"]), ob["z_objects"])),
        colours_per_camera=col,
        residual_placement_error=resid,
        residual_note="brick: IoU / centroid distance between the model silhouette and the colour mask on each episode's first frame; box: mean distance of detected rim pixels to the model rim; cm = px x depth / focal; two_view_ray_gap = distance of the triangulated brick centre from each camera's ray",
        arm_identity_check=ident,
        unchanged=dict(files=check_keep(), meta="every key except 'objects' (arm_iou, views, joint, lag ...) re-serialised byte-identically"),
        images_backup="so101_twin/pickplace_objects/backup/images_{top,side}.npy (the previous twin2 images: builder box/brick through the old cameras)")
    m["objects"] = objects
    txt = json.dumps(m, indent=1)
    blk = lambda t, k: t[t.index(f'"{k}":'):].split('\n }', 1)[0]
    for k in ("arm_iou", "views", "joint"):
        assert blk(txt, k) == blk(orig_txt, k), k
    (SET / "meta.json").write_text(txt)
    print("meta.json objects section written; arm_iou/views/joint blocks byte-identical")


if __name__ == "__main__":
    {"install": install, "meta": meta}[sys.argv[1]]()
