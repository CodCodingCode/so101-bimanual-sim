"""Install the rendered images into so101_ricky_twin2 and add the meta.json "objects" section.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/finalize_ricky.py install   # copy images (backup must exist)
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python so101_twin/ricky_objects/finalize_ricky.py meta      # write the objects section

Only images_top.npy / images_side.npy and the "objects" key of meta.json change; everything else is verified
byte-identical against backup/checksums_before.txt and backup/meta.json.orig."""
import hashlib, json, pathlib, shutil, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SET = ROOT / "neuroscope/molmoact2/store/sets/so101_ricky_twin2"
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
    sys.path.insert(0, str(HERE))
    import build_ricky as B
    eps = ob["episodes"]
    resid = {}
    for n in ("bowl", "block"):
        resid[n] = {c: dict(iou=summary([eps[e][n][c]["iou"] for e in eps]), centroid_px=summary([eps[e][n][c]["centroid_px"] for e in eps]),
                            centroid_cm=summary([eps[e][n][c]["centroid_cm"] for e in eps])) for c in ("top", "side")}
        resid[n]["top_only_vs_side_only_fit_cm"] = summary([eps[e][n]["top_vs_side_cm"] for e in eps])
    frames = {k: v for k, v in bs.items() if k.isdigit()}
    states = {s: sum(v["state"] == s for v in frames.values()) for s in ("table", "held", "plate")}
    ident = json.loads((HERE / "out" / "arm_identity_check.json").read_text())["summary"] if (HERE / "out" / "arm_identity_check.json").exists() else None
    w = B.walls()
    objects = dict(
        status="MODELLED (so101_twin/ricky_objects, %s): fitted table/mat/walls/chair + per-episode bowl and block; arm, cameras, lens, lag and joint map untouched (frozen fit %s)" % (time.strftime("%Y-%m-%d %H:%M"), m.get("fit")),
        previous="NOT MODELLED: minimal scene (table, wall); objects missing",
        renderer="so101_twin/ricky_objects/build_ricky.py (objtwin.ObjTwin = fitkit twin2.Twin2 build + scene hook; <statistic> pinned to the placeholder model so near/far clip and shadow extent are unchanged; lights unchanged; per-camera material colours, arm materials never touched)",
        frame="world = twin arm base frame (base at the origin, base bottom z = 0)",
        static=dict(
            table=dict(kind="box (wood)", center_xy=st["table"]["center"], half_xy=st["table"]["half"], yaw_deg=st["table"]["yaw_deg"], top_z=0.0,
                       how="IoU fit of a rectangle (rays of every 4th pixel cast onto z=0) to wood|mat colour masks of the 5 per-session median frames",
                       iou_per_view=st["table"]["iou_per_view"]),
            mat=dict(kind="box 2 mm (dark grey)", center_xy=st["mat"]["center"], half_xy=st["mat"]["half"], yaw_deg=st["mat"]["yaw_deg"], top_z=0.0005, thickness_m=0.002,
                     how="same rectangle fit to the dark-grey mask (logo closed)", iou_per_view=st["mat"]["iou_per_view"]),
            wall_back=dict(kind="vertical wall (plane through the table's far edge, table yaw)", table_local_x=w["back_local_x"], top_z=1.0),
            wall_side=dict(kind="vertical wall behind the arm in the top view", table_local_y=w["side_local_y"], corner_world=w["corner"],
                           how="room-corner column (940 px) of the top_c median cast onto the back-wall plane", top_z=1.0,
                           note="walls stop at z = 1.0 so the placeholder key light (0.3, -0.4, 1.5) still lights the arm exactly as before; nearly unlit material (emission 4) so the spot cone leaves no ellipse"),
            chair=dict(kind="box (chair back, top view only)", center=list(B.CHAIR["center"]), half=list(B.CHAIR["half"]),
                       how="image box of the chair (cols 0-200, rows 108-439, same in the three top sessions) cast onto y = -0.25 (depth not observable; behind the side camera)"),
            not_modelled=["arm power cable and wrist camera bracket on the real arm's left (side view)", "printed text on the bowl", "mat logo", "door frame / skirting boards"]),
        bowl=dict(kind="paper bowl: truncated-cone shell, navy outside, white inside + rim, 2.5 mm wall",
                  r_bottom_m=ob["shapes"]["bowl"][0], r_top_m=ob["shapes"]["bowl"][1], height_m=ob["shapes"]["bowl"][2],
                  placement="per episode (static within an episode): Grounding-DINO 'a white bowl.' + SAM 2.1 mask on the episode's first frame in both cameras; (x, y) per episode and the shared shape by IoU of the projected convex silhouette through the frozen cameras, both views jointly",
                  per_episode_xy={e: eps[e]["bowl"]["pose"] for e in eps}),
        block=dict(kind="toy block: box", half_size_m=ob["shapes"]["block"], size_cm=[round(200 * x, 2) for x in ob["shapes"]["block"]],
                   placement="table pose per episode (x, y, yaw) as for the bowl ('a small brown block.'); per frame state table -> held -> plate from Grounding-DINO + SAM block detections in every frame of both cameras vs the three hypotheses projected through the frozen cameras (monotone dynamic programme per episode, gripper reading < 8 % = open cannot hold); held = between the jaw tips of the twin gripper (midpoint of fixed-jaw tip site and moving-jaw tip, 1.4 cm back toward the wrist, gripper orientation); plate = on the bowl floor at the side-view detection (bowl centre if unseen)",
                   state_counts=states, per_frame_state={k: v["state"] for k, v in frames.items()},
                   per_episode_table_pose={e: eps[e]["block"]["pose"] for e in eps}),
        support_plane=dict(z_objects_m=ob["z_objects"], note="objects stand on z = %.4f m, fitted jointly with the shapes: with the frozen cameras the two views agree best when the objects sit %.1f mm above the arm-base plane (the table top stays at z = 0 and the mat top at z = 0.0005, so the arm base keeps its fitted pixels); a camera-fit inconsistency away from the arm, not refitted" % (ob["z_objects"], 1000 * ob["z_objects"])),
        colours_per_camera=col,
        residual_placement_error=resid,
        residual_note="IoU / centroid distance between the model silhouette and the SAM mask on each episode's first frame; cm = centroid px x depth / focal; top_only_vs_side_only = distance between single-camera fits (the frozen top camera, 10 cm above the table, sees the table at a grazing angle: its object silhouettes come out taller than the real ones, so the joint fit leaves ~11 px (bowl) / ~7 px (block) centroid error in the top view and ~1-5 px in the side view)",
        arm_identity_check=ident,
        unchanged=dict(files=check_keep(), meta="every key except 'objects' (arm_iou, views, joint, lag ...) re-serialised byte-identically"),
        images_backup="so101_twin/ricky_objects/backup/images_{top,side}.npy (the placeholder-scene twin2 images)")
    m["objects"] = objects
    txt = json.dumps(m, indent=1)
    blk = lambda t, k: t[t.index(f'"{k}":'):].split('\n }', 1)[0]
    for k in ("arm_iou", "views", "joint"):
        assert blk(txt, k) == blk(orig_txt, k), k
    (SET / "meta.json").write_text(txt)
    print("meta.json objects section written; arm_iou/views/joint blocks byte-identical")


if __name__ == "__main__":
    {"install": install, "meta": meta}[sys.argv[1]]()
