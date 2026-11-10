"""ballcup real-arm masks with SAM 2.1, two modes (one model load, one process = one SAM slot):

    neuroscope/molmoact2/.venv/bin/python so101_twin/ballcup/segment.py --mode dino [--tag _d1] [--limit N]
    neuroscope/molmoact2/.venv/bin/python so101_twin/ballcup/segment.py --mode twin --prompts DIR [--tag _t1]
    neuroscope/molmoact2/.venv/bin/python so101_twin/ballcup/segment.py --mode occ --tag _occ   (real occluder masks)

dino (twin-independent): Grounding DINO (tiny) boxes for "a black robot arm." (every box with score >= max(0.2,
  0.5 * best)), SAM 2.1 cuts a mask inside each box (box prompt only), union = the arm mask. A second DINO pass with
  "a blue ball. a paper cup." gives the ball / cup boxes and their SAM masks (saved separately: object placement and
  optional ball removal). Nothing from the twin is used.
twin (re-prompt): consensus prompting (as common/segment.py with several prompt dirs) from two silhouettes, the twin's
  arm silhouette DIR/ballcup_<cam>.npy (visible_prompts.py: minus the real cup / ball masks) and the cleaned
  twin-independent mask (--cons-tag, default _d1c): box from their union dilated 12 px, positives from their
  intersection, negatives far from the union plus hard negatives on dark pixels 20-90 px outside (power cable / plug)
  and on the dino-run ball / cup masks; 3 prompt variants x 3 multimask outputs, kept = best MEAN 4-px-tolerant IoU
  with the silhouettes + 0.25 * SAM predicted IoU.
Smoothing as common/segment.py: 3x3 average of the 256x256 logits, then 1-px opening and 2-px closing.

Output so101_twin/ballcup/masks/ballcup_<cam><tag>.npy (packed bool [n, 480, 80]); dino mode also _ball / _cup masks
and ballcup_dino<tag>.json (boxes, scores per frame / camera).
Check first that no other SAM process runs: pgrep -fl "segment.py|segment_arm.py|segment_lc2.py|segment_parts.py".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(ROOT / "so101_twin" / "common"))
from segment import dilate, erode, prompts_from, spread_points  # noqa: E402  (common/segment.py, read-only import)

MASKS = HERE / "masks"
CAMS = ("top", "side")


class Models:
    def __init__(self, sam_id: str, dino: bool):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.sp = Sam2Processor.from_pretrained(sam_id)
        self.sm = Sam2Model.from_pretrained(sam_id).to(self.dev).eval()
        if dino:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(
                self.dev).eval()

    def boxes(self, img: Image.Image, text: str, thr: float = 0.2):
        torch = self.torch
        with torch.inference_mode():
            x = self.dp(images=img, text=text, return_tensors="pt").to(self.dev)
            out = self.dm(**x)
            r = self.dp.post_process_grounded_object_detection(out, x["input_ids"], threshold=thr, text_threshold=thr,
                                                               target_sizes=[img.size[::-1]])[0]
        labs = r.get("text_labels", r.get("labels"))
        return r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), [str(s) for s in labs]

    def embed(self, img: Image.Image):
        torch = self.torch
        with torch.inference_mode():
            enc = self.sp(images=img, return_tensors="pt")
            emb = self.sm.get_image_embeddings(enc["pixel_values"].to(self.dev))
        return enc, emb

    def decode(self, enc, emb, W, H, box=None, pts=None, labs=None, multi=True):
        """-> (masks [k, H, W] bool, scores [k])"""
        torch = self.torch
        ts = float(self.sp.target_size)
        sx, sy = ts / W, ts / H
        kw = {}
        if pts is not None and len(pts):
            kw["input_points"] = torch.tensor([[[[x * sx, y * sy] for x, y in pts]]], dtype=torch.float32,
                                              device=self.dev)
            kw["input_labels"] = torch.tensor([[list(labs)]], dtype=torch.int32, device=self.dev)
        if box is not None:
            kw["input_boxes"] = torch.tensor([[[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]]],
                                             dtype=torch.float32, device=self.dev)
        with torch.inference_mode():
            out = self.sm(image_embeddings=emb, multimask_output=multi, **kw)
            logits = out.pred_masks.float().cpu()
            sh = logits.shape
            logits = torch.nn.functional.avg_pool2d(logits.reshape(-1, 1, *sh[-2:]), 3, 1, 1,
                                                    count_include_pad=False).reshape(sh)
            masks = self.sp.post_process_masks(logits, enc["original_sizes"])[0]
        return masks[0].numpy().astype(bool), out.iou_scores.float().cpu().numpy().reshape(-1)


def clean(m: np.ndarray) -> np.ndarray:
    m = dilate(erode(m, 1), 1)
    return erode(dilate(m, 2), 2)


def run_dino(M: Models, fs, n: int, tag: str):
    arm = {c: np.zeros((n, 480, 640), bool) for c in CAMS}
    ball = {c: np.zeros((n, 480, 640), bool) for c in CAMS}
    cup = {c: np.zeros((n, 480, 640), bool) for c in CAMS}
    info = {c: [] for c in CAMS}
    t0 = time.time()
    for i in range(n):
        obs = fs.obs(i)
        for c in CAMS:
            rgb = np.asarray(obs["images"][c])
            img = Image.fromarray(rgb)
            H, W = rgb.shape[:2]
            ab, asc, _ = M.boxes(img, "a black robot arm.")
            ob, osc, ol = M.boxes(img, "a blue ball. a paper cup.")
            enc, emb = M.embed(img)
            rec = dict(arm=[], ball=None, cup=None)
            if len(ab):
                keep = asc >= max(0.25, 0.5 * asc.max())
                for b, s in zip(ab[keep], asc[keep]):
                    ms, sc = M.decode(enc, emb, W, H, box=b, multi=False)
                    arm[c][i] |= ms[0]
                    rec["arm"].append(dict(box=[round(float(v), 1) for v in b], score=round(float(s), 3),
                                           sam=round(float(sc[0]), 3)))
                arm[c][i] = clean(arm[c][i])
            for name, store in (("ball", ball), ("cup", cup)):
                sel = [k for k, l in enumerate(ol) if name in l and not (("ball" in l) and ("cup" in l))]
                if not sel:
                    continue
                k = max(sel, key=lambda k: osc[k])
                ms, sc = M.decode(enc, emb, W, H, box=ob[k], multi=False)
                store[c][i] = clean(ms[0])
                rec[name] = dict(box=[round(float(v), 1) for v in ob[k]], score=round(float(osc[k]), 3),
                                 sam=round(float(sc[0]), 3))
            info[c].append(rec)
        if i % 20 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s " + " ".join(
                f"{c}: arm {len(info[c][-1]['arm'])} px {arm[c][i].sum()}" for c in CAMS), flush=True)
    for c in CAMS:
        np.save(MASKS / f"ballcup_{c}{tag}.npy", np.packbits(arm[c], axis=-1))
        np.save(MASKS / f"ballcup_{c}{tag}_ball.npy", np.packbits(ball[c], axis=-1))
        np.save(MASKS / f"ballcup_{c}{tag}_cup.npy", np.packbits(cup[c], axis=-1))
    (MASKS / f"ballcup_dino{tag}.json").write_text(json.dumps(info))
    print(f"dino done {n} frames in {time.time() - t0:.0f}s", flush=True)


def run_twin(M: Models, fs, n: int, tag: str, pdir: pathlib.Path, obj_tag: str, cons_tag: str):
    """Consensus prompting (as common/segment.py with several prompt dirs): silhouettes = the fitted twin's arm mask
    and (if cons_tag) the cleaned twin-independent DINO mask. Box / negatives from their union, positives from their
    intersection, candidate chosen by the MEAN 4-px-tolerant IoU with them + 0.25 * SAM score. Extra negatives: dark
    pixels (max channel < 70: the black power cable / plug, not the white table) 20-90 px outside the union, and
    points on the ball / cup masks of the dino run (outside the union)."""
    rng = np.random.default_rng(0)
    out = {c: np.zeros((n, 480, 640), bool) for c in CAMS}
    score = {c: np.zeros((n, 2), np.float32) for c in CAMS}
    t0 = time.time()
    unpack = lambda a: np.unpackbits(a, axis=-1)[:n, :, :640].astype(bool)   # noqa: E731
    twin = {c: unpack(np.load(pdir / f"ballcup_{c}.npy")) for c in CAMS}
    cons = {c: unpack(np.load(MASKS / f"ballcup_{c}{cons_tag}.npy")) for c in CAMS} if cons_tag else None
    objs = {c: [unpack(np.load(MASKS / f"ballcup_{c}{obj_tag}_{o}.npy")) for o in ("ball", "cup")] for c in CAMS}
    for i in range(n):
        rgb_all = fs.obs(i)["images"]
        for c in CAMS:
            Ps = [twin[c][i]] + ([cons[c][i]] if cons is not None and cons[c][i].sum() >= 300 else [])
            Ps = [p for p in Ps if p.sum() >= 30]
            if not Ps:
                continue
            U = np.logical_or.reduce(Ps)
            pr = prompts_from(U, rng, core=np.logical_and.reduce(Ps) if len(Ps) > 1 else None)
            if pr is None:
                continue
            box, pos, neg = pr
            rgb = np.asarray(rgb_all[c])
            near = dilate(U, 20)
            band = dilate(U, 90) & ~near
            dark = rgb.max(-1) < 70
            yy, xx = np.nonzero(band & dark)
            if len(xx) > 50:
                sel = rng.choice(len(xx), size=min(len(xx), 600), replace=False)
                neg += [list(map(float, p)) for p in spread_points(np.stack([xx[sel], yy[sel]], 1),
                                                                    rng.random(len(sel)), 3, 30)]
            for om in objs[c]:
                cand = erode(om[i], 3) & ~dilate(U, 6)
                yy, xx = np.nonzero(cand)
                if len(xx) > 30:
                    sel = rng.choice(len(xx), size=min(len(xx), 400), replace=False)
                    neg += [list(map(float, p)) for p in spread_points(np.stack([xx[sel], yy[sel]], 1),
                                                                        rng.random(len(sel)), 2, 20)]
            img = Image.fromarray(rgb)
            enc, emb = M.embed(img)
            pts = pos + neg
            labs = [1] * len(pos) + [0] * len(neg)
            cands, scs = [], []
            for kw in (dict(box=box, pts=pts, labs=labs), dict(pts=pts, labs=labs), dict(box=box)):
                ms, sc = M.decode(enc, emb, 640, 480, multi=True, **kw)
                cands.append(ms)
                scs.append(sc)
            cands = np.concatenate(cands, 0)
            scs = np.concatenate(scs, 0)
            Pds = [dilate(p, 4) for p in Ps]

            def agree(cm):
                return float(np.mean([(cm & pd).sum() / max((cm | p).sum(), 1) for p, pd in zip(Ps, Pds)]))

            best, bi = -1.0, 0
            for k, cm in enumerate(cands):
                v = agree(cm) + 0.25 * scs[k]
                if v > best:
                    best, bi = v, k
            m = clean(cands[bi])
            out[c][i] = m
            score[c][i] = (agree(m), scs[bi])
        if i % 20 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s " + " ".join(f"{c} agree {score[c][i, 0]:.2f}" for c in CAMS),
                  flush=True)
    for c in CAMS:
        np.save(MASKS / f"ballcup_{c}{tag}.npy", np.packbits(out[c], axis=-1))
        np.save(MASKS / f"ballcup_{c}{tag}_score.npy", score[c])
    print(f"twin done in {time.time() - t0:.0f}s; median agreement " + " ".join(
        f"{c} {np.median(score[c][:, 0]):.3f}" for c in CAMS), flush=True)


def run_occ(M: Models, fs, n: int, tag: str):
    """Twin-independent REAL occluder masks (cup, ball) for every frame / camera: Grounding DINO with each prompt alone
    ("a paper cup.", "a blue ball."), EVERY box with score >= 0.3, SAM box prompt per box; a mask is kept only if its
    pixels are bright (median max-channel >= 110: the white cup / white-blue ball, not the black arm or gripper) and
    it covers < 20 % of the image. Union per object. Fixed constants only; no twin, no fitted parameter."""
    out = {c: {o: np.zeros((n, 480, 640), bool) for o in ("cup", "ball")} for c in CAMS}
    info = {c: [] for c in CAMS}
    t0 = time.time()
    for i in range(n):
        obs = fs.obs(i)
        for c in CAMS:
            rgb = np.asarray(obs["images"][c])
            img = Image.fromarray(rgb)
            H, W = rgb.shape[:2]
            enc, emb = M.embed(img)
            vmax = rgb.max(-1)
            rec = {}
            for o, text in (("cup", "a paper cup."), ("ball", "a blue ball.")):
                bx, sc, _ = M.boxes(img, text, thr=0.3)
                kept = []
                for b, s_ in zip(bx, sc):
                    ms, iou = M.decode(enc, emb, W, H, box=b, multi=False)
                    m = clean(ms[0])
                    area = int(m.sum())
                    ok = area >= 50 and area < 0.2 * H * W and float(np.median(vmax[m])) >= 110 if area else False
                    kept.append(dict(box=[round(float(v), 1) for v in b], score=round(float(s_), 3), area=area,
                                     kept=bool(ok)))
                    if ok:
                        out[c][o][i] |= m
                rec[o] = kept
            info[c].append(rec)
        if i % 20 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s " + " ".join(
                f"{c}: cup {out[c]['cup'][i].sum()} ball {out[c]['ball'][i].sum()}" for c in CAMS), flush=True)
    for c in CAMS:
        for o in ("cup", "ball"):
            np.save(MASKS / f"ballcup_{c}{tag}_{o}.npy", np.packbits(out[c][o], axis=-1))
        np.save(MASKS / f"ballcup_{c}{tag}.npy", np.packbits(out[c]["cup"] | out[c]["ball"], axis=-1))
    (MASKS / f"ballcup_occ{tag}.json").write_text(json.dumps(info))
    print(f"occ done {n} frames in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dino", choices=("dino", "twin", "occ"))
    ap.add_argument("--prompts", default="")
    ap.add_argument("--obj-tag", default="_d1", help="dino-run tag whose ball / cup masks give extra negatives")
    ap.add_argument("--cons-tag", default="_d1c", help="twin-independent mask used as a second consensus silhouette "
                    "('' = twin silhouette only)")
    ap.add_argument("--tag", default="_d1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    a = ap.parse_args()
    from sets import Store

    fs = Store().frames("so101_ballcup_real")
    n = fs.n if not a.limit else min(a.limit, fs.n)
    MASKS.mkdir(exist_ok=True)
    t0 = time.time()
    M = Models(a.sam, dino=a.mode in ("dino", "occ"))
    print(f"models loaded in {time.time() - t0:.0f}s on {M.dev}", flush=True)
    if a.mode == "dino":
        run_dino(M, fs, n, a.tag)
    elif a.mode == "occ":
        run_occ(M, fs, n, a.tag)
    else:
        pdir = pathlib.Path(a.prompts)
        run_twin(M, fs, n, a.tag, pdir if pdir.is_absolute() else ROOT / pdir, a.obj_tag, a.cons_tag)


if __name__ == "__main__":
    main()
