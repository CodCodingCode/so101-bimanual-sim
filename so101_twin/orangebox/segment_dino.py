"""Twin-INDEPENDENT real-arm masks for orangebox: Grounding-DINO (tiny) box for "an orange robot arm." + SAM 2.1.

    neuroscope/molmoact2/.venv/bin/python so101_twin/orangebox/segment_dino.py [--tag _dino] [--limit N]
        [--sam facebook/sam2.1-hiera-small] [--prompts DIR]   # --prompts: also use twin silhouettes (consensus)

Per frame and camera:
  box   = union of the confident DINO boxes (score >= max(0.2, 0.5 * best)), else the bbox of the hue mask;
  SAM 2.1 candidates: box only (3 multimask outputs) and box + points (3): positives = up to 4 deep-interior points of
  the orange hue mask (hue_masks.py) inside the box, negatives = up to 4 bright low-saturation table pixels inside the
  box, >= 25 px from any orange pixel;
  choice (no twin involved): score = recall of the orange hue pixels in the box - 2 * fraction of the candidate that
  is bright white table + 0.25 * SAM predicted IoU (with --prompts: + IoU with the twin silhouette, consensus mode).
  Post: 3x3 logit smoothing (as common/segment.py), 1-px opening + 2-px closing, then the yellow-green cube's pixels
  are removed (the cube is not the arm).
  --dark-neg (consensus only, as legocup2 did for its black cup): negatives also on up to 4 dark pixels 15-60 px
  outside the twin silhouette (the black box when the gripper is over it) and the candidate score loses
  2 x the fraction of the candidate made of dark pixels > 15 px outside the twin silhouette.
Output: so101_twin/orangebox/masks/orangebox_<cam><tag>.npy (packed bool [n, 480, 80]) + _score.npy + sheet in out/.
Check `pgrep -fl "segment"` shows no other SAM process before running (one segmentation process at a time).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(HERE))
from segment import dilate, erode, depth_map, spread_points, prompts_from  # noqa: E402
from hue_masks import hsv  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PROMPT = "an orange robot arm."
CAMS = ("top", "side")


def unpack(a):
    return np.unpackbits(a, axis=-1)[..., :640].astype(bool)


class Seg:
    def __init__(self, sam_id):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEV).eval()
        self.sp = Sam2Processor.from_pretrained(sam_id)
        self.sm = Sam2Model.from_pretrained(sam_id).to(DEV).eval()

    @torch.inference_mode()
    def boxes(self, img: Image.Image, thr=0.2):
        x = self.dp(images=img, text=PROMPT, return_tensors="pt").to(DEV)
        out = self.dm(**x)
        r = self.dp.post_process_grounded_object_detection(out, x.input_ids, threshold=thr, text_threshold=thr,
                                                           target_sizes=[img.size[::-1]])[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()

    @torch.inference_mode()
    def candidates(self, img: Image.Image, box, pos, neg, extra=()):
        """extra: further (box, pos, neg) prompt sets (twin-prompted, consensus mode): box+points and box only each."""
        enc = self.sp(images=img, return_tensors="pt")
        emb = self.sm.get_image_embeddings(enc["pixel_values"].to(DEV))
        W, H = img.size
        ts = float(self.sp.target_size)
        sx, sy = ts / W, ts / H

        def kw_for(box, pos, neg, box_only=True):
            bx = torch.tensor([[[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]]], dtype=torch.float32, device=DEV)
            out = [dict(input_boxes=bx)] if box_only else []
            if pos:
                pts = pos + neg
                pt = torch.tensor([[[[x * sx, y * sy] for x, y in pts]]], dtype=torch.float32, device=DEV)
                lb = torch.tensor([[[1] * len(pos) + [0] * len(neg)]], dtype=torch.int32, device=DEV)
                out.append(dict(input_points=pt, input_labels=lb, input_boxes=bx))
            return out

        kws = kw_for(box, pos, neg)
        for b2, p2, n2 in extra:
            kws += kw_for(b2, p2, n2)
        cands, scores = [], []
        for kw in kws:
            out = self.sm(image_embeddings=emb, multimask_output=True, **kw)
            logits = out.pred_masks.float().cpu()
            sh = logits.shape
            logits = torch.nn.functional.avg_pool2d(logits.reshape(-1, 1, *sh[-2:]), 3, 1, 1,
                                                    count_include_pad=False).reshape(sh)
            m = self.sp.post_process_masks(logits, enc["original_sizes"])[0]
            cands.append(m[0].numpy().astype(bool))
            scores.append(out.iou_scores.float().cpu().numpy().reshape(-1))
        return np.concatenate(cands, 0), np.concatenate(scores, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_dino")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--prompts", default="", help="dir with orangebox_<cam>.npy twin silhouettes (consensus)")
    ap.add_argument("--frames", default="")
    ap.add_argument("--dark-neg", action="store_true")
    a = ap.parse_args()
    from sets import Store

    fs = Store().frames("so101_orangebox_real")
    idx = [int(x) for x in a.frames.split(",")] if a.frames else list(range(fs.n if not a.limit else a.limit))
    hue = {c: unpack(np.load(HERE / "masks" / f"orangebox_{c}_hue.npy")) for c in CAMS}
    for c in CAMS:   # static false positives (e.g. the wooden floor edge in the side view): on in > 90 % of frames
        static = hue[c].mean(0) > 0.9
        hue[c] &= ~static[None]
    tw = {c: unpack(np.load(pathlib.Path(a.prompts) / f"orangebox_{c}.npy")) for c in CAMS} if a.prompts else None
    t0 = time.time()
    seg = Seg(a.sam)
    print(f"models loaded in {time.time() - t0:.0f}s on {DEV}", flush=True)
    rng = np.random.default_rng(0)
    masks = {c: np.zeros((fs.n, 480, 640), bool) for c in CAMS}
    score = {c: np.zeros((fs.n, 4), np.float32) for c in CAMS}
    for k, i in enumerate(idx):
        obs = fs.obs(i)["images"]
        for c in CAMS:
            rgb = np.asarray(obs[c])
            img = Image.fromarray(rgb)
            h, s, v = hsv(rgb)
            white = (s < 0.18) & (v > 0.45)
            cube = (h > 48) & (h < 85) & (s > 0.35) & (v > 0.25)
            Hm = hue[c][i]
            b, sc = seg.boxes(img)
            keep = sc >= max(0.2, 0.5 * sc.max()) if len(sc) else np.zeros(0, bool)
            if keep.any():
                bb = b[keep]
                box = [float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 2].max()), float(bb[:, 3].max())]
                dscore = float(sc.max())
            elif Hm.sum() > 50:
                ys, xs = np.nonzero(dilate(Hm, 10))
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                dscore = 0.0
            else:
                continue
            x0, y0, x1, y1 = [int(round(t)) for t in box]
            inbox = np.zeros_like(Hm)
            inbox[max(y0, 0):y1, max(x0, 0):x1] = True
            Hb = Hm & inbox
            pos, neg = [], []
            if Hb.sum() > 40:
                D = depth_map(Hb, 12)
                ys, xs = np.nonzero(D >= max(2, D.max() // 2))
                if len(xs):
                    sel = rng.choice(len(xs), size=min(len(xs), 3000), replace=False)
                    cand = np.stack([xs[sel], ys[sel]], 1)
                    pos = [list(map(float, p)) for p in spread_points(cand, D[ys[sel], xs[sel]] + rng.random(len(sel)), 4, 30)]
                far = white & inbox & ~dilate(Hm, 25)
                yb, xb = np.nonzero(far[::4, ::4])
                if len(xb):
                    cb = np.stack([xb * 4, yb * 4], 1)
                    neg = [list(map(float, p)) for p in spread_points(cb, rng.random(len(cb)), 4, 50)]
            extra = []
            darkfar = None
            if tw is not None and tw[c][i].sum() >= 30:
                pr = prompts_from(tw[c][i], rng)
                if a.dark_neg:
                    dark = (v < 0.25) & (s < 0.5)
                    darkfar = dark & ~dilate(tw[c][i], 15)
                    band = darkfar & dilate(tw[c][i], 60)
                    yb, xb = np.nonzero(band[::3, ::3])
                    if len(xb):
                        dn = [list(map(float, q)) for q in spread_points(np.stack([xb * 3, yb * 3], 1), rng.random(len(xb)), 4, 30)]
                        neg = neg + dn
                        if pr is not None:
                            pr = (pr[0], pr[1], pr[2] + dn)
                if pr is not None:
                    extra.append(pr)
            cands, sams = seg.candidates(img, box, pos, neg, extra)
            best, bi = -9.0, 0
            for j, m in enumerate(cands):
                rec = (m & Hb).sum() / max(Hb.sum(), 1)
                wf = (m & white).sum() / max(m.sum(), 1)
                val = rec - 2.0 * wf + 0.25 * sams[j]
                if tw is not None:
                    P = tw[c][i]
                    val += (m & dilate(P, 4)).sum() / max((m | P).sum(), 1)
                if darkfar is not None:
                    val -= 2.0 * (m & darkfar).sum() / max(m.sum(), 1)
                if val > best:
                    best, bi = val, j
            m = cands[bi]
            m = dilate(erode(m, 1), 1)
            m = erode(dilate(m, 2), 2)
            m &= ~cube
            masks[c][i] = m
            score[c][i] = (dscore, float(sams[bi]), (m & Hb).sum() / max(Hb.sum(), 1), (m & white).sum() / max(m.sum(), 1))
        if k % 20 == 0:
            print(f"{k}/{len(idx)} {time.time() - t0:.0f}s " + " ".join(
                f"{c}: dino {score[c][i][0]:.2f} sam {score[c][i][1]:.2f} rec {score[c][i][2]:.2f} white {score[c][i][3]:.2f}"
                for c in CAMS), flush=True)
    for c in CAMS:
        np.save(HERE / "masks" / f"orangebox_{c}{a.tag}.npy", np.packbits(masks[c], axis=-1))
        np.save(HERE / "masks" / f"orangebox_{c}{a.tag}_score.npy", score[c])
    # sheet: 16 frames x 2 cams, green = SAM outline, red = hue outline
    sel = np.array(idx)[np.linspace(0, len(idx) - 1, 16).astype(int)]
    tiles = []
    for i in sel:
        row = []
        for c in CAMS:
            im = np.asarray(fs.obs(int(i))["images"][c]).copy()
            o = masks[c][i] & ~erode(masks[c][i], 1)
            im[dilate(o, 1)] = (40, 255, 40)
            row.append(im)
        tiles.append(np.concatenate(row, 1))
    grid = [np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(4)]
    g = Image.fromarray(np.concatenate(grid, 0))
    g = g.resize((g.width // 3, g.height // 3), Image.BILINEAR)
    ImageDraw.Draw(g).text((4, 4), f"orangebox{a.tag}: green = SAM arm mask", fill=(255, 0, 0))
    g.save(HERE / "out" / f"masks{a.tag}_sheet.jpg", quality=88)
    print(f"done {len(idx)} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
