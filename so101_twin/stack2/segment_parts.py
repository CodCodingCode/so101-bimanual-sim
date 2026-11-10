"""SAM 2.1 real-arm masks prompted PART BY PART from the twin (stack: white arm on a white table, where a single
whole-arm prompt often returns only the gripper / forearm).

    neuroscope/molmoact2/.venv/bin/python so101_twin/stack2/segment_parts.py --parts DIR --tag _s2

Inputs: DIR/stack_<cam>_parts.npz (labels uint8 [n,480,640]: 0 background, 1 base+shoulder, 2 upper arm, 3 forearm,
4 wrist+gripper+jaws+camera; export_parts.py from a fitted twin). Per frame and camera, one image embedding, then
  - the whole-arm prompt of common/segment.py (box + 5 deep-interior positives + 4 far negatives; 3 prompt variants x
    3 multimask outputs; candidate = best IoU with the twin silhouette (4 px tolerant) + 0.25 SAM score), and
  - one prompt per visible part (box of the part + 3 interior positives + the same far negatives), its candidate chosen
    the same way against the PART silhouette and kept only if it agrees with the part (>= 0.45) and stays inside the
    twin's arm silhouette dilated by 25 px (<= 20 % outside) -- so a part prompt never adds table / cloth;
  - mask = union, 1-px opening + 2-px closing (as common/segment.py).
Output: so101_twin/masks/stack_<cam><tag>.npy (packed, same format as common/segment.py) + _score.npy +
so101_twin/masks/sheets/stack<tag>.jpg. Check `pgrep -fl "[Pp]ython.*segment"` before running (one SAM at a time).
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
TWIN = HERE.parent
ROOT = TWIN.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
from segment import Segmenter, depth_map, dilate, erode, prompts_from, sheet, spread_points  # noqa: E402

MASKS = TWIN / "masks"
CAMS = ("top", "side")


class PartSegmenter(Segmenter):
    def _cands(self, emb, enc, box, pos, neg, H, W, variants):
        torch = self.torch
        ts = float(self.proc.target_size)
        sx, sy = ts / W, ts / H
        pts = pos + neg
        labs = [1] * len(pos) + [0] * len(neg)
        pt = torch.tensor([[[[x * sx, y * sy] for x, y in pts]]], dtype=torch.float32, device=self.device)
        lb = torch.tensor([[labs]], dtype=torch.int32, device=self.device)
        bx = torch.tensor([[[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]]], dtype=torch.float32, device=self.device)
        kws = {"bp": dict(input_points=pt, input_labels=lb, input_boxes=bx), "p": dict(input_points=pt, input_labels=lb),
               "b": dict(input_boxes=bx)}
        cands, scores = [], []
        for v in variants:
            out = self.model(image_embeddings=emb, multimask_output=True, **kws[v])
            logits = out.pred_masks.float().cpu()
            sh = logits.shape
            logits = torch.nn.functional.avg_pool2d(logits.reshape(-1, 1, *sh[-2:]), 3, 1, 1,
                                                    count_include_pad=False).reshape(sh)
            masks = self.proc.post_process_masks(logits, enc["original_sizes"])[0]
            cands.append(masks[0].numpy().astype(bool))
            scores.append(out.iou_scores.float().cpu().numpy().reshape(-1))
        return np.concatenate(cands, 0), np.concatenate(scores, 0)

    @staticmethod
    def _pick(cands, scores, P):
        Pd = dilate(P, 4)
        best, bi, ag = -1.0, 0, 0.0
        for i, c in enumerate(cands):
            a = (c & Pd).sum() / max((c | P).sum(), 1)
            s = a + 0.25 * scores[i]
            if s > best:
                best, bi, ag = s, i, a
        return cands[bi], ag, float(scores[bi])

    def segment(self, img: np.ndarray, labels: np.ndarray, rng):
        torch = self.torch
        whole = labels > 0
        H, W = whole.shape
        if whole.sum() < 30:
            return np.zeros_like(whole), 0.0, 0.0
        with torch.inference_mode():
            enc = self.proc(images=Image.fromarray(img), return_tensors="pt")
            emb = self.model.get_image_embeddings(enc["pixel_values"].to(self.device))
            box, pos, neg = prompts_from(whole, rng)
            cands, scores = self._cands(emb, enc, box, pos, neg, H, W, ("bp", "p", "b"))
            mask, ag, sc = self._pick(cands, scores, whole)
            inside = dilate(whole, 12)
            for k in (1, 2, 3, 4):
                Pk = labels == k
                if Pk.sum() < 150:
                    continue
                ys, xs = np.nonzero(dilate(Pk, 8))
                bk = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                D = depth_map(Pk, 12)
                yk, xk = np.nonzero(Pk)
                sel = rng.choice(len(xk), size=min(len(xk), 2000), replace=False)
                pk = spread_points(np.stack([xk[sel], yk[sel]], 1), D[yk[sel], xk[sel]] + rng.random(len(sel)) * 0.5, 3, 15)
                pk = [list(map(float, p)) for p in pk]
                ck, sk = self._cands(emb, enc, bk, pk, neg, H, W, ("bp", "p"))
                mk, agk, _ = self._pick(ck, sk, Pk)
                outside = (mk & ~inside).sum() / max(mk.sum(), 1)
                if agk >= 0.45 and outside <= 0.1 and mk.sum() <= 2.5 * Pk.sum():
                    mask = mask | (mk & inside)
        mask = dilate(erode(mask, 1), 1)
        mask = erode(dilate(mask, 2), 2)
        a = (mask & dilate(whole, 4)).sum() / max((mask | whole).sum(), 1)
        return mask, float(a), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from sets import Store

    t0 = time.time()
    seg = PartSegmenter(a.model, a.device, smooth=True)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    fs = Store().frames("so101_stack_real")
    n = fs.n if not a.limit else min(a.limit, fs.n)
    pdir = pathlib.Path(a.parts) if pathlib.Path(a.parts).is_absolute() else ROOT / a.parts
    labels = {c: np.load(pdir / f"stack_{c}_parts.npz")["labels"] for c in CAMS}
    rng = np.random.default_rng(0)
    masks = {c: np.zeros((n, 480, 640), bool) for c in CAMS}
    score = {c: np.zeros((n, 2), np.float32) for c in CAMS}
    for i in range(n):
        obs = fs.obs(i)
        for c in CAMS:
            m, ag, s = seg.segment(np.asarray(obs["images"][c]), labels[c][i], rng)
            masks[c][i] = m
            score[c][i] = (ag, s)
        if (i + 1) % 80 == 0:
            print(f"{i + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    for c in CAMS:
        np.save(MASKS / f"stack_{c}{a.tag}.npy", np.packbits(masks[c], axis=-1))
        np.save(MASKS / f"stack_{c}{a.tag}_score.npy", score[c])
    sheet("stack", fs, masks, {c: labels[c][:n] > 0 for c in CAMS}, MASKS / "sheets" / f"stack{a.tag}.jpg")
    st = {c: dict(prompt_iou_med=float(np.median(score[c][:, 0])), area=float(masks[c].mean())) for c in CAMS}
    (MASKS / f"segment_stats_stack{a.tag}.json").write_text(json.dumps(dict(model=a.model, parts=str(pdir), n=n,
                                                                           seconds=time.time() - t0, cams=st), indent=1))
    print("stack:", json.dumps(st), f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
