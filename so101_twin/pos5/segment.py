"""pos5 COPY of so101_twin/common/segment.py (changes marked POS5: pos5 only, masks in pos5/masks, chroma-key consensus
(positives from twin AND chroma interior, candidate chosen by agreement with the twin(s) and the twin-free chroma
mask), hard negatives on the static non-arm objects (box, wall corner, mat marks) where no arm is near and on the
real cube pixels).

    neuroscope/molmoact2/.venv/bin/python so101_twin/pos5/segment.py --prompts _tw1[,_tw2] --ck _ck --tag _s1

Real-arm segmentation of every real frame (both cameras) with SAM 2.1, prompted from the twin's arm silhouette.

    neuroscope/molmoact2/.venv/bin/python so101_twin/common/segment.py [--scenes lemon,sort] [--prompts DIR[,DIR2]]
        [--model facebook/sam2.1-hiera-small] [--device mps] [--limit N] [--tag ""] [--no-smooth]

ONE model load for all requested scenes (run it once, not per scene in parallel: memory; check first that no other
segment.py / segment_arm.py is alive: pgrep -fl "segment"). Per frame and camera the prompt silhouette P (twin arm
mask, packed bool from twin_prompts.py = the builder twin, or fit.py --export-prompts = a refitted twin) gives
  - a box = bbox of P dilated by 12 px (clipped to the image),
  - positives = up to 5 deep-interior points of P (distance-transform maxima, spread out),
  - negatives = 4 background points far (> 40 px) from P, spread out.
Three prompt variants (box+points, points only, box only) x 3 multimask outputs = 9 candidates; the kept mask is the
candidate with the best  IoU(candidate, P_dilated-tolerant) + 0.25 * SAM predicted IoU,  i.e. SAM supplies the real
boundary and the twin decides which object/part granularity is "the arm". Frames whose prompt is empty (arm not in
view of the twin camera) get an empty mask.
Several prompt dirs (comma-separated) = consensus prompting: box / negatives from the union of the silhouettes,
positives from their intersection, candidate chosen by the MEAN IoU with all of them. Use it to compare two twins
fairly (masks chosen by one twin's silhouette favour that twin: on lemon, masks prompted by the builder twin gave
the builder twin +0.016 median IoU over an equally good refit).
Smoothing (default on): 3x3 average of the 256x256 mask logits before upsampling (removes the decoder's checkerboard
speckle in uncertain regions), then a 1-px opening and 2-px closing of the chosen mask.

Output: so101_twin/masks/<scene>_<cam><tag>.npy  uint8 [n, 480, 80] = np.packbits(mask, axis=-1)
        (unpack: np.unpackbits(a, axis=-1).astype(bool) -> [n, 480, 640]), plus <scene>_<cam><tag>_score.npy
        float32 [n, 2] (chosen candidate's IoU with the prompt, SAM predicted IoU), and
        so101_twin/masks/sheets/<scene><tag>.jpg: 16 frames x 2 cams (real + SAM outline green + twin outline red).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

TWIN = pathlib.Path(__file__).resolve().parent.parent
ROOT = TWIN.parent
MASKS = pathlib.Path(__file__).resolve().parent / "masks"   # POS5
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
SCENES = ("lemon", "pickplace", "legocup", "drawer", "ringpole", "bincube", "sort", "stack")
CAMS = ("top", "side")


def unpack(a: np.ndarray, w: int = 640) -> np.ndarray:
    return np.unpackbits(a, axis=-1)[..., :w].astype(bool)


# ---------------------------------------------------------------------------------------------------- numpy morphology
def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r px via cumulative sums (no scipy in this venv)."""
    if r <= 0:
        return m
    H, W = m.shape
    c = np.pad(m.astype(np.int32), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]
    return s > 0


def erode(m: np.ndarray, r: int) -> np.ndarray:
    return ~dilate(~m, r)


def depth_map(m: np.ndarray, rmax: int = 24) -> np.ndarray:
    """Approximate chessboard distance to the background (0 outside), capped at rmax, from repeated erosion."""
    d = np.zeros(m.shape, np.int32)
    cur = m.copy()
    for r in range(1, rmax + 1):
        d += cur
        cur = erode(cur, 1)
        if not cur.any():
            break
    return d


def spread_points(cand: np.ndarray, weight: np.ndarray, k: int, min_dist: float) -> list:
    """Greedy: highest weight first, keep points at least min_dist apart. cand [N,2] (x,y)."""
    order = np.argsort(-weight)
    out = []
    for i in order:
        p = cand[i]
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_dist ** 2 for q in out):
            out.append(p)
            if len(out) == k:
                break
    return out


def prompts_from(P: np.ndarray, rng: np.random.Generator, core: np.ndarray | None = None, hard=None):
    """box from P (dilated), positives from the deep interior of `core` (default P), negatives far from P.
    POS5: hard = bool mask of pixels that are certainly not arm (static objects far from the arm, the cube):
    up to 3 extra negatives there, inside the box."""
    H, W = P.shape
    ys, xs = np.nonzero(P)
    if len(xs) < 30:
        return None
    if core is not None and core.sum() >= 30:
        ys, xs = np.nonzero(core)
    else:
        core = P
    Pd = dilate(P, 12)
    yd, xd = np.nonzero(Pd)
    box = [float(xd.min()), float(yd.min()), float(xd.max() + 1), float(yd.max() + 1)]
    D = depth_map(core)
    sel = rng.choice(len(xs), size=min(len(xs), 4000), replace=False)
    cand = np.stack([xs[sel], ys[sel]], 1)
    pos = spread_points(cand, D[ys[sel], xs[sel]] + rng.random(len(sel)) * 0.5, 5, 25)
    far = ~dilate(P, 40)
    yb, xb = np.nonzero(far[::4, ::4])
    neg = []
    if len(xb):
        cb = np.stack([xb * 4, yb * 4], 1)
        # prefer background points near-ish the box (more informative than image corners)
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        dist = np.hypot(cb[:, 0] - cx, cb[:, 1] - cy)
        neg = spread_points(cb, -dist + rng.random(len(cb)) * 40, 4, 60)
    if hard is not None and hard.any():   # POS5 hard negatives inside the prompt box
        hb = hard.copy()
        hb[: int(box[1])] = False
        hb[int(box[3]):] = False
        hb[:, : int(box[0])] = False
        hb[:, int(box[2]):] = False
        hb = erode(hb, 3)
        yh, xh = np.nonzero(hb[::2, ::2])
        if len(xh):
            ch = np.stack([xh * 2, yh * 2], 1)
            neg = list(neg) + spread_points(ch, rng.random(len(ch)), 3, 30)
    return box, [list(map(float, p)) for p in pos], [list(map(float, p)) for p in neg]


# ---------------------------------------------------------------------------------------------------- SAM
class Segmenter:
    def __init__(self, model_id: str, device: str, smooth: bool = True):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device
        self.smooth = smooth
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def __call__(self, img: np.ndarray, Ps, rng, ck=None, hard=None) -> tuple[np.ndarray, float, float]:
        """Ps: one prompt silhouette [H, W] or a list of them (consensus: box / negatives from their union,
        positives from their intersection, candidate chosen by the mean IoU with all of them)."""
        torch = self.torch
        Ps = [Ps] if isinstance(Ps, np.ndarray) else [p for p in Ps if p.sum() >= 30] or list(Ps[:1])
        P = np.logical_or.reduce(Ps)
        core = np.logical_and.reduce(Ps) if len(Ps) > 1 else P
        if ck is not None and (core & ck).sum() >= 200:   # POS5: positives where the twin(s) AND the chroma agree
            core = core & ck
        pr = prompts_from(P, rng, core=core, hard=hard)
        if pr is None:
            return np.zeros(P.shape, bool), 0.0, 0.0
        box, pos, neg = pr
        pts = pos + neg
        labs = [1] * len(pos) + [0] * len(neg)
        with torch.inference_mode():
            enc = self.proc(images=Image.fromarray(img), return_tensors="pt")
            pv = enc["pixel_values"].to(self.device)
            emb = self.model.get_image_embeddings(pv)
            cands, scores = [], []
            H, W = P.shape
            ts = float(self.proc.target_size)
            sx, sy = ts / W, ts / H
            pt = torch.tensor([[[[x * sx, y * sy] for x, y in pts]]], dtype=torch.float32, device=self.device)
            lb = torch.tensor([[labs]], dtype=torch.int32, device=self.device)
            bx = torch.tensor([[[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]]], dtype=torch.float32,
                              device=self.device)
            for kw in (dict(input_points=pt, input_labels=lb, input_boxes=bx),
                       dict(input_points=pt, input_labels=lb),
                       dict(input_boxes=bx)):
                out = self.model(image_embeddings=emb, multimask_output=True, **kw)
                logits = out.pred_masks.float().cpu()                       # [1, 1, 3, 256, 256]
                if self.smooth:   # the decoder's transposed convs leave a checkerboard where logits are near 0
                    sh = logits.shape
                    logits = torch.nn.functional.avg_pool2d(logits.reshape(-1, 1, *sh[-2:]), 3, 1, 1,
                                                            count_include_pad=False).reshape(sh)
                masks = self.proc.post_process_masks(logits, enc["original_sizes"])[0]
                cands.append(masks[0].numpy().astype(bool))
                scores.append(out.iou_scores.float().cpu().numpy().reshape(-1))
        cands = np.concatenate(cands, 0)
        scores = np.concatenate(scores, 0)
        Pds = [dilate(p, 4) for p in Ps]

        def agree(c):   # mean over prompts of IoU(c, prompt), tolerant by 4 px; POS5: + IoU with the chroma mask
            a = float(np.mean([(c & pd).sum() / max((c | p).sum(), 1) for p, pd in zip(Ps, Pds)]))
            if ck is None or ck.sum() < 30:
                return a
            return 0.5 * a + 0.5 * float((c & ck).sum() / max((c | ck).sum(), 1))

        best, bi = -1.0, 0
        for i, c in enumerate(cands):
            s = agree(c) + 0.25 * scores[i]
            if s > best:
                best, bi = s, i
        c = cands[bi]
        if self.smooth:   # drop 1-px speckles and close 2-px gaps
            c = dilate(erode(c, 1), 1)
            c = erode(dilate(c, 2), 2)
        return c, agree(c), float(scores[bi])


def outline(m: np.ndarray) -> np.ndarray:
    return m & ~erode(m, 1)


def main():   # POS5
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="_tw1", help="twin silhouette tags (pos5/masks/pos5_<cam><tag>.npy), comma list")
    ap.add_argument("--ck", default="_ck", help="twin-free chroma mask tag ('' = none)")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cams", default="top,side")
    ap.add_argument("--tag", default="_s1")
    a = ap.parse_args()
    from sets import Store

    t0 = time.time()
    seg = Segmenter(a.model, a.device, smooth=True)
    print(f"model {a.model} on {a.device} loaded in {time.time() - t0:.0f}s", flush=True)
    fs = Store().frames("so101_pos5_real")
    n = fs.n if not a.limit else min(a.limit, fs.n)
    for c in a.cams.split(","):
        rng = np.random.default_rng(0)
        plist = [unpack(np.load(MASKS / f"pos5_{c}{t}.npy")[:n]) for t in a.prompts.split(",")]
        ck = unpack(np.load(MASKS / f"pos5_{c}{a.ck}.npy")[:n]) if a.ck else None
        static = unpack(np.load(MASKS / f"pos5_{c}_static.npy"))[0]
        cube = unpack(np.load(MASKS / f"pos5_{c}_cube.npy")[:n])
        masks = np.zeros((n, 480, 640), bool)
        score = np.zeros((n, 2), np.float32)
        t1 = time.time()
        for i in range(n):
            img = np.asarray(fs.obs(i)["images"][c])
            near = dilate(np.logical_or.reduce([pl[i] for pl in plist]) | (ck[i] if ck is not None else False), 12)
            hard = (static & ~near) | erode(cube[i], 2)
            m, iou, sc = seg(img, [pl[i] for pl in plist], rng, ck=None if ck is None else ck[i], hard=hard)
            masks[i] = m
            score[i] = (iou, sc)
            if i % 80 == 0:
                print(f"{c} {i}/{n} agree {iou:.2f} sam {sc:.2f} {time.time() - t1:.0f}s", flush=True)
        np.save(MASKS / f"pos5_{c}{a.tag}.npy", np.packbits(masks, axis=-1))
        np.save(MASKS / f"pos5_{c}{a.tag}_score.npy", score)
        print(f"{c}: {n} frames in {time.time() - t1:.0f}s, agree med {np.median(score[:, 0]):.3f} sam med "
              f"{np.median(score[:, 1]):.3f}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB",
              flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
