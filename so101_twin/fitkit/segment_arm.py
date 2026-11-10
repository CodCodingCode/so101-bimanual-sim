"""Real arm masks with open models, independent of the twin: Grounding DINO (tiny) boxes 'a robot arm', SAM 2.1
(small) cuts the mask inside the box. Run with the MolmoAct2 venv (transformers 5.3):

    neuroscope/molmoact2/.venv/bin/python so101_twin/fitkit/segment_arm.py <set> <cam> [--frames 0,5,9] [--preview]

Writes so101_twin/fitkit/masks/<set>__<cam>.npz (bits = packed bool [n, H, W], score = box score per frame, empty
frames have score 0) or, with --preview, a contact sheet in fitkit/out/."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PROMPT = "a robot arm."


class Segmenter:
    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEV).eval()
        self.sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
        self.sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(DEV).eval()

    @torch.no_grad()
    def boxes(self, img: Image.Image, prompt: str, thr: float = 0.25):
        x = self.dp(images=img, text=prompt, return_tensors="pt").to(DEV)
        out = self.dm(**x)
        r = self.dp.post_process_grounded_object_detection(out, x.input_ids, threshold=thr, text_threshold=thr,
                                                           target_sizes=[img.size[::-1]])[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()

    @torch.no_grad()
    def mask(self, img: Image.Image, box) -> np.ndarray:
        x = self.sp(images=img, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(DEV)
        out = self.sm(**x, multimask_output=False)
        m = self.sp.post_process_masks(out.pred_masks.cpu(), x["original_sizes"].cpu())[0]
        return m[0, 0].numpy() > 0

    def arm(self, rgb: np.ndarray, prompt: str = PROMPT) -> tuple[np.ndarray, float]:
        img = Image.fromarray(rgb)
        b, s = self.boxes(img, prompt)
        if len(b) == 0:
            return np.zeros(rgb.shape[:2], bool), 0.0
        # union of the masks of every confident box (an arm partly occluded can come out as two boxes)
        keep = s >= max(0.25, 0.6 * s.max())
        m = np.zeros(rgb.shape[:2], bool)
        for bb in b[keep]:
            m |= self.mask(img, bb)
        return m, float(s.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("cam")
    ap.add_argument("--frames", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--prompts", default="", help="npy of (frame, box, pos_xy, neg_xy) rows from armfit.export_prompts;"
                                                 " SAM only, no Grounding DINO; output named <set>__<view>.npz")
    ap.add_argument("--view", default="")
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    a = ap.parse_args()
    from sets import Store

    fs = Store().frames(a.set)
    if a.prompts:
        return prompted(fs, a)
    idx = [int(x) for x in a.frames.split(",")] if a.frames else list(range(fs.n))
    seg = Segmenter()
    masks, scores = [], []
    for k, i in enumerate(idx):
        rgb = np.asarray(fs.obs(i)["images"][a.cam])
        m, s = seg.arm(rgb, a.prompt)
        masks.append(m)
        scores.append(s)
        if k % 40 == 0:
            print(f"{a.set} {a.cam} {k}/{len(idx)} score {s:.2f} px {m.sum()}", flush=True)
    if a.preview:
        rows = []
        for i, m in zip(idx, masks):
            rgb = np.asarray(fs.obs(i)["images"][a.cam]).copy()
            ov = rgb.copy()
            ov[~m] = (ov[~m] * 0.3).astype(np.uint8)
            rows.append(np.concatenate([rgb, ov], 1))
        im = Image.fromarray(np.concatenate(rows, 0))
        im.thumbnail((1000, 3000))
        im.save(HERE / "out" / f"segprev_{a.set}__{a.cam}.jpg", quality=85)
        return
    np.savez_compressed(HERE / "masks" / f"{a.set}__{a.cam}.npz", bits=np.packbits(np.stack(masks), axis=-1),
                        shape=np.array(masks[0].shape), score=np.array(scores), frames=np.array(idx))
    print("saved", len(idx), "masks")


@torch.no_grad()
def prompted(fs, a):
    from transformers import Sam2Model, Sam2Processor

    sp = Sam2Processor.from_pretrained(a.sam_model)
    sm = Sam2Model.from_pretrained(a.sam_model).to(DEV).eval()
    rows = np.load(a.prompts, allow_pickle=True)
    H, W = fs.meta["image_shape"][:2]
    masks, frames, scores = [], [], []
    prev = []
    for k, (i, box, pos, neg) in enumerate(rows):
        frames.append(int(i))
        if box is None:
            masks.append(np.zeros((H, W), bool))
            scores.append(0.0)
            continue
        img = Image.fromarray(np.asarray(fs.obs(int(i))["images"][a.cam]))
        pts = np.concatenate([pos, neg], 0).astype(float)
        lab = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        x = sp(images=img, input_boxes=[[list(map(float, box))]], input_points=[[pts.tolist()]],
               input_labels=[[lab.tolist()]], return_tensors="pt").to(DEV)
        out = sm(**x, multimask_output=False)
        m = sp.post_process_masks(out.pred_masks.cpu(), x["original_sizes"].cpu())[0][0, 0].numpy() > 0
        masks.append(m)
        scores.append(float(out.iou_scores.flatten()[0]))
        if len(prev) < 6 and k % max(1, len(rows) // 6) == 0:
            prev.append((np.asarray(img), m, box, pos, neg))
        if k % 50 == 0:
            print(f"{a.set} {a.view} {k}/{len(rows)} iou_pred {scores[-1]:.2f} px {m.sum()}", flush=True)
    name = a.view or a.cam
    np.savez_compressed(HERE / "masks" / f"{a.set}__{name}.npz", bits=np.packbits(np.stack(masks), axis=-1),
                        shape=np.array([H, W]), score=np.array(scores), frames=np.array(frames))
    from PIL import ImageDraw

    tiles = []
    for rgb, m, box, pos, neg in prev:
        ov = rgb.copy()
        ov[~m] = (ov[~m] * 0.3).astype(np.uint8)
        im = Image.fromarray(np.concatenate([rgb, ov], 1))
        d = ImageDraw.Draw(im)
        d.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 255, 0), width=2)
        for x_, y_ in pos:
            d.ellipse([x_ - 4, y_ - 4, x_ + 4, y_ + 4], fill=(0, 255, 0))
        for x_, y_ in neg:
            d.ellipse([x_ - 4, y_ - 4, x_ + 4, y_ + 4], fill=(255, 0, 0))
        tiles.append(np.asarray(im))
    if tiles:
        im = Image.fromarray(np.concatenate(tiles, 0))
        im.thumbnail((1000, 3000))
        im.save(HERE / "out" / f"sam_{a.set}__{name}.jpg", quality=85)
    print("saved", len(frames), "masks")


if __name__ == "__main__":
    main()
