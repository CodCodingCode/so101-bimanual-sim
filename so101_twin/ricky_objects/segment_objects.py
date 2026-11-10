"""Object masks on chosen real frames: Grounding DINO (tiny) box for each text prompt, SAM 2.1 (small) mask in the
best box. Run with the MolmoAct2 venv (transformers 5.3); never loads the policy.

    neuroscope/molmoact2/.venv/bin/python so101_twin/ricky_objects/segment_objects.py <set> <out.npz> \
        --frames 0,8,16 --prompts "a white bowl.|a small brown block."

Writes <out.npz>: for every (frame, cam, prompt): best box, its score, and the SAM mask (packed bits); all boxes
above threshold are kept too (boxes_all) so a caller can pick a different candidate.
"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("out")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--cams", default="top,side")
    ap.add_argument("--thr", type=float, default=0.2)
    a = ap.parse_args()
    from sets import Store
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    fs = Store().frames(a.set)
    frames = [int(x) for x in a.frames.split(",")]
    prompts = a.prompts.split("|")
    cams = a.cams.split(",")
    dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEV).eval()
    sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
    sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(DEV).eval()
    rec = dict(frame=[], cam=[], prompt=[], box=[], score=[], bits=[], boxes_all=[], scores_all=[])
    H, W = fs.meta["image_shape"][:2]
    with torch.no_grad():
        for k, i in enumerate(frames):
            for c in cams:
                img = Image.fromarray(np.asarray(fs._img[c][i]))
                for pr in prompts:
                    x = dp(images=img, text=pr, return_tensors="pt").to(DEV)
                    out = dm(**x)
                    r = dp.post_process_grounded_object_detection(out, x.input_ids, threshold=a.thr, text_threshold=a.thr,
                                                                  target_sizes=[img.size[::-1]])[0]
                    b, s = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
                    rec["frame"].append(i); rec["cam"].append(c); rec["prompt"].append(pr)
                    rec["boxes_all"].append(b.astype(np.float32)); rec["scores_all"].append(s.astype(np.float32))
                    if len(b) == 0:
                        rec["box"].append(np.full(4, np.nan, np.float32)); rec["score"].append(0.0)
                        rec["bits"].append(np.packbits(np.zeros((H, W), bool), axis=-1))
                        continue
                    j = int(np.argmax(s))
                    xs = sp(images=img, input_boxes=[[list(map(float, b[j]))]], return_tensors="pt").to(DEV)
                    o = sm(**xs, multimask_output=False)
                    m = sp.post_process_masks(o.pred_masks.cpu(), xs["original_sizes"].cpu())[0][0, 0].numpy() > 0
                    rec["box"].append(b[j].astype(np.float32)); rec["score"].append(float(s[j]))
                    rec["bits"].append(np.packbits(m, axis=-1))
            if k % 10 == 0:
                print(f"{a.set} {k + 1}/{len(frames)}", flush=True)
    np.savez_compressed(a.out, frame=np.array(rec["frame"]), cam=np.array(rec["cam"]), prompt=np.array(rec["prompt"]),
                        box=np.stack(rec["box"]), score=np.array(rec["score"]), bits=np.stack(rec["bits"]),
                        shape=np.array([H, W]), boxes_all=np.array(rec["boxes_all"], dtype=object),
                        scores_all=np.array(rec["scores_all"], dtype=object))
    print("saved", a.out, len(rec["frame"]))


if __name__ == "__main__":
    main()
