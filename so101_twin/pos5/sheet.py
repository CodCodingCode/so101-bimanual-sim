"""Diagnostic sheet: per frame, real | real with mask (green) + twin (red) outlines | IoU overlay
(white both, red real only, blue twin only).   sheet.py <cam> <mask tag> <twin tag> <frames|odd8> <out.jpg>"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import fit as F
import p5

cam, mt, tt, fr, out = sys.argv[1:6]
fs = F.real_set(p5.scene())
R, T = p5.load(cam, mt), p5.load(cam, tt)
if fr == "odd8":
    odd = np.flatnonzero(p5.odd())
    idx = odd[np.linspace(0, len(odd) - 1, 10).astype(int)[1:-1]]
else:
    idx = [int(x) for x in fr.split(",")]
v = p5.iou(T, R)
rows = []
for i in idx:
    im = np.asarray(fs.obs(int(i))["images"][cam])
    a = im.copy()
    a[ndi.binary_dilation(R[i] & ~ndi.binary_erosion(R[i]))] = (0, 255, 0)
    a[T[i] & ~ndi.binary_erosion(T[i])] = (255, 0, 0)
    ov = (im * 0.35).astype(np.uint8)
    ov[R[i] & T[i]] = 255
    ov[R[i] & ~T[i]] = (255, 40, 40)
    ov[~R[i] & T[i]] = (40, 90, 255)
    t = Image.fromarray(np.concatenate([im, a, ov], 1))
    ImageDraw.Draw(t).text((5, 5), f"{cam} frame {i} ep {int(fs.labels['episode'][i])} IoU {v[i]:.3f}", fill=(255, 255, 0))
    rows.append(np.asarray(t))
g = Image.fromarray(np.concatenate(rows, 0))
g = g.resize((g.width * 2 // 3, g.height * 2 // 3), Image.BILINEAR)
g.save(out, quality=85)
