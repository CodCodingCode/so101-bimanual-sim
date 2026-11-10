"""Static background per camera from the real frames only (no twin): per-pixel median image over all 320 frames,
and the green-screen key. Writes out/bg_<cam>.png, masks/bg_<cam>.npy (uint8 HxWx3)."""
import sys, pathlib
import numpy as np
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5, fit as F


def green(rgb):
    """green-screen key on PIL HSV (hue 0..255): green hue band with some saturation."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    return (hsv[..., 0] >= 70) & (hsv[..., 0] <= 140) & (hsv[..., 1] >= 45)


if __name__ == "__main__":
    fs = F.real_set(p5.scene())
    for c in p5.CAMS:
        imgs = np.asarray(fs._img[c])
        med = np.median(imgs, axis=0).astype(np.uint8)
        np.save(HERE / "masks" / f"bg_{c}.npy", med)
        g = green(med)
        vis = med.copy()
        vis[~g] = (vis[~g] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
        Image.fromarray(np.concatenate([med, vis], 1)).save(HERE / "out" / f"bg_{c}.png")
        print(c, "not-green frac in bg", round(float((~g).mean()), 4))
