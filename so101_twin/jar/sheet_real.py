"""Contact sheet of real frames (per camera) for a quick look."""
import sys, numpy as np
from PIL import Image, ImageDraw
S = "neuroscope/molmoact2/store/sets/so101_jar_real"
cam = sys.argv[1]; idx = [int(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
im = np.load(f"{S}/images_{cam}.npy", mmap_mode="r")
L = np.load(f"{S}/labels.npz", allow_pickle=True)
tiles = []
for i in idx:
    t = Image.fromarray(np.asarray(im[i])).resize((320, 240))
    ImageDraw.Draw(t).text((4, 4), f"{i} ep{L['episode'][i]} t{L['t'][i]} {L['phase'][i]}", fill=(255, 0, 0))
    tiles.append(np.asarray(t))
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
Image.fromarray(np.concatenate(rows, 0)).save(out, quality=85)
