#!/usr/bin/env python3
"""Per-image iris texture scores from ArcIris's own polar unwrap (inference only).

For each image: unwrap with ArcIris's circles, keep only visible-iris pixels
(dataset segmentation mask, from polar_masks.npz) and measure
  iris_mean      mean gray level (darkness of the iris)
  iris_std       raw contrast
  hf_std         std of (polar - gaussian blur sigma=3): fine texture energy
  hf_rel         hf_std / iris_mean: texture relative to brightness
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent; WEEK1 = HERE.parent
sys.path.insert(0, str(HERE))
from embeddings import make_iris_rec
torch.set_grad_enabled(False)
R = WEEK1 / "Task 1 results"
z = np.load(R / "polar_masks.npz")
man = {r["image_id"]: r for r in csv.DictReader(open(R / "manifest.csv"))}
rec = make_iris_rec(WEEK1 / "OpenSourceIrisRecognition/methods/ArcIris/Python")
rows = []
for n, (i, mk, c, vis) in enumerate(zip(z["ids"], z["masks"], z["circles"], z["visible"])):
    im = rec.fix_image(Image.open(man[i]["filepath"]).convert("RGB").split()[0])
    pol = rec.cartToPol_torch(im, c[:3], c[3:]).astype(np.float32)
    hf = pol - cv2.GaussianBlur(pol, (0, 0), 3)
    if mk.sum() < 200:            # essentially no visible iris
        rows.append([i, vis, "", "", "", ""]); continue
    v = pol[mk]; h = hf[mk]
    rows.append([i, vis, v.mean(), v.std(), h.std(), h.std() / max(v.mean(), 1)])
    if n % 1000 == 0: print(n, flush=True)
with open(R / "texture_scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image_id", "visible_frac", "iris_mean", "iris_std", "hf_std", "hf_rel"]); w.writerows(rows)
print("done", len(rows))
