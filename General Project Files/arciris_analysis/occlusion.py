#!/usr/bin/env python3
"""Per-image occlusion inside ArcIris's own iris ring (inference only, no training).

Unwraps the dataset's binary iris segmentation mask with the *same* circles
ArcIris predicts, so mask coverage is measured in the exact region the
embedding network sees. Writes polar masks + visible fraction per image_id.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent
sys.path.insert(0, str(HERE))
from embeddings import make_iris_rec  # noqa: E402

ARC = WEEK1 / "OpenSourceIrisRecognition/methods/ArcIris/Python"
OUT = WEEK1 / "Task 1 results"

rec = make_iris_rec(ARC)
rows = list(csv.DictReader(open(OUT / "manifest.csv")))
ids, fracs, masks, circles = [], [], [], []
torch.set_grad_enabled(False)
for n, r in enumerate(rows):
    fp = Path(r["filepath"])
    mp = WEEK1 / "NDCVRL_002 Dataset/segmentation_masks" / fp.parent.parent.name / fp.parent.name / f"{fp.stem}.png"
    if not mp.is_file():
        continue
    im = rec.fix_image(Image.open(fp).convert("RGB").split()[0])
    p, i = rec.circApprox(im)
    if not rec.checkQuality(p, i):
        continue
    m = Image.open(mp).convert("L").point(lambda v: 255 if v else 0)
    m = rec.fix_image(m) if m.size != (640, 480) else m
    pm = rec.cartToPol_torch(m, p, i) > 127
    ids.append(r["image_id"]); fracs.append(pm.mean()); masks.append(pm); circles.append(np.r_[p, i])
    if n % 500 == 0:
        print(n, flush=True)
np.savez_compressed(OUT / "polar_masks.npz", ids=np.array(ids), visible=np.array(fracs), masks=np.array(masks), circles=np.array(circles))
print("done", len(ids), "mean visible", np.mean(fracs))
