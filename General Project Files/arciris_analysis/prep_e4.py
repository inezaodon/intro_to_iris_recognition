#!/usr/bin/env python3
"""Dump images + polar unwraps for the E4 example figure (needs torch env; plotting is done by make_evidence.py)."""
import csv, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(__file__).resolve().parent; WEEK1 = HERE.parent
sys.path.insert(0, str(HERE))
from embeddings import make_iris_rec
torch.set_grad_enabled(False)
R = WEEK1 / "Task 1 results"
z = np.load(R / "polar_masks.npz"); occ = dict(zip(z["ids"], 1 - z["visible"])); circ = dict(zip(z["ids"], z["circles"]))
man = {r["image_id"]: r for r in csv.DictReader(open(R / "manifest.csv"))}
ids = sorted(occ, key=occ.get); pick = [ids[10], ids[len(ids) // 2], ids[-40]]
rec = make_iris_rec(WEEK1 / "OpenSourceIrisRecognition/methods/ArcIris/Python")
out = {"ids": np.array(pick)}
for n, i in enumerate(pick):
    im = rec.fix_image(Image.open(man[i]["filepath"]).convert("RGB").split()[0]); c = circ[i]
    out[f"img{n}"] = np.asarray(im); out[f"polar{n}"] = rec.cartToPol_torch(im, c[:3], c[3:])
np.savez_compressed(R / "evidence" / "_e4_data.npz", **out)
