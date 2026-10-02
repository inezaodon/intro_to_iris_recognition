#!/usr/bin/env python3
"""Inference-only test: blank occluded polar pixels (mid-gray) before extractVector.

No weights are changed or trained. Compares d' and error rates on a subset,
unmasked (cached embeddings) vs masked (re-extracted).
"""
import csv, itertools, pickle, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent
sys.path.insert(0, str(HERE))
from embeddings import make_iris_rec

ARC = WEEK1 / "OpenSourceIrisRecognition/methods/ArcIris/Python"
OUT = WEEK1 / "Task 1 results"
torch.set_grad_enabled(False)
rec = make_iris_rec(ARC)
z = np.load(OUT / "polar_masks.npz")
mask = dict(zip(z["ids"], z["masks"])); circ = dict(zip(z["ids"], z["circles"]))
man = {r["image_id"]: r for r in csv.DictReader(open(OUT / "manifest.csv"))}
base = pickle.load(open(OUT / "embeddings.pkl", "rb"))

by = {}
for i in mask:
    by.setdefault((man[i]["subject_id"], man[i]["eye"]), []).append(i)
rng = np.random.RandomState(0)
keys = [k for k, v in sorted(by.items()) if len(v) >= 10]
keys = [keys[j] for j in rng.choice(len(keys), 60, replace=False)]
ids = [i for k in keys for i in by[k][:15]]
print(len(keys), "subject-eyes,", len(ids), "images", flush=True)

masked = {}
for n, i in enumerate(ids):
    im = rec.fix_image(Image.open(man[i]["filepath"]).convert("RGB").split()[0])
    c = circ[i]
    polar = rec.cartToPol_torch(im, c[:3], c[3:]).copy()
    polar[~mask[i]] = 127
    masked[i] = np.asarray(rec.extractVector(polar))
    if n % 100 == 0: print(n, flush=True)

def score(a, b):
    cs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(np.clip(cs, -1.0, 1.0)))  # same acos as matchVectors, clamped for float error

def run(emb):
    g, imp, occ_g = [], [], []
    for a, b in itertools.combinations(ids, 2):
        s = score(emb[a], emb[b])
        same = man[a]["subject_id"] == man[b]["subject_id"] and man[a]["eye"] == man[b]["eye"]
        (g if same else imp).append(s)
    return np.array(g), np.array(imp)

res = {}
for name, emb in [("unmasked", {i: base[i] for i in ids}), ("masked", masked)]:
    g, imp = run(emb)
    dp = abs(g.mean() - imp.mean()) / np.sqrt(0.5 * (g.var(ddof=1) + imp.var(ddof=1)))
    t = np.percentile(imp, 1)
    print(f"{name:9} genuine {g.mean():.3f} impostor {imp.mean():.3f} d'={dp:.2f}  FNMR@FMR1%={np.mean(g > t):.2%}  cut={t:.3f}")
    res[name] = (g, imp)
np.savez(OUT / "masked_rescore_subset.npz", ids=np.array(ids), **{f"{k}_g": v[0] for k, v in res.items()}, **{f"{k}_i": v[1] for k, v in res.items()})
