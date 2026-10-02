#!/usr/bin/env python3
"""Genuine-pair miss rate by occlusion quartile, with counts and a subject-eye cluster bootstrap.

Occlusion of a pair = max(occlusion of image A, image B); occlusion = 1 - visible fraction of
ArcIris's iris ring per the dataset segmentation mask (polar_masks.npz, from occlusion.py).
A genuine pair is "missed" if its score > the 1% impostor-tail cut (impostor_tail_summary.json).
"""
import csv, json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "Task 1 results"
z = np.load(R / "polar_masks.npz"); occ = dict(zip(z["ids"], 1 - z["visible"]))
cut = json.load(open(R / "impostor_tail_summary.json"))["threshold"]
rows = [r for r in csv.DictReader(open(R / "pairwise_scores.csv"))
        if r["label"] == "genuine" and r["image_id_a"] in occ and r["image_id_b"] in occ]
sc = np.array([float(r["score"]) for r in rows])
mo = np.array([max(occ[r["image_id_a"]], occ[r["image_id_b"]]) for r in rows])
grp = np.array([r["subject_a"] + r["eye_a"] for r in rows])
q = np.clip(np.digitize(mo, np.quantile(mo, [.25, .5, .75])), 0, 3)
for k in range(4):
    m = q == k; miss = int((sc[m] > cut).sum())
    print(f"Q{k+1} occ {mo[m].min():.2f}-{mo[m].max():.2f}  n={m.sum()}  missed={miss}  rate={miss/m.sum():.3%}")
rng = np.random.RandomState(0); G = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in G}; res = []
for _ in range(1000):
    p = np.concatenate([idx[g] for g in rng.choice(G, len(G))]); a = (sc[p][q[p] == 0] > cut).mean(); b = (sc[p][q[p] == 3] > cut).mean()
    res.append((a, b, b / a if a else np.nan))
res = np.array(res); print("95% CI  Q1", np.percentile(res[:, 0], [2.5, 97.5]), " Q4", np.percentile(res[:, 1], [2.5, 97.5]), " ratio", np.nanpercentile(res[:, 2], [2.5, 97.5]))
