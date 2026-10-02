#!/usr/bin/env python3
"""Genuine vs impostor score distributions for HDBIF, same visual style as
Week 1's `score_distributions.png` (ArcIris) so the two methods are directly comparable.

HDBIF score = fractional Hamming distance (bit mismatch rate over the unoccluded,
overlapping iris region). Lower = more similar; 0.5 = chance level (unlike ArcIris's
acos-cosine score, which centres near pi/2 for chance).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "hdbif_pairwise_scores.csv")))
def usable(r):
    v = float(r["score"])
    return v >= 0 and np.isfinite(v)

g = np.array([float(r["score"]) for r in rows if r["label"] == "genuine" and usable(r)])
i = np.array([float(r["score"]) for r in rows if r["label"] == "impostor" and usable(r)])
n_dropped_g = sum(1 for r in rows if r["label"] == "genuine" and not usable(r))
n_dropped_i = sum(1 for r in rows if r["label"] == "impostor" and not usable(r))

dprime = abs(g.mean() - i.mean()) / np.sqrt(0.5 * (g.var(ddof=1) + i.var(ddof=1)))
# threshold: same rule as Week 1 (1st percentile of impostor scores), on the LOW side here too
cut = np.percentile(i, 1)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 0.5, 120)
ax.hist(g, bins, density=True, alpha=.6, color="tab:green", label=f"Genuine (n={len(g):,})")
ax.hist(i, bins, density=True, alpha=.6, color="tab:red", label=f"Impostor (n={len(i):,})")
ax.axvline(cut, color="k", ls="--", label=f"1% impostor tail cut = {cut:.3f}")
ax.set_xlabel("HDBIF score (fractional Hamming distance) - lower = more similar")
ax.set_ylabel("Density")
ax.set_title(f"NDCVRL_002 subset (HDBIF): genuine vs impostor scores (d' = {dprime:.2f})")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "hdbif_score_distributions.png", dpi=150)

print(f"genuine n={len(g)} mean={g.mean():.4f}  impostor n={len(i)} mean={i.mean():.4f}  d'={dprime:.2f}")
print(f"cut={cut:.4f}  impostor<=cut: {np.mean(i<=cut):.3%}  genuine>cut: {np.mean(g>cut):.2%}")
print(f"dropped (mask overlap too small): genuine {n_dropped_g}, impostor {n_dropped_i}")
