#!/usr/bin/env python3
"""Genuine vs impostor score distributions, same style as Week 1 `score_distributions.png`, one per capture-format group."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CUT = json.load(open(HERE.parent / "Week 1" / "Task 1 results" / "impostor_tail_summary.json"))["threshold"]
z = np.load(HERE / "group_scores.npz")
G = {  # key, title, file suffix
    "all": ("All pairs", "all"),
    0: ("640x480 only (both images)", "640x480_only"),
    1: ("Mixed formats (one 640x480, one 960x640)", "mixed_formats"),
    2: ("960x640 only (both images)", "960x640_only"),
}
gen = {0: z["gen0"], 1: z["gen1"], 2: z["gen2"]}; imp = {0: z["imp0"], 1: z["imp1"], 2: z["imp2"]}
gen["all"] = np.concatenate([gen[k] for k in (0, 1, 2)]); imp["all"] = np.concatenate([imp[k] for k in (0, 1, 2)])
bins = np.linspace(0, 2, 120)
YMAX = 0

def dprime(g, i):
    g = g.astype(np.float64); i = i.astype(np.float64)
    return abs(g.mean() - i.mean()) / np.sqrt(0.5 * (g.var(ddof=1) + i.var(ddof=1)))

def draw(ax, k):
    g, i = gen[k], imp[k]
    ax.hist(g, bins, density=True, alpha=.6, color="tab:green", label=f"Genuine (n={len(g):,})")
    ax.hist(i, bins, density=True, alpha=.6, color="tab:red", label=f"Impostor (n={len(i):,})")
    ax.axvline(CUT, color="k", ls="--", label=f"1% impostor tail cut = {CUT:.3f}")
    ax.set_xlabel("ArcIris score (acos cosine similarity, radians) - lower = more similar"); ax.set_ylabel("Density")
    ax.set_title(f"NDCVRL_002: {G[k][0]}\n(d' = {dprime(g, i):.2f};  impostors under the cut {np.mean(i <= CUT):.2%};  genuine over the cut {np.mean(g > CUT):.1%})")
    ax.legend()

ylim = 0
for k in G:
    fig, ax = plt.subplots(figsize=(9, 5)); draw(ax, k); fig.tight_layout()
    ylim = max(ylim, ax.get_ylim()[1]); fig.savefig(HERE / "plots" / f"W2_dist_{G[k][1]}.png", dpi=150); plt.close(fig)
# combined figure with identical axes for direct comparison
fig, axs = plt.subplots(2, 2, figsize=(17, 10), sharex=True, sharey=True)
for ax, k in zip(axs.ravel(), G): draw(ax, k)
for ax in axs[0]: ax.set_xlabel("")
for ax in axs[:, 1]: ax.set_ylabel("")
fig.tight_layout(); fig.savefig(HERE / "plots" / "W2_dist_all_groups_side_by_side.png", dpi=130)
for k in G:
    g, i = gen[k], imp[k]
    print(f"{G[k][0]:45} genuine n={len(g):>9,} mean {g.mean():.3f} | impostor n={len(i):>10,} mean {i.mean():.3f} | d'={dprime(g, i):.2f} | imp<cut {np.mean(i<=CUT):.2%} | gen>cut {np.mean(g>CUT):.1%}")
