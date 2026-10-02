#!/usr/bin/env python3
"""Impostor (and genuine) score distributions split by capture format, over ALL pairs.

Groups by the image format of the two images: 640x480 only ("std"), 960x640 only
("wide") and mixed. Uses cached ArcIris embeddings; no model is run or trained.
"""
import csv, json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent / "Week 1" / "Task 1 results"
CUT = json.load(open(R / "impostor_tail_summary.json"))["threshold"]

e = pickle.load(open(R / "embeddings.pkl", "rb")); ids = list(e); N = len(ids)
X = np.array([np.asarray(e[k], np.float64) for k in ids]); X /= np.linalg.norm(X, axis=1, keepdims=True)
man = {r["image_id"]: r for r in csv.DictReader(open(R / "manifest.csv"))}
lab = np.array([man[k]["subject_id"] + man[k]["eye"] for k in ids])
wide = np.array([Image.open(man[k]["filepath"]).size[0] == 960 for k in ids])

S = X @ X.T
iu = np.triu(np.ones((N, N), bool), 1)
same = lab[:, None] == lab[None, :]
gid = wide[:, None].astype(int) + wide[None, :].astype(int)          # 0 std-std, 1 mixed, 2 wide-wide
A = np.arccos(np.clip(S, -1, 1)).astype(np.float32)
names = {0: "640x480 only", 1: "mixed formats", 2: "960x640 only"}
cols = {0: "tab:blue", 1: "tab:purple", 2: "tab:orange"}
imp = {g: A[iu & ~same & (gid == g)] for g in names}
gen = {g: A[iu & same & (gid == g)] for g in names}
np.savez_compressed(HERE / "group_scores.npz", **{f"imp{g}": imp[g] for g in names}, **{f"gen{g}": gen[g] for g in names})

rows = []
print(f"cut = {CUT:.4f}")
for g in names:
    v = imp[g].astype(np.float64); n = len(v)
    mu, sd = v.mean(), v.std()
    # tail expectation from the group's own body: normal fitted to the central 50% (robust to the tail)
    q1, q3 = np.percentile(v, [25, 75]); rsd = (q3 - q1) / 1.349; med = np.median(v)
    exp_cut = stats.norm.cdf(CUT, med, rsd)
    obs_cut = (v <= CUT).mean()
    qs = [0.0001, 0.001, 0.01, 0.05]
    obs_q = np.quantile(v, qs); exp_q = stats.norm.ppf(qs, med, rsd)
    rows.append(dict(group=names[g], n_pairs=n, mean=mu, std=sd, robust_std=rsd, skew=stats.skew(v), excess_kurtosis=stats.kurtosis(v),
                     frac_below_cut=obs_cut, normal_frac_below_cut=exp_cut, tail_ratio=obs_cut / exp_cut,
                     **{f"q{q}": o for q, o in zip(qs, obs_q)}, **{f"normal_q{q}": o for q, o in zip(qs, exp_q)},
                     n_genuine=len(gen[g]), genuine_median=float(np.median(gen[g])) if len(gen[g]) else np.nan))
    print(f"\n{names[g]}: {n:,} impostor pairs  mean {mu:.3f} std {sd:.3f} (robust {rsd:.3f})  skew {stats.skew(v):.2f}  exc.kurt {stats.kurtosis(v):.2f}")
    print(f"  below cut: observed {obs_cut:.3%}  vs normal-from-body {exp_cut:.3%}  -> {obs_cut/exp_cut:.1f}x heavier")
    for q, o, x in zip(qs, obs_q, exp_q): print(f"  {q:.2%} quantile: observed {o:.3f}  normal-from-body {x:.3f}  (gap {x-o:+.3f})")
with open(HERE / "tail_shape_by_group.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# ---- Figure 1: three groups overlaid (linear + log) -------------------------------------
bins = np.linspace(0.6, 2.0, 141)
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
for a, logy in zip(ax, [False, True]):
    for g in names:
        a.hist(imp[g], bins, density=True, histtype="step", lw=2, color=cols[g], label=f"{names[g]} (n={len(imp[g]):,})")
    a.axvline(CUT, c="k", ls="--", label=f"cut {CUT:.3f}"); a.set_xlabel("impostor score (radians, lower = more similar)")
    a.set_ylabel("density" + (" (log)" if logy else "")); a.legend()
    if logy: a.set_yscale("log"); a.set_ylim(1e-4, 20)
ax[0].set_title("Impostor scores by capture format (all 13.3M pairs)"); ax[1].set_title("Same, log y: the tails")
fig.tight_layout(); fig.savefig(HERE / "plots" / "W2_1_impostor_by_format_overlay.png", dpi=130); plt.close(fig)

# ---- Figure 2: one panel per group, with genuine and the normal predicted by the body ------
fig, ax = plt.subplots(2, 3, figsize=(19, 9))
b2 = np.linspace(0, 2.0, 161)
for k, g in enumerate(names):
    r = rows[k]; v = imp[g]
    for j, logy in enumerate([False, True]):
        a = ax[j, k]
        a.hist(v, b2, density=True, color=cols[g], alpha=.65, label=f"impostor (n={len(v):,})")
        if len(gen[g]) > 50: a.hist(gen[g], b2, density=True, color="tab:green", alpha=.5, label=f"genuine (n={len(gen[g]):,})")
        xs = np.linspace(0, 2, 400); a.plot(xs, stats.norm.pdf(xs, np.median(v), r["robust_std"]), "k-", lw=1.5, label="normal fitted to the body")
        a.axvline(CUT, c="r", ls="--", label=f"cut {CUT:.3f}"); a.set_xlabel("score (radians)")
        if logy: a.set_yscale("log"); a.set_ylim(1e-4, 30)
        if j == 0: a.set_title(f"{names[g]}\nbelow cut: {r['frac_below_cut']:.2%} observed vs {r['normal_frac_below_cut']:.2%} normal ({r['tail_ratio']:.1f}x)")
        a.legend(fontsize=8)
    ax[0, k].set_ylabel("density"); ax[1, k].set_ylabel("density (log)")
fig.suptitle("Impostor left tail, by capture format: observed vs what the distribution's own body predicts", fontsize=14)
fig.tight_layout(); fig.savefig(HERE / "plots" / "W2_2_tail_shape_per_group.png", dpi=120); plt.close(fig)

# ---- Figure 3: QQ of the left tail vs normal fitted to the body -------------------------
fig, a = plt.subplots(figsize=(7.5, 6.5))
p = np.logspace(-5, np.log10(0.2), 60)
for g in names:
    v = imp[g]; a.plot(stats.norm.ppf(p, np.median(v), rows[g]["robust_std"]), np.quantile(v, p), lw=2, color=cols[g], label=names[g])
lo, hi = 1.1, 1.75; a.plot([lo, hi], [lo, hi], "k--", label="perfect normal tail"); a.axhline(CUT, c="r", ls=":", label="cut")
a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xlabel("quantile predicted by normal fitted to the body"); a.set_ylabel("observed quantile")
a.set_title("Left-tail QQ (lower 0.001% to 20%): curve below the line = heavier tail"); a.legend()
fig.tight_layout(); fig.savefig(HERE / "plots" / "W2_3_left_tail_qq.png", dpi=130)
