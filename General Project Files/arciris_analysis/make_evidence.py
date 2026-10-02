#!/usr/bin/env python3
"""Regenerate the evidence figures cited in `Task 1 results/findings_log.md`."""
import csv
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

WEEK1 = Path(__file__).resolve().parent.parent
R = WEEK1 / "Task 1 results"; EV = R / "evidence"; EV.mkdir(exist_ok=True)
z = np.load(R / "polar_masks.npz")
occ = dict(zip(z["ids"], 1 - z["visible"])); masks = dict(zip(z["ids"], z["masks"])); circ = dict(zip(z["ids"], z["circles"]))
man = {r["image_id"]: r for r in csv.DictReader(open(R / "manifest.csv"))}
CUT = float(__import__("json").load(open(R / "impostor_tail_summary.json"))["threshold"])

# --- E2: occlusion vs score -------------------------------------------------
S = {"genuine": [], "impostor": []}
for r in csv.DictReader(open(R / "pairwise_scores.csv")):
    a, b = r["image_id_a"], r["image_id_b"]
    if a in occ and b in occ: S[r["label"]].append((float(r["score"]), max(occ[a], occ[b])))
tail = [r["image_id"] for r in csv.DictReader(open(R / "impostor_tail_images.csv")) if r["image_id"] in occ]
base = np.random.RandomState(0).choice(list(occ), len(tail), replace=False)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
bins = np.linspace(0, 1, 40)
ax[0].hist([occ[i] for i in base], bins, alpha=.6, density=True, label="random images", color="gray")
ax[0].hist([occ[i] for i in tail], bins, alpha=.6, density=True, label=f"impostor-tail images (n={len(tail)})", color="tab:red")
ax[0].set_xlabel("fraction of ArcIris iris ring that is occluded"); ax[0].set_ylabel("density"); ax[0].legend()
ax[0].set_title("Tail images are only slightly more occluded\n(mean 0.30 vs 0.28, p=0.027)")
for k, (lab, col) in enumerate([("genuine", "tab:green"), ("impostor", "tab:red")], start=1):
    v = np.array(S[lab]); q = np.quantile(v[:, 1], [0, .25, .5, .75, 1]); xs, ys, ns = [], [], []
    for lo, hi in zip(q[:-1], q[1:]):
        s = v[(v[:, 1] >= lo) & (v[:, 1] <= hi), 0]
        xs.append(f"{lo:.2f}-{hi:.2f}"); ys.append(np.mean(s < CUT) * 100)
    ax[k].bar(xs, ys, color=col, alpha=.75)
    for x, y in zip(xs, ys): ax[k].text(x, y, f"{y:.1f}%", ha="center", va="bottom")
    ax[k].set_xlabel("occlusion of the more-occluded image (quartiles)")
    ax[k].set_ylabel(f"% of {lab} pairs scoring below the cut ({CUT:.3f})")
    ax[k].set_title("Genuine: heavy occlusion loses true matches" if lab == "genuine" else "Impostor: occlusion barely changes false matches")
fig.tight_layout(); fig.savefig(EV / "E2_occlusion_vs_score.png", dpi=130); plt.close(fig)

# --- E3: masked vs unmasked ------------------------------------------------
m = np.load(R / "masked_rescore_subset.npz")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for a, name in zip(ax, ["unmasked", "masked"]):
    g, i = m[f"{name}_g"], m[f"{name}_i"]
    dp = abs(g.mean() - i.mean()) / np.sqrt(.5 * (g.var(ddof=1) + i.var(ddof=1)))
    bb = np.linspace(0, 2, 80)
    a.hist(g, bb, density=True, alpha=.6, color="tab:green", label="genuine"); a.hist(i, bb, density=True, alpha=.6, color="tab:red", label="impostor")
    a.set_title(f"{name} polar input  (d'={dp:.2f})"); a.set_xlabel("score (lower = more similar)"); a.legend()
fig.suptitle("Blanking occluded pixels (no retraining) makes separation worse: subset of 900 images"); fig.tight_layout()
fig.savefig(EV / "E3_masked_vs_unmasked.png", dpi=130); plt.close(fig)

# --- E4: what the network sees (low vs high occlusion) ---------------------
d = np.load(EV / "_e4_data.npz"); pick = list(d["ids"])
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for row, i in enumerate(pick):
    im, polar, c = d[f"img{row}"], d[f"polar{row}"], circ[i]
    ax[row, 0].imshow(im, cmap="gray"); ax[row, 0].set_title(f"{i}  occlusion={occ[i]:.0%}")
    for (x, y, r), col in [(c[:3], "cyan"), (c[3:], "yellow")]:
        ax[row, 0].add_patch(plt.Circle((x, y), r, fill=False, color=col, lw=1.5))
    ax[row, 1].imshow(polar, cmap="gray", aspect="auto"); ax[row, 1].set_title("polar unwrap fed to the network (no mask)")
    ov = np.stack([polar] * 3, -1).astype(float) / 255; ov[~masks[i]] = ov[~masks[i]] * .4 + np.array([.6, 0, 0])
    ax[row, 2].imshow(ov, aspect="auto"); ax[row, 2].set_title("red = not visible iris (lids, lashes, reflections)")
    for a in ax[row]: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(EV / "E4_occlusion_examples.png", dpi=110); plt.close(fig)
print("wrote", sorted(p.name for p in EV.iterdir()))
