#!/usr/bin/env python3
"""Distribution matching analysis: look at distributions to find genuine/impostor separability.

Analyzes whether distributions allow reliable matching, and identifies
score ranges where genuine and impostor overlap occurs.
"""

import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R_PILOT = HERE.parent / "Week 1" / "Task 1 pilot"
R_RESULTS = HERE.parent / "Week 1" / "Task 1 results"

# Load data
print("="*70)
print("DISTRIBUTION MATCHING ANALYSIS")
print("="*70)

# Load ArcIris scores from pilot (has full pairwise scores)
scores_data = {}
for row in csv.DictReader(open(R_PILOT / "pairwise_scores.csv")):
    label = row["label"]
    score = float(row["score"])
    if label not in scores_data:
        scores_data[label] = []
    scores_data[label].append(score)

genuine = np.array(scores_data.get("genuine", []))
impostor = np.array(scores_data.get("impostor", []))

print(f"\nArcIris scores loaded:")
print(f"  Genuine pairs: {len(genuine)}")
print(f"  Impostor pairs: {len(impostor)}")

# Load cut (both pilot and results have this)
with open(R_PILOT / "impostor_tail_summary.json") as f:
    tail_info = json.load(f)
    cut = tail_info["threshold"]

print(f"  Cut (1% impostor tail): {cut:.4f} rad")

# Compute key statistics
genuine_mean, genuine_std = genuine.mean(), genuine.std()
impostor_mean, impostor_std = impostor.mean(), impostor.std()

# Cohen's d (effect size)
cohens_d = (impostor_mean - genuine_mean) / np.sqrt((genuine_std**2 + impostor_std**2) / 2)

# d-prime (discriminability) - already computed in findings but let's verify
d_prime = abs(impostor_mean - genuine_mean) / np.sqrt(0.5 * (genuine_std**2 + impostor_std**2))

print(f"\nDistribution statistics:")
print(f"  Genuine: mean={genuine_mean:.4f}, std={genuine_std:.4f}")
print(f"  Impostor: mean={impostor_mean:.4f}, std={impostor_std:.4f}")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  d': {d_prime:.3f}")

# Overlap analysis: where do genuine and impostor distributions overlap?
print(f"\nOverlap analysis:")

# Find score where genuine > impostor distribution density
x_range = np.linspace(0, 2, 1000)
genuine_pdf = stats.norm.pdf(x_range, genuine_mean, genuine_std)
impostor_pdf = stats.norm.pdf(x_range, impostor_mean, impostor_std)

crossover_idx = np.argmin(np.abs(genuine_pdf - impostor_pdf))
crossover_score = x_range[crossover_idx]

print(f"  Distribution crossover: {crossover_score:.4f} rad")
print(f"  Cut vs crossover: cut={cut:.4f} {'>' if cut > crossover_score else '<'} crossover={crossover_score:.4f}")

# False match rate (FMR) and false non-match rate (FNMR) at various thresholds
thresholds = np.linspace(0, 2, 201)
fmr = []
fnmr = []
tpr = []  # true positive rate (1 - FNMR)
fpr = []  # false positive rate (FMR)

for threshold in thresholds:
    fmr_val = (impostor <= threshold).mean() * 100
    fnmr_val = (genuine > threshold).mean() * 100
    fmr.append(fmr_val)
    fnmr.append(fnmr_val)
    tpr.append(100 - fnmr_val)
    fpr.append(fmr_val)

# Find Equal Error Rate (EER)
eer_idx = np.argmin(np.abs(np.array(fmr) - np.array(fnmr)))
eer_threshold = thresholds[eer_idx]
eer_value = fmr[eer_idx]

print(f"\nError rates:")
print(f"  At cut={cut:.4f}: FMR={fmr[np.argmin(np.abs(thresholds-cut))]:.2f}%, FNMR={fnmr[np.argmin(np.abs(thresholds-cut))]:.2f}%")
print(f"  Equal Error Rate (EER): threshold={eer_threshold:.4f}, error={eer_value:.2f}%")

# Overlap percentage: how many scores fall in the overlap region?
overlap_region = (genuine > np.min(impostor)) & (genuine < np.max(impostor))
genuine_in_impostor_range = overlap_region.mean() * 100

print(f"  Genuine scores in impostor range: {genuine_in_impostor_range:.1f}%")

# Visualization
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histograms with cut line
ax[0, 0].hist(genuine, bins=60, alpha=0.6, density=True, label="Genuine", color="tab:green")
ax[0, 0].hist(impostor, bins=60, alpha=0.6, density=True, label="Impostor", color="tab:red")
ax[0, 0].axvline(cut, color="k", linestyle="--", linewidth=2, label=f"Cut={cut:.3f}")
ax[0, 0].axvline(eer_threshold, color="orange", linestyle="--", linewidth=2, label=f"EER={eer_threshold:.3f}")
ax[0, 0].set_xlabel("Score (radians)")
ax[0, 0].set_ylabel("Density")
ax[0, 0].set_title("Score distributions with cut and EER")
ax[0, 0].legend()

# 2. PDFs (smooth)
ax[0, 1].plot(x_range, genuine_pdf, linewidth=2, label="Genuine PDF", color="tab:green")
ax[0, 1].plot(x_range, impostor_pdf, linewidth=2, label="Impostor PDF", color="tab:red")
ax[0, 1].axvline(cut, color="k", linestyle="--", alpha=0.5, label=f"Cut")
ax[0, 1].axvline(crossover_score, color="purple", linestyle="--", alpha=0.5, label=f"Crossover")
ax[0, 1].set_xlabel("Score (radians)")
ax[0, 1].set_ylabel("Probability density")
ax[0, 1].set_title("Fitted normal distributions")
ax[0, 1].legend()

# 3. ROC curve (FPR vs TPR)
ax[1, 0].plot(fpr, tpr, linewidth=2, color="steelblue", label="ROC")
ax[1, 0].plot([0, 100], [0, 100], "k--", alpha=0.3, label="Chance")
ax[1, 0].scatter([fpr[eer_idx]], [tpr[eer_idx]], color="orange", s=100, zorder=5, label=f"EER ({eer_value:.1f}%)")
ax[1, 0].set_xlabel("False Positive Rate (%)")
ax[1, 0].set_ylabel("True Positive Rate (%)")
ax[1, 0].set_title("ROC Curve (Receiver Operating Characteristic)")
ax[1, 0].legend()
ax[1, 0].grid(alpha=0.3)

# 4. FMR and FNMR vs threshold
ax[1, 1].plot(thresholds, fmr, linewidth=2, label="FMR (false match rate)", color="tab:red")
ax[1, 1].plot(thresholds, fnmr, linewidth=2, label="FNMR (false non-match rate)", color="tab:green")
ax[1, 1].axvline(cut, color="k", linestyle="--", alpha=0.5, label=f"Current cut")
ax[1, 1].axvline(eer_threshold, color="orange", linestyle="--", alpha=0.5, label=f"EER")
ax[1, 1].scatter([cut], [fmr[np.argmin(np.abs(thresholds-cut))]], color="k", s=100, zorder=5)
ax[1, 1].set_xlabel("Threshold (radians)")
ax[1, 1].set_ylabel("Error rate (%)")
ax[1, 1].set_title("Error rates vs threshold")
ax[1, 1].legend()
ax[1, 1].grid(alpha=0.3)

fig.suptitle(f"Distribution Matching Analysis (ArcIris)\nd'={d_prime:.2f}  Cohen's d={cohens_d:.2f}  Overlap={genuine_in_impostor_range:.1f}%",
             fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "plots" / "W2_6_distribution_matching_analysis.png", dpi=130)
print(f"\nVisualization saved to plots/W2_6_distribution_matching_analysis.png")

# Save summary to CSV
summary = [{
    "metric": "genuine_mean",
    "value": genuine_mean,
}, {
    "metric": "genuine_std",
    "value": genuine_std,
}, {
    "metric": "impostor_mean",
    "value": impostor_mean,
}, {
    "metric": "impostor_std",
    "value": impostor_std,
}, {
    "metric": "cohens_d",
    "value": cohens_d,
}, {
    "metric": "d_prime",
    "value": d_prime,
}, {
    "metric": "crossover_score",
    "value": crossover_score,
}, {
    "metric": "eer_threshold",
    "value": eer_threshold,
}, {
    "metric": "eer_value_%",
    "value": eer_value,
}, {
    "metric": "genuine_in_impostor_range_%",
    "value": genuine_in_impostor_range,
}, {
    "metric": "current_cut",
    "value": cut,
}]

with open(HERE / "distribution_matching_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "value"])
    w.writeheader()
    w.writerows(summary)

print(f"Summary saved to distribution_matching_summary.csv")
print("\nDone.")
