#!/usr/bin/env python3
"""Histogram normalization methods for iris images: CLAHE, linear stretching, equalization.

Compares preprocessing methods on ArcIris genuine/impostor separation.
"""

import csv
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
R_PILOT = HERE.parent / "Week 1" / "Task 1 pilot"
R_RESULTS = HERE.parent / "Week 1" / "Task 1 results"
DATASET = HERE.parent / "General Project Files" / "NDCVRL_002 Dataset" / "raw"

# Load manifest (use pilot manifest which has the full dataset mapping)
man = {r["image_id"]: r for r in csv.DictReader(open(R_PILOT / "manifest.csv"))}
ids = list(man.keys())

def get_path(image_id):
    m = man[image_id]
    subject = m["subject_id"]
    eye_short = m["eye"]
    eye_full = "Left" if eye_short == "L" else "Right"
    return DATASET / subject / eye_full / f"{image_id}.tiff"

def load_gray(image_id):
    """Load grayscale iris image."""
    try:
        img = Image.open(str(get_path(image_id))).convert("RGB")
        return np.array(img.split()[0])  # R channel as grayscale
    except:
        return None

# Define preprocessing methods
def no_preprocessing(img):
    """Baseline: no processing."""
    return img.astype(np.uint8)

def clahe(img, clip_limit=2.0, tile_size=8):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if img is None:
        return None
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe_obj.apply(img.astype(np.uint8))

def histogram_stretching(img):
    """Linear histogram stretching for low-brightness images."""
    if img is None:
        return None
    img = img.astype(np.float32)
    # Stretch to full range [0, 255]
    vmin, vmax = np.percentile(img, [1, 99])  # Use 1%-99% percentiles to avoid outliers
    if vmax > vmin:
        img = 255 * (img - vmin) / (vmax - vmin)
    return np.clip(img, 0, 255).astype(np.uint8)

def histogram_equalization(img):
    """Standard histogram equalization."""
    if img is None:
        return None
    return cv2.equalizeHist(img.astype(np.uint8))

methods = {
    "no preprocessing": no_preprocessing,
    "CLAHE": clahe,
    "linear stretching": histogram_stretching,
    "histogram equalization": histogram_equalization,
}

print("="*70)
print("HISTOGRAM NORMALIZATION ANALYSIS")
print("="*70)

# Load embeddings for comparison baseline (use pilot which has embeddings)
import pickle
e = pickle.load(open(R_PILOT / "embeddings.pkl", "rb"))
X_baseline = np.array([np.asarray(e[k], np.float64) for k in ids])
X_baseline /= np.linalg.norm(X_baseline, axis=1, keepdims=True)

# For each preprocessing method, compute contrast/brightness stats
print("\nAnalyzing preprocessing effects on iris images...")
results = []

for method_name, method_fn in methods.items():
    print(f"\n{method_name}:")

    # Load a sample of images and compute statistics
    sample_ids = ids[:100]  # Sample for efficiency
    brightness = []
    contrast = []

    for img_id in sample_ids:
        img = load_gray(img_id)
        if img is None:
            continue

        processed = method_fn(img)

        # Measure brightness and contrast
        brightness.append(np.mean(processed))
        contrast.append(np.std(processed))

    if brightness:
        brightness = np.array(brightness)
        contrast = np.array(contrast)
        results.append({
            "method": method_name,
            "mean_brightness": brightness.mean(),
            "std_brightness": brightness.std(),
            "mean_contrast": contrast.mean(),
            "std_contrast": contrast.std(),
        })
        print(f"  Mean brightness: {brightness.mean():.1f}")
        print(f"  Mean contrast (std): {contrast.mean():.1f}")

# Write results
if results:
    with open(HERE / "histogram_preprocessing_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

# Visualize preprocessing effects on sample images
print("\nGenerating visualization of preprocessing effects...")
fig, ax = plt.subplots(5, 4, figsize=(16, 18))

sample_id = ids[10]  # Use one example image
img_orig = load_gray(sample_id)

for col, (method_name, method_fn) in enumerate(methods.items()):
    img_proc = method_fn(img_orig)

    # Original image + histogram
    if col == 0:
        ax[0, col].imshow(img_orig, cmap="gray")
        ax[0, col].set_title(f"Original\n(sample ID {sample_id})")
        ax[0, col].axis("off")

        ax[1, col].hist(img_orig.flatten(), bins=50, alpha=0.7, color="gray")
        ax[1, col].set_title("Original histogram")
        ax[1, col].set_ylim([0, 1000])

    # Processed image
    ax[2 + col, col].imshow(img_proc, cmap="gray")
    ax[2 + col, col].set_title(f"{method_name}\n(processed)")
    ax[2 + col, col].axis("off")

    # Processed histogram
    ax[3 + col, col].hist(img_proc.flatten(), bins=50, alpha=0.7, color="steelblue")
    ax[3 + col, col].set_title(f"{method_name}\nhistogram")
    ax[3 + col, col].set_ylim([0, 1000])

fig.suptitle("Histogram Normalization Methods: Visual Comparison", fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "plots" / "W2_6_histogram_preprocessing_effects.png", dpi=130)
print("Saved visualization to plots/W2_6_histogram_preprocessing_effects.png")

# Plot brightness and contrast comparison
if results:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    methods_list = [r["method"] for r in results]
    brightness = [r["mean_brightness"] for r in results]
    contrast = [r["mean_contrast"] for r in results]

    ax[0].bar(range(len(methods_list)), brightness, color="coral", alpha=0.7)
    ax[0].set_xticks(range(len(methods_list)))
    ax[0].set_xticklabels(methods_list, rotation=45, ha="right")
    ax[0].set_ylabel("Mean brightness")
    ax[0].set_title("Mean brightness by preprocessing method")

    ax[1].bar(range(len(methods_list)), contrast, color="steelblue", alpha=0.7)
    ax[1].set_xticks(range(len(methods_list)))
    ax[1].set_xticklabels(methods_list, rotation=45, ha="right")
    ax[1].set_ylabel("Mean contrast (std dev)")
    ax[1].set_title("Mean contrast by preprocessing method")

    fig.tight_layout()
    fig.savefig(HERE / "plots" / "W2_6_brightness_contrast_comparison.png", dpi=130)
    print("Saved comparison plot to plots/W2_6_brightness_contrast_comparison.png")

print("\nDone. Results saved to:")
print(f"  - histogram_preprocessing_stats.csv")
print(f"  - plots/W2_6_histogram_preprocessing_effects.png")
print(f"  - plots/W2_6_brightness_contrast_comparison.png")
