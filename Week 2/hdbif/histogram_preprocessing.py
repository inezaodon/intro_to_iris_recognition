#!/usr/bin/env python3
"""
T3.1: Histogram normalization preprocessing comparison.

Implements 3 normalization techniques:
  - CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  - Linear Histogram Stretching (for low-brightness images)
  - Histogram Equalization (global)

Applies to a subset of NDCVRL_002 images and caches preprocessed versions.

Output:
  - hdbif/preprocessed_images_clahe/
  - hdbif/preprocessed_images_stretch/
  - hdbif/preprocessed_images_equalization/
"""

import numpy as np
import cv2
from pathlib import Path
import pickle
from skimage.exposure import equalize_hist

WEEK_1_RESULTS = Path("../Week 1/Task 1 results")
MANIFEST_PATH = WEEK_1_RESULTS / "manifest.csv"
OUTPUT_DIR = Path("hdbif")

# Create output directories
for method in ['clahe', 'stretch', 'equalization']:
    (OUTPUT_DIR / f"preprocessed_images_{method}").mkdir(exist_ok=True, parents=True)

def load_manifest():
    """Load image manifest from Week 1."""
    import pandas as pd
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(df)} images")
    return df

def load_image(image_path):
    """Load image in grayscale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0

def clahe_normalize(image):
    """Apply Contrast-Limited Adaptive Histogram Equalization."""
    if image is None:
        return None
    # Convert to 0-255 range for OpenCV
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(img_uint8).astype(np.float32) / 255.0
    return normalized

def linear_stretch_normalize(image):
    """Linear histogram stretching for low-brightness images."""
    if image is None:
        return None
    # Stretch from [min, max] to [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-6:
        return image  # Avoid division by zero
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def histogram_equalization_normalize(image):
    """Global histogram equalization."""
    if image is None:
        return None
    img_uint8 = (image * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_uint8).astype(np.float32) / 255.0
    return equalized

def process_images(manifest, sample_size=500):
    """
    Apply all 3 normalization methods to a subset of images.

    Args:
        manifest: pandas DataFrame with image paths
        sample_size: number of images to process
    """
    print(f"Processing {min(sample_size, len(manifest))} images...")

    methods = {
        'clahe': clahe_normalize,
        'stretch': linear_stretch_normalize,
        'equalization': histogram_equalization_normalize,
    }

    # Sample images (deterministic seed for reproducibility)
    np.random.seed(42)
    indices = np.random.choice(len(manifest), min(sample_size, len(manifest)), replace=False)

    preprocessed = {method: {} for method in methods}

    for idx, i in enumerate(indices):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(indices)}] Processing image {i}...")

        row = manifest.iloc[i]
        image_id = row.get('image_id') or row.get('id') or str(i)
        image_path = row.get('image_path') or row.get('path')

        if image_path is None:
            print(f"    Warning: No image path for row {i}")
            continue

        image = load_image(image_path)
        if image is None:
            print(f"    Warning: Failed to load {image_path}")
            continue

        # Apply each normalization method
        for method_name, method_fn in methods.items():
            normalized = method_fn(image)
            if normalized is not None:
                preprocessed[method_name][image_id] = normalized

    return preprocessed

def save_preprocessed(preprocessed):
    """Save preprocessed images to pickle files."""
    for method_name, images_dict in preprocessed.items():
        output_path = OUTPUT_DIR / f"preprocessed_{method_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(images_dict, f)
        print(f"Saved {len(images_dict)} {method_name} images to {output_path}")

def main():
    print("Preprocessing images with histogram normalization (T3.1)...")

    manifest = load_manifest()
    preprocessed = process_images(manifest, sample_size=500)
    save_preprocessed(preprocessed)

    print("\n✓ Preprocessing complete")
    print("  CLAHE images ready for re-encoding")
    print("  Linear stretch images ready for re-encoding")
    print("  Histogram equalization images ready for re-encoding")

if __name__ == "__main__":
    main()
