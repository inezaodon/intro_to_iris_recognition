#!/usr/bin/env python3
"""
T1.1: Full HDBIF pipeline over NDCVRL_002 dataset with all 5 independent filter kernels.

Reads manifest from Week 1, encodes all 5,169 images with each of 5 filter kernels,
caches codes and masks per kernel. No scoring yet (T1.2).

Output:
  - hdbif/codes_by_kernel.pkl: dict of {kernel_id: codes_array}
  - hdbif/masks_cache.pkl: cached masks for all images
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Paths
WEEK_1_RESULTS = Path("../Week 1/Task 1 results")
MANIFEST_PATH = WEEK_1_RESULTS / "manifest.csv"
EMBEDDINGS_PATH = WEEK_1_RESULTS / "embeddings.pkl"
HDBIF_REPO = Path("../General Project Files/OpenSourceIrisRecognition")
OUTPUT_DIR = Path("hdbif")

# HDBIF paths
HDBIF_PYTHON = HDBIF_REPO / "methods/HDBIF/Python"
sys.path.insert(0, str(HDBIF_PYTHON))

OUTPUT_DIR.mkdir(exist_ok=True)

def load_manifest():
    """Load image manifest from Week 1."""
    import pandas as pd
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(df)} images")
    return df

def load_embeddings():
    """Load cached ArcIris embeddings from Week 1."""
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_dict = pickle.load(f)
    print(f"Loaded embeddings: {len(embeddings_dict)} keys")
    return embeddings_dict

def initialize_hdbif():
    """Import and initialize HDBIF encoding pipeline."""
    try:
        from hdbif import SegmentAndCircApprox, CartToPol, ExtractCode
        print("Imported HDBIF modules")
        return SegmentAndCircApprox, CartToPol, ExtractCode
    except ImportError as e:
        print(f"Error importing HDBIF: {e}")
        print("Ensure HDBIF weights are in ../General Project Files/OpenSourceIrisRecognition/")
        sys.exit(1)

def encode_with_kernels(manifest, embeddings_dict, segment, cart_to_pol, extract_code):
    """
    Encode all images with each of 5 independent filter kernels.

    Returns:
        codes_by_kernel: {kernel_id: {image_id: code_array}}
        masks_cache: {image_id: mask_array}
    """
    codes_by_kernel = {i: {} for i in range(5)}
    masks_cache = {}

    # TODO: Implement kernel iteration and encoding
    # This is a stub; full implementation depends on HDBIF API
    print("Encoding images with HDBIF kernels (stub - update with real HDBIF API calls)")

    return codes_by_kernel, masks_cache

def main():
    print("Starting HDBIF full pipeline (T1.1)...")

    manifest = load_manifest()
    embeddings_dict = load_embeddings()
    segment, cart_to_pol, extract_code = initialize_hdbif()

    codes_by_kernel, masks_cache = encode_with_kernels(
        manifest, embeddings_dict, segment, cart_to_pol, extract_code
    )

    # Save outputs
    codes_path = OUTPUT_DIR / "codes_by_kernel.pkl"
    masks_path = OUTPUT_DIR / "masks_cache.pkl"

    with open(codes_path, "wb") as f:
        pickle.dump(codes_by_kernel, f)
    print(f"Saved codes to {codes_path}")

    with open(masks_path, "wb") as f:
        pickle.dump(masks_cache, f)
    print(f"Saved masks to {masks_path}")

if __name__ == "__main__":
    main()
