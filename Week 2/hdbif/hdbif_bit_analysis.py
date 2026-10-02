#!/usr/bin/env python3
"""
T2.1: Bit-usage analysis per kernel.

For each kernel: identify which bit positions/clusters are never/rarely used in genuine pairs.
Cross-reference with wide-field, low-texture, and hub identities from Week 1.

Inputs:
  - hdbif/codes_by_kernel.pkl (from T1.1)
  - hdbif/kernel_variance_summary.csv (from T1.3)
  - ../Week 1/Task 1 results/manifest.csv

Outputs:
  - hdbif/kernel_bit_usage.csv
  - hdbif/bit_heatmaps_per_kernel.png (visualization)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

CODES_FILE = Path("hdbif/codes_by_kernel.pkl")
VARIANCE_FILE = Path("hdbif/kernel_variance_summary.csv")
MANIFEST_FILE = Path("../Week 1/Task 1 results/manifest.csv")
OUTPUT_CSV = Path("hdbif/kernel_bit_usage.csv")
OUTPUT_PLOT = Path("hdbif/bit_heatmaps_per_kernel.png")

def load_codes():
    """Load encoded codes for all kernels."""
    if not CODES_FILE.exists():
        print(f"Error: {CODES_FILE} not found. Run T1.1 first.")
        return None

    with open(CODES_FILE, 'rb') as f:
        codes = pickle.load(f)
    print(f"Loaded codes for {len(codes)} kernels")
    return codes

def load_variance_summary():
    """Load kernel variance ranking from T1.3."""
    if not VARIANCE_FILE.exists():
        print(f"Error: {VARIANCE_FILE} not found. Run T1.3 first.")
        return None

    df = pd.read_csv(VARIANCE_FILE)
    return df

def load_manifest():
    """Load image manifest with metadata (format, sensor, identity)."""
    if not MANIFEST_FILE.exists():
        print(f"Error: {MANIFEST_FILE} not found.")
        return None

    df = pd.read_csv(MANIFEST_FILE)
    return df

def analyze_bit_usage(codes_dict, manifest_df, kernel_id):
    """
    Analyze bit usage for a single kernel.

    Returns dict with:
      - bits_per_position: how many genuine pairs use each bit
      - bit_entropy: entropy of bit usage
      - unused_positions: bit positions rarely/never used
    """
    if kernel_id not in codes_dict or len(codes_dict[kernel_id]) == 0:
        return None

    all_codes = list(codes_dict[kernel_id].values())
    if len(all_codes) == 0:
        return None

    # Stack codes to get (n_images, n_bits)
    try:
        codes_array = np.array(all_codes)
        if codes_array.ndim < 2:
            return None

        n_bits = codes_array.shape[1]

        # Count bit usage per position (1s)
        bit_counts = np.sum(codes_array == 1, axis=0)
        bit_usage_pct = (bit_counts / len(all_codes)) * 100

        # Entropy per bit position
        p_1 = bit_counts / len(all_codes)
        p_0 = 1 - p_1
        entropy = -np.sum(np.where(p_1 > 0, p_1 * np.log2(p_1), 0) +
                         np.where(p_0 > 0, p_0 * np.log2(p_0), 0)) / n_bits

        # Unused positions (usage < 5%)
        unused = np.where(bit_usage_pct < 5)[0]

        return {
            'kernel_id': kernel_id,
            'n_images': len(all_codes),
            'n_bits': n_bits,
            'bit_usage_mean_pct': np.mean(bit_usage_pct),
            'bit_usage_std_pct': np.std(bit_usage_pct),
            'entropy_mean': entropy,
            'n_unused_positions': len(unused),
            'unused_positions': ','.join(map(str, unused[:20])),  # First 20
        }
    except Exception as e:
        print(f"Error processing kernel {kernel_id}: {e}")
        return None

def main():
    print("Analyzing kernel bit usage (T2.1)...")

    codes = load_codes()
    if codes is None:
        return

    variance_df = load_variance_summary()
    manifest = load_manifest()

    results = []

    # Analyze each kernel
    for kernel_id in range(5):
        analysis = analyze_bit_usage(codes, manifest, kernel_id)
        if analysis:
            results.append(analysis)
            print(f"Kernel {kernel_id}: {analysis['n_bits']} bits, "
                  f"{analysis['n_unused_positions']} rarely-used positions")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values('kernel_id')

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved bit usage analysis to {OUTPUT_CSV}")

    print("\nBit Usage Summary per Kernel:")
    print(df[['kernel_id', 'n_bits', 'bit_usage_mean_pct',
              'n_unused_positions', 'entropy_mean']])

    print(f"\n✓ Bit analysis complete")
    print("  Next: Generate bit heatmaps (TODO: visualization)")

if __name__ == "__main__":
    main()
