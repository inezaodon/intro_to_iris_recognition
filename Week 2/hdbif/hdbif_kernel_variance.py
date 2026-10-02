#!/usr/bin/env python3
"""
T1.3: Analyze impostor variance per kernel.

For each of 5 kernels:
  - Impostor score distribution (mean, std, skew, kurtosis)
  - Genuine score distribution
  - d' metric
  - Rank kernels by impostor variance

Input: hdbif/kernel_scores.npz (from T1.2)
Output: hdbif/kernel_variance_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

INPUT_FILE = Path("hdbif/kernel_scores.npz")
OUTPUT_FILE = Path("hdbif/kernel_variance_summary.csv")

def load_kernel_scores():
    """Load precomputed kernel scores from T1.2."""
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run T1.2 first.")
        return None

    data = np.load(INPUT_FILE)
    print(f"Loaded kernel_scores: {data.files}")
    return data

def compute_distribution_stats(scores, label=""):
    """Compute mean, std, skew, kurtosis for a score distribution."""
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'std_robust': stats.median_abs_deviation(scores) / 0.6745,  # MAD conversion
        'skew': stats.skew(scores),
        'kurtosis_excess': stats.kurtosis(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        '1pct_quantile': np.percentile(scores, 1),
        '50pct_quantile': np.percentile(scores, 50),
        '99pct_quantile': np.percentile(scores, 99),
    }

def compute_d_prime(genuine_scores, impostor_scores):
    """Compute d' (Cohen's d) between genuine and impostor distributions."""
    mean_g = np.mean(genuine_scores)
    mean_i = np.mean(impostor_scores)
    std_g = np.std(genuine_scores)
    std_i = np.std(impostor_scores)

    pooled_std = np.sqrt((std_g**2 + std_i**2) / 2)
    if pooled_std == 0:
        return 0

    return (mean_i - mean_g) / pooled_std

def main():
    print("Analyzing kernel variance (T1.3)...")

    data = load_kernel_scores()
    if data is None:
        return

    results = []

    # Iterate over 5 kernels
    for kernel_id in range(5):
        kernel_key_genuine = f"genuine_kernel_{kernel_id}"
        kernel_key_impostor = f"impostor_kernel_{kernel_id}"

        if kernel_key_genuine not in data or kernel_key_impostor not in data:
            print(f"Warning: kernel {kernel_id} not found in data")
            continue

        genuine_scores = data[kernel_key_genuine]
        impostor_scores = data[kernel_key_impostor]

        gen_stats = compute_distribution_stats(genuine_scores, f"genuine_k{kernel_id}")
        imp_stats = compute_distribution_stats(impostor_scores, f"impostor_k{kernel_id}")
        d_prime = compute_d_prime(genuine_scores, impostor_scores)

        results.append({
            'kernel_id': kernel_id,
            'genuine_n': len(genuine_scores),
            'impostor_n': len(impostor_scores),
            'genuine_mean': gen_stats['mean'],
            'genuine_std': gen_stats['std'],
            'impostor_mean': imp_stats['mean'],
            'impostor_std': imp_stats['std'],
            'impostor_std_robust': imp_stats['std_robust'],
            'impostor_skew': imp_stats['skew'],
            'impostor_kurtosis_excess': imp_stats['kurtosis_excess'],
            'd_prime': d_prime,
            'impostor_1pct': imp_stats['1pct_quantile'],
            'impostor_median': imp_stats['50pct_quantile'],
            'impostor_99pct': imp_stats['99pct_quantile'],
        })

    df = pd.DataFrame(results)

    # Rank by impostor variance (lower is better)
    df['impostor_variance'] = df['impostor_std'] ** 2
    df['variance_rank'] = df['impostor_variance'].rank()

    # Sort by variance rank
    df = df.sort_values('variance_rank')

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved summary to {OUTPUT_FILE}")
    print("\nKernel Variance Summary (ranked by impostor variance):")
    print(df[['kernel_id', 'impostor_std', 'impostor_std_robust', 'd_prime', 'variance_rank']])

    # Identify best kernel (lowest impostor variance)
    best_kernel = df.loc[df['variance_rank'].idxmin()]
    print(f"\n✓ Best kernel (lowest impostor variance): Kernel {int(best_kernel['kernel_id'])} (σ={best_kernel['impostor_std']:.4f})")

if __name__ == "__main__":
    main()
