# Task Implementation Plan

## 1. HDBIF Per-Kernel Analysis (W2-4 continuation)
- Load full NDCVRL_002 dataset with ArcIris manifest
- Run HDBIF with each of 5 independent filter kernels
- Compute impostor score variance per kernel → identify lowest-variance kernel
- Visualize bit-level analysis: which bits/positions each kernel ignores

## 2. Histogram Normalization Methods
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Linear Histogram Stretching (for low-brightness images)
- Standard Histogram Equalization
- Apply to iris images, compare genuine/impostor separation

## 3. Distribution Analysis & Visualization
- Plot histograms for each preprocessing method
- Compare d' (discriminability) across methods
- Identify best-performing preprocessing

## 4. Integration
- Combine best preprocessing with ArcIris and HDBIF
- Final comparison table: all methods + preprocessing combos
