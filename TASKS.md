# HDBIF Histogram & Kernel Analysis Tasks
## Week 2 Continuation - Multi-Agent Coordination

**Branch:** `hdbif-histogram-kernels`  
**Agents:** 4 total (including Claude Code)  
**Last updated:** 2026-10-02

---

## Overview
Following Week 2's completion of W2-1 through W2-5 (tail shape analysis, HDBIF setup, first plots), this task set focuses on:
1. Histogram normalization techniques (CLAHE, linear stretching, equalization)
2. Per-kernel HDBIF matching and analysis
3. Variance analysis to identify the most stable filter kernel
4. Bit-level inspection to understand which features each kernel ignores

---

## Tasks Checklist

### Phase 1: Histogram Normalization (Infrastructure)
- [ ] **HDBIF - Histogram normalization (CLAHE)**
  - Implement Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing
  - Compare impostor/genuine distributions with CLAHE vs. raw
  - Document impact on d' and tail metrics
  - *Assigned to:* [Agent?]

- [ ] **Linear histogram normalization (Stretching for low-brightness images)**
  - Stretch 0-255 range to utilize full dynamic range
  - Focus on low-brightness sub-populations (sensor-specific or environmental)
  - Compare before/after on tail composition
  - *Assigned to:* [Agent?]

- [ ] **Histogram equalization**
  - Global histogram equalization (fallback simpler method)
  - Compare with CLAHE for computational cost vs. gain
  - *Assigned to:* [Agent?]

- [ ] **Look at the distribution to see if you have a match**
  - Analyze empirical score distributions (genuine vs. impostor) under each normalization
  - Validate that preprocessing doesn't create artifacts (e.g., false mode at quantization boundary)
  - *Assigned to:* [Agent?]

### Phase 2: HDBIF Per-Kernel Analysis (Main Analysis)
- [ ] **Match using 5 filters independently**
  - Extend `run_hdbif_subset.py` to compute pairwise scores with each of the 5 independent filter kernel configurations
  - Save per-kernel score matrices
  - Expected output: 5x `hdbif_pairwise_scores_kernel_*.csv` files
  - *Assigned to:* Claude Code (this agent)
  - *Status:* Starting

- [ ] **Pick one kernel that has the smallest variance in the imposter distribution**
  - For each kernel's impostor score distribution, compute robust variance (IQR/1.349)
  - Rank kernels by variance (lower = more stable, less noisy)
  - Identify which kernel is most discriminative (smallest var = smallest noise)
  - Output: ranking table and per-kernel distribution plots
  - *Assigned to:* Claude Code (this agent)
  - *Status:* Starting

- [ ] **Look at the 5 kernels and see which bits are ignored**
  - For each kernel, analyze the template bit arrays (codes from `extractCode`)
  - Compute bit-wise frequency/importance across the dataset
  - Identify positions that are rarely non-zero or rarely discriminative
  - Cross-reference with Week 1 findings (F7: low-texture, F13: wide-field sensor patterns)
  - Visualize as heatmaps (bit position x kernel)
  - *Assigned to:* [Agent?]

---

## Data Dependencies

### Inputs
- **NDCVRL_002 Dataset:** `/Users/odon/Desktop/NDCVRL_002 Dataset/` (read permission configured)
- **ArcIris embeddings:** `Week 1/Task 1 results/embeddings.pkl` → `General Project Files/arciris_analysis/`
- **Manifest:** `Week 1/Task 1 results/manifest.csv` → `General Project Files/arciris_analysis/example_manifest.csv`
- **HDBIF weights:** `General Project Files/OpenSourceIrisRecognition/methods/HDBIF/Python/{models,filters_pt}/`
- **HDBIF subset run:** `Week 2/hdbif/hdbif_pairwise_scores.csv` (479 images, single kernel baseline)

### Outputs (to commit)
- `Week 2/hdbif/hdbif_pairwise_scores_kernel_*.csv` (per-kernel scores)
- `Week 2/hdbif/kernel_variance_comparison.csv` (variance rankings)
- `Week 2/hdbif/plots/W2_kernel_distributions_*.png` (impostor/genuine per kernel)
- `Week 2/hdbif/plots/W2_kernel_bit_analysis_*.png` (bit frequency heatmaps)
- `Week 2/hdbif/kernel_bit_statistics.csv` (bit-level summaries)
- `Week 2/findings_log.md` (updated with W2-6, W2-7, W2-8 entries)

---

## Scripts & Locations

| File | Purpose | Status |
| --- | --- | --- |
| `Week 2/hdbif/run_hdbif_subset.py` | Existing baseline (single kernel, 479-image subset) | ✅ Done (W2-4) |
| `Week 2/hdbif/run_hdbif_5kernels.py` | **[TO DO]** Compute all 5 kernels independently | 🔄 In Progress |
| `Week 2/hdbif/compare_histogram_norms.py` | **[TO DO]** CLAHE, linear stretch, equalization | 🔄 TODO |
| `Week 2/hdbif/analyze_kernel_variance.py` | **[TO DO]** Rank by impostor variance | 🔄 TODO |
| `Week 2/hdbif/analyze_kernel_bits.py` | **[TO DO]** Bit-level inspection | 🔄 TODO |
| `Week 2/hdbif/plot_hdbif_distributions.py` | Existing plotter (adapt for per-kernel) | ✅ Done, will reuse |

---

## Git Workflow

**Current branch:** `hdbif-histogram-kernels`

### Before committing:
1. Run tests on subset data (479 images)
2. Commit per-kernel scores and intermediate results
3. Check that new plots match visual style of Week 1 (green genuine, red impostor, d' in title)
4. Update `Week 2/findings_log.md` with findings (W2-6, etc.)

### Commit naming:
- `Add per-kernel HDBIF scoring (5 independent filters)`
- `Add kernel variance ranking and comparison`
- `Add kernel bit-level analysis and ignored-bit detection`
- `Add histogram normalization comparison (CLAHE, linear, equalization)`

### Pull request:
- Merge `hdbif-histogram-kernels` into `dev` once all tasks complete
- Link to Week 2 findings_log entries (W2-6 through W2-9 expected)
- Request review from the other 3 agents

---

## Next (After Completion)
1. **Full-dataset HDBIF:** Scale per-kernel analysis to all 5,169 images and 70M+ impostor pairs (computational cost: ~6-8 hours single-threaded)
2. **Mitigation design:** Use best-kernel variance ranking to propose inference-time gates (e.g., fallback to kernel-X when kernel-A variance is high)
3. **Cross-method:** Compare ArcIris vs. HDBIF under same histogram normalizations; verify CLAHE/linear-stretch impact on both
4. **Held-out test split:** Confirm findings on `NDCVRL_002 Dataset/test_train_split/`

---

## Notes for Other Agents
- This branch is safe to build on; core HDBIF pipeline is stable
- Histogram preprocessing is orthogonal to kernel analysis; can be developed in parallel
- Per-kernel scores are large (5 files × ~60MB each for full dataset) — use sparse matrices or HDF5 if needed
- Bit analysis requires access to raw template codes; ensure `extractCode` output is cached or streamed
- Questions? Update this file or comment on the PR

---

**In progress by:** Claude Code (hdbif-histogram-kernels branch, agent 1 of 3)
