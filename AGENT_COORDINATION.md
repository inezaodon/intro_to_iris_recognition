# Multi-Agent Work Coordination
## HDBIF Histogram & Kernel Analysis (Week 2 Continuation)

**Status:** Phase 2 framework complete; ready for parallel work  
**Branch:** `hdbif-histogram-kernels` / `feature/hdbif-full-analysis`  
**Last sync:** 2026-10-02 11:35 UTC

---

## Summary for Other Agents

This document coordinates work across 4 agents. See **TASKS.md** for the full breakdown.

### What Claude Code (Agent 1) Has Done
✅ Task tracking setup (`TASKS.md`)  
✅ Per-kernel analysis script created (`Week 2/hdbif/run_hdbif_5kernels.py`)  
🔄 Ready to run per-kernel variance analysis on subset (479 images, ~80 subject-eyes)

### What Remains (Parallel Work)

| Phase | Task | Owner | Status |
|-------|------|-------|--------|
| **1A** | Histogram normalization: CLAHE | Agent 2 | 🔄 TODO |
| **1B** | Histogram normalization: Linear stretching | Agent 3 | 🔄 TODO |
| **1C** | Histogram normalization: Equalization | Agent 3 | 🔄 TODO |
| **1D** | Distribution comparison (histograms) | Agent 2 | 🔄 TODO |
| **2A** | Run per-kernel scoring (subset: 5 kernels) | Agent 1 | 🔄 In Progress |
| **2B** | Rank kernels by impostor variance | Agent 1 | 🔄 In Progress |
| **2C** | Bit-level analysis (which bits ignored) | Agent 4 | 🔄 TODO |

---

## Code Locations & Dependencies

### HDBIF Setup (Already Complete)
- **Weights:** `General Project Files/OpenSourceIrisRecognition/methods/HDBIF/Python/{models,filters_pt}/`
  - 2 model files (.pth): segmentation mask model, iris circle approximation model
  - 3 filter banks (288 filters .pt files each): independent ICA filter sets
  - Config fixed in W2-3; ready to use

### Scripts
- **Baseline (single kernel, 479-image subset):** `Week 2/hdbif/run_hdbif_subset.py` (W2-4, working)
- **New (5 kernels independently):** `Week 2/hdbif/run_hdbif_5kernels.py` (Agent 1, ready to run)
- **Histogram preprocessing (TBD):** `Week 2/hdbif/compare_histogram_norms.py` (Agent 2/3 to create)
- **Kernel variance ranking (TBD):** `Week 2/hdbif/analyze_kernel_variance.py` (Agent 1 to create)
- **Bit analysis (TBD):** `Week 2/hdbif/analyze_kernel_bits.py` (Agent 4 to create)

### Data & Outputs
- **Input:** `Week 1/Task 1 results/` (embeddings, manifest, cached results)
- **Output:** `Week 2/hdbif/` (score CSVs, plots, statistics)
- **Manifest:** Reused from Week 1 (`ndcvrl_manifest.csv`); subset same seed=0 as baseline

---

## Parallel Work Suggestions

### For Agent 2 (Histogram Normalization - Preprocessing)
1. Create `compare_histogram_norms.py`
2. Load images from manifest
3. Apply CLAHE, linear stretching, histogram equalization
4. Save preprocessed images to temp cache
5. Re-run a quick HDBIF encode/score on a small subset (10 images, ~45 pairs) with each normalization
6. Plot genuine/impostor distributions side-by-side (same style as Week 2 W2-2)
7. Commit findings with note on which method helps most (if any)

**Inputs needed:** Image filenames from manifest  
**Outputs:** Preprocessed image cache, comparison plot, summary CSV

### For Agent 3 (Histogram Normalization - Full Comparison)
1. Refine Agent 2's findings across all 5 kernels (use Agent 1's run_hdbif_5kernels.py with preprocessed images)
2. Build histograms of pixel values before/after normalization
3. Analyze impact on low-brightness images (sensor-specific or environmental correlation)
4. Document which images benefit most from each normalization

**Inputs:** Agent 2's preprocessed images, Agent 1's per-kernel scores  
**Outputs:** Full histogram analysis, per-kernel comparison under normalization

### For Agent 4 (Bit-Level Analysis)
1. Run Agent 1's per-kernel script to completion (generates 5 score CSVs + codes)
2. For each kernel's template codes, compute bit-wise frequency (how many times bit j is 1 across all images)
3. Identify "ignored" bits: high variance or always-0 positions
4. Visualize as heatmaps (bit position x kernel size x filter bank)
5. Cross-reference with Week 1 findings (F7: low-texture regions, F13: wide-field sensor patterns)

**Inputs:** Cached HDBIF codes (from Agent 1's run), template templates  
**Outputs:** Bit frequency tables, heatmap PNGs, ignored-bit summary

---

## Git Workflow for Multi-Agent

### Before Pushing
1. **Local test first** on subset (479 images max; runs in ~5-10 min per script)
2. **Check paths** match this repo structure (HDBIF_PY, WEEK1, WEEK2 constants)
3. **Output format** must be CSV for scores, PNG for plots (match Week 2 style)

### Committing
```bash
# Example commit messages:
git add -f Week\ 2/hdbif/compare_histogram_norms.py
git commit -m "Add histogram normalization comparison (CLAHE, linear stretching, equalization)"

git add -f Week\ 2/hdbif/plots/W2_kernel_distributions_*.png
git commit -m "Add per-kernel score distributions and variance ranking"

git add -f Week\ 2/hdbif/kernel_bit_analysis.csv
git commit -m "Add bit-level analysis: ignored bits per kernel"
```

### Pull Request
Target: `dev` (via `hdbif-histogram-kernels` or working branch)  
Include:  
- Link to TASKS.md tasks completed  
- Summary of findings (1-2 sentences per task)  
- Link to evidence plots/tables in Week 2/findings_log.md

---

## Notes & Gotchas

### HDBIF-Specific
- **Filter sizes:** 5x5, 7x7, 9x9, 11x11, 13x13 (5 independent "kernels")
- **Scoring scale:** Fractional Hamming distance [0, 0.5], where 0.5 = chance
  - NOT the same as ArcIris's arc-cos score [0, π]; only d' is directly comparable
- **Mask handling:** HDBIF's `matchCodesEfficient` returns `inf` if overlap too small; these are auto-filtered in the script
- **Rotation:** Tested with rotation_range=3 (±3 discrete shifts); baseline used same

### Performance
- **Subset (479 images):** ~5 min per kernel; 25 min total for all 5 kernels
- **Full dataset (5,169 images):** ~2-3 hours per kernel; ~12-15 hours total (not yet attempted)
- Run subset first to validate changes before full-scale runs

### Manifest & Data
- **Subject-eye format:** `{subject_id}_L` / `{subject_id}_R` (or just subject_id + eye column)
- **Subset seeding:** Same `seed=0` as W2-4 baseline to ensure reproducibility across agents
- **Image paths:** Relative to `Week 1/Task 1 results/`; script auto-resolves via manifest

---

## Communication & Blockers

### If You're Blocked
1. Check this file for data path / HDBIF setup details
2. Check TASKS.md for task dependencies
3. Ping in comments if:
   - Scripts fail on model loading (check GPU/CPU, torch version)
   - Manifest format differs from expected
   - Paths to Week 1 data are missing

### Handoff Points
- Agent 1 (per-kernel) → Agent 1 (variance ranking) [sequential, same agent]
- Agent 1 (variance ranking) → Agent 4 (bit analysis) [can overlap; needs Agent 1's codes]
- Agent 2 (CLAHE) → Agent 3 (full comparison) [sequential; Agent 2 validates method first]

---

## Success Criteria (End of Phase)

- [ ] 5 per-kernel score CSVs generated and committed
- [ ] Kernel variance ranking table with d' and impostor tail metrics
- [ ] Visualization: impostor score distribution per kernel (5 subplots)
- [ ] Bit analysis: heatmaps and summary of ignored bits per kernel
- [ ] Histogram normalization comparison showing impact on d' and tail (if any)
- [ ] Week 2/findings_log.md updated with W2-6 through W2-9 entries
- [ ] All scripts committed with clear docstrings
- [ ] PR merged to `dev`, ready for next phase (full-dataset scale-up)

---

**Questions?** Update this file or comment on TASKS.md.  
**Ready?** Follow git workflow above and commit with clear messages.
