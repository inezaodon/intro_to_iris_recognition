# Week 2 HDBIF Analysis — Team Coordination Guide

**Status:** Ready for team work  
**Branch:** `feature/hdbif-full-analysis`  
**Commit:** 0e18a0d (infrastructure + task skeletons)  

## Quick Start

1. **Fetch and checkout the feature branch:**
   ```bash
   git fetch origin
   git checkout feature/hdbif-full-analysis
   ```

2. **Read the task plan:**
   ```bash
   cat Week\ 2/TASK_PLAN.md
   ```

3. **Pick your task** (T1.1-T4.2) and update TASK_PLAN.md with your assignment

## Team Assignments

### Recommended (can adjust):
| Agent | Task Range | Focus |
| --- | --- | --- |
| **Agent 1** | T1.1–T1.3 | Core HDBIF pipeline (encoding, scoring, variance) |
| **Agent 2** | T1.4–T2.2 | Visualization & bit analysis |
| **Agent 3** | T3.1–T3.2 | Histogram normalization comparison |
| **Agent 4 / User** | T4.1–T4.2 | Integration & findings |

## Git Workflow

### Per-Task Commits
Each completed task gets a single commit:
```bash
git checkout feature/hdbif-full-analysis
# Work on T1.1
git commit -m "[Week2-T1.1] Build full HDBIF pipeline over NDCVRL_002"
# Work on T1.2
git commit -m "[Week2-T1.2] Score all pairs per kernel (genuine + impostor)"
```

### Data Sharing
Intermediate outputs (`.npz`, `.pkl`, `.csv`) are cached in `Week 2/hdbif/` and committed:
- `codes_by_kernel.pkl` (T1.1 → T1.2, T2.1)
- `kernel_scores.npz` (T1.2 → T1.3)
- `kernel_variance_summary.csv` (T1.3 → visualization tasks)

Each task reads from and writes to these shared files.

### PR to `dev`
When all tasks are complete:
```bash
git push origin feature/hdbif-full-analysis
gh pr create --title "Week 2: Full HDBIF analysis + histogram normalization" \
  --body "Completes W2 plan: kernel variance, bit analysis, histogram methods comparison"
```

## Critical Paths

**Path A (Core):** T1.1 → T1.2 → T1.3 (must be sequential)  
**Path B (Visual):** T1.4, T2.1 (depends on T1.3 for variance ranking)  
**Path C (Preprocessing):** T3.1 → T3.2 (independent, can run in parallel)  
**Path D (Integration):** T4.1, T4.2 (merge results after others complete)

## Resources & Paths

### Input Data (Read-Only)
- **Manifest:** `Week 1/Task 1 results/manifest.csv` (5,169 images)
- **Cached embeddings:** `Week 1/Task 1 results/embeddings.pkl` (ArcIris, for reference)
- **HDBIF weights:** `General Project Files/OpenSourceIrisRecognition/methods/HDBIF/Python/`

### Output Structure
```
Week 2/hdbif/
├── codes_by_kernel.pkl          ← T1.1 output
├── kernel_scores.npz            ← T1.2 output
├── kernel_variance_summary.csv  ← T1.3 output (reference in visualizations)
├── kernel_bit_usage.csv         ← T2.1 output
├── histogram_methods_comparison.csv  ← T3.2 output
│
└── [script files for each task]
```

## Gotchas & Notes

1. **HDBIF API:** Review `../General Project Files/OpenSourceIrisRecognition/methods/HDBIF/` README before implementing T1.1. The filter-kernel parameter syntax may differ from the placeholder in `hdbif_full_pipeline.py`.

2. **Memory:** Full NDCVRL_002 encoding (5,169 images × 5 kernels) will be ~500MB–1GB cached. Ensure sufficient RAM.

3. **Reproducibility:** All sampling uses `np.random.seed(42)` or explicit seeds. Document any changes.

4. **Manifest columns:** Week 1's manifest may have columns like `image_id`, `image_path`, `subject_id`, `capture_format`, `sensor`. Check with:
   ```bash
   python3 -c "import pandas as pd; print(pd.read_csv('Week 1/Task 1 results/manifest.csv').head())"
   ```

5. **Cross-references:** 
   - Week 1 findings (F1-F13) are in `Week 1/findings_log.md`
   - Week 2 existing findings (W2-1 through W2-5) are in `findings_log.md`
   - This work adds W2-6 (HDBIF full analysis) and W2-7 (histogram methods)

## Questions?

- **Architecture:** Check `Week 2/README.md` and `TASK_PLAN.md`
- **Findings context:** See `findings_log.md` and cross-references to Week 1
- **API usage:** Check existing working scripts (`run_hdbif_subset.py`) as templates

---

**Ready to start?** Pick a task from TASK_PLAN.md, mark it as in progress, and begin!
