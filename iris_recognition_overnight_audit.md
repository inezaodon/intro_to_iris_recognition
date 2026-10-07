# Iris recognition: overnight technical audit

October 6, 2026. Prepared for Odon's morning review.

## What was done

Cloned https://github.com/inezaodon/intro_to_iris_recognition and audited final-push main commit `80f0ef8e5d7f4a8f88fc69cdc379e8abbb3d8276` (11:36pm EDT). Rechecked remote main after numerical work. No source files in the repository were edited, no commit or push was made, and git working tree remained clean.

Main contains results, weights, filter banks and bytecode but **no Python source**. Recovered 26 historical Python scripts (3,113 lines) from existing `feature/hdbif-full-analysis` commit `10efd4e7cc44783333c6bafd62d51de7d52d27ab` (October 2) into a separate reference directory. Read all modules and documents, parsed all scripts, inspected current official ArcIris/HDBIF interfaces, and wrote a 7,000-plus-word implementation brief. Historical code was not treated as proven final runtime source.

Ran real cache-only experiments: restricted NumPy-only loading of cached vectors, exhaustive blockwise angular matching, metadata/label joins, tail stratification, train/test split integrity, stored predictor rank/error analysis, method complementarity, duplicate-vector and embedding spectrum checks, kernel archive inspection, gallery numeric consistency and targeted pure-function/synthetic regression checks. No model archive code was executed. No new segmentation/CLAHE/model inference or training was performed.

## Results that matter

### The cached extreme is real; its cause is not yet established

5,169 usable cached 512-dimensional vectors yield 70,876 genuine and **13,285,820 impostor** pairs. The full 1% impostor quantile is 1.406318 rad. At the old sampled cut 1.405748 rad, FMR is **0.980203%** (130,228 impostors accepted), FNMR **5.403804%**. Lower angular distance means closer matches.

The worst impostor is **241619 vs 437343**, 0.433921 rad (cosine 0.907324), recorded different persons `nd1S05555` / `nd1S05675`, both Left. Its two sensors are `nd1N00016` / `nd1N00006`, not the wide-sensor group. Pixels are missing, so this could be a capture/label/model problem; no cause was guessed. There are no exact duplicate embedding byte arrays.

The sampled Week 1 gallery's best impostor is only 1.214818 rad. A random 100k-pair sample misses rare severe outliers. Full-tail diagnosis is needed before changing the model.

Wide-sensor-pair FMR is 5.2490%, versus 0.8321% for neither-wide pairs, at the same historical cut. Mixed capture groups have 11.5831% genuine FNMR. Groups were recreated using recorded sensor IDs as proxies, not by reading missing original image dimensions. This confirms a useful group pattern but is not a resolution-causality claim. The most extreme pair is elsewhere.

### A concrete historical gallery bug can mislead interpretation

`Week 2/render_the_1_percent.py` filters scores to impostors, then picks image IDs using indices from the unfiltered upper triangle. A small synthetic example reproduces attaching a genuine pair's IDs to the best impostor score. The brief gives the exact correction and required regression test.

This does **not** establish the tracked galleries are corrupt: independently re-scored all 1,026 listed Week 1 gallery records and their ID-score values match within 0.000000533 rad. Their referenced JPG pixels cannot be verified. The defect matters for future use of that historical renderer.

### Fusion is promising, but the saved results are not a deployable claim

The 412,321-pair table has no duplicate unordered pairs, self-pairs or metadata label disagreement. Train/test share **zero images and zero persons**: 144 train persons / 36 test persons. Cross-partition impostor rows are separate.

The 249,571 saved test predictions numerically align in order with the table's test subset. But the file lacks pair IDs, selected-bank definition, trained-model files and fit/calibration source. The plan requires explicit pair-ID joins and validation-only selection/thresholds.

Descriptive AUC / FNMR at **test-selected** ~1% FMR:

| Existing output | AUC | FNMR |
|---|---:|---:|
| ArcIris | 0.989609 | 5.9287% |
| HDBIF best, bank unspecified | 0.970701 | 13.1266% |
| HDBIF fusion | 0.981409 | 11.7931% |
| ArcIris + HDBIF | 0.995752 | 4.7076% |
| Boosted all | 0.995536 | 4.4023% |

Because these thresholds were selected on test for this audit, these are **descriptive, not independently calibrated performance**. The learned non-linear boundary and its provenance remain unavailable. No new boundary was fitted tonight.

Only 68 impostors pass both individual test-selected cuts, but only 5,315/6,224 genuine pass both. Complementarity is useful; a strict AND rule sacrifices genuine recall. Train/validation-controlled linear vs RBF fusion is a better next experiment than blindly ANDing outputs.

### "Five kernels" does not describe the tracked inventory

Inspected **288** available filter archives: 3 families x 12 patch sizes x 8 bit depths. All contain finite coefficients and distinct numeric storage, with individual learned filters unit-L2 to within 8.6e-12. These are archive integrity results, not recognition-quality results.

Current official upstream default selects one eye-tracker **17x17 / 5-bit bank**, normalization off. Final-run log lists **all 12 patch sizes**, but does not specify family/depth. Older five-bank wrapper's proposed settings are not proof of what ran. The brief makes actual selected path/config/hash mandatory and explains banks versus individual filters, response shapes, overlap masks and ignored-bit analysis.

### Masks and images must share geometry

ArcIris baseline unwraps using circles but feeds an unmasked polar image to its encoder. HDBIF uses predicted segmentation and pairwise rotated mask intersection. Original-frame masks cannot simply be stretched or placed over polar images. Historical `occlusion.py` uses generic image transformation on masks, risking gray padding and interpolated labels. The plan specifies shared recorded pad/resize geometry, zero mask padding, nearest-neighbor masks, separately saved Cartesian/polar artifacts and visually inspected genuine/impostor overlays.

### CLAHE scores are still outstanding

Existing scripts preprocess images or measure brightness/contrast, but do not recompute requested recognition scores. One also has out-of-bounds subplot indices. The brief specifies real re-encoding of both ArcIris and HDBIF on exactly the same pair IDs, separate raw/CLAHE caches, a fixed-geometry primary experiment and end-to-end geometry secondary experiment. Missing pixels cannot be worked around by transforming distances or vectors.

## Tests and limits

- All 26 recovered scripts parsed successfully; parsing is not runtime correctness.
- Seven pure-function assertions passed for angular matching and identity labels. Unknown labels are currently classified as impostor by old pairwise code; change to invalid evaluation status.
- Synthetic upper-triangle mapping test reproduced the gallery bug. Histogram plotting index failure verified statically.
- All saved pair scores agree with vector recalculation within 0.000001633 rad.
- Full cached scores were computed in blocks. One overly memory-heavy attempt was killed; revised bounded indexing succeeded. Final numbers derive only from the successful run.
- Embedding geometry: effective rank ~142.03/512, top-ten eigenvalue energy ~14.70%, mean normalized vector norm 0.09191. These are diagnostics, not proof of embedding collapse.
- Generated and directly inspected actual-pixel diagnostic plots: full score histograms, dense-rank full-tail CDF, exploratory stored-score scatter and actual 17x17 kernel heatmap. No iris image or original-mask overlay was visually validated because no pixels were available.

Missing: raw images, dataset masks, source/config/weights for final runtime (including ArcIris encoder), per-bank score CSV explicitly excluded from Git, full-run code, trained separators and calibration provenance. Some may exist on Odon's computer. The plan begins by finding them without overwriting working changes.

The current typed request has **four** items. No matching fifth was found in main README or the old PR. Historical tasks contain variance and ignored-bit analysis, included under kernel work; the brief does not invent a fifth. The follow-up false-match/experiment request is covered in the plan.

## Recommended next steps

1. Recover actual local source and ignored per-bank scores; fix ID/score/gallery integrity before drawing conclusions from images.
2. Save and inspect per-image mask/geometry/polar artifacts, then render all requested pairs with explicit missing/failure records.
3. Compare raw versus CLAHE on the identical pair list and run subject-disjoint validation-controlled linear/RBF fusion.
4. Use tail hubs, measured quality, sensor/session strata, native-mask support and controlled resolution experiments to decide whether retraining is warranted.

Companion file: `iris_recognition_execution_brief.md` contains the full module audit, specific fixes, CLI/package/data contracts, experiments, test cases, acceptance criteria and scientific reporting rules. It is ready to paste into a coding session. It explicitly separates implemented work from proposed work.

## Sources

- Project: https://github.com/inezaodon/intro_to_iris_recognition
- Historical PR: https://github.com/inezaodon/intro_to_iris_recognition/pull/1
- Actual main CSV/vector/log/filter artifacts and recovered source at the exact revisions above, freshly checked and calculated.
- Official upstream interfaces/configs: https://github.com/CVRL/OpenSourceIrisRecognition
- HDBIF paper lead only (full-page fetch failed; no findings rely on it): https://arxiv.org/html/1807.05248v2
- CLAHE API: https://docs.opencv.org/4.12.0/d6/db6/classcv_1_1CLAHE.html
- Validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Calibration: https://scikit-learn.org/stable/modules/calibration.html
- RBF guidance: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

No missing-data result, independent deployment guarantee or new model improvement is claimed.
