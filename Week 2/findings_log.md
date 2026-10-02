# Week 2 findings log (running)

Same conventions as Week 1: numbers, why it matters, evidence image, how to reproduce. No retraining; cached ArcIris embeddings from `../Week 1/Task 1 results/embeddings.pkl` (5,169 images). Cut = 1.4057 rad (Week 1, 1% impostor cut on the sampled set).

## W2-1. Impostor tail shape by capture format (all 13,285,820 impostor pairs)

Method (`tail_shape_by_group.py`): every impostor pair is grouped by the image formats of its two images: **640x480 only** (9,021,691 pairs), **mixed** (3,855,462) and **960x640 only** (408,667; the two wide-field sensors `nd1N00059` / `nd1N00073`). For each group: the score distribution, skew / excess kurtosis, and a **normal fitted to the group's own body** (median and IQR/1.349, so the tail does not influence the fit) to see how much heavier than a bell curve the left tail is.

| Group | Mean | Std (robust) | Skew | Excess kurtosis | Below cut, observed | Below cut, normal from body | Heavier by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 only | 1.567 | 0.066 (0.065) | -0.08 | 0.08 | 0.832% | 0.661% | 1.3x |
| mixed formats | 1.567 | 0.066 (0.065) | -0.08 | 0.06 | 0.874% | 0.661% | 1.3x |
| 960x640 only | **1.528** | **0.074** (0.073) | -0.17 | 0.08 | 5.249% | 4.530% | 1.2x |

Left-tail quantiles (observed vs normal from body): 640x480 1% quantile 1.411 vs 1.416, 0.1% 1.351 vs 1.366, 0.01% 1.295 vs 1.325. 960x640: 1% 1.347 vs 1.359, 0.1% 1.281 vs 1.303, 0.01% 1.219 vs 1.257.

What this shows:
- **Each group is very close to a bell curve** (excess kurtosis about 0.08, small negative skew). The tail is only 1.2-1.3x heavier than the group's own normal predicts, and only far out (0.1% and below): the observed quantiles sit 0.01-0.04 rad **below** the normal's.
- **640x480 and mixed-format impostor distributions are indistinguishable** (same mean 1.567, same std 0.066; curves overlap exactly). The format effect only appears when **both** images are wide-field.
- **Wide-field pairs are shifted left by 0.039 rad and slightly wider** (mean 1.528 vs 1.567; std 0.074 vs 0.066). That shift, not a fat tail, is why 5.25% of them fall under a cut chosen at the 1% point of the whole population.
- **Where the 130,228 tail hits come from** (approximate decomposition against the 640x480 group's normal): about **67% (87.8k)** is just the left side of a normal impostor distribution at a cut 2.4 sigma below the mean (0.66% of every pair); about **12% (15.8k)** is the wide-field shift; about **20% (26.6k)** is tail heaviness beyond each group's own normal. (Sum 130.2k.)
- Interpretation for the "long tail": most of it is the *natural spread* of impostor scores (std about 0.066 rad); one identifiable population (wide-field pairs) shifts and widens it; the genuinely "extra-long" part beyond a bell curve is a minority (about 20% of tail hits, far tail).
- Caveats: "normal from body" is one choice of null (median/IQR fit); the decomposition assumes the 640x480 body applies to all groups; sensor and format are confounded (Week 1 F13); the 960x640 group is small (408,667 pairs, 133 identity-eyes) and its own hub structure is not modelled; scores of the same image appear in many pairs, so pairs are not independent.

Side finding (genuine pairs by group, same script): genuine median score is **0.692** for 640x480 pairs (49,979 pairs) but **1.053** for mixed-format pairs (16,878) and **0.995** for 960x640 pairs (4,019). Cross-format and wide-field genuine pairs sit much closer to the cut, so those captures also cost false non-matches (right panel of the figure).

Evidence (analytic views, secondary; moved to `plots/extra/`):
- `plots/extra/W2_1_impostor_by_format_overlay.png` (three groups overlaid, linear and log y; the mixed and 640x480 curves lie on top of each other, wide-field is shifted left).
- `plots/extra/W2_2_tail_shape_per_group.png` (one panel per group with genuine distribution, normal fitted to the body, and the cut; top linear, bottom log).
- `plots/extra/W2_3_left_tail_qq.png` (left-tail QQ vs a normal fitted to each body).
- Data: `tail_shape_by_group.csv`, `group_scores.npz` (all pair scores by group).

## W2-2. Genuine vs impostor distributions per group (same style as Week 1 `score_distributions.png`)

The W2-1 figures were hard to read (overlays, fitted curves, log axes), so the main plots now match Week 1's `score_distributions.png` exactly: genuine (green) vs impostor (red) histograms, x = score, y = density, dashed line = the 1% impostor tail cut (1.406, same in every panel), one panel per capture-format group. Script: `score_distributions_by_group.py` (reads `group_scores.npz`).

| Group | Genuine n / mean | Impostor n / mean | d' | Impostors under cut | Genuine over cut |
| --- | --- | --- | --- | --- | --- |
| All pairs (= Week 1) | 70,876 / 0.837 | 13,285,820 / 1.566 | 3.42 | 0.98% | 5.4% |
| 640x480 only | 49,979 / 0.745 | 9,021,691 / 1.567 | **4.40** | 0.83% | **2.9%** |
| Mixed formats | 16,878 / 1.063 | 3,855,462 / 1.567 | 2.66 | 0.87% | **11.6%** |
| 960x640 only | 4,019 / 1.019 | 408,667 / 1.528 | 2.50 | **5.25%** | 11.1% |

What the shapes say:
- **640x480 only (clean case):** the two humps are well separated, d' = 4.40. The headline d' = 3.42 is a mixture and is pulled down by the other two groups.
- **Mixed formats:** the impostor hump is unchanged (same place, same width as 640x480), but the **genuine hump moves right** (mean 0.745 -> 1.063, over the cut 11.6%). Same eye photographed in two formats matches poorly: an issue for genuine pairs, not for the impostor tail.
- **960x640 only:** **both** humps move: the impostor hump shifts **left** (1.567 -> 1.528, toward the cut: this is where the impostor-tail excess comes from) and the genuine hump also sits right (1.019). Least separation, d' = 2.50.
- Why it matters: format explains two different things at once: (1) the wide-field impostor shift that feeds the impostor tail; (2) a genuine-side loss whenever a wide-field image is involved.
- Caveat: same as W2-1 (sensor and format are confounded; pairs share images).
- Evidence: `plots/W2_dist_all_groups_side_by_side.png` (2x2, shared axes, best single view), plus one file each: `plots/W2_dist_all.png`, `W2_dist_640x480_only.png`, `W2_dist_mixed_formats.png`, `W2_dist_960x640_only.png`.

## W2-3. HDBIF set up as a second method (mask-aware, for comparison)

Everything from this session (2026-09-27). HDBIF ("Human-Driven Binary Image Features") is a separate method in the same repo (`OpenSourceIrisRecognition/methods/HDBIF`), unrelated to ArcIris. It matters here because, unlike ArcIris, it is **mask-aware**: it matches only the overlapping unoccluded iris region between two templates using fractional Hamming distance, with Daugman's bit-count score normalization, and it exposes N independently-swappable ICA filter kernels (5-12, at 12 patch sizes 5x5-39x39) with per-kernel bit codes.

- **Weights obtained (65MB models + 288 filter files), not via automated download.** An anonymous `gdown` script pulled most of the small filter files but hit Google Drive's per-file "too many downloads" quota partway through (about 90 rapid small-file requests in a burst looks like abuse to Drive, and this then blocked even previously-successful files and the 2 large model files). A same-session attempt to bulk-download via the browser pane's own UI ("Download" on both folders) reported success but silently failed (`Cello status was: 3` in the browser's own error log; no file ever reached disk) — the browser pane appears not to expose real file downloads to this environment. The user then downloaded the shared folder manually through their own browser (not subject to the anonymous quota) and dropped it in `~/Desktop/intro_to_iris_recognition/`; all 288 filter files (`filters_pt/`) and both model files (`models/`, `nestedsharedatrousresunet-...pth` 19.7MB + `resnet18-...pth` 42.7MB) were then moved into `methods/HDBIF/Python/{models,filters_pt}`.
- **Config fix:** `cfg.yaml`'s `circle_model_path` named a checkpoint (`resnet18-1543-0.047488-maskIoU-0.934494.pth`) that is not the file the shared folder actually serves (`resnet18-027-0.008222-maskIoU-0.967159.pth`, a different training epoch/maskIoU) - repo drift between the readme link and the config default. Repointed the config to the file that exists. Both `.pth` files load as valid torch state dicts; the `.pt` filter files load as valid TorchScript modules.
- **Verified working:** ran the repo's own `hdbif.py` demo unmodified (pretrained weights, no training) on the 3 bundled synthetic images (the same ones ArcIris's dry run used). It completed and printed 3 impostor scores with an estimated rotation, e.g. `synthetic1 <-> synthetic3 : 0.459 (mutual rot: 11.95 deg)`.
- **Why it matters for Week 2's open questions:**
  - Mask-awareness is the natural test of whether Week 1's occlusion finding (F5: eyelids drive false non-matches, not the impostor tail; F6: gray-fill masking on ArcIris without retraining made things worse) is an ArcIris-specific limitation or a more fundamental one. HDBIF's masking is native (not a bolted-on inference hack), so it is a fairer comparison.
  - The independent 5-12 filter kernels let the impostor tail be decomposed per kernel: match with each filter's bits independently, find which kernel has the lowest impostor-score variance (most stable, possibly least noisy for a threshold), and inspect which bits/positions each kernel tends to leave out (low weight / rarely discriminative) versus the wide-field, low-texture and hub-identity findings from Week 1 (F13, F7).
- **Not yet done:** a manifest-driven pipeline for HDBIF over the full NDCVRL_002 dataset (analogous to `run_pipeline.py` for ArcIris - not written yet) and the per-kernel variance/bit analysis itself.
- Files: weights and filters live in `../General Project Files/OpenSourceIrisRecognition/methods/HDBIF/Python/{models,filters_pt}` (shared resource, not duplicated per week); no new script in `Week 2/` yet for this.

## W2-4. First HDBIF genuine/impostor plot (same style as ArcIris's)

Built `hdbif/run_hdbif_subset.py`: reuses the ArcIris manifest, encodes each image once with HDBIF's own pretrained weights (`segment_and_circApprox` -> `cartToPol_torch` -> `extractCode`, single filter bank: 17x17, 5-bit, `finetuned_bsif_eyetracker_data`), caches codes+masks, then scores all genuine pairs and up to 40,000 impostor pairs with `matchCodesEfficient` (fractional Hamming distance over the unoccluded overlap, lower = more similar; 0.5 = chance). Subset: 80 subject-eyes (min 4 images each), up to 6 images per eye, 479 images total, same random seed convention as Week 1's masking subset. Plotted with `plot_hdbif_distributions.py` in the exact visual style of Week 1's `score_distributions.png` (same green/red histogram, dashed 1%-impostor-tail cut, d' in the title) for direct comparison.

| | HDBIF (this subset) | ArcIris (full set, Week 1) | ArcIris (640x480-only, Week 2 W2-2) |
| --- | --- | --- | --- |
| Genuine mean | 0.299 | 0.837 | 0.745 |
| Impostor mean | 0.445 | 1.566 | 1.567 |
| d' | **2.50** | 3.42 | 4.40 |
| Impostor <= cut | 1.00% (by construction) | 0.98% | 0.83% |
| Genuine > cut | 20.25% | 5.4% | 2.9% |

- **Not directly comparable to the d' numbers above without care**: this is a single 17x17/5-bit filter bank and a 479-image subset (weighted toward subjects with >=4 images), not the full 5,169-image, all-filter-size HDBIF configuration, and ArcIris's d' is on a different, much larger pair set. Take this as a first look, not a head-to-head verdict.
- Score scale differs from ArcIris: HDBIF's fractional Hamming distance runs 0-0.5 (0.5 = chance/orthogonal bits), vs ArcIris's acos-cosine score which runs 0-pi (~pi/2 = chance). The two are not on the same axis; only d', not raw scores, is comparable.
- 15 genuine and 512 impostor pairs (of 1,195 / 40,000) returned `inf` ("too small overlap between masks" - HDBIF's own quality signal, printed by the library) and were excluded from the plot; not yet cross-checked against Week 1's occlusion numbers for the same images.
- 5-bit/17x17 is only the default filter; the user's original request (match with the 5 independent filters, find the lowest-variance kernel, inspect ignored bits) is not done yet - this run establishes that the pipeline works end-to-end first.
- Evidence: `hdbif/hdbif_score_distributions.png`. Data: `hdbif/hdbif_pairwise_scores.csv` (41,195 rows, columns image_id_a/b, label, score, shift).

## W2-5. True worst 1,000 impostor pairs, rendered as a gallery

Built `render_the_1_percent.py`: computes all ~13.3M impostor scores from cached embeddings (no model rerun), selects the 1,000 with **lowest** scores (most confused pairs across the full dataset), and renders them as side-by-side JPEG pairs in the same format as Week 1's `evidence/plots/imposters/` gallery, but with sensor and format metadata per the Week 2 focus. Saved to `../the 1%/` (project root, alongside Week 1 and Week 2) as 1,000 JPEGs ranked 0001-1000 plus `index.csv`.

| Metric | Week 1 imposters/ (sampled) | the 1% (full dataset worst) |
| --- | --- | --- |
| Lowest score | 1.215 rad | **0.434 rad** |
| n pairs | 1,000 | 1,000 |
| Context | 1-in-130 random sample from 100k impostor pairs | True 1,000 worst of all 13.3M impostor pairs |
| Highest score in gallery | 1.406 | 1.281 |

What this shows:
- **Much deeper tail exists**: the true worst pair (0.434 rad) is 0.78 rad lower than Week 1's worst-in-sample (1.215). This is more extreme than the sampled 1% cut would suggest.
- **Gallery composition:** all 1,000 in `the 1%/` score strictly below 1.281 (the best-of-worst), so even the "least bad" pair in this gallery is indistinguishable from many genuine pairs (genuine median = 0.837, so 1.281 is in the tail of genuine). All are far left of Week 1's imposed 1% cut.
- **Format and sensor breakdown (from index.csv):** count 640x480-only, mixed-format, 960x640-only among the 1,000; also find which sensors and identities are over-represented in the true worst (hubs like `nd1S05271` may cluster here differently than in the sampled set).
- Why it matters: the sampled impostor gallery from Week 1 is a *representative sample*, but not the true worst. For understanding the most problematic false matches and for design of inference-only mitigations, the true worst 1,000 are the tightest constraints. Hubs, low-texture identities, or format combinations in this gallery are more likely to be the limiting factor for any threshold-based or cohort-based gate.
- Caveat: rendering 1,000 JPEGs (all 13.3M would be ~130k and infeasible) remains a sampling, but a *ranked* sample of the true extremes, not a random sample.
- Evidence: `the 1%/0001__*__score-0.434.jpg` (worst pair) and `the 1%/index.csv` (full index with subjects, eyes, sensors, formats). Data in `the 1%/index.csv` (1,000 rows); to reproduce, run `python3 render_the_1_percent.py` (~2 min on this machine).

## Next
See `README.md` plan: attribution, resolution-vs-sensor control, identity-level hubs, mitigations (per-format thresholds, cohort normalization), held-out confirmation; HDBIF (W2-3/W2-4 done: weights set up, first genuine/impostor plot on a subset) still needs the full-dataset run and the per-kernel variance/bit analysis.
