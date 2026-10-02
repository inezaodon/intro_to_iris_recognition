# Week 2: diagnosing the long left tail of the impostor distribution

Continues Week 1 (ArcIris on NDCVRL_002; inference only, **no retraining**). Inputs are read from `../Week 1/Task 1 results/` (`embeddings.pkl`, `manifest.csv`, `findings_log.md`). Running record of results: `findings_log.md` (same convention as Week 1: numbers, why it matters, evidence image, how to reproduce).

## What Week 1 established (short)
- d' = 3.42; impostor mean 1.566 rad; 1% impostor cut = 1.4057; 130,228 of 13,285,820 impostor pairs (0.98%) fall under it.
- Occlusion mainly causes false **non**-matches (genuine right tail), not the impostor left tail.
- Both-960x640 pairs (sensors `nd1N00059` / `nd1N00073`, small wide-field irises) false-match at 5.25% vs 0.83% for 640x480 pairs, but explain only about 16% of the tail hits.

## Week 2 plan
1. **Tail shape by group** (this step): impostor distribution split into 640x480-only, 960x640-only and mixed-format pairs; compare each tail to what its own body predicts.
2. Attribution: how much of the tail each factor explains (format, sensor, environment, eye side, occlusion, texture, glasses, contacts) and how much is left over.
3. Control experiment: shrink 640x480 irises to wide-field size and re-embed (resolution vs sensor signature).
4. Identity-level: why hubs like `nd1S05271 L` false-match; what separates hubs within 640x480.
5. Inference-only mitigations: per-format thresholds, cohort score normalization, minimum-iris-size gate.
6. Confirm main findings on the dataset's held-out split (`NDCVRL_002 Dataset/test_train_split/`); finish Task 1 deliverables (ROC/DET, threshold, false_positives.csv, report).
7. **HDBIF as a second, mask-aware method** (weights set up this session, see `findings_log.md` W2-3): build a manifest-driven HDBIF pipeline over NDCVRL_002, compare its genuine/impostor separation and occlusion behaviour to ArcIris, and match with its 5 ICA filter kernels independently to find which kernel has the lowest impostor variance and which bits each kernel tends to ignore.

## Layout
| Path | What |
| --- | --- |
| `tail_shape_by_group.py` | step 1 script |
| `score_distributions_by_group.py` | genuine vs impostor per format group (same style as Week 1 `score_distributions.png`) |
| `render_the_1_percent.py` | render true worst 1,000 impostor pairs across all 13.3M pairs (finding W2-5) |
| `plots/` | main figures (`W2_dist_*`); `plots/extra/` = secondary analytic views |
| `findings_log.md` | results log |

## Shared resources (not duplicated per week)
HDBIF weights and filter kernels (downloaded this session) live in `../General Project Files/OpenSourceIrisRecognition/methods/HDBIF/Python/{models,filters_pt}`, alongside ArcIris in the same repo checkout.
