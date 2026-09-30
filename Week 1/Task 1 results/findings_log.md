# Task 1 findings log (running)

Living record for the report. Each finding lists the numbers, **why it matters**, the **evidence image**, and how to reproduce it.
Newest sections go at the bottom. Nothing here involved retraining: ArcIris weights are untouched and all models run in `eval()` / inference mode.

## Setup (what was run)

| Item | Value |
| --- | --- |
| Method | ArcIris (`OpenSourceIrisRecognition/methods/ArcIris/Python`, `cfg_baseline.yaml`), pretrained weights |
| Data | NDCVRL_002: 5,175 images, 248 subject-eyes (subject + eye), from `recording_metadata.csv` |
| Manifest | `ndcvrl_manifest.csv` (built by `arciris_analysis/build_ndcvrl_manifest.py`); all 5,175 images matched metadata |
| Quality gate | 5,169 / 5,175 passed `checkQuality` (failures: off-centre pupil/iris circles, pupil too small) |
| Pairs | 70,876 genuine (all kept) + 100,000 impostor sampled from 13,285,820 (seed 0) |
| Tail | impostor pairs with score <= 1st percentile = **1.4057** rad (1,000 pairs, 1,582 images, 178 subjects) |
| Score | `acos(cosine similarity)` of 512-d embeddings, radians, **lower = more similar** |

Pilot before the full run: 100 images / 20 subject-eyes, all passed, d' = 4.58.

## F1. Overall separation

- Genuine mean **0.837**, impostor mean **1.566** (close to pi/2, i.e. unrelated eyes are near-orthogonal), lowest impostor score **1.215**.
- **d' = 3.42** on the full set (pilot was 4.58, so the full set has harder genuine pairs).
- At the 1% impostor cut (1.406), **5.4% of genuine pairs score above it** (false non-matches).
- Why it matters: this is the headline number for the score-distribution section and the Daugman "test of statistical independence" comparison.
- Evidence: `evidence/E0_score_distributions.png`

## F2. Embedding space looks like identity clusters

- t-SNE of the 512-d embeddings (5,169 points, coloured by subject+eye): most identities form a tight single-colour cluster; a few identities are spread out and there is a crowded central knot where identities blur together.
- Left and right eyes are interleaved (no left/right region).
- PCA (PC1+PC2 about 3% of variance) shows only a few clean outlier clusters, a linear 2-D view cannot separate 248 identities.
- Caveats: t-SNE inter-cluster distances are not meaningful. sklearn printed divide-by-zero / overflow warnings during PCA (suspected macOS BLAS issue). **Re-checked in float64 via `eigh`: PC1+PC2 = 3.4% of variance, top-50 PCs = 49.9%, all embeddings finite, so the earlier figure stands.**
- Evidence: `evidence/E1_embedding_space.png`

## F3. What the impostor false matches have in common

Compared the 1,000 tail pairs against all 100k sampled impostor pairs (metadata joined from `recording_metadata.csv`):

| Same attribute | Tail pairs | All impostor pairs |
| --- | --- | --- |
| sensor | 34.3% | 24.2% |
| capture environment | 28.3% | 18.1% |
| eye side (L-L / R-R) | 60.5% | 50.2% |
| eye colour | 34.2% | 26.9% |
| gender / race / contacts / glasses | about equal to baseline | |

- Sensors `nd1N00059` and `nd1N00073` are 1.48x / 1.42x overrepresented in the tail (about 400-500 images each).
- Tail images are slightly darker (mean gray 146 vs 152); sharpness about the same.
- "Hub" subjects: `nd1S05271` is in 68 of the 1,000 tail pairs (13 each with `nd1S05619` and `nd1S04851`); all its images come from sensor `nd1N00006`.
- Interpretation: the embedding carries a little capture-condition signal (sensor / lighting / environment) on top of identity. Effects are modest and untested for significance.
- Caveat: impostor pairs are a 100k sample of about 13M, so which specific images land in the tail is partly sampling.
- Evidence: `evidence/E5_worst_impostor_pairs.png` (4 lowest-scoring impostor pairs: nd1S06943/nd1S05857 1.215, nd1S05044/nd1S05970 1.256, nd1S05271/nd1S05619 1.263, nd1S05487/nd1S05857 1.265). Visibly different irises; frames are dark / low contrast with heavy lashes or partly closed lids.

## F4. ArcIris has no eyelid / occlusion masking

- Code check (`modules/irisRecognition.py`): `circApprox` returns only pupil and iris circles; `cartToPol_torch` unwraps the whole ring between them; `extractVector` takes only the polar image; `checkQuality` checks radii and centre distance only, never coverage. Daugman's method, by contrast, masks lids and lashes and compares only valid bits.
- Test: unwrapped the dataset's own binary segmentation masks (`segmentation_masks/`) with **ArcIris's own predicted circles**, giving the visible-iris fraction in exactly the region the network sees. Mean visible iris = **72.6%** (occlusion about 27%).
- Evidence: `evidence/E4_occlusion_examples.png` (2%, 23% and 100% occlusion; red = not visible iris). Image `268188` is **100% occluded yet passed `checkQuality`**, so the quality gate does not catch occlusion.

## F5. Occlusion causes false NON-matches, not false matches

- Impostor-tail images are only slightly more occluded than random images: mean **0.301 vs 0.280**, Mann-Whitney p = 0.027.
- Impostor pairs: score vs occlusion is essentially flat (Spearman **-0.03**; false-match rate at the cut goes 0.9% -> 1.3% only in the most-occluded quartile).
- Genuine pairs: strong effect. Spearman **0.38** (mean occlusion) / **0.45** (more-occluded image). Fraction of genuine pairs scoring under the cut falls from **99.7%** (least occluded quartile) to **82.3%** (most occluded quartile).
- Conclusion: eyelids/lashes are the main driver of the genuine overlap tail (F1's 5.4%), not of the impostor false matches (F3).
- Evidence: `evidence/E2_occlusion_vs_score.png`

### F5 audit: exact counts behind the "82% vs 99.7%" numbers (script `arciris_analysis/occlusion_miss_rate.py`)

Genuine pairs only (70,876). Pair occlusion = the larger of the two images' occlusion (occlusion = 1 - visible fraction of ArcIris's ring, dataset masks). "Missed" = score above the 1.4057 cut. Quartiles are of pair occlusion.

| Quartile | Occlusion range | Pairs | Missed | Miss rate | Median score |
| --- | --- | --- | --- | --- | --- |
| Q1 | 0.02-0.20 | 17,684 | 59 | 0.334% | 0.681 |
| Q2 | 0.20-0.31 | 17,749 | 161 | 0.907% | 0.695 |
| Q3 | 0.31-0.45 | 17,685 | 458 | 2.590% | 0.760 |
| Q4 | 0.45-1.00 | 17,758 | 3,152 | 17.750% | 1.074 |

- Q4 / Q1 miss-rate ratio is **53x** (an earlier chat message said "roughly 60x" from rounded percentages; 53x is the exact figure).
- Cluster bootstrap (resampling the 248 subject-eyes, 1,000 draws, because pairs share images): Q1 miss 0.12-0.60%, Q4 miss 14.4-21.0%, ratio 28-144x (95% CI). The direction and size hold; the exact ratio is uncertain.
- The miss rate rises monotonically across the four quartiles, and the median score rises with it, so the result does not depend on where the cut sits.
- Limits: this is an association (observational), not an experiment; occlusion co-varies with other capture differences (lid position, pupil size).

## F6. Naive masking without retraining makes it worse

- Test (inference only): blank occluded polar pixels to mid-gray (127) before `extractVector`, re-embedding a 900-image subset (60 subject-eyes x up to 15 images), identical pairs for both conditions.

| | genuine mean | impostor mean | d' | genuine rejected at 1% FMR |
| --- | --- | --- | --- | --- |
| unmasked | 0.811 | 1.568 | **3.73** | **3.99%** |
| masked (gray fill) | 0.920 | 1.550 | **3.13** | **8.03%** |

- Interpretation: the network never saw gray-filled regions in training, so blanking gives it an out-of-distribution input. Masking would need a mask-aware matcher or training with masks (out of scope: no retraining).
- Caveats: subset only; one fill choice (127); other fills (noise, mean-iris texture) untested.
- Evidence: `evidence/E3_masked_vs_unmasked.png`

## F7. Where the worst impostors sit in the t-SNE (natural clusters)

t-SNE re-run with the same settings (perplexity 30, cosine); coordinates saved in `tsne_coords.npz`. Tail = the 1,582 images in the 1% impostor tail.

- **No single confusable cluster.** Tail images (red) are scattered across essentially every identity cluster; the tail-pair graph is one connected component over 243 identity-eyes. The false matches are a diffuse property of weak identities, not a few look-alike groups.
- **Tail identities have looser clusters:** mean same-identity cosine similarity 0.466 (tail) vs 0.524 (rest), p = 3e-22; 5-NN same-identity purity 0.945 vs 0.976, p = 4e-10. Identity size is the same (28.6 vs 28.4 images), so it is not a small-sample effect.
- **Tail images sit slightly more centrally / hub-like:** t-SNE radius 43.0 vs 46.1 (p = 2e-7); 27% of tail images fall in the innermost 25% of the map; mean similarity to all images 0.010 vs 0.007. Small effects, but consistent with the crowded central knot seen in F2.
- **Worst-100 pairs:** mean t-SNE distance 34.8 (all 1,000 tail pairs 43.0) vs 62.8 for random pairs, so confused eyes are closer than chance but mostly in different clusters (long lines in the figure).
- **Hub identities:** 13 of the worst 100 pairs involve `nd1S05271 L`, 10 involve `nd1S04851 L`. Both are female, sensor `nd1N00006`, dark irises (Brown / Black); their frequent partners are `nd1S05619 L` (also dark) and each other. Their images looked dark and low-contrast by eye, and some are overexposed or cropped (see the montage). **Observation from 4 identities by eye. The texture test in F10 did NOT confirm it for the top hub: `nd1S05271 L` has above-median texture.** (eye colour is only weakly enriched in the metadata comparison in F3: Black 1.13x, Brown 1.04x).
- **Same person, other eye:** 1.6% of tail pairs (16) are the same subject's left vs right eye, vs 0.31% of all sampled impostor pairs, about 5x enriched. Expected to be low but not zero; the two eyes of one person share sensor, session, skin, eyelid shape, so some capture-level similarity survives.
- Evidence: `evidence/E6_tsne_impostor_tail.png` (left: tail images in red on the full map; right: 100 worst pairs joined by lines), `evidence/E7_hub_identities.png` (3 images each of `nd1S05271 L`, `nd1S04851 L`, `nd1S05619 L`, `nd1S05308 L`).

## F8. Side-by-side gallery of every tail pair (`imposters/`)

- `Task 1 results/imposters/` holds **1,000 side-by-side JPEGs**, one per impostor pair the model scored at or below the 1% cut (1.4057), named `{rank}__{id_a}__{id_b}__score-{x.xxx}.jpg`, ranked worst first. Each shows both images with score, image ids, subject, eye, sensor and eye colour; `imposters/index.csv` lists them all. 16 are the same person's other eye and are labelled as such.
- **Scope caveat:** the 1,000 are the pairs in the sampled 100k impostor set. Scoring **all 13,285,820** impostor pairs gives **130,228 (0.98%) below the same cut**, so the gallery is a 1-in-130 sample of the full confused set, not the whole thing. A gallery of the full set would be about 130k images.
- Evidence: `imposters/0001__1081607__1082443__score-1.215.jpg` is the worst pair (dark, low-contrast frames of two different people).

## F9. Ten genuine (true-match) examples (`true matches/`)

- `Task 1 results/true matches/` holds 10 side-by-side JPEGs of the **same eye** (different subject for each), picked at evenly spaced percentiles (2%-98%) of the 70,876 genuine scores instead of cherry-picking the best. Same layout as the impostor gallery; `index.csv` lists them.
- Scores: 0.384, 0.529, 0.604, 0.669, 0.737, 0.816, 0.918, 1.048, 1.220, 1.534 (cut = 1.4057).
- Pair 1 (score 0.384): same session, two different sensors (`nd1N00049` vs `nd1N00048`): the model still matches, showing sensor differences alone do not break a match when the iris is well visible.
- **Pair 10 (score 1.534) is a miss**: same eye, sensor `nd1N00074`, captured 8/14/12 vs 10/29/13 (about 14 months apart); the second frame is dark with a heavy lid and lash shadow covering the upper iris, and the pupil size differs. This matches F5 (occlusion drives false non-matches). About 5.4% of genuine pairs look like this.
- Evidence: `true matches/01__861756__862714__score-0.384.jpg` (easy match), `true matches/10__1148984__1190031__score-1.534.jpg` (missed match).

## F10. Iris texture vs false matches (test of the "low-texture irises" hypothesis)

Method (inference only): per image, unwrap with ArcIris's circles, keep only visible-iris pixels (dataset mask), and measure fine-texture energy `hf_std` = std of (polar - Gaussian blur sigma 3). Also mean gray (`iris_mean`), raw contrast and `hf_rel` (see `texture_scores.csv`, script `arciris_analysis/texture.py`). 5,125 of 5,169 images usable (44 have essentially no visible iris in the mask). False-match rate = share of impostor pairs at or below the 1.4057 cut, computed on **all** 13.3M impostor pairs.

- **Only the low-with-low corner is elevated.** FMR is **2.99%** when both images are in the lowest texture quartile, vs 0.67-1.03% for every other combination (roughly flat). About 3x worse, on about 6% of pairs (812k). The lowest-quartile image alone barely matters (1.13% vs 0.8-0.9%).
- Same pattern for `iris_std` (2.61%) and `hf_rel` (2.80%); `iris_mean` (darkness) is weak (1.40%).
- **Per identity-eye:** Spearman -0.25 between mean texture and the identity's own false-match rate (p = 5e-5, n = 248), but the extremes differ little (lowest-texture 10% of identities 1.12% vs highest 10% 1.00%).
- **Correction to F7:** `nd1S05271 L` (worst hub, own FMR 2.61%) has mean `hf_std` 5.99, **above** the dataset median of 4.66; `nd1S04851 L` 4.76 (median); `nd1S05619 L` 3.14 (low); `nd1S05308 L` 5.68 (own FMR 1.86%). So low texture explains some hubs but not the top one; my earlier visual impression was wrong for it.
- **Logistic regression** on the 98,300 sampled impostor pairs (962 positives), effect per 1 SD: max texture of the pair OR 0.74, min texture 0.93, same eye side 1.25, same sensor 1.23, same environment 1.17, max occlusion 1.17. All modest, none dominant.
- **UPDATE (see F13): most of this texture effect is the 960x640 wide-field capture format.** 61% of the lowest-texture quartile are 960x640 images. Restricted to 640x480 images only, both-low-texture pairs have FMR 0.92% vs 0.88% for the rest: **no texture effect**. The texture effect survives only inside the 960x640 format (10.0% lower-half vs 2.6% upper-half). Read the conclusion below with that in mind.
- **Conclusion (original):** low iris texture is a real but partial contributor: it raises false matches when both irises are low-texture, and the other factors (sensor, environment, eye side, occlusion) each add a little. No single cause explains the impostor tail. What makes `nd1S05271 L` a hub is still unexplained.
- Caveats: `hf_std` is one texture proxy and also responds to lash edges and reflections inside the mask; regression coefficients came from sklearn under the same suspected BLAS warnings as F2 (unverified); positives are few (962).
- Evidence: `evidence/E8_texture_vs_false_matches.png` (left: 4x4 FMR grid by texture quartile; right: per-identity scatter).

## F11. Label audit: is a "false match" secretly a true match? (imposter #22)

Question: pair #22 (`243066` vs `934720`, score 1.305, gallery file `0022__243066__934720__score-1.305.jpg`) is the same subject `nd1S05271`, labelled Right vs Left. Could it be a mislabelled eye and therefore a real match?

- **Test 1, consistency with own labelled eye:** `243066` (labelled Right) has median score 0.932 (min 0.565) against the subject's other 24 Right images, but 1.360 (min 1.289) against her 50 Left images. `934720` (Left) has median 0.789 vs her Left images and 1.359 vs her Right images. Each image sits with the eye it is labelled as, so the labels are consistent and the pair is two different eyes of one person, not a mislabelled pair.
- **Test 2, whole-dataset scan:** for every one of the 5,169 images, compared the median score to its own subject-eye against every other subject-eye. Images whose best-matching other identity beats their own *and* is under 1.0: **0 of 5,169**. No sign of eye-label errors or duplicate enrolments among embeddings. (Limit: this cannot detect a subject whose *all* images are mislabelled consistently.)
- **Side finding:** this subject's right eye is unusually scattered (R-vs-R median 1.083 vs the dataset genuine median about 0.84), and her R-vs-L median (1.414) is close to the cut. That helps explain why `nd1S05271` is the top hub (F7); images span sensors `nd1N00006`, `00016`, `00020` and 2006-2009 captures.
- Evidence: `evidence/plots/imposters/0022__243066__934720__score-1.305.jpg` (the pair). Note the folders `imposters/` and `true matches/` now live under `evidence/plots/`.

## F12. All same-person / other-eye false matches (`imposters/interesting false matches/`)

- Scanned the 1,000-pair gallery for pairs where both images are the **same subject, opposite eyes**: **16 pairs** (1.6%), copied (originals kept) into `evidence/plots/imposters/interesting false matches/` with an `index.csv`. Scores 1.304-1.405 (all just under the cut; none among the very worst).
- **Subjects:** `nd1S05271` 6 pairs (ranks 22, 31, 307, 530, 633, 955), `nd1S04261` 3 (226, 873, 895), `nd1S05881` 2 (609, 799), and one each `nd1S07015` (20), `nd1S05096` (52), `nd1S05379` (111), `nd1S05974` (260), `nd1S05216` (917). 6 of 16 come from the hub subject `nd1S05271`.
- **Label check (same test as F11) on all 32 images:** each image is closer to the other images of its labelled eye than to the subject's opposite eye in every case (0 of 32 fail), so none of these is a mislabelled true match.
- **One marginal case:** rank 52, `659279` (R, subject `nd1S05096`): median score 1.40 to its own eye vs 1.43 to the other eye, i.e. this image is barely closer to its own eye, so it is a weak or atypical capture. Worth a manual look.
- Interpretation: the model finds a person's two eyes slightly more alike than two strangers' eyes (5x enrichment in the tail vs random impostor pairs, F7). The scores are only just under the cut, so these are borderline false matches, not confident ones.

## F13. Other factors (format, glasses, contacts, colour, eyelids) and the wide-field capture finding

Method: on **all 13,285,820 impostor pairs** (full set, not the sample), FMR at the 1.4057 cut among pairs where **both** images share an attribute value, vs the 0.980% baseline (130,228 hits). Metadata from `recording_metadata.csv`; occlusion from F4; image size read from the files.

**Biggest finding: the 960x640 capture format.**
- 909 images (133 identity-eyes) are 960x640 (3:2) and come only from sensors `nd1N00059` (516) and `nd1N00073` (393); every other sensor is 640x480 (4:3).
- FMR when both are 960x640: **5.25%** (21,451 hits on 408,667 pairs, **5.4x baseline**); cross-sensor 59 vs 73 pairs 4.99%; 640x480-only pairs **0.83%**; mixed-format pairs (960 vs 640) **0.87%**, back to baseline.
- 165 of the 1,000 tail pairs (16.5%) are both-960x640, though such pairs are only 3.1% of all pairs.
- Why: measured on the same circles, these irises are much smaller and flatter: iris radius about **66 px vs 119 px** (in ArcIris's 640x480 frame), mean fine-texture `hf_std` **2.10 vs 5.27**, contrast std 7.1 vs 15.5, visible iris 67% vs 75%. So they are wide-field, low-resolution iris captures; the unwrapped iris is a blurry upsample and the embedding falls back on generic capture features shared across these people. The effect is spread over 97 (sensor 59) and 82 (sensor 73) identity-eyes, so it is not one hub subject. Within the format, lower texture still hurts (10.0% vs 2.6%).
- Evidence: `evidence/plots/imposters/interesting false matches/other factors/both_wide_960x640_format/` (12 example pairs, incl. #1 the worst pair, #6, #7): dark, low-contrast frames with small irises.
- Caveat: sensor and image format are perfectly confounded here (59 and 73 are the only 960x640 sensors), so this data cannot separate "small irises" from "sensor signature". Separating them would need resized 640x480 images as a control (inference only), not done yet.

**Other attributes** (both images share it; baseline 0.98%):
- Sensor `nd1N00059` 5.93% (6.1x), `nd1N00073` 4.76% (4.9x); these are the same effect as above.
- **Asian-Southern** 6.48% (6.6x) but only **9 identity-eyes**; excluding the hub `nd1S05271` it is still 4.89% over 7 identity-eyes. Too few identities to call a demographic effect; likely driven by a few subjects. Not a fairness conclusion.
- Eye colour Black 1.97% (2.0x, 30 identity-eyes); Brown, Blue, Hazel, Green about baseline. Gray 0.61%.
- **Cosmetic contacts** 4.4x but only 41 images / 4 identity-eyes / 27 hits: anecdotal. Any contacts 1.08% (about baseline); toric contacts 0.19x on tiny n.
- **Glasses** both: 1.43% (1.45x, 193 images, 50 identity-eyes, 259 hits): a small effect. 3 of the 1,000 tail pairs have both wearing glasses, 79 have exactly one. The 3 both-glasses pairs (`.../other factors/both_wearing_glasses/`) show frames and very large specular glare blobs covering the iris in at least one image, so the model may be matching on glare/frame artefacts.
- **Both heavily occluded (>0.45):** 2.76% (2.8x, 757 images). Consistent with F5/F10 but larger than the earlier estimate; note occlusion also co-occurs with wide-field captures.
- **Eyelid/lash layout:** correlation of the angular occlusion profile between the two images vs FMR: about flat (1.09% for the least similar layouts, 0.92-0.93% in the middle, 1.07% for the most similar). **No sign that the model matches on eyelid layout / orientation of the occlusion.** (Only a proxy for head tilt; true rotation of the eye was not measured.)
- Same-person opposite-eye pairs: see F12.

Caveats: hits within a subgroup share images and identities (not independent); small groups (cosmetic contacts, some races, toric) have very few identity-eyes; the sensor/format effect is confounded (above).

## Method notes worth keeping

- `matchVectors` can raise `math.acos` domain errors when cosine = 1.0000001 (identical images); the masked test clamps to [-1, 1] instead. The main run did not hit this.
- Environments: the `arciris` mamba env has torch but no matplotlib; system `python3` has matplotlib / sklearn / scipy but no torch. Plotting scripts run under system python; extraction under the mamba env (`prep_e4.py` bridges the two for E4).
- Templates cache: `OpenSourceIrisRecognition/.../templates/<image_id>_tmpl.npz` (side effect of the pipeline, reused on re-runs).

## Files

| File | Purpose |
| --- | --- |
| `arciris_analysis/build_ndcvrl_manifest.py` | metadata -> manifest |
| `arciris_analysis/occlusion.py` | polar visible-iris masks -> `polar_masks.npz` |
| `arciris_analysis/masked_rescore.py` | gray-fill re-embedding test -> `masked_rescore_subset.npz` |
| `arciris_analysis/prep_e4.py`, `make_evidence.py` | regenerate `evidence/E2-E4` (E0, E1, E5 were copied from `plots/`) |
| `Task 1 results/run.log` | full-run console log |

## Still to do

- ROC / DET curve, threshold from a target FMR, `false_positives.csv`, pair renders (Task 1 handoff steps 7-9).
- Same-sensor vs different-sensor false-match rate on the full pair set (to confirm the F3 sensor/environment effect).
- Why is `nd1S05271 L` a hub if not low texture? (exposure/cropping variation across its frames? ask: inspect its images against its top partners)
- Optional: other mask-fill strategies (F6 caveat).
- F13 control: resize 960x640 images to look like 640x480 (or crop the iris to equal pixel size) and re-embed (inference only) to separate small-iris resolution from sensor signature.
