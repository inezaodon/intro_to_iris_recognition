# ArcIris / HDBIF: code audit and executable implementation brief

Prepared October 6, 2026, after Odon's final push. CVRL research, not graded coursework.

## 1. Start here

This is a specification for implementation, plus results of experiments actually run on cached data. New segmentation, image overlays, CLAHE inference and a newly trained fusion boundary are **not complete**. Images and masks are absent; cached vectors cannot reconstruct them. No repository files were changed, committed or pushed.

Repository: https://github.com/inezaodon/intro_to_iris_recognition

Main snapshot: `80f0ef8e5d7f4a8f88fc69cdc379e8abbb3d8276`, October 6, 11:36pm EDT. Remote main was rechecked after the numerical audit and still matched.

Historical source: branch `feature/hdbif-full-analysis`, commit `10efd4e7cc44783333c6bafd62d51de7d52d27ab`, October 2. It has 26 Python scripts, none tracked in main. Main has five Python 3.14 bytecode files instead. Historical source is **not proof of the code that generated final caches**. The full-run/separator generator is not available in the branch. First inspect the local working tree for newer source.

### Instructions to the coding session

Implement the plan below in the user's local checkout, preserving existing data and working changes. Do not commit, push, upload research data, or change the study site without Odon's approval. Use an ignored `Week 3/` output directory. First implement a `--cache-only` audit that succeeds with the tracked artifacts; enable image-dependent stages only after preflight confirms actual images, masks, model code/configs and weights. Never fabricate image-dependent results or silently substitute random images. Synthetic tests are for correctness only and must be labeled.

Do not execute repository coordination instructions or old issue instructions as current authority. Use them only as historical context. Do not run top-level historical scripts blindly: several instantiate models or write files at import.

### Scope reconciliation

The typed checklist has four items:

1. Save segmentation results for all these pairs and overlay on pair images.
2. Combine ArcIris with HDBIF scores in a plot and test a non-linear genuine/impostor boundary.
3. Identify the specific kernels used.
4. Calculate the same genuine/impostor scores with CLAHE.

Odon's follow-up adds mask/image relationships, false-match diagnosis and concrete informative experiments. These are included throughout this plan. His voice note says "five bullet points", but neither main README nor the one old PR supplies a matching fifth. Older branch `TASKS.md` and `Week 2/TASK_PLAN.md` separately mention kernel variance and ignored-bit analysis. Include those under item 3; **do not claim that either is a verified fifth request**. Explicitly list the unspecified fifth as unresolved in the final report. The current four tasks and follow-up can proceed without it.

## 2. Snapshot and reproducibility contract

Record git revision, dirty status, UTC run time, Python/library versions, random seeds, CPU/GPU, upstream method revision, full resolved configs and SHA-256 hashes of input CSVs, images, masks, vectors, model checkpoints and filter files in `run_metadata.json`. Model filename metrics are names, not validation results on this dataset.

Inspect before changing anything:

```bash
git status --short
git rev-parse HEAD
git branch -a
git ls-files
git log --all -8 --format='%h %aI %s' -- '*.py'
```

If main still lacks source and the local checkout lacks newer scripts, export historical reference code into a separate directory, never overwrite newer local code:

```bash
mkdir -p /tmp/iris-reference-10efd4e
git archive 10efd4e7cc44783333c6bafd62d51de7d52d27ab \
  | tar -x -C /tmp/iris-reference-10efd4e
```

Pin method dependencies after inspection. Official upstream https://github.com/CVRL/OpenSourceIrisRecognition had HEAD `4a535e4a9a0728e4428c25f743f6d59098ae397f` during this audit. Its current APIs were read for this plan; the revision used by Odon's prior local run is unknown. Preserve his actual source/config if present and record differences rather than silently updating it.

## 3. Whole-repository map

### Main: tracked artifacts

| Path | Contents and interpretation |
|---|---|
| `README.md`, `index.html`, `assets/app.js`, `assets/styles.css` | Static iris study site, separate from the experiment pipeline. JS controls tabs, hash routing, slides and quiz expansion. No model inference. Leave unchanged. |
| `.gitignore` | Globally excludes common image extensions and `.npz`; explicitly excludes `Week 2/hdbif_full_run/scores/hdbif_kernel_scores.csv`. Explains missing experimental evidence; not proof it exists locally. It does not generally exclude `.pkl`, weights or CSV metadata. |
| `General Project Files/NDCVRL_002 Dataset/recording_metadata.csv` | 5,175 metadata rows; `fileid` connects to image IDs, `subjectid` and eye define iris identity. |
| `.../iso_metrics.csv` | Per-image recorded quality metrics. Treat undocumented sentinel values, especially repeated 255, as unknown until the dataset schema confirms meaning. |
| `.../sensors.csv`, `.../bxgrid-results.csv` | Sensor metadata; not image pixels or masks. |
| `General Project Files/arciris_analysis/__pycache__/` | Five Python 3.14 bytecode files: embeddings, manifest, pairwise, render, tails. Not portable executable source. |
| `Kernels and other models/filters_pt/` | 288 ICA filter archives: three families, twelve patch sizes, eight bit depths. |
| `Kernels and other models/models/` | Two `.pth` archives: nested U-Net mask model and `resnet18-027...` circle model. ArcIris ResNet100 encoder weights are absent. Upstream current config references a different circle checkpoint (`resnet18-1543...`), so resolve the local model/config deliberately. |
| `Week 1/Task 1 results/embeddings.pkl` | 5,169 vectors, each shape `(512,)`, the basis of all-pairs cache-only audit. |
| `.../_texture_ok.npy` | A 5,125-element Unicode image-ID array (dtype `<U7`), not texture measurements or masks. No tracked generating script/provenance links it to a documented quality criterion; preserve, do not silently filter evaluation with it. |
| `.../run.log` | Historical run: 5,169/5,175 passed, 70,876 genuine and 100,000 sampled impostors, sampled 1% cut 1.405748 rad. |
| `.../evidence/plots/*/index.csv` | 1,000 sampled impostor records, 16 selected interesting false matches and ten genuine examples; referenced JPGs are absent. Numeric ID-score mapping was checked and is correct to rounding. |
| `Week 2/hdbif/hdbif_run.log` | Old 479-image subset run, including overlap failures. Does not supply raw codes or masks. |
| `Week 2/hdbif_full_run/cache/pairs.csv` | 412,321 distinct unordered pairs with ArcIris distance, labels, source and train/test/cross split. |
| `.../cache/encode_failures.csv`, `.../encode.log` | Header-only failure table; log says 5,175 encoded, zero failed. This does not imply all pair overlaps are valid. |
| `.../match.log` | Logs twelve patch sizes, including lengthy 39x39 matching. Missing per-kernel CSV was reportedly written locally. Family, bit depth and normalization are not encoded in this log. |
| `.../scores/separator_test_predictions.csv` | 249,571 test rows with five predictor outputs. No pair IDs, model files, fit/calibration code or selected-bank identity. Numeric order aligns with the test subset of pairs.csv, but this is not a durable join contract. |

No raw image or mask files, YAML method configs, dependency lockfile, full-run source, source for trained separators, HDBIF code cache, per-kernel raw score CSV or ArcIris encoder checkpoint is tracked in this main snapshot.

### Recovered analysis modules (26 scripts, 3,113 lines)

These were parsed and read as historical reference, not installed into main:

| Module under `General Project Files/arciris_analysis/` | Function and required fixes |
|---|---|
| `manifest.py` | Manifest scanning, normalization, strict existing-path loading, CSV helpers. Add metadata-only mode; reject unlabeled evaluation pairs instead of silently calling them impostors. |
| `build_ndcvrl_manifest.py` | Join metadata `fileid` to raw TIFF stem. Detect duplicate stems and output missing-image records, not silently skip all absent files. Resolve output from repo root, not its old parent directory. |
| `embeddings.py` | ArcIris template reuse, circle quality check, polar unwrap, vector extraction. Cache key is only image ID, insufficient for CLAHE/config/model variants. Add provenance key and artifact validation; do not infer quality pass from cache existence. |
| `pairwise.py` | `acos(cosine)` distance and subject-eye labels. Correct basic score direction. Avoid enumerating all pairs into Python lists before sampling; stream/block instead. Fail on nonfinite/zero-norm vectors and unknown identity. |
| `tails.py` | Percentile selection, image hub tracker, tail outputs. Distinguish descriptive percentile from deployed threshold; small-n median fallback must never become a claimed biometric operating point. Add validated finite inputs and selection provenance. |
| `render.py` | Tail-only renderer with circles/polar images, hardcoded "Impostor" title. No segmentation saving. Extend to both labels, mask overlays and pair statuses, reading per-image caches rather than repeated inference. |
| `run_pipeline.py` | Orchestrates manifest, cache, scoring, tail rendering. Defaults became stale after folder relocation; default out path can land under General Project Files. Add explicit rooted configuration and cache-only path; model is None on cache hits, so old renderer drops geometry. |
| `occlusion.py` | Unwraps dataset masks using ArcIris circles; unsafe mask transformation currently uses image `fix_image` with gray fill and default resampling. Paths/output reflect old layout. Apply identical geometry with zero mask padding and nearest-neighbor sampling. |
| `texture.py` | Visible iris brightness/contrast and high-frequency residual. Useful when masks arrive; root paths stale. Log finite support and invalid images, avoid unmasked whole-image proxies. |
| `masked_rescore.py` | Gray-fills occluded polar pixels then re-embeds a subset. Not native ArcIris masking, may create distribution shift. Keep as explicit ablation, no blanket fix. Rigid sampling requires enough eligible groups. |
| `occlusion_miss_rate.py` | Occlusion quartiles and subject-eye bootstrap. Fix output root, finite checks, empty quartiles; don't choose thresholds on test. Cluster by person where both eyes should stay together. |
| `prep_e4.py` | Dumps three image/polar examples. Depends on images/circles/masks, has rigid picks and stale paths. Replace with explicit selected image IDs. |
| `make_evidence.py` | E2/E3/E4 figures; hardcoded numeric titles can lie after reruns, quartile bins include shared edges twice. Compute captions and use disjoint bins. |

| Module under `Week 2/` | Function and required fixes |
|---|---|
| `tail_shape_by_group.py` | Full cached pair distributions, normal-body fit and QQ plots. Requires original image dimensions, missing manifest/summary. Blocks or memmaps instead of multiple full NxN arrays; sensors are only proxies when dimensions missing. |
| `score_distributions_by_group.py` | Histogram group plotter. Reuse with validated group score files and readable units/counts. |
| `render_the_1_percent.py` | Selects worst 1,000, **not 1% of all impostors**. Critical filtered-index mapping defect described below; misleading zero-image placeholders. Replace with audited ID-score rows and explicit missing states. |
| `distribution_matching_analysis.py` | One-dimensional metrics and normal fits, not cross-method fusion. `argmin(abs(pdf_g-pdf_i))` can pick a place both densities approach zero rather than their useful intersection. Empirical ROC and validation thresholds take precedence. Coarse EER grid is descriptive only. |
| `histogram_normalization.py` | Image brightness/contrast comparison only; does not calculate new recognition scores. Loads baseline but never re-encodes variants. A five-row figure indexes rows 5 and 6, causing IndexError. Use separate layout and real paired scoring. |
| `hdbif_per_kernel_analysis.py` | Placeholder/custom nonexistent `hdbif` API, broad swallowed errors, no genuine analysis per bank. Retire in favor of actual upstream adapter. |
| `hdbif/run_hdbif_subset.py` | Best historical API reference: fix image -> segment/circles -> unwrap image+mask -> list of codes -> match `(score,shift)`. Only subset/default bank, placeholder cache `{codes: None}`. Add real persistent artifacts and all requested pairs. |
| `hdbif/run_hdbif_5kernels.py` | Broken wrapper: treats segmentation output as polar; wrong extraction unpacking; wrong matcher argument order and unsupported `rotation_range`; uses missing manifest `filename` not `filepath`; repeated random impostor sampling can duplicate pairs. See exact API section. |
| `hdbif/hdbif_full_pipeline.py` | Empty encoding stub, imports nonexistent `SegmentAndCircApprox/CartToPol/ExtractCode`, writes empty dictionaries. Do not treat successful exit as completion. |
| `hdbif/hdbif_kernel_variance.py` | Expects absent `kernel_scores.npz` and exactly five banks. Generalize keyed bank list, finite failures, uniform pair set, train-only selection; variance alone is not discriminative power. |
| `hdbif/hdbif_bit_analysis.py` | Counts ones without masks and misreads multidimensional code shape as bit count. Low 1-frequency is not "ignored"; entropy averaging is wrong for multi-axis arrays. Use eligible support, binary entropy and match contributions. |
| `hdbif/histogram_preprocessing.py` | CLAHE/stretch/equalization preprocessing only; looks for `image_path/path` instead of manifest `filepath`; imports unused dependency. No re-encoding/matching; replace with unified variants stage. |
| `hdbif/plot_hdbif_distributions.py` | Filters nonfinite/negative scores, plots sampled Hamming distances and descriptive cut. Preserve failure counts and rates, no arbitrary 0.5 truncation if normalization differs. |

Branch documents conflict with implementation: `W2_comprehensive_summary.md` says all tasks completed while also describing per-kernel infrastructure as incomplete; `TASKS.md` claims 70M+ impostors although 5,169 images produce ~13.3M. Treat prose as leads, not results.

## 4. Results actually reproduced tonight

All values below come from tracked cached vectors and recorded labels, not fresh model inference.

- 5,169 finite nonzero float vectors, dimension 512; norms 3.821005 to 36.024514. All metadata IDs uniquely map. Six missing vectors: `234278, 391778, 433438, 666137, 760990, 869806`.
- 180 persons, 248 subject-eyes. Identity is `(subject_id, normalized eye)`. Same person/different eye is impostor for iris verification, but keep both eyes of a person in the same dataset split.
- All `N(N-1)/2 = 13,356,696` comparisons: 70,876 genuine, 13,285,820 impostor. Normalized-dot/angular computation was blockwise; stored six-decimal pair scores differ by at most 0.000001633 rad.
- Full impostor mean 1.565916 rad, SD 0.066254; genuine mean 0.836525, SD 0.294251. Full 1% impostor quantile 1.406318.
- At historical sample cut 1.405748: FMR 0.980203% (130,228 comparisons), FNMR 5.403804%. This is descriptive on these caches, not a calibrated deployment result.
- Impostor 0.1% quantile 1.344508; 0.01% 1.286571; 0.001% 1.224882. Worst 0.433921 rad: **241619 vs 437343**, recorded subjects `nd1S05555` and `nd1S05675`, Left/Left. Cosine 0.907324. Sensors `nd1N00016` and `nd1N00006`; dates April 23, 2008 and March 26, 2009. Pixels unavailable, so no confirmed duplicate, labeling error or capture explanation.
- Sampled impostor index minimum is 1.214818 rad. Random 100k sampling missed the true extreme. Do not equate a sample tail gallery with the all-pairs worst cases.
- No exact duplicate embedding byte arrays. Uncentered covariance participation ratio/effective rank ~142.03 (512 dimensions), top-ten eigenvalues ~14.70% of energy, unit-vector mean norm 0.09191. These describe anisotropy, not proof of collapse or its cause.

Sensor proxies for historical formats (`nd1N00059/nd1N00073` = wide group) reproduce old count/mean patterns, but original image sizes cannot be rechecked:

| Pair proxy | Impostors | FMR at historical cut | Genuine FNMR |
|---|---:|---:|---:|
| Neither wide sensor | 9,021,691 | 0.8321% | 2.8612% |
| Mixed | 3,855,462 | 0.8743% | 11.5831% |
| Both wide sensors | 408,667 | 5.2490% | 11.0724% |

The single most extreme impostor is **not** in the wide-sensor group. Format shift is a partial explanation, not the whole false-match problem.

### Existing pair and prediction integrity

`pairs.csv`: 412,321 rows; no self-pairs, duplicate canonical unordered pairs or label disagreement. Train 138,937 (64,652 genuine, 74,285 impostor); test 249,571 (6,224 genuine, 243,347 impostor); cross 23,813 impostor. Sources: week1_sample 170,876 and test_all 241,445. Train contains 4,462 images/144 persons; test 707 images/36 persons. No shared train/test images, persons or pairs. This validates split membership only, not unseen training/model selection.

`separator_test_predictions.csv` has matching test length, ordered labels and negated ArcIris scores within 0.000005 rad of test rows. This supports an **explicitly verified provisional order join**, but add pair IDs and checksums before future use. Ties may still conceal permutations; don't silently rely on row order.

Stored outputs are genuine-oriented scores, not all probabilities. Raw ArcIris and HDBIF distance columns are negated. Fused outputs may be margins/logits; their training definitions are unavailable.

| Stored predictor | Rank AUC | FNMR at test-recalibrated ~1% FMR |
|---|---:|---:|
| ArcIris | 0.989609 | 5.9287% |
| HDBIF selected bank (unspecified) | 0.970701 | 13.1266% |
| HDBIF fusion | 0.981409 | 11.7931% |
| ArcIris + HDBIF | 0.995752 | 4.7076% |
| Boosted all | 0.995536 | 4.4023% |

**All thresholds in this table were picked on test outputs for descriptive comparison. They are not independent held-out threshold results and must not be promoted as deployable claims.** Model training/calibration source, chosen bank and fit data are unavailable.

At individual test-recalibrated ~1% cuts, ArcIris and selected HDBIF jointly accept only 68/243,347 impostors (0.02794%), but jointly accept 5,315/6,224 genuine (85.3952%). ArcIris-only accepts 540 genuine, HDBIF-only 92. This is evidence of useful complementarity and a severe genuine-rejection cost for a strict AND rule, **not permission to choose that rule on test**. Evaluate learned fusion on untouched identities with validation-only thresholds.

### Critical gallery defect, distinct from tracked gallery data

Historical `render_the_1_percent.py` filters the upper-triangle scores by impostor label, then uses positions in that filtered array to index the *unfiltered* upper-triangle ID arrays. This attaches wrong images to scores. Fix before interpreting newly generated galleries:

```python
ui, uj = np.triu_indices(N, 1)
keep = labels[ui] != labels[uj]
ii, jj = ui[keep], uj[keep]
d = distances[ii, jj]
k = min(1000, len(d))
selected = np.argpartition(d, k - 1)[:k] if k else np.array([], dtype=int)
rows = [(ids[ii[t]], ids[jj[t]], float(d[t])) for t in selected]
rows.sort(key=lambda x: (x[2], x[0], x[1]))
```

Better: derive gallery directly from canonical scored rows and re-score each selected pair as an assertion. Synthetic reproduction: with `(a,b)` genuine and `(a,c),(b,c)` impostor, the old code selects `(a,b)` for the best impostor score; corrected code selects `(a,c)`.

This defect is in historical source. It **does not prove tracked Week 1 gallery CSVs are wrong**: all 1,026 tracked index records were independently re-scored and match their listed IDs within 0.000000533 rad. Their pixels remain unverified.

## 5. Target design and contracts

Create a small import-safe package at `Week 3/iris_audit/` and tests in `Week 3/tests/`. Resolve all paths from one CLI config, never current working directory. Suggested modules: `config.py`, `preflight.py`, `manifest.py`, `pairs.py`, `adapters.py`, `segmentation.py`, `render_pairs.py`, `kernels.py`, `preprocessing.py`, `metrics.py`, `fusion.py`, `experiments.py`, `cli.py`. Add `__init__.py` and `__main__.py`. Do not copy historical stubs as working implementations.

### Data contracts

- `images.csv`: image_id (string), relative raw path, subject_id, eye (`L/R`), capture session/date, sensor_id, original width/height if read, raw/mask SHA-256, image_status, mask_status. Filename stems are IDs, never subject IDs. Metadata-only mode retains missing rows with explicit status.
- `pairs.csv`: pair_id, image_id_a/b sorted as strings, subject/eye for each endpoint, label, split, selection_source. Use SHA-256 of a stable delimited ID tuple as pair_id (or safe stable tuple serialization). Test identity equality, no self-pairs, unique canonical pairs. Do not train on cross pairs bridging subject partitions.
- `image_artifacts.csv`: image_id, variant, cache_key, raw/fixed image path, raw/fixed mask path, polar image/mask paths, geometry JSON path, quality status, visible fraction, segmentation_source (`dataset` or `predicted`), reason if unavailable.
- `scores.csv`: pair_id, method, variant, bank_id, score, score_direction (`distance_lower_is_match` or `margin_higher_is_match`), shift, compared_bits, status/reason, run/config hash. Invalid values are missing scores with failure status, never negative distances that look like strong genuine matches.
- `pair_renders.csv`: one row for every requested pair with artifact paths, render status and reason. Distinguish scored pairs from rendered ones. No omitted failed/missing pairs.
- `kernel_inventory.csv`: family, patch_size, bit_depth, selected boolean, file hash, tensor shape/dtype, coefficient statistics, source revision. Runtime selection must be explicit.

Default pair universe: preserve all 412,321 rows of the tracked pairs table for apples-to-apples raw/CLAHE/HDBIF/fusion comparisons, and add the sampled evidence pairs plus corrected all-cache extremes if absent, tagged `diagnostic_extra`. Also support `--pair-universe all-cache` for all 13,356,696 valid vector comparisons and `--pairs-csv PATH` for specific supervisor-selected pairs. The phrase "all these pairs" need not mean a whole quadratic image gallery: expose the universe in config and report counts. Never silently cap pair scoring/rendering. Saving unique per-image masks once and linking them per pair is required regardless of render count. All-cache rendering may be very expensive; estimate disk/time and use resumable batching, not hidden top-K selection.

Use non-executable numeric storage for new caches, e.g. `.npy/.npz` with `allow_pickle=False`, strings/numbers in JSON, partitioned score tables. Existing pickle vectors may be read only after inspecting globals, using a restricted loader allowing the known NumPy reconstruction functions and verifying dict/array/dtype/shape. Do not load arbitrary pickle or TorchScript models merely to inspect coefficients. Our filter audit read ZIP serialization metadata and raw double storage, without executing archive code.

Cache keys must include raw image hash, preprocessing variant/settings/location, method/upstream revision, weights hashes, geometry policy, bank hash, resolved config and mask hash/source. Same image ID with a different preprocessing/model/config must not hit the raw-vector cache.

## 6. Task 1: save segmentation and correct overlays

### What mask means relative to image

A circle locates pupil/iris geometry. It is **not** an occlusion segmentation mask. The dataset mask represents visible/eligible iris pixels (confirm actual encoding when files arrive). HDBIF predicts a usable-region mask; `255` is valid and `0` invalid in current upstream. The normalized polar mask must use exactly the same coordinate transform/circles as its polar image. ArcIris baseline uses predicted circles and feeds an unmasked polar image to its encoder; external mask coverage is diagnostic unless deliberately testing a separate masking variant. HDBIF uses the intersection of the two rotated masks, cropped radially to match the selected filter response shape, to decide which bits contribute.

Original image dimensions may differ. Both current upstream methods `fix_image` to ISO `(640,480)` by padding to 4:3 and resizing. For a 960x640 source, this pads vertically toward 960x720, then resizes. Image padding uses gray 127, **mask padding must be zero**. Applying generic `fix_image` to a mask can introduce gray valid regions and default interpolation can create intermediate labels. Independently resizing a mask to 640x480 without the same padding geometry also shifts alignment.

Implement one recorded transform with original dimensions, integer pad offsets, padded dimensions and output dimensions. Use bilinear/image-policy interpolation for pixels, nearest-neighbor for categorical masks. Validate raw mask width/height matches its source; don't stretch unknown mismatches to force success. If mapping is unknown, record `mask_alignment_unknown`. Preserve multiclass masks and specify which classes are eligible; do not assume all nonzero classes are iris without checking.

Save per image:

1. Original mask unchanged and its hash, when available.
2. Fixed-frame binary usable mask, original/fixed image dimensions and transform JSON.
3. ArcIris pupil/iris circles and quality status; HDBIF circles separately when different.
4. Polar image and polar usable mask for each geometry/method policy, expected 64x512 in current config.
5. Model-mask prediction when dataset masks are absent but image+mask model available, labeled `predicted`, not ground truth.
6. Visible fraction, valid-pixel count, circle containment/radius checks, out-of-bounds coverage.

Never draw a polar mask directly over a Cartesian image. Use Cartesian fixed-frame mask there; separately show polar image+polar mask. To overlay on original pixels, inverse the recorded transform with correct nearest-neighbor mask mapping and circle coordinate scaling/offsets.

### Actual upstream call sequences

The following contracts were verified from current official upstream. Confirm against the pinned local version before execution:

```python
# HDBIF modules.irisRecognition.irisRecognition(cfg)
fixed = rec.fix_image(raw_gray_pil)
mask_fixed, pupil, iris = rec.segment_and_circApprox(fixed)
polar, polar_mask = rec.cartToPol_torch(fixed, mask_fixed, pupil, iris)
codes = rec.extractCode(polar)  # LIST: one ndarray per configured patch bank
score, shift = rec.matchCodesEfficient(codes_a, codes_b, mask_a, mask_b)

# ArcIris has a DIFFERENT interface under the same modules package name.
fixed = arc.fix_image(raw_gray_pil)
pupil, iris = arc.circApprox(fixed)
quality_ok = arc.checkQuality(pupil, iris)
polar = arc.cartToPol_torch(fixed, pupil, iris)
vector = arc.extractVector(polar)
```

Prevent `modules.irisRecognition` import collisions: use isolated subprocess adapters or proper package namespaces; merely switching `sys.path` can reuse the wrong module in `sys.modules`. Historical five-kernel wrapper has exactly this style of brittle integration. Implement tests proving each adapter loads its intended module.

`matchCodesEfficient` arguments are `(codes1,codes2,mask1,mask2)`, no `rotation_range` keyword in current upstream. Set `recog_max_shift` in config. Return is `(score,shift)`, not a scalar. It searches even shifts then neighboring odd shifts; at endpoint winners this can evaluate one pixel beyond configured max shift. Preserve baseline behavior and record it, or introduce an explicitly named bounded/exhaustive ablation, never silently change it. Other upstream match functions may return `-1` sentinels; normalize all invalid/nonfinite outputs to failure.

To diagnostic-unwrap a dataset mask with ArcIris geometry, use its `cartToPol_torch` with `interpolation='nearest'` and the exact image circles, then threshold explicitly. Preserve upstream coordinate conventions (including radius grid from 1/H to 1 and its grid-sample correction), don't replace the unwrap with a visually similar mapping without parity tests.

### Rendering requirements

Every requested pair has labeled panels: image A and B fixed-frame originals, usable-mask overlays and pupil/iris circles, polar images with invalid pixels marked, and HDBIF selected-shift overlap/cropped mask where available. Show label, IDs, sensor/session, score units and direction, actual bank, preprocessing, quality/failure and number of compared bits. Use consistent RGB/BGR handling. Render missing evidence as clearly unavailable, not a black image presented as a real capture.

Read artifacts once per image, no re-segmentation per pair. Save high-fidelity masks (PNG/binary arrays), overlays and pair index. Run rendering in chunks with resumable index and checksummed inputs; separate `--render-all` from `--render-top-k` and label diagnostic subsets. Completion requires all requested pairs represented in index, including failures, with requested available-image overlays actually generated.

## 7. Task 2: paired plot and non-linear separation

### First recover existing full-run evidence

Look locally for the explicitly ignored `Week 2/hdbif_full_run/scores/hdbif_kernel_scores.csv`, upstream source/config, HDBIF code/mask cache, splitter and separator scripts/models. Read actual column definitions and model provenance. Logs alone cannot identify kernel score columns or trained boundaries.

If per-bank raw scores are missing but images/weights exist, recompute them. If only tracked prediction outputs exist, make the **exploratory** scatter using negated `arciris` and `hdbif_best`, verifying ordered labels and ArcIris-score equality against test rows before annotating any pair. Mark chosen HDBIF bank unknown. Do not fit a deployable boundary from already transformed fused outputs or from unverified row joins.

### Proper pairwise experiment

Join canonical pair IDs with raw ArcIris angular distances and HDBIF bank distances, one-to-one. Assert labels and split consistent. Report missing/failure counts by method and class; compare on a common valid pair set **and** separately report end-to-end failure coverage. Features for the core two-dimensional plot are only `[arciris_rad, hdbif_distance]`, not subject ID/sensor/label/source. Color genuine/impostor, show density and low-distance corner, give support counts. Avoid plots that hide extremes by dense point overlap.

Split by **persons across both pair endpoints** before fitting: train/validation/test subjects disjoint, both eyes and all captures stay together. Build within-partition pairs only; exclude cross pairs. A naive GroupKFold using just subject A leaks subject B. Existing train/test partition is clean; subdivide the 144 training persons into train/validation without ever examining test to pick settings. Keep original 36-person test untouched; now that test outputs have been explored, acknowledge they are reused benchmark data and seek fresh identities for final scientific confirmation.

Compare:

- ArcIris alone and each HDBIF bank alone.
- StandardScaler -> logistic regression on the two raw features (linear boundary).
- StandardScaler -> RBF SVC on same two features (non-linear); bounded training subset if needed, no millions-pair kernel matrix. Search `C=[0.1,1,10]`, `gamma=['scale',0.1,1]` using validation persons, balanced class weights with explicit sampling weights if distributions changed.
- Optional HistGradientBoosting on the same two inputs as a second non-linear model. Add quality/sensor only as a **separate** experiment, not a hidden change in the plotted classifier.

Choose bank and hyperparameters on training/validation, not on test. Fit scaler only on training. Threshold margins using validation at predeclared FMR targets (1%, 0.1%, 0.01% where support permits). Calibration, if probabilities are required, uses a separate validation subset with genuine operational priors, not class-balanced fit priors. Report confidence intervals clustered by persons/pair endpoints, not naive IID millions-pair certainty. If zero false matches, report sample denominator and a cautious upper bound; comparisons sharing images are dependent. These are verification FMR/FNMR, not IREX identification FPIR/FNIR.

Plot actual learned two-input boundary over a mesh, legend and threshold contour; label validation-selected threshold, and show separate untouched test scatter. For nonlinear model performance to count as improvement, require lower test FNMR at fixed validation-calibrated FMR, stable by sensor/session, no identity leakage and a useful tail rescue pattern. A curved line in a scatter or higher AUC alone is insufficient. Overlap may remain; do not promise perfect separation.

Save fitted model/scaler with version metadata, numeric predictions including pair IDs, train/validation/test subject lists, ROC/DET and low-FMR zoom, calibration curves if used, metrics with counts and per-group results. Record exactly what produced old `hdbif_fusion`, `arciris_hdbif` and `boosted_all` if recovered; otherwise leave definitions unknown.

## 8. Task 3: specific kernels, variance and ignored bits

### Verified inventory

288 files = 3 families x 12 sizes x 8 bit depths:

- families: `finetuned_bsif_eyetracker_data`, `finetuned_bsif_random_iris_patches`, `finetuned_bsif_user_annotations`;
- sizes: `5,7,9,11,13,15,17,19,21,27,33,39`;
- bits/filter counts per bank: `5,6,7,8,9,10,11,12`.

`ICAtextureFilters_KxK_Bbit.pt` is a bank containing B learned filters of spatial shape KxK, not one B-bit score and not B independent bank configurations. Official loader reads TorchScript attribute `ICAtextureFilters`, stored shape `(K,K,B)` and moves to convolution shape `(B,1,K,K)`. Response sign yields code `(B,64-K+1,512)`, with circular angular padding and no radial padding. Example 17x17/5-bit response shape `(5,48,512)`. For 39x39 it has only 26 radial rows, so compared support differs substantially.

All 288 coefficient arrays were inspected as raw numeric storage: finite, unique storage hashes, every individual filter L2 norm within `8.6e-12` of 1. This validates archive integrity, not recognition quality. Actual kernels can be displayed as coefficient heatmaps; one eye-tracker 17x17/5-bit filter was visually checked.

Current upstream HDBIF cfg selects **one eye-tracker 17x17/5-bit bank**, `recog_max_shift=16`, `score_norm=False`, `threshold_frac_avg_bits=0`, polar 64x512. Historical subset log/docs are consistent with this baseline. The broken five-bank wrapper proposes sizes 5/7/9/11/13 at 10 bits, but it is not evidence those ran. Final `match.log` lists **all twelve sizes**, so don't describe the final run as five kernels; bit depth/family unknown until config or raw-cache metadata recovered. Do not claim Daugman normalization was used: current default is false and historical prose conflicts with that.

Inventory actual selected bank paths and hashes at runtime. Save full config rather than infer selection from filenames present on disk. Keep a short smoke subset and one fixed pair set across banks before scaling. Encode segmentation/polar once per geometry/preprocessing policy, then derive each bank's codes; do not repeat segmentation for every bank.

### Kernel analysis

For each bank on the identical valid pair set, output genuine/impostor mean, sample SD/variance, robust SD `IQR/1.349` and its square, skew/kurtosis, tail quantiles, d' with consistent positive separation convention, FMR/FNMR at validation cuts, compared support and failures. Do not rank by smallest impostor variance alone: a constant score has zero variance and no useful matching. Select by low-FMR genuine recovery and confidence intervals; keep variance as diagnostic.

Bit analysis must separate:

1. Excluded because mask/overlap is invalid ("ignored").
2. Eligible but code nearly always 0 or always 1 (low entropy).
3. Eligible and contributing genuine/impostor disagreement.

Count per `(filter,row,angle)` eligible support before probability of one. `H(p)=-p log2 p -(1-p)log2(1-p)` per position, explicit 0/1 limits, then aggregate eligible positions. A frequent zero is not ignored. Apply the **selected pair shift** to both code and mask B, crop mask to the correct response radial support, and count eligible mismatches for genuine/impostor separately. Save participation maps and difference maps. Synthetic tests: all-valid mask, all-zero mask, partial overlap, wraparound shifts, each bank radial crop, negative/infinite sentinel, matchCodes vs matchCodesEfficient parity where their search semantics coincide. Avoid tautological bank success with zero encoded images.

## 9. Task 4: CLAHE on the same genuine/impostor pairs

CLAHE changes image pixels. It cannot be applied to embedding distances, output CSVs, masks or cached vectors to infer new biometric scores. Existing `histogram_normalization.py` measures brightness/contrast, not recognition improvement. Implement real re-encoding and matching for both methods.

Primary controlled arm: estimate/save raw-image geometry and masks once; apply CLAHE to the image while holding those masks/circles fixed, unwrap with the baseline map, re-encode, match the **exact same canonical pairs**. Record CLAHE location (Cartesian fixed image or polar). Start with Cartesian fixed-frame intensity CLAHE `clipLimit=2.0`, `tileGridSize=(8,8)`, matching old intended settings. OpenCV tileGridSize is the number of tiles in each axis, not patch pixels. Input uint8 gray; don't repeatedly quantize or apply twice. Do not enhance padding as if it were real iris: record padded regions, and use an explicitly documented padding policy shared across arms. Save raw vs processed examples.

Second arm: CLAHE before segmentation, allowing new predicted geometry/masks, to measure end-to-end pipeline effect. Never conflate texture improvement with geometry changes. Optional polar-only CLAHE is a separate ablation; ordinary CLAHE has seam issues around the angular wrap, so don't silently treat it as equivalent to Cartesian CLAHE. Never CLAHE the binary segmentation mask.

Pilot with a person-stratified training/validation subset, fixed pairs and two methods. If tuning is justified, try clip `[1,2,4]`, grid `[(4,4),(8,8)]` on validation only, cap cost and choose once. Preserve raw baseline and cache isolation. When fresh raw inference differs from tracked raw vectors, reconcile weights/upstream/pixel/geometry provenance before attributing the difference to CLAHE.

For each matched pair report raw distance, CLAHE distance, delta, label, split, method/bank, status, support and whether baseline false match or false non-match was rescued/harmed. Compare FMR and FNMR at **fixed raw validation threshold** to measure calibration drift, then at **separately validation-calibrated same target FMR** for each variant to measure achievable operating-point benefit. Both are needed. Keep failures in coverage tables, no favorable survivor-only comparison. Plot paired deltas by class, ROC/DET, histograms/tail ECDF and sensor/quality strata. Bootstrap by identity, including dependence between pair endpoints. Run nonlinear fusion comparison again only if first-arm results and budget justify it.

## 10. Experiments to learn why false matches happen

Priority order, with hypotheses separated from established results:

| Experiment | Concrete action | Inputs | Decision it informs |
|---|---|---|---|
| A. Label and gallery integrity | Re-score every gallery pair from saved vector IDs; compare labels to metadata, flag unknown IDs, fix filtered-index bug, inspect extreme images when present; hash raw images for exact duplicates and check same capture aliases. | Cache+metadata now, pixels later | Whether tail/gallery is an accounting problem before altering a model. Worst cached pair is genuine-labeled different persons, but labels/pixels need review. |
| B. Exhaustive vs sampled tail | Recompute all vector comparisons in blocks, track worst K correct IDs and per-image/identity hub degrees; show full ECDF and sampled minima; normalize hub counts by available impostor comparisons. | Tracked vectors+metadata | Random sample misses worst outliers. Avoid selection-biased evidence. Do not call worst 1,000 "the 1%": true 1% is ~132,858 pairs. |
| C. Capture-group attribution | Stratify sensor-pair, session-gap and measured format groups; estimate FMR/FNMR at one validation cut and per-group validation cuts; use counts/CIs. Hold out sensors where feasible. | Metadata now; dimensions/masks later | Confirm wide-sensor shift, mixed genuine degradation; test other sensor/quality hubs, not only resolution. No causal claim from proxies. |
| D. Occlusion and geometry | Compare predicted vs dataset masks on audited images, visible fraction, circle area/radius, out-of-bounds polar support; pair errors by max occlusion and overlap; save worst false matches and false non-matches. | Images+mask+model | Does missing texture cause false accepts, false rejects, or both? Existing old claims about occlusion were not reproduced from absent masks. |
| E. HDBIF complementarity | Compare which exact false accepts/rejects ArcIris vs each HDBIF bank makes; evaluate linear then RBF fusion on fixed pairs and subject splits. | Local per-bank scores or fresh inference | Native mask-aware method can reject distinct mistakes; avoid naive AND rule's genuine loss. |
| F. Bank support and bit entropy | Same pair set for each runtime bank; support-aware entropy, disagreement and overlap; normalization off/on as explicitly named arm. | Polar/masks/codes | Bank-size variance differences might come from response support or rotations, not better feature learning. |
| G. CLAHE controlled/end-to-end | Two arms above; same image IDs/pairs and isolated caches; report rescued/harmed pairs and genuine cost. | Images+methods | Is brightness/contrast truly limiting, or does preprocessing shift learned encoder distribution? |
| H. Resolution vs sensor control | On validation subjects, downsample selected standard captures to the measured effective iris size of wide captures, reconstruct to same fixed frame, hold geometry where appropriate, then re-embed. Compare sensor groups matched on quality/iris radius; preserve originals. | Images+dimensions+circle data | Distinguish spatial resolution from acquisition/device signatures. Synthetic resize is a controlled perturbation, not proof of real sensor causality. |
| I. Embedding geometry/hubs | Check finite/norm/duplicates, eigen spectrum/effective rank and hub concentration. Optionally fit centered/whitened or cohort-normalized scores using training vectors only, choose validation settings, test untouched. | Vectors+metadata | Does common-direction/anisotropy contribute to tail? Effective rank ~142 is a lead, not a collapse verdict. No test-fitted normalization. |
| J. Retraining only if evidence supports | After fixes and ablations, propose hard-impostor mining from **training subjects only**, augmentation matching measured degradations, mask-aware auxiliary objective or fine-tuning with frozen baseline control. New held-out people for confirmation. | Original training pipeline, approved model/data access, compute | Changes to learned representation should be justified by diagnostics, not used to hide labeling/selection bugs. Do not silently replace research model. |

### False-match interpretation rules

A long impostor left tail is harmful because smaller distance means a false candidate looks genuine. Its size alone is not "wrong": percentile selection deliberately includes a fraction of impostors. Evaluate wrong accepts at an operating threshold and whether genuine recall suffers, not merely tail length on a histogram. Mixed acquisition populations, extreme outliers, model distribution shift, incorrect IDs, mislabeled data and insufficient overlap are distinct mechanisms. Missing masks cannot be inferred from vectors. Tightening a threshold reduces FMR but increases FNMR; goal is better real matching at a stated risk, not deleting hard cases from evaluation.

Do not extrapolate sample FMR to large identification galleries without a supported identification experiment. IID formula `1-(1-p)^N` is a conceptual multiple-comparison warning, not a reliable estimate under shared-template dependence. Verification and identification metrics remain separate.

## 11. Implementation order and exact success checks

Implement these proposed CLI commands; they **do not exist in current main**. `python -m iris_audit` should work from `Week 3/` once package is created. Every command reads a config JSON containing repo-root paths and writes to a run directory, supports `--resume`, and validates hashes before resume.

```bash
cd 'Week 3'
python -m iris_audit preflight --config config.json --cache-only
python -m iris_audit audit-cache --config config.json
python -m iris_audit inventory-kernels --config config.json
python -m iris_audit build-pairs --config config.json
python -m iris_audit segment --config config.json
python -m iris_audit score --config config.json --variant raw
python -m iris_audit render --config config.json --all
python -m iris_audit score --config config.json --variant clahe-fixed-geometry
python -m iris_audit score --config config.json --variant clahe-end-to-end
python -m iris_audit evaluate --config config.json
python -m iris_audit fit-fusion --config config.json
python -m iris_audit report --config config.json
python -m pytest tests -q
```

1. **Preflight and source recovery.** Preserve dirty work. Find actual source/manifest, raw/masks, upstream code/config, weights, bank scores/models. Record unavailable items. Normalize metadata, no guessing images from internet. Read-only cache mode must not require missing TIFFs.
2. **Integrity before model changes.** Build canonical pair IDs and split tables, verify current tracked counts, all cached vector IDs and labels. Fix gallery mapping and unknown-label handling. Add blockwise scoring and finite/status rules. Score-only audit should reproduce values in section 4 within rounding.
3. **Kernel inventory.** Confirm families/sizes/depths and actual selected config/hash, archive shapes and filters. Inventory can complete even when bank scores absent. Avoid reporting bank performance from archive integrity.
4. **Segmentation artifacts.** Validate geometry/mask source, categorical padding/interpolation and correct polar mapping. Save unique per-image artifacts. Complete index for all requested pairs; explicitly block pixels that don't exist. Generated masks are not ground-truth substitutes.
5. **Raw HDBIF scores.** Fixed pair set across selected banks, correct tuple/API, support/failure counts, native mask overlap. Reconcile historical logs with actual local artifacts.
6. **Paired plot and baseline metrics.** Raw distance two-feature join, correct signs. Save exploratory and validation/test plots separately. Recreate old predictor definitions only if actual code found.
7. **CLAHE.** Isolated caches, raw parity check, fixed-geometry primary arm and end-to-end secondary arm; same pairs. Add rescue/harm tables and validation-calibrated metrics.
8. **Fusion and experiments.** Subject-disjoint validation, linear baseline before nonlinear, no test hyperparameter selection, save actual contour and provenance. High-cost arms start on training pilot, scale only when useful.
9. **Scientific report.** Separate completed, blocked, descriptive and validated outcomes. Include unspecified fifth, sources, hashes, failures, coverage, runtime, plots and recommendations grounded in results. No commit/push unless approved.

### Required automated tests

- Identical/orthogonal/opposite vectors give 0/pi/2/pi; zero/nonfinite vector fails explicitly.
- Same subject/same normalized eye genuine; different eye impostor; unknown identity invalid. Labels preserved across scoring variants.
- Canonical unordered pair key invariant to endpoint order, no duplicates/self-pairs, ID-score mapping after any filtering/sorting.
- Metadata-only manifest works without images; image stage reports exact missing IDs and does not proceed with zero images as success.
- Mask transform alignment with a synthetic marked rectangle/circle through pad/resize/inverse; no gray padding is usable; values remain categorical. Image/polar mask correspondence and same geometry in fixed arm.
- Each HDBIF code bank has correct `(B,H-K+1,W)`, radial mask crop and shift alignment; invalid overlap cannot yield false low score.
- Adapter isolation: ArcIris and HDBIF resolve their intended modules and signatures.
- Raw and CLAHE caches differ; changing config/weights/mask/hash invalidates cache; resume rejects stale artifacts.
- Subject sets disjoint across train/validation/test for both pair endpoints; test cannot appear in fit/scaler/bank-selection/calibration stages.
- Recompute actual metrics/figure captions, no out-of-range subplot indices and no omitted failures. Render one genuine, one impostor, one mixed-format and one missing pair, visually inspect actual output.
- Save scores to partitioned numeric tables without needing full NxN distance + ID arrays simultaneously. The audit encountered a memory kill with large fancy-index copies and succeeded after blockwise reductions; test constrained memory behavior.

### Acceptance artifacts

`run_metadata.json`, `preflight.json`, `images.csv`, `pairs.csv`, `subject_splits.json`, `image_artifacts.csv`, per-image masks/geometry/polars, `pair_renders.csv`, full requested available pair overlays, `kernel_inventory.csv`, bank bit/variance tables, raw+CLAHE scores with exact pair IDs, validation/test metrics and coverage, genuine/impostor scatter with learned boundary, low-FMR ROC/DET, rescue/harm table, `report.md`, test results.

Done means files exist, input hashes/configs are recorded, numeric readbacks pass and actual plotted/rendered pixels were inspected. A stub that returns empty codes, preprocessing-only brightness plots, a generic curve drawn over scores or an absent pair gallery is not completion.

## 12. Minimal safe reproducer for cache-only scoring

Use a vetted vector loader first. This block takes already validated `embeddings` and metadata indexed by fileid. It avoids the old ID-score filtering error and expensive all-pair Python tuple lists:

```python
ids = np.array(sorted(embeddings), dtype=str)
X = np.stack([embeddings[i] for i in ids]).astype(np.float64)
norm = np.linalg.norm(X, axis=1)
assert np.isfinite(X).all() and (norm > 0).all()
X /= norm[:, None]
subject = metadata.loc[ids, 'subjectid'].to_numpy()
eye = metadata.loc[ids, 'eye'].map({'Left': 'L', 'Right': 'R'}).to_numpy()
assert all(isinstance(s, str) and s for s in subject)
assert np.isin(eye, ['L', 'R']).all()
for start in range(0, len(ids), 256):
    cosine = X[start:start + 256] @ X.T
    distances = np.arccos(np.clip(cosine, -1, 1))
    local_i, j = np.where(
        np.arange(start, start + len(distances))[:, None]
        < np.arange(len(ids))[None, :]
    )
    i = local_i + start
    score = distances[local_i, j]
    genuine = (subject[i] == subject[j]) & (eye[i] == eye[j])
    # Emit these EXACT i,j,score,genuine arrays together to a chunk writer.
    # Compute summaries/ranks in bounded storage. Never apply a score mask
    # without applying the same mask to both endpoint arrays.
```

Re-run on this snapshot: 70,876 genuine and 13,285,820 impostor, minimum impostor ~0.433921, historical-cut tail count 130,228. Exact score comparison tolerance accounts for saved float precision; use float64 for calculations, float32 only as deliberate result storage. Ties and rounding near thresholds require explicit comparison convention (`<=` means accept for distance).

## 13. Sources and evidentiary limits

Primary project and old PR: https://github.com/inezaodon/intro_to_iris_recognition and https://github.com/inezaodon/intro_to_iris_recognition/pull/1 . Repo snapshots and runtime artifact paths above are the exact internal source for code/numeric claims; calculations were freshly performed after final push.

Official upstream project/API/config: https://github.com/CVRL/OpenSourceIrisRecognition . Read HDBIF `Python/modules/irisRecognition.py`, `Python/cfg.yaml` and ArcIris `Python/modules/irisRecognition.py`, `Python/cfg_baseline.yaml` under their respective `methods` directories. Contracts are current upstream, not proof of Odon's prior runtime version.

- HDBIF primary paper lead (search result only; full-page fetch returned no content, not used for load-bearing claims): https://arxiv.org/html/1807.05248v2
- OpenCV CLAHE API (clip/grid semantics): https://docs.opencv.org/4.12.0/d6/db6/classcv_1_1CLAHE.html
- Grouped validation/leakage background: https://scikit-learn.org/stable/modules/cross_validation.html
- Calibration/background: https://scikit-learn.org/stable/modules/calibration.html
- RBF C/gamma guidance: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

Official upstream and the listed library docs were read during this audit; the paper fetch failed and remains a background lead only. Core findings rest on the project's exact tracked data and directly read code; general docs support proposed implementation. Missing local data/source/version, unspecified fifth and unavailable separator fit provenance remain explicit uncertainties. No newly trained model, CLAHE recognition result, original-mask overlay or corrected image gallery was produced tonight.

## Appendix: runnable cache-only audit

The following standalone script was run successfully against the audited snapshot and reproduced the full counts, 130,228 historical-cut impostor hits and worst pair. Save it as `cache_audit.py` outside tracked source, then run with explicit root/output paths. It needs NumPy only, reads the known vector serialization through a restricted loader and writes to your chosen output directory. It does not require TIFFs, masks, PyTorch or fresh model inference. It is a safe starting artifact, not implementation of the image-dependent tasks.

```bash
OPENBLAS_NUM_THREADS=2 python3 cache_audit.py \
  --repo-root /path/to/intro_to_iris_recognition \
  --out-dir /path/to/local/audit-results
```

```python
# Save as cache_audit.py; writes only the explicit output directory.
import argparse, csv, io, json, pickle
from collections import Counter
from pathlib import Path
import numpy as np

class VectorLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy._core.numeric', '_frombuffer'):
            return np._core.numeric._frombuffer
        if (module, name) == ('numpy', 'dtype'):
            return np.dtype
        raise pickle.UnpicklingError(f'Unexpected global {module}.{name}')

ap = argparse.ArgumentParser()
ap.add_argument('--repo-root', type=Path, required=True)
ap.add_argument('--out-dir', type=Path, required=True)
args = ap.parse_args()
root, out = args.repo_root.resolve(), args.out_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
vector_file = root / 'Week 1/Task 1 results/embeddings.pkl'
emb = VectorLoader(io.BytesIO(vector_file.read_bytes())).load()
assert isinstance(emb, dict) and all(
    isinstance(k, str) and isinstance(v, np.ndarray)
    and v.dtype.kind == 'f' and v.shape == (512,)
    for k, v in emb.items()
)
metadata_file = root / 'General Project Files/NDCVRL_002 Dataset/recording_metadata.csv'
with metadata_file.open(newline='') as f:
    records = list(csv.DictReader(f))
meta = {r['fileid']: r for r in records}
assert len(meta) == len(records), 'Duplicate metadata image IDs'
ids = np.array(sorted(emb), dtype=str)
assert set(ids) <= set(meta), 'Embedding IDs missing from metadata'
subject = np.array([meta[i]['subjectid'] for i in ids])
eye = np.array([meta[i]['eye'] for i in ids])
assert np.isin(eye, ['Left', 'Right']).all()
assert all(subject)
X = np.stack([emb[i] for i in ids]).astype(np.float64)
norm = np.linalg.norm(X, axis=1)
assert np.isfinite(X).all() and (norm > 0).all()
X /= norm[:, None]
N = len(ids)
# Store at most one float32 value per unordered pair on disk, not many copies.
capacity = N * (N - 1) // 2
score_mm = np.lib.format.open_memmap(out / 'angular_scores.npy',
    mode='w+', dtype=np.float32, shape=(capacity,))
label_mm = np.lib.format.open_memmap(out / 'genuine_mask.npy',
    mode='w+', dtype=bool, shape=(capacity,))
cut = 1.405748
count = Counter()
worst = []
degree = np.zeros(N, dtype=np.int64)
pos = 0
for start in range(0, N, 128):
    d = np.arccos(np.clip(X[start:start + 128] @ X.T, -1, 1))
    li, j = np.where(np.arange(start, start + len(d))[:, None]
                     < np.arange(N)[None, :])
    i = li + start
    scores = d[li, j]
    genuine = (subject[i] == subject[j]) & (eye[i] == eye[j])
    n = len(scores)
    score_mm[pos:pos+n] = scores
    label_mm[pos:pos+n] = genuine
    pos += n
    count['genuine'] += int(genuine.sum())
    count['impostor'] += int((~genuine).sum())
    count['false_match_at_historical_cut'] += int(((~genuine) & (scores <= cut)).sum())
    count['false_nonmatch_at_historical_cut'] += int((genuine & (scores > cut)).sum())
    tail = (~genuine) & (scores <= cut)
    np.add.at(degree, i[tail], 1)
    np.add.at(degree, j[tail], 1)
    eligible = np.where(~genuine)[0]
    selected = eligible[np.argsort(scores[eligible])[:20]]
    worst += [(float(scores[t]), str(ids[i[t]]), str(ids[j[t]])) for t in selected]
    worst = sorted(worst)[:20]
assert pos == capacity
score_mm.flush(); label_mm.flush()
result = dict(images=N, count=dict(count), historical_cut_rad=cut,
    fmr=count['false_match_at_historical_cut']/count['impostor'],
    fnmr=count['false_nonmatch_at_historical_cut']/count['genuine'],
    worst_impostors=[dict(score=s, image_id_a=a, image_id_b=b)
                      for s, a, b in worst],
    missing_vector_ids=sorted(set(meta)-set(ids)))
(out / 'cache_summary.json').write_text(json.dumps(result, indent=2))
with (out / 'tail_hubs.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'subject_id', 'eye', 'tail_degree'])
    for t in np.argsort(-degree):
        writer.writerow([ids[t], subject[t], eye[t], degree[t]])
print(json.dumps(result, indent=2))

```
