# ArcIris analysis (image_id through the pipeline)

Tracks a stable `image_id` from ingest → cached embedding → pairwise score →
impostor-tail report → pair-image filenames. When Adam's labeled set arrives,
change `--data-dir` / `--manifest` only.

Uses `irisRecognition` from the ArcIris clone. Does **not** copy the method
tree or retrain. Existing `templates/<image_id>_tmpl.npz` files are reused.

## Environment

```bash
micromamba activate arciris
# or: /Users/odon/mamba/envs/arciris/bin/python
```

Weights stay in `OpenSourceIrisRecognition/methods/ArcIris/Python/models/`.
Do not re-download them.

## How `image_id` is assigned

- **Folder ingest:** `image_id` = filename stem (made unique with a parent
  prefix if two files share a stem).
- **Adam's CSV:** use his `image_id` column; if missing, fall back to the stem.
- That same string is the key in `embeddings.pkl`, both sides of
  `pairwise_scores.csv`, the tail tracker, and render names
  `{id_a}__{id_b}__score-{x.xxx}.png`.

**Genuine pair** = same non-empty `subject_id` **and** same eye (`L` or `R`).
Left ≠ right even for one person. Blank / `unknown` subject or eye → every
pair involving that image is **impostor** (the three bundled synthetics).

## How the impostor tail is defined

Lower score = more similar (`acos` of cosine similarity). The **left** tail
is the low-score impostor pairs.

| situation | cut |
|---|---|
| `--tail-threshold` set | that score (radians). Not the guide's placeholder 0.35. |
| fewer than 10 impostor pairs | `score <= median` of *impostor* scores (the n=3 synthetic case) |
| otherwise | `score <=` the `--tail-percentile` of *impostor* scores (default **5th**) |

Query: *which images appeared in the low-score impostor tail?* → every
`image_id` in `Task 1 results/impostor_tail_images.csv`. Cross-check
`quality_passed` if a hit might be a segmentation failure.

## Synthetics dry-run (already run once)

```bash
/Users/odon/mamba/envs/arciris/bin/python \
  "Week 1/arciris_analysis/run_pipeline.py" \
  --data-dir "Week 1/OpenSourceIrisRecognition/methods/ArcIris/Python/data" \
  --out-dir "Week 1/Task 1 results"
```

Three files (`synthetic1/2/3.png`): `image_id` = `subject_id` = stem, `eye` =
unknown → 3 impostor pairs, 0 genuine. Tail = scores at or below the median.

## When Adam's dataset arrives

1. Put images in a folder (or keep his folder layout).
2. If he sends labels, save them as a CSV with the columns in `schema.md`
   (`image_id,filepath,subject_id,eye,session`).
3. Run:

```bash
/Users/odon/mamba/envs/arciris/bin/python \
  "Week 1/arciris_analysis/run_pipeline.py" \
  --data-dir "/path/to/adam/images" \
  --manifest "/path/to/adam/manifest.csv" \
  --out-dir "Week 1/Task 1 results"
```

Omit `--manifest` to scan `--data-dir` and parse `L`/`R` out of filenames
when possible. For a large set the pipeline keeps **all** genuine pairs and
samples impostor pairs (`--max-impostor-pairs`, default 100000).

Outputs land in `Week 1/Task 1 results/`:

| file | what |
|---|---|
| `manifest.csv` | ids used for the run |
| `quality_flags.csv` | `checkQuality` per `image_id` |
| `embeddings.pkl` | vectors keyed by `image_id` |
| `pairwise_scores.csv` | every scored pair |
| `impostor_tail_pairs.csv` | impostor pairs on the tail |
| `impostor_tail_images.csv` | **image-level tracker** |
| `impostor_tail_summary.json` | threshold + rule used |
| `impostor_images/` | tail-pair renders |

See `schema.md` for column definitions.
