# Manifest and score schemas

All tables are UTF-8 CSV with a header row. `image_id` is the join key
from ingest through tail reports and pair-render filenames.

## `manifest.csv`

| column | required | notes |
|---|---|---|
| `image_id` | yes | Unique, stable. Filename stem is fine if unique in the run. |
| `filepath` | yes | Absolute or relative to `--data-dir`. |
| `subject_id` | yes | Blank if unknown. Same value means same person. |
| `eye` | yes | `L`, `R`, or `unknown`. Left ≠ right even for the same person. |
| `session` | no | Capture / visit id if Adam provides one. |

Adam can send this CSV as-is. If he only sends a folder, the pipeline
builds a manifest (see `manifest.py`). Unparseable subject/eye stay
blank/`unknown`; those images still get an `image_id` and every pair
involving them is labeled **impostor**.

## `quality_flags.csv`

| column | notes |
|---|---|
| `image_id` | |
| `quality_passed` | `True` / `False` from ArcIris `checkQuality` |
| `source` | `cache` if loaded from `templates/<image_id>_tmpl.npz`, else `extract` |

## `pairwise_scores.csv`

One row per compared pair. Lower `score` = more similar
(`acos` of cosine similarity, radians).

| column | notes |
|---|---|
| `image_id_a`, `image_id_b` | Sorted so `a < b` lexicographically |
| `subject_a`, `subject_b` | From the manifest |
| `eye_a`, `eye_b` | |
| `label` | `genuine` or `impostor` |
| `score` | ArcIris match score |

**Genuine** = same non-empty `subject_id` **and** same eye in `{L, R}`.
Anything else (unknown eye, blank subject, left vs right) is **impostor**.

## `impostor_tail_pairs.csv`

Impostor rows from `pairwise_scores.csv` with `score <= threshold`.
Same columns as pairwise, plus `threshold` and `tail_rule`.

## `impostor_tail_images.csv`

One row per `image_id` that appears in at least one tail pair.

| column | notes |
|---|---|
| `image_id` | |
| `n_tail_pairs` | How many tail pairs this image is in |
| `min_tail_score` | Lowest tail-pair score for this image |
| `partner_image_ids` | `;`-separated partner ids, lowest score first |
| `filepath` | |
| `subject_id` | |
| `eye` | |
| `quality_passed` | From `quality_flags.csv` if present |

Query: *which images fell on the left (low-score) tail of the impostor
distribution?* → every row of this file.
