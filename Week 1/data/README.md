# Adam’s iris image set (drop here)

Put the labeled images (and optional `manifest.csv`) in this folder, then run:

```bash
python3 "Week 1/arciris_analysis/run_pipeline.py" \
  --data-dir "Week 1/data" \
  --manifest "Week 1/data/manifest.csv" \
  --out-dir "Week 1/Task 1 results"
```

Omit `--manifest` if subject/eye can be parsed from filenames or folder layout
(see `Week 1/arciris_analysis/schema.md`).

Do **not** commit raw iris images to git.
