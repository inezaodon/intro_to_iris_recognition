#!/usr/bin/env python3
"""Build a manifest.csv for NDCVRL_002 from recording_metadata.csv + raw/ layout."""

import argparse
import csv
from pathlib import Path

WEEK1 = Path(__file__).resolve().parent.parent
DATASET = WEEK1 / "NDCVRL_002 Dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=WEEK1 / "Task 1 results" / "ndcvrl_manifest.csv")
    args = ap.parse_args()

    files = {p.stem: p for p in (args.dataset / "raw").rglob("*.tif*")}
    rows, missing = [], 0
    with open(args.dataset / "recording_metadata.csv", newline="") as f:
        for r in csv.DictReader(f):
            p = files.get(r["fileid"])
            if p is None:
                missing += 1
                continue
            rows.append({
                "image_id": r["fileid"],
                "filepath": str(p),
                "subject_id": r["subjectid"],
                "eye": {"Left": "L", "Right": "R"}.get(r["eye"], "unknown"),
                "session": r["date"].split(" ")[0],
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "filepath", "subject_id", "eye", "session"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({missing} metadata rows without an image, "
          f"{len(files) - len(rows)} images without metadata) -> {args.out}")


if __name__ == "__main__":
    main()
