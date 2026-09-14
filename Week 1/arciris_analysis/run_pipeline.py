#!/usr/bin/env python3
"""Carry image_id through ArcIris: ingest → embedding → pair score → impostor tail."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent
ARCIRIS_PYTHON = WEEK1 / "OpenSourceIrisRecognition" / "methods" / "ArcIris" / "Python"
DEFAULT_DATA = ARCIRIS_PYTHON / "data"
DEFAULT_TEMPLATES = ARCIRIS_PYTHON / "templates"
DEFAULT_OUT = WEEK1 / "Task 1 results"
DEFAULT_CFG = ARCIRIS_PYTHON / "cfg_baseline.yaml"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embeddings import load_or_extract, save_embeddings_pickle, write_quality_flags
from manifest import build_manifest_from_dir, index_by_id, load_manifest_csv, write_manifest
from pairwise import pairwise_scores, write_pairwise_scores
from render import render_tail_pairs
from tails import select_tail_pairs, image_tail_tracker, write_tail_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ArcIris embeddings once per image_id, score pairs, and "
            "list images that appear on the low-score impostor tail."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Folder of iris images (default: ArcIris Python/data).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV with image_id,filepath,subject_id,eye[,session]. Built from --data-dir if omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help='Output folder (default: "Week 1/Task 1 results").',
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=DEFAULT_TEMPLATES,
        help="Existing ArcIris templates/*_tmpl.npz cache.",
    )
    parser.add_argument("--cfg-path", type=Path, default=DEFAULT_CFG)
    parser.add_argument(
        "--arciris-python",
        type=Path,
        default=ARCIRIS_PYTHON,
        help="ArcIris Python/ directory (modules + models). Not copied.",
    )
    parser.add_argument(
        "--tail-percentile",
        type=float,
        default=5.0,
        help="Impostor-score percentile used as the tail cut when n >= 10 (default: 5).",
    )
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=None,
        help="Absolute score cut (radians). Overrides percentile / small-n median.",
    )
    parser.add_argument(
        "--max-impostor-pairs",
        type=int,
        default=100_000,
        help="Cap on impostor pairs when N is large. Genuine pairs are always kept in full.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Ignore cached templates and re-run circApprox / extractVector.",
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Do not write pair images under impostor_images/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest = load_manifest_csv(args.manifest, data_dir=args.data_dir)
        print(f"Loaded manifest {args.manifest} ({len(manifest)} images).")
    else:
        manifest = build_manifest_from_dir(args.data_dir)
        print(f"Built manifest from {args.data_dir} ({len(manifest)} images).")
    write_manifest(manifest, out_dir / "manifest.csv")

    embeddings, quality_rows, iris_rec = load_or_extract(
        manifest,
        templates_dir=args.templates_dir,
        arciris_python=args.arciris_python,
        cfg_path=args.cfg_path,
        reextract=args.reextract,
    )
    write_quality_flags(quality_rows, out_dir / "quality_flags.csv")
    save_embeddings_pickle(embeddings, out_dir / "embeddings.pkl")
    print(f"Embeddings: {len(embeddings)} / {len(manifest)} passed quality.")

    records = pairwise_scores(
        embeddings,
        manifest,
        max_impostor_pairs=args.max_impostor_pairs,
        seed=args.seed,
    )
    write_pairwise_scores(records, out_dir / "pairwise_scores.csv")
    n_gen = sum(1 for row in records if row["label"] == "genuine")
    n_imp = sum(1 for row in records if row["label"] == "impostor")
    print(f"Pairwise scores: {len(records)} rows ({n_gen} genuine, {n_imp} impostor).")

    tail_pairs, threshold, rule = select_tail_pairs(
        records,
        percentile=args.tail_percentile,
        threshold=args.tail_threshold,
    )
    quality_by_id = {row["image_id"]: row["quality_passed"] for row in quality_rows}
    tail_images = image_tail_tracker(tail_pairs, index_by_id(manifest), quality_by_id)
    write_tail_outputs(
        tail_pairs,
        tail_images,
        out_dir,
        threshold=threshold,
        rule=rule,
        n_impostor=n_imp,
        n_genuine=n_gen,
    )
    print(f"Impostor tail rule: {rule}")
    print(f"Tail pairs: {len(tail_pairs)}  |  tail images: {len(tail_images)}")
    for row in tail_images:
        print(
            f"  {row['image_id']}: n_tail_pairs={row['n_tail_pairs']} "
            f"min={row['min_tail_score']} partners={row['partner_image_ids']}"
        )

    if not args.skip_renders:
        print(f"Writing tail pair renders to {out_dir / 'impostor_images'}")
        render_tail_pairs(tail_pairs, index_by_id(manifest), out_dir, iris_rec=iris_rec)

    print(f"Done. Tracker CSV: {out_dir / 'impostor_tail_images.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
