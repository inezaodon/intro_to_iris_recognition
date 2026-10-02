"""Impostor-tail pairs and the image-level tracker keyed by image_id."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from manifest import write_csv
from pairwise import PAIRWISE_FIELDS

# Below this many impostor pairs, a 5th-percentile cut is too thin to be
# useful (n=3 synthetics → one or zero rows). Fall back to ≤ median so the
# tracker still returns images on the left half of the impostor scores.
SMALL_N_IMPOSTOR = 10

TAIL_PAIR_FIELDS = PAIRWISE_FIELDS + ["threshold", "tail_rule"]
TAIL_IMAGE_FIELDS = [
    "image_id",
    "n_tail_pairs",
    "min_tail_score",
    "partner_image_ids",
    "filepath",
    "subject_id",
    "eye",
    "quality_passed",
]


def choose_threshold(
    impostor_scores: list[float],
    percentile: float = 5.0,
    threshold: float | None = None,
) -> tuple[float, str]:
    """Return (threshold, rule_description).

    Priority:
    1. Explicit --tail-threshold (a score in radians, not the guide's 0.35).
    2. Tiny set (n < 10): score ≤ median of *impostor* scores (inclusive).
    3. Otherwise: the given percentile of *impostor* scores (default 5th).
    """
    if not impostor_scores:
        raise ValueError("No impostor scores; cannot define a tail.")
    scores = np.asarray(impostor_scores, dtype=float)
    n = int(scores.size)
    if threshold is not None:
        rule = f"score <= {threshold:.6f} (explicit --tail-threshold); n_impostor={n}"
        return float(threshold), rule
    if n < SMALL_N_IMPOSTOR:
        cut = float(np.median(scores))
        rule = (
            f"score <= median({cut:.6f}) of {n} impostor scores "
            f"(small-n fallback; n < {SMALL_N_IMPOSTOR}; "
            "a 5th-percentile cut would be unreliable)"
        )
        return cut, rule
    cut = float(np.percentile(scores, percentile))
    rule = (
        f"score <= {percentile:g}th percentile of impostor scores "
        f"({cut:.6f}); n_impostor={n}"
    )
    return cut, rule


def select_tail_pairs(
    pairwise: list[dict],
    percentile: float = 5.0,
    threshold: float | None = None,
) -> tuple[list[dict], float, str]:
    impostor = [row for row in pairwise if row.get("label") == "impostor"]
    scores = [float(row["_score"] if "_score" in row else row["score"]) for row in impostor]
    cut, rule = choose_threshold(scores, percentile=percentile, threshold=threshold)
    tails: list[dict] = []
    for row in impostor:
        score = float(row["_score"] if "_score" in row else row["score"])
        if score <= cut:
            out = {key: row.get(key, "") for key in PAIRWISE_FIELDS}
            out["score"] = f"{score:.6f}"
            out["_score"] = score
            out["threshold"] = f"{cut:.6f}"
            out["tail_rule"] = rule
            tails.append(out)
    tails.sort(key=lambda row: (row["_score"], row["image_id_a"], row["image_id_b"]))
    return tails, cut, rule


def image_tail_tracker(
    tail_pairs: list[dict],
    by_id: dict[str, dict[str, str]],
    quality_by_id: dict[str, str] | None = None,
) -> list[dict]:
    """One row per image_id that participated in at least one impostor-tail pair."""
    partners: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in tail_pairs:
        score = float(row["_score"] if "_score" in row else row["score"])
        a, b = row["image_id_a"], row["image_id_b"]
        partners[a].append((score, b))
        partners[b].append((score, a))

    rows: list[dict] = []
    for image_id, hits in partners.items():
        hits.sort()
        meta = by_id.get(image_id, {})
        quality = (quality_by_id or {}).get(image_id, "")
        rows.append(
            {
                "image_id": image_id,
                "n_tail_pairs": str(len(hits)),
                "min_tail_score": f"{hits[0][0]:.6f}",
                "partner_image_ids": ";".join(partner for _, partner in hits),
                "filepath": meta.get("filepath", ""),
                "subject_id": meta.get("subject_id", ""),
                "eye": meta.get("eye", ""),
                "quality_passed": quality,
            }
        )
    rows.sort(key=lambda row: (float(row["min_tail_score"]), -int(row["n_tail_pairs"]), row["image_id"]))
    return rows


def write_tail_outputs(
    tail_pairs: list[dict],
    tail_images: list[dict],
    out_dir: Path,
    threshold: float,
    rule: str,
    n_impostor: int,
    n_genuine: int,
) -> None:
    out_dir = Path(out_dir)
    write_csv(out_dir / "impostor_tail_pairs.csv", tail_pairs, TAIL_PAIR_FIELDS)
    write_csv(out_dir / "impostor_tail_images.csv", tail_images, TAIL_IMAGE_FIELDS)
    summary = {
        "threshold": threshold,
        "tail_rule": rule,
        "n_impostor_pairs": n_impostor,
        "n_genuine_pairs": n_genuine,
        "n_tail_pairs": len(tail_pairs),
        "n_tail_images": len(tail_images),
        "query": (
            "Images on the low-score (left) tail of the impostor distribution "
            "are the image_id rows in impostor_tail_images.csv."
        ),
    }
    (out_dir / "impostor_tail_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
