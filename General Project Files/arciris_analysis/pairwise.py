"""All-vs-all (or sampled-impostor) scores, labeled from the manifest."""

from __future__ import annotations

import itertools
import math
import random
from typing import Iterable

import numpy as np

from manifest import normalize_eye, normalize_subject, write_csv

PAIRWISE_FIELDS = [
    "image_id_a",
    "image_id_b",
    "subject_a",
    "subject_b",
    "eye_a",
    "eye_b",
    "label",
    "score",
]


def match_vectors(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Same formula as irisRecognition.matchVectors: acos(cosine). Lower = more similar."""
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0:
        return float("nan")
    cosine = float(np.dot(vector_a, vector_b) / denom)
    cosine = min(1.0, max(-1.0, cosine))
    return math.acos(cosine)


def is_genuine(row_a: dict[str, str], row_b: dict[str, str]) -> bool:
    subject_a = normalize_subject(row_a.get("subject_id"))
    subject_b = normalize_subject(row_b.get("subject_id"))
    eye_a = normalize_eye(row_a.get("eye"))
    eye_b = normalize_eye(row_b.get("eye"))
    if not subject_a or not subject_b or subject_a != subject_b:
        return False
    if eye_a not in {"L", "R"} or eye_b not in {"L", "R"}:
        return False
    return eye_a == eye_b


def _pair_record(
    id_a: str,
    id_b: str,
    by_id: dict[str, dict[str, str]],
    embeddings: dict[str, np.ndarray],
) -> dict:
    if id_a > id_b:
        id_a, id_b = id_b, id_a
    row_a = by_id[id_a]
    row_b = by_id[id_b]
    label = "genuine" if is_genuine(row_a, row_b) else "impostor"
    score = match_vectors(embeddings[id_a], embeddings[id_b])
    return {
        "image_id_a": id_a,
        "image_id_b": id_b,
        "subject_a": row_a.get("subject_id", ""),
        "subject_b": row_b.get("subject_id", ""),
        "eye_a": row_a.get("eye", ""),
        "eye_b": row_b.get("eye", ""),
        "label": label,
        "score": f"{score:.6f}",
        "_score": score,
    }


def pairwise_scores(
    embeddings: dict[str, np.ndarray],
    manifest: list[dict[str, str]],
    max_impostor_pairs: int | None = 100_000,
    seed: int = 0,
) -> list[dict]:
    by_id = {row["image_id"]: row for row in manifest}
    usable = [image_id for image_id in embeddings if image_id in by_id]
    usable.sort()
    if len(usable) < 2:
        raise ValueError("Need at least two embeddings that appear in the manifest.")

    genuine_ids: list[tuple[str, str]] = []
    impostor_ids: list[tuple[str, str]] = []
    for id_a, id_b in itertools.combinations(usable, 2):
        if is_genuine(by_id[id_a], by_id[id_b]):
            genuine_ids.append((id_a, id_b))
        else:
            impostor_ids.append((id_a, id_b))

    chosen_impostor = impostor_ids
    if max_impostor_pairs is not None and len(impostor_ids) > max_impostor_pairs:
        rng = random.Random(seed)
        chosen_impostor = rng.sample(impostor_ids, max_impostor_pairs)
        chosen_impostor.sort()
        print(
            f"Sampled {max_impostor_pairs} of {len(impostor_ids)} impostor pairs "
            f"(kept all {len(genuine_ids)} genuine pairs)."
        )

    records = [
        _pair_record(id_a, id_b, by_id, embeddings)
        for id_a, id_b in itertools.chain(genuine_ids, chosen_impostor)
    ]
    records.sort(key=lambda row: (row["_score"], row["image_id_a"], row["image_id_b"]))
    return records


def write_pairwise_scores(records: Iterable[dict], path) -> None:
    write_csv(path, list(records), PAIRWISE_FIELDS)
