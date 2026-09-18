"""Genuine vs impostor score distributions and decidability (d')."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decidability(genuine: np.ndarray, impostor: np.ndarray) -> float | None:
    """d' = |μ_g − μ_i| / sqrt(0.5 (σ_g² + σ_i²)). None if either side is empty."""
    if genuine.size == 0 or impostor.size == 0:
        return None
    denom = float(np.sqrt(0.5 * (genuine.var(ddof=0) + impostor.var(ddof=0))))
    if denom == 0:
        return None
    return float(abs(genuine.mean() - impostor.mean()) / denom)


def score_arrays(pairwise: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray(
        [
            float(row["_score"] if "_score" in row else row["score"])
            for row in pairwise
            if row.get("label") == "genuine"
        ],
        dtype=float,
    )
    impostor = np.asarray(
        [
            float(row["_score"] if "_score" in row else row["score"])
            for row in pairwise
            if row.get("label") == "impostor"
        ],
        dtype=float,
    )
    return genuine, impostor


def write_distribution_plot(
    pairwise: list[dict],
    out_path: Path,
    threshold: float | None = None,
    title: str = "Genuine vs Impostor Score Distributions",
) -> dict:
    """Histogram overlay. Lower score = more similar (ArcIris acos)."""
    genuine, impostor = score_arrays(pairwise)
    d_prime = decidability(genuine, impostor)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = 40
    if genuine.size:
        ax.hist(
            genuine,
            bins=bins,
            alpha=0.55,
            density=True,
            label=f"Genuine (n={genuine.size})",
            color="tab:green",
        )
    if impostor.size:
        ax.hist(
            impostor,
            bins=bins,
            alpha=0.55,
            density=True,
            label=f"Impostor (n={impostor.size})",
            color="tab:red",
        )
    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Tail / accept cut @ {threshold:.3f}",
        )

    ax.set_xlabel("ArcIris match score (acos cosine) — lower = more similar")
    ax.set_ylabel("Density")
    d_txt = f"{d_prime:.3f}" if d_prime is not None else "n/a (need both classes)"
    ax.set_title(f"{title}\nd′ = {d_txt}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    stats = {
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "genuine_mean": float(genuine.mean()) if genuine.size else None,
        "genuine_std": float(genuine.std(ddof=0)) if genuine.size else None,
        "impostor_mean": float(impostor.mean()) if impostor.size else None,
        "impostor_std": float(impostor.std(ddof=0)) if impostor.size else None,
        "decidability_d_prime": d_prime,
        "threshold": threshold,
        "plot": str(out_path),
    }
    return stats


def write_distribution_outputs(
    pairwise: list[dict],
    out_dir: Path,
    threshold: float | None = None,
) -> dict:
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    stats = write_distribution_plot(
        pairwise,
        plots_dir / "genuine_impostor_distribution.png",
        threshold=threshold,
    )
    (out_dir / "distribution_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats
