"""Extract embeddings once per image_id, reusing ArcIris templates when present."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manifest import write_csv

QUALITY_FIELDS = ["image_id", "quality_passed", "source"]


def add_arciris_to_path(arciris_python: Path) -> None:
    resolved = str(Path(arciris_python).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_cfg(arciris_python: Path, cfg_path: Path | None = None) -> dict[str, Any]:
    add_arciris_to_path(arciris_python)
    from modules.utils import get_cfg

    cfg_file = Path(cfg_path) if cfg_path else Path(arciris_python) / "cfg_baseline.yaml"
    cfg = get_cfg(str(cfg_file))
    root = Path(arciris_python).resolve()
    for key in ("circle_model_path", "nn_model_path"):
        raw = Path(cfg[key])
        cfg[key] = str(raw if raw.is_absolute() else (root / raw).resolve())
    return cfg


def make_iris_rec(arciris_python: Path, cfg_path: Path | None = None):
    add_arciris_to_path(arciris_python)
    from modules.irisRecognition import irisRecognition

    return irisRecognition(load_cfg(arciris_python, cfg_path))


def template_path(templates_dir: Path, image_id: str) -> Path:
    return Path(templates_dir) / f"{image_id}_tmpl.npz"


def load_cached_vector(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as data:
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return np.asarray(data[key])


def save_template(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vector)


def _open_gray(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB").split()[0]


def extract_one(iris_rec, filepath: str) -> tuple[bool, np.ndarray | None]:
    image = iris_rec.fix_image(_open_gray(filepath))
    pupil_xyr, iris_xyr = iris_rec.circApprox(image)
    ok = bool(iris_rec.checkQuality(pupil_xyr, iris_xyr))
    if not ok:
        return False, None
    polar = iris_rec.cartToPol_torch(image, pupil_xyr, iris_xyr)
    vector = iris_rec.extractVector(polar)
    return True, np.asarray(vector)


def load_or_extract(
    manifest: list[dict[str, str]],
    templates_dir: Path,
    arciris_python: Path,
    cfg_path: Path | None = None,
    reextract: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], Any]:
    """Return embeddings keyed by image_id, quality rows, and irisRec (or None).

    Cached `templates/<image_id>_tmpl.npz` / `arr_0` are reused. The model is
    constructed only if at least one image still needs extraction.
    """
    embeddings: dict[str, np.ndarray] = {}
    quality_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for row in manifest:
        image_id = row["image_id"]
        cached = None if reextract else load_cached_vector(template_path(templates_dir, image_id))
        if cached is not None:
            embeddings[image_id] = cached
            quality_rows.append(
                {"image_id": image_id, "quality_passed": "True", "source": "cache"}
            )
        else:
            missing.append(row)

    iris_rec = None
    if missing:
        print(f"Extracting embeddings for {len(missing)} image(s) (others reused from cache).")
        iris_rec = make_iris_rec(arciris_python, cfg_path)
        for row in missing:
            image_id = row["image_id"]
            print(f"  extract {image_id}")
            ok, vector = extract_one(iris_rec, row["filepath"])
            quality_rows.append(
                {
                    "image_id": image_id,
                    "quality_passed": str(ok),
                    "source": "extract",
                }
            )
            if ok and vector is not None:
                embeddings[image_id] = vector
                save_template(template_path(templates_dir, image_id), vector)

    return embeddings, quality_rows, iris_rec


def save_embeddings_pickle(embeddings: dict[str, np.ndarray], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(embeddings, handle)


def write_quality_flags(rows: list[dict[str, str]], path: Path) -> None:
    write_csv(path, rows, QUALITY_FIELDS)
