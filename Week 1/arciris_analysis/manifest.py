"""Build or load an image manifest keyed by a stable image_id."""

from __future__ import annotations

import csv
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".png", ".gif", ".jpg", ".jpeg", ".tiff", ".tif"}
MANIFEST_FIELDS = ["image_id", "filepath", "subject_id", "eye", "session"]

# subject_L_session, subject_R.png, subject-left-01, trailing _L / _R
_EYE_IN_NAME = re.compile(
    r"(?:^|[_\-\s])(?P<eye>L|R|left|right)(?:[_\-\s]|$|\.)",
    re.IGNORECASE,
)


def normalize_eye(value: str | None) -> str:
    text = (value or "").strip()
    if not text or text.lower() in {"unknown", "nan", "none", "na"}:
        return "unknown"
    lowered = text.lower()
    if lowered in {"l", "left"}:
        return "L"
    if lowered in {"r", "right"}:
        return "R"
    return "unknown"


def normalize_subject(value: str | None) -> str:
    text = (value or "").strip()
    if not text or text.lower() in {"unknown", "nan", "none", "na"}:
        return ""
    return text


def _parse_eye_from_name(stem: str) -> str:
    match = _EYE_IN_NAME.search(stem)
    if not match:
        return "unknown"
    return normalize_eye(match.group("eye"))


def _parse_subject_from_name(stem: str, eye: str) -> str:
    """Best-effort: leading token before an eye tag, else empty."""
    if eye in {"L", "R"}:
        parts = re.split(r"[_\-\s]+(?:L|R|left|right)(?:[_\-\s]|$)", stem, maxsplit=1, flags=re.IGNORECASE)
        candidate = parts[0].strip("_- ")
        if candidate and candidate.lower() != stem.lower():
            return candidate
    return ""


def _unique_image_id(stem: str, path: Path, used: set[str], data_dir: Path) -> str:
    candidate = stem
    if candidate not in used:
        return candidate
    try:
        rel = path.resolve().relative_to(data_dir.resolve())
        candidate = "__".join(rel.with_suffix("").parts)
    except ValueError:
        candidate = f"{path.parent.name}__{stem}"
    if candidate not in used:
        return candidate
    n = 2
    while f"{candidate}__{n}" in used:
        n += 1
    return f"{candidate}__{n}"


def iter_image_paths(data_dir: Path) -> list[Path]:
    paths = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths


def build_manifest_from_dir(data_dir: Path) -> list[dict[str, str]]:
    """Scan data_dir. Unparseable subject/eye stay blank/unknown.

    Synthetics (`synthetic*.png` with no eye tag): image_id = stem,
    subject_id = stem (each file is its own fake identity), eye = unknown.
    """
    data_dir = Path(data_dir)
    rows: list[dict[str, str]] = []
    used: set[str] = set()
    for path in iter_image_paths(data_dir):
        stem = path.stem
        eye = _parse_eye_from_name(stem)
        subject = _parse_subject_from_name(stem, eye)
        if not subject and stem.lower().startswith("synthetic"):
            subject = stem
        image_id = _unique_image_id(stem, path, used, data_dir)
        used.add(image_id)
        rows.append(
            {
                "image_id": image_id,
                "filepath": str(path.resolve()),
                "subject_id": subject,
                "eye": eye,
                "session": "",
            }
        )
    if not rows:
        raise FileNotFoundError(f"No iris images found under {data_dir}")
    return rows


def load_manifest_csv(csv_path: Path, data_dir: Path | None = None) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        have = {name.strip() for name in reader.fieldnames}
        if "filepath" not in have:
            raise ValueError(f"{csv_path} must include a filepath column")
        rows: list[dict[str, str]] = []
        used: set[str] = set()
        for raw in reader:
            filepath = (raw.get("filepath") or "").strip()
            if not filepath:
                continue
            path = Path(filepath)
            if not path.is_absolute():
                base = Path(data_dir) if data_dir else csv_path.parent
                path = base / path
            if not path.is_file():
                raise FileNotFoundError(f"Manifest filepath does not exist: {path}")
            image_id = (raw.get("image_id") or path.stem).strip()
            if image_id in used:
                raise ValueError(f"Duplicate image_id in manifest: {image_id}")
            used.add(image_id)
            rows.append(
                {
                    "image_id": image_id,
                    "filepath": str(path.resolve()),
                    "subject_id": normalize_subject(raw.get("subject_id")),
                    "eye": normalize_eye(raw.get("eye")),
                    "session": (raw.get("session") or "").strip(),
                }
            )
    if not rows:
        raise ValueError(f"{csv_path} has no usable rows")
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    write_csv(path, rows, MANIFEST_FIELDS)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["image_id"]: row for row in rows}
