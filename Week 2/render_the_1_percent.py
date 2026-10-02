#!/usr/bin/env python3
"""Render the true worst 1,000 impostor pairs (lowest scores) to 'the 1%' folder.

Computes all ~13.3M impostor scores from cached embeddings (no model rerun).
Selects 1,000 with lowest scores. Renders as side-by-side JPEG pairs matching
Week 1's imposters/ gallery layout, with sensor and eye color info (Week 2 focus).
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
R = HERE.parent / "Week 1" / "Task 1 results"
DATASET = HERE.parent / "General Project Files" / "NDCVRL_002 Dataset"
OUT = HERE / "the 1%"
CUT = json.load(open(R / "impostor_tail_summary.json"))["threshold"]

# Load embeddings and manifest
import pickle
e = pickle.load(open(R / "embeddings.pkl", "rb"))  # e is a dict {image_id: embedding}
ids = np.array(list(e.keys()))
X = np.array([np.asarray(e[k], np.float64) for k in ids])
X /= np.linalg.norm(X, axis=1, keepdims=True)
N = len(ids)

# Load manifest and metadata
man = {r["image_id"]: r for r in csv.DictReader(open(R / "manifest.csv"))}
meta = {r["fileid"]: r for r in csv.DictReader(open(DATASET / "recording_metadata.csv"))}

# Compute all impostor scores
S = X @ X.T
scores = np.arccos(np.clip(S, -1, 1)).astype(np.float32)

# Mark same-identity pairs (genuine)
lab = np.array([man[k]["subject_id"] + man[k]["eye"] for k in ids])
same = lab[:, None] == lab[None, :]

# Get only the upper triangle (avoid duplicates and self-pairs)
ui, uj = np.triu_indices(N, 1)
impostor_mask = ~same[ui, uj]
impostor_scores = scores[ui, uj][impostor_mask]

# Get 1,000 lowest impostor scores (via argpartition)
k = 1000
if len(impostor_scores) >= k:
    worst_idx = np.argpartition(impostor_scores, k - 1)[:k]
    worst_pairs = sorted(
        [(ids[ui[i]], ids[uj[i]], float(impostor_scores[i])) for i in worst_idx],
        key=lambda x: x[2]
    )
else:
    worst_pairs = sorted(
        [(ids[ui[i]], ids[uj[i]], float(impostor_scores[i])) for i in range(len(impostor_scores))],
        key=lambda x: x[2]
    )[:k]

print(f"Found {len(worst_pairs)} worst impostor pairs")
if worst_pairs:
    print(f"Worst score: {worst_pairs[0][2]:.4f}, Best of worst: {worst_pairs[-1][2]:.4f}")
    print(f"Cut = {CUT:.4f}")

# Helper functions for rendering
def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

def _load_gray(filepath: str) -> np.ndarray:
    try:
        img = Image.open(filepath).convert("RGB")
        return np.array(img.split()[0])
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

def get_image_path(image_id):
    """Map image_id to the actual file path in the dataset."""
    m = man[image_id]
    subject = m["subject_id"]
    eye_short = m["eye"]
    eye_full = "Left" if eye_short == "L" else "Right"
    return DATASET / "raw" / subject / eye_full / f"{image_id}.tiff"

def _resize_and_panel(image: np.ndarray, label: str, width: int = 516) -> Image.Image:
    """Resize grayscale or RGB to width, add label header."""
    if image is None:
        # Placeholder for missing image
        image = np.zeros((480, 640), dtype=np.uint8)
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image
    pil = Image.fromarray(rgb.astype(np.uint8))
    w, h = pil.size
    new_h = max(1, int(round(h * (width / w))))
    pil = pil.resize((width, new_h), Image.Resampling.BILINEAR)

    header_h = 20
    canvas = Image.new("RGB", (width, new_h + header_h), (30, 30, 30))
    canvas.paste(pil, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(220, 220, 220), font=_font(11))
    return canvas

def render_pair(rank: int, id_a: str, id_b: str, score: float) -> bool:
    """Render one impostor pair as a JPEG. Return True if success."""
    OUT.mkdir(parents=True, exist_ok=True)

    path_a = get_image_path(id_a)
    path_b = get_image_path(id_b)

    if not path_a.exists() or not path_b.exists():
        print(f"  [{rank:4d}] SKIP {id_a} vs {id_b}: image file not found")
        return False

    # Load images
    gray_a = _load_gray(str(path_a))
    gray_b = _load_gray(str(path_b))
    if gray_a is None or gray_b is None:
        return False

    # Get metadata
    m_a = man[id_a]
    m_b = man[id_b]
    same_person = m_a["subject_id"] == m_b["subject_id"]
    sensor_a = meta.get(id_a, {}).get("sensorid", "?")
    sensor_b = meta.get(id_b, {}).get("sensorid", "?")
    color_a = meta.get(id_a, {}).get("color", "?")
    color_b = meta.get(id_b, {}).get("color", "?")

    # Image sizes for format
    w_a, h_a = Image.open(path_a).size
    w_b, h_b = Image.open(path_b).size
    fmt_a = f"{w_a}x{h_a}"
    fmt_b = f"{w_b}x{h_b}"

    # Create panels
    label_a = f"{id_a}  {m_a['subject_id']} {m_a['eye']}  sensor {sensor_a}  {color_a}"
    label_b = f"{id_b}  {m_b['subject_id']} {m_b['eye']}  sensor {sensor_b}  {color_b}"
    panel_a = _resize_and_panel(gray_a, label_a)
    panel_b = _resize_and_panel(gray_b, label_b)

    # Stack horizontally
    height = max(panel_a.height, panel_b.height)
    width = panel_a.width + panel_b.width + 8
    body = Image.new("RGB", (width, height), (20, 20, 22))
    body.paste(panel_a, (0, 0))
    body.paste(panel_b, (panel_a.width + 8, 0))

    # Add header
    header_h = 40
    canvas = Image.new("RGB", (body.width, body.height + header_h), (18, 18, 20))
    canvas.paste(body, (0, header_h))
    draw = ImageDraw.Draw(canvas)

    person_label = " (SAME person, other eye)" if same_person else " DIFFERENT people, model says similar"
    title = f"#{rank}  score {score:.3f} (cut {CUT:.3f}, lower = more similar){person_label}"
    draw.text((8, 6), title, fill=(240, 240, 240), font=_font(13))

    # Save
    if id_a > id_b:
        id_a, id_b = id_b, id_a
    out_path = OUT / f"{rank:04d}__{id_a}__{id_b}__score-{score:.3f}.jpg"
    canvas.save(out_path, quality=90)
    return True

# Render all pairs
print(f"Rendering to {OUT}...")
n_success = 0
rows = []
for rank, (id_a, id_b, score) in enumerate(worst_pairs, 1):
    if render_pair(rank, id_a, id_b, score):
        n_success += 1
        if rank % 100 == 0:
            print(f"  [{rank:4d}] rendered")

        # For index.csv
        m_a, m_b = man[id_a], man[id_b]
        sensor_a = meta.get(id_a, {}).get("sensorid", "?")
        sensor_b = meta.get(id_b, {}).get("sensorid", "?")
        w_a, h_a = Image.open(str(get_image_path(id_a))).size if id_a in man else (0, 0)
        w_b, h_b = Image.open(str(get_image_path(id_b))).size if id_b in man else (0, 0)

        if id_a > id_b:
            id_a_out, id_b_out = id_b, id_a
        else:
            id_a_out, id_b_out = id_a, id_b

        rows.append({
            "rank": rank,
            "file": f"{rank:04d}__{id_a_out}__{id_b_out}__score-{score:.3f}.jpg",
            "image_id_a": id_a_out,
            "image_id_b": id_b_out,
            "subject_a": m_a["subject_id"] if id_a in man else "?",
            "eye_a": m_a["eye"] if id_a in man else "?",
            "subject_b": m_b["subject_id"] if id_b in man else "?",
            "eye_b": m_b["eye"] if id_b in man else "?",
            "score": score,
            "sensor_a": sensor_a,
            "sensor_b": sensor_b,
            "format_a": f"{w_a}x{h_a}",
            "format_b": f"{w_b}x{h_b}",
        })

# Write index.csv
if rows:
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nRendered {n_success} of {len(worst_pairs)} pairs to {OUT}/")
    print(f"Wrote index.csv with {len(rows)} rows")
