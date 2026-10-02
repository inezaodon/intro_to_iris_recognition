"""Write impostor-tail pair renders named with both image_ids and the score."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAIR_DIRNAME = "impostor_images"


def pair_filename(id_a: str, id_b: str, score: float) -> str:
    if id_a > id_b:
        id_a, id_b = id_b, id_a
    return f"{id_a}__{id_b}__score-{score:.3f}.png"


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_gray(filepath: str) -> np.ndarray:
    image = np.array(Image.open(filepath).convert("RGB").split()[0])
    return image


def _circles_and_polar(iris_rec, gray: np.ndarray):
    pil = Image.fromarray(gray)
    fixed = iris_rec.fix_image(pil)
    pupil_xyr, iris_xyr = iris_rec.circApprox(fixed)
    polar = iris_rec.cartToPol_torch(fixed, pupil_xyr, iris_xyr)
    canvas = cv2.cvtColor(np.array(fixed), cv2.COLOR_GRAY2RGB)
    px, py, pr = [int(round(v)) for v in pupil_xyr]
    ix, iy, ir = [int(round(v)) for v in iris_xyr]
    cv2.circle(canvas, (ix, iy), ir, (80, 170, 255), 2)
    cv2.circle(canvas, (px, py), pr, (255, 220, 60), 2)
    return canvas, polar


def _panel(image: np.ndarray, title: str, width: int = 420) -> Image.Image:
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image
    pil = Image.fromarray(rgb)
    w, h = pil.size
    new_h = max(1, int(round(h * (width / w))))
    pil = pil.resize((width, new_h), Image.Resampling.BILINEAR)
    header = 28
    canvas = Image.new("RGB", (width, new_h + header), (18, 18, 20))
    canvas.paste(pil, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(230, 230, 230), font=_font(14))
    return canvas


def _stack_horizontal(images: list[Image.Image], gap: int = 12, bg=(12, 12, 14)) -> Image.Image:
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _stack_vertical(images: list[Image.Image], gap: int = 12, bg=(12, 12, 14)) -> Image.Image:
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def render_pair(
    id_a: str,
    id_b: str,
    score: float,
    by_id: dict[str, dict[str, str]],
    dest_dir: Path,
    iris_rec=None,
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if id_a > id_b:
        id_a, id_b = id_b, id_a
    path_a = by_id[id_a]["filepath"]
    path_b = by_id[id_b]["filepath"]
    gray_a = _load_gray(path_a)
    gray_b = _load_gray(path_b)

    rows = []
    if iris_rec is not None:
        circ_a, polar_a = _circles_and_polar(iris_rec, gray_a)
        circ_b, polar_b = _circles_and_polar(iris_rec, gray_b)
        rows.append(
            _stack_horizontal(
                [
                    _panel(circ_a, f"{id_a} original + circles"),
                    _panel(polar_a, f"{id_a} polar unwrap", width=512),
                ]
            )
        )
        rows.append(
            _stack_horizontal(
                [
                    _panel(circ_b, f"{id_b} original + circles"),
                    _panel(polar_b, f"{id_b} polar unwrap", width=512),
                ]
            )
        )
    else:
        rows.append(
            _stack_horizontal(
                [
                    _panel(gray_a, id_a),
                    _panel(gray_b, id_b),
                ]
            )
        )

    body = _stack_vertical(rows)
    header_h = 56
    canvas = Image.new("RGB", (body.width, body.height + header_h), (12, 12, 14))
    canvas.paste(body, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    title = f"Impostor pair  {id_a}  vs  {id_b}"
    subtitle = f"ArcIris match score = {score:.3f}  (acos cosine; lower = more similar)"
    draw.text((12, 8), title, fill=(240, 240, 240), font=_font(18))
    draw.text((12, 32), subtitle, fill=(180, 180, 180), font=_font(14))

    out_path = dest_dir / pair_filename(id_a, id_b, score)
    canvas.save(out_path)
    return out_path


def render_tail_pairs(
    tail_pairs: list[dict],
    by_id: dict[str, dict[str, str]],
    out_dir: Path,
    iris_rec=None,
) -> list[Path]:
    dest = Path(out_dir) / PAIR_DIRNAME
    written: list[Path] = []
    for row in tail_pairs:
        score = float(row["_score"] if "_score" in row else row["score"])
        path = render_pair(
            row["image_id_a"],
            row["image_id_b"],
            score,
            by_id,
            dest,
            iris_rec=iris_rec,
        )
        written.append(path)
        print(f"  wrote {path.name}")
    return written
