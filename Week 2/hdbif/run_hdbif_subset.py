#!/usr/bin/env python3
"""HDBIF genuine/impostor scores on a subset of NDCVRL_002 (inference only, no retraining).

Reuses the manifest ArcIris built (Week 1). Segments + encodes each image once with
HDBIF's own pretrained weights, caches codes+masks, then scores all genuine pairs and
a capped number of impostor pairs with matchCodesEfficient (fractional Hamming distance,
lower = more similar). Mirrors run_pipeline.py's genuine/impostor split, but for HDBIF.
"""
import argparse, csv, itertools, pickle, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

WEEK1 = Path(__file__).resolve().parents[2] / "Week 1"
HDBIF_PY = Path(__file__).resolve().parents[2] / "General Project Files" / "OpenSourceIrisRecognition" / "methods" / "HDBIF" / "Python"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HDBIF_PY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=WEEK1 / "Task 1 results" / "ndcvrl_manifest.csv")
    ap.add_argument("--min-images-per-eye", type=int, default=4)
    ap.add_argument("--images-per-eye", type=int, default=6)
    ap.add_argument("--n-eyes", type=int, default=80)
    ap.add_argument("--max-impostor-pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    import yaml
    sys.path.insert(0, str(HDBIF_PY))
    from modules.irisRecognition import irisRecognition

    cfg = yaml.safe_load(open(HDBIF_PY / "cfg.yaml"))
    for k in ("mask_model_path", "circle_model_path"):
        cfg[k] = str((HDBIF_PY / cfg[k]).resolve())
    cfg["recog_bsif_dir"] = str((HDBIF_PY / cfg["recog_bsif_dir"]).resolve()) + "/"
    rec = irisRecognition(cfg)

    rows = list(csv.DictReader(open(args.manifest)))
    by_eye = {}
    for r in rows:
        by_eye.setdefault((r["subject_id"], r["eye"]), []).append(r)
    rng = random.Random(args.seed)
    eyes = [k for k, v in by_eye.items() if len(v) >= args.min_images_per_eye]
    eyes = rng.sample(eyes, min(args.n_eyes, len(eyes)))
    subset = []
    for k in eyes:
        vs = by_eye[k][:]
        rng.shuffle(vs)
        subset.extend(vs[: args.images_per_eye])
    print(f"{len(eyes)} subject-eyes, {len(subset)} images", flush=True)

    codes, masks, ids, lab = {}, {}, [], {}
    t0 = time.time()
    for n, r in enumerate(subset):
        fp = r["filepath"].replace("Week 1/NDCVRL_002 Dataset", "General Project Files/NDCVRL_002 Dataset")
        im = Image.fromarray(np.array(Image.open(fp).convert("RGB"))[:, :, 0], "L")
        im = rec.fix_image(im)
        mask, pupil_xyr, iris_xyr = rec.segment_and_circApprox(im)
        im_polar, mask_polar = rec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
        code = rec.extractCode(im_polar)
        codes[r["image_id"]] = code
        masks[r["image_id"]] = mask_polar
        ids.append(r["image_id"])
        lab[r["image_id"]] = (r["subject_id"], r["eye"])
        if n % 50 == 0:
            print(f"  encoded {n}/{len(subset)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"encoding done in {time.time()-t0:.0f}s", flush=True)

    genuine_pairs = [(a, b) for a, b in itertools.combinations(ids, 2) if lab[a] == lab[b]]
    impostor_pairs_all = [(a, b) for a, b in itertools.combinations(ids, 2) if lab[a] != lab[b]]
    rng.shuffle(impostor_pairs_all)
    impostor_pairs = impostor_pairs_all[: args.max_impostor_pairs]
    print(f"genuine pairs {len(genuine_pairs)}, impostor pairs sampled {len(impostor_pairs)} of {len(impostor_pairs_all)}", flush=True)

    def score_all(pairs, label):
        out = []
        t1 = time.time()
        for i, (a, b) in enumerate(pairs):
            s, shift = rec.matchCodesEfficient(codes[a], codes[b], masks[a], masks[b])
            out.append({"image_id_a": a, "image_id_b": b, "label": label, "score": s, "shift": shift})
            if i % 2000 == 0 and i:
                print(f"  {label} {i}/{len(pairs)}  ({time.time()-t1:.0f}s)", flush=True)
        return out

    records = score_all(genuine_pairs, "genuine") + score_all(impostor_pairs, "impostor")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "hdbif_pairwise_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id_a", "image_id_b", "label", "score", "shift"])
        w.writeheader(); w.writerows(records)
    pickle.dump({"codes": None}, open(args.out_dir / "hdbif_meta.pkl", "wb"))  # placeholder, codes not pickled (large)
    print("wrote", args.out_dir / "hdbif_pairwise_scores.csv")


if __name__ == "__main__":
    main()
