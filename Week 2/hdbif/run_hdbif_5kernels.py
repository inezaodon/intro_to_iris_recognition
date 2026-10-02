#!/usr/bin/env python3
"""HDBIF per-kernel impostor/genuine analysis: match using 5 filter sizes independently.

Extends run_hdbif_subset.py to score pairs using each of 5 patch-size kernels in isolation.
Each kernel (5x5, 7x7, 9x9, 11x11, 13x13) is tested separately to measure:
- Impostor score variance (robust: IQR/1.349)
- d' (discriminability)
- Impostor tail tail behavior (% below 1% cut)

Output: per-kernel score CSVs and summary statistics.
"""
import argparse, csv, itertools, pickle, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

WEEK1 = Path(__file__).resolve().parents[2] / "Week 1"
WEEK2 = Path(__file__).resolve().parents[1]
HDBIF_PY = Path(__file__).resolve().parents[2] / "General Project Files" / "OpenSourceIrisRecognition" / "methods" / "HDBIF" / "Python"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HDBIF_PY))

# Use manifest from Week 1
MANIFEST = WEEK1 / "Task 1 results" / "ndcvrl_manifest.csv"


class HDBIFSingleKernel:
    """HDBIF wrapper that uses only one patch size at a time."""
    def __init__(self, cfg_dict, filter_size, num_filters, hdbif_py_path, device='cpu'):
        import yaml
        import torch
        from modules.irisRecognition import irisRecognition

        # Modify config to use only one filter size
        cfg = cfg_dict.copy()
        cfg['recog_filter_size'] = str(filter_size)
        cfg['recog_num_filters'] = str(num_filters)

        # Resolve paths
        for k in ("mask_model_path", "circle_model_path"):
            cfg[k] = str((hdbif_py_path / cfg[k]).resolve())
        cfg["recog_bsif_dir"] = str((hdbif_py_path / cfg["recog_bsif_dir"]).resolve()) + "/"

        self.rec = irisRecognition(cfg)
        self.filter_size = filter_size

    def segment_and_encode(self, image_path):
        """Segment iris and extract code."""
        img = Image.open(image_path).convert('L')
        # Use the irisRecognition module's pipeline
        try:
            polar, mask, _ = self.rec.segment_and_circApprox(img)
            code, _ = self.rec.extractCode(polar)
            return code, mask, None
        except Exception as e:
            return None, None, str(e)

    def match_codes(self, code_a, mask_a, code_b, mask_b):
        """Match two codes using fractional Hamming distance on unoccluded overlap."""
        try:
            score = self.rec.matchCodesEfficient(code_a, mask_a, code_b, mask_b, rotation_range=3)
            return score
        except Exception as e:
            return None


def load_hdbif_config(hdbif_py_path):
    """Load HDBIF config."""
    import yaml
    cfg = yaml.safe_load(open(hdbif_py_path / "cfg.yaml"))
    return cfg


def run_kernel(kernel_size, num_filters, manifest_path, n_eyes=80, min_images_per_eye=4,
               images_per_eye=6, max_impostor_pairs=40000, seed=0, out_dir=None):
    """Run HDBIF scoring with a single kernel size."""

    if out_dir is None:
        out_dir = HERE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Kernel {kernel_size}x{kernel_size} ({num_filters} filters) ===")
    start = time.time()

    # Load config and initialize single-kernel HDBIF
    cfg = load_hdbif_config(HDBIF_PY)
    hdbif = HDBIFSingleKernel(cfg, kernel_size, num_filters, HDBIF_PY)

    # Load manifest and select subset
    rows = list(csv.DictReader(open(manifest_path)))
    by_eye = {}
    for r in rows:
        by_eye.setdefault((r["subject_id"], r["eye"]), []).append(r)
    rng = random.Random(seed)
    eyes = [k for k, v in by_eye.items() if len(v) >= min_images_per_eye]
    eyes = rng.sample(eyes, min(n_eyes, len(eyes)))
    subset = []
    for k in eyes:
        eye_rows = by_eye[k]
        subset.extend(rng.sample(eye_rows, min(images_per_eye, len(eye_rows))))
    subset = sorted(subset, key=lambda r: (r["subject_id"], r["eye"]))

    print(f"Subset: {len(subset)} images from {len(eyes)} subject-eyes")

    # Encode all images
    codes = {}
    masks = {}
    errors = 0
    for i, r in enumerate(subset):
        img_path = WEEK1 / "Task 1 results" / r["filename"]
        if not img_path.exists():
            print(f"  Skip {r['filename']}: not found")
            continue

        code, mask, err = hdbif.segment_and_encode(img_path)
        if err:
            print(f"  Error {r['filename']}: {err}")
            errors += 1
        else:
            codes[r["filename"]] = code
            masks[r["filename"]] = mask

        if (i + 1) % 50 == 0:
            print(f"  Encoded {i + 1}/{len(subset)}")

    print(f"Encoded {len(codes)}/{len(subset)} images ({errors} errors)")

    # Score all genuine pairs and impostor pairs
    scores = []
    genuine_pair_count = 0
    impostor_pair_count = 0

    # Genuine pairs: same subject-eye
    for eye, eye_rows in by_eye.items():
        eye_subset = [r for r in eye_rows if r["filename"] in codes]
        if len(eye_subset) < 2:
            continue
        for r1, r2 in itertools.combinations(eye_subset, 2):
            score = hdbif.match_codes(codes[r1["filename"]], masks[r1["filename"]],
                                      codes[r2["filename"]], masks[r2["filename"]])
            if score is not None:
                scores.append({
                    'image_id_a': r1["filename"],
                    'image_id_b': r2["filename"],
                    'label': 'genuine',
                    'score': float(score),
                    'filter_size': kernel_size
                })
                genuine_pair_count += 1

    # Impostor pairs: random different subject-eyes
    all_filenames = list(codes.keys())
    n_images = len(all_filenames)
    rng = random.Random(seed + 1)  # Different random order
    impostor_attempts = 0
    max_attempts = max_impostor_pairs * 3  # Upper limit to avoid infinite loop

    while impostor_pair_count < max_impostor_pairs and impostor_attempts < max_attempts:
        i1, i2 = rng.sample(range(n_images), 2)
        f1, f2 = all_filenames[i1], all_filenames[i2]
        r1, r2 = next(r for r in rows if r["filename"] == f1), next(r for r in rows if r["filename"] == f2)

        # Check they're from different subject-eyes
        if (r1["subject_id"], r1["eye"]) == (r2["subject_id"], r2["eye"]):
            impostor_attempts += 1
            continue

        score = hdbif.match_codes(codes[f1], masks[f1], codes[f2], masks[f2])
        if score is not None:
            scores.append({
                'image_id_a': f1,
                'image_id_b': f2,
                'label': 'impostor',
                'score': float(score),
                'filter_size': kernel_size
            })
            impostor_pair_count += 1
        impostor_attempts += 1

    print(f"Genuine pairs: {genuine_pair_count}, Impostor pairs: {impostor_pair_count}")

    # Save scores
    out_file = out_dir / f"hdbif_pairwise_scores_kernel_{kernel_size}x{kernel_size}.csv"
    with open(out_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id_a', 'image_id_b', 'label', 'score', 'filter_size'])
        w.writeheader()
        w.writerows(scores)
    print(f"Wrote {out_file}")

    # Compute statistics
    genuine_scores = np.array([s['score'] for s in scores if s['label'] == 'genuine'])
    impostor_scores = np.array([s['score'] for s in scores if s['label'] == 'impostor'])

    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        gen_mean, gen_std = genuine_scores.mean(), genuine_scores.std()
        imp_mean, imp_std = impostor_scores.mean(), impostor_scores.std()

        # Robust stats (IQR / 1.349 for normal approximation)
        gen_iqr = np.percentile(genuine_scores, 75) - np.percentile(genuine_scores, 25)
        imp_iqr = np.percentile(impostor_scores, 75) - np.percentile(impostor_scores, 25)
        gen_robust_std = gen_iqr / 1.349
        imp_robust_std = imp_iqr / 1.349

        d_prime = (gen_mean - imp_mean) / np.sqrt((gen_std**2 + imp_std**2) / 2)

        # Count below cut (0.445 = empirical 1% impostor cut from W2-4 baseline)
        cut = 0.445
        pct_below_cut = 100.0 * (impostor_scores < cut).sum() / len(impostor_scores) if len(impostor_scores) > 0 else 0
        pct_genuine_above_cut = 100.0 * (genuine_scores > cut).sum() / len(genuine_scores) if len(genuine_scores) > 0 else 0

        stats = {
            'filter_size': kernel_size,
            'num_filters': num_filters,
            'genuine_n': len(genuine_scores),
            'impostor_n': len(impostor_scores),
            'genuine_mean': gen_mean,
            'genuine_std': gen_std,
            'genuine_robust_std': gen_robust_std,
            'impostor_mean': imp_mean,
            'impostor_std': imp_std,
            'impostor_robust_std': imp_robust_std,
            'impostor_variance': imp_robust_std**2,
            'd_prime': d_prime,
            'pct_impostor_below_cut': pct_below_cut,
            'pct_genuine_above_cut': pct_genuine_above_cut,
        }

        print(f"Genuine: mean={gen_mean:.3f}, std={gen_std:.3f} (robust={gen_robust_std:.3f})")
        print(f"Impostor: mean={imp_mean:.3f}, std={imp_std:.3f} (robust={imp_robust_std:.3f})")
        print(f"d' = {d_prime:.2f}")
        print(f"Impostor below cut (0.445): {pct_below_cut:.2f}%")
        print(f"Genuine above cut (0.445): {pct_genuine_above_cut:.2f}%")
        print(f"Time: {time.time() - start:.1f}s\n")

        return stats

    return None


def main():
    ap = argparse.ArgumentParser(description="HDBIF per-kernel analysis (5 filter sizes independently)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--n-eyes", type=int, default=80)
    ap.add_argument("--min-images-per-eye", type=int, default=4)
    ap.add_argument("--images-per-eye", type=int, default=6)
    ap.add_argument("--max-impostor-pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    # 5 independent filter sizes (based on cfg.yaml: recog_filter_size and recog_num_filters)
    # Typical config: "5,7,9,11,13" with "10,10,10,10,10" filters each
    filter_configs = [
        (5, 10),
        (7, 10),
        (9, 10),
        (11, 10),
        (13, 10),
    ]

    all_stats = []
    for filter_size, num_filters in filter_configs:
        stats = run_kernel(
            filter_size, num_filters, args.manifest,
            n_eyes=args.n_eyes,
            min_images_per_eye=args.min_images_per_eye,
            images_per_eye=args.images_per_eye,
            max_impostor_pairs=args.max_impostor_pairs,
            seed=args.seed,
            out_dir=args.out_dir
        )
        if stats:
            all_stats.append(stats)

    # Save summary
    if all_stats:
        summary_file = args.out_dir / "kernel_variance_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            w.writeheader()
            w.writerows(all_stats)
        print(f"\nWrote kernel variance summary to {summary_file}")

        # Print ranking by impostor variance
        print("\n=== Kernel Ranking by Impostor Variance (lower = more stable) ===")
        sorted_stats = sorted(all_stats, key=lambda s: s['impostor_variance'])
        for i, s in enumerate(sorted_stats, 1):
            print(f"{i}. {s['filter_size']}x{s['filter_size']}: "
                  f"variance={s['impostor_variance']:.6f}, "
                  f"d'={s['d_prime']:.2f}, "
                  f"impostor below cut={s['pct_impostor_below_cut']:.2f}%")


if __name__ == '__main__':
    main()
