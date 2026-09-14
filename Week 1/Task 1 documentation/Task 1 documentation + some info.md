# ArcIris Task Guide: Match Scores, Distribution Plots, and False-Positive Identification

**Repo:** https://github.com/CVRL/OpenSourceIrisRecognition
**Method:** ArcIris — `methods/ArcIris/Python`

This is grounded directly in the repo's own `arciris.py`, `irisRecognition.py`, and
readme files (confirmed by cloning and reading the code, not guessed), plus your
current access status as of today.

## Current status

- ✅ **Weights**: found — `https://notredame.app.box.com/s/3palriosrj34fzg2uok58p34au6rvenq`
  (Box, not the Google Drive link the top-level `readme.md` mentions — two
  different links point to the same models, use this one). Download and drop
  the contents into `methods/ArcIris/Python/models/`.
- ⏳ **Image dataset**: pending — Adam is sending you the actual dataset
  directly. Until it arrives, the repo's own bundled samples
  (`methods/ArcIris/Python/data/synthetic1.png`, `synthetic2.png`,
  `synthetic3.png` — three StyleGAN3-synthetic, non-real iris images from
  Tinsley/Czajka/Flynn's 2022 IJCB paper) are enough to prove the pipeline runs,
  but not enough for a real genuine/impostor distribution or false-positive
  analysis — three images, three different (fake) identities, one image each,
  so there are **zero genuine pairs** in that sample set.

So right now you can do steps 1–2 below (setup, run the demo pipeline
end-to-end, confirm it works). Steps 3 onward need Adam's dataset.

---

## 0. What you're actually building

This is a genuine/impostor score-distribution analysis — the same shape of
experiment Daugman ran to justify his HD ≤ 0.32 threshold, just using ArcIris's
embedding distance instead of Hamming distance.

- **Genuine pair** = two images of the *same* eye (same subject, same eye label).
- **Impostor pair** = two images of *different* eyes.
- **Score** = what `matchVectors` returns: `acos(cosine similarity)` between two
  embeddings. **Lower score = more similar.** (Not a 0–1 "confidence" — it's an
  angle in radians, roughly 0 to π.)
- **False positive (false match)** = an impostor pair whose score is *low*
  enough to fall on the "same eye" side of your threshold — the model
  mistaking two different eyes for one eye. This is what you're hunting for.
- The "tails and peaks" plot is the genuine-score histogram and the
  impostor-score histogram overlaid. Two clean, separated peaks = good
  separability. The **overlap region** — impostor scores dipping low enough to
  brush the genuine peak, or vice versa — is where false positives/negatives
  come from.

---

## 1. Set up the repo

```bash
git clone https://github.com/CVRL/OpenSourceIrisRecognition.git
cd OpenSourceIrisRecognition/methods/ArcIris/Python
conda create -n arciris --file minimal_environment.yml
conda activate arciris
```

Then:
1. Download the weights from the Box link above and put them in `./models/`.
2. Leave `./data/` as-is for now (the three synthetic samples) until Adam's
   dataset arrives — then you'll clear it out and drop the real images in.

---

## 2. Run the demo pipeline as-is, on the synthetic samples

`arciris.py` already does almost the entire pipeline for you — read it once
before you write anything of your own:

```python
# from arciris.py, the actual loop:
for im, fn in zip(image_list, filename_list):
    im = irisRec.fix_image(im)                          # resize to ISO 640x480, 4:3
    pupil_xyr, iris_xyr = irisRec.circApprox(im)         # segmentation + circle fit
    if irisRec.checkQuality(pupil_xyr, iris_xyr):        # bool: pupil/iris radius + alpha ratio + center-dist gate
        im_polar = irisRec.cartToPol_torch(im, pupil_xyr, iris_xyr)  # rubber-sheet unwrap
        vector = irisRec.extractVector(im_polar)         # embedding (numpy array)
        # saved to ./templates/<name>_tmpl.npz

# then all-vs-all matching over everything that passed quality:
for (vector1, fn1), (vector2, fn2) in itertools.combinations(items, 2):
    score = irisRec.matchVectors(vector1, vector2)       # acos(cosine_sim) — lower = more similar
```

Run it:

```bash
python arciris.py --cfg_path cfg_baseline.yaml
```

It will print every pairwise `fn1 <-> fn2 : score` for the three synthetic
images to stdout, and drop embeddings into `./templates/`. **Do this now** — it
confirms your environment, weights, and config actually work, independent of
whether you have the real dataset yet.

Two things worth noting from the real code, versus what you might assume:

- `checkQuality` returns a plain `True`/`False` and **prints its own reason**
  for failing (pupil too small, alpha ratio out of range, centers too far
  apart) — you don't need to inspect the flags yourself, just capture stdout
  or grep the print statements if you want them logged to a file instead of
  the console.
- There's no separate "load all embeddings then match" step — `arciris.py`
  keeps `vectors_list` and `filtered_filename_list` in memory and matches
  everything that passed quality, in one pass, right after extraction. You'll
  want to restructure this once there are real subject/eye labels involved
  (see step 3), but the core embedding + matching calls are exactly these four
  functions: `circApprox` → `cartToPol_torch` → `extractVector` →
  `matchVectors`.

---

## 3. When Adam's dataset arrives: build a manifest with identifiers

The demo script only tracks filenames — it has no concept of "same eye" vs.
"different eye," because the synthetic samples don't need one. Once you have
real images, the first thing to add is a manifest that a filename alone can't
give you:

| image_id | filepath | subject_id | eye (L/R) | session/capture |
|---|---|---|---|---|

- `image_id` — carry this through every plot and score row (filename stem is
  usually fine, as long as it's unique).
- `subject_id` + `eye` — defines what counts as a genuine pair. Two images of
  the same subject's *different* eyes are impostor pairs, not genuine ones
  (left ≠ right, even for the same person — this is Daugman's point).
- Ask Adam whether the dataset he sends already comes with this labeling
  (subject/eye per file, or per-folder) or whether you'll need to parse it out
  of filenames/directory structure yourself — worth confirming when he sends
  it rather than guessing at a naming convention.

---

## 4. Extract embeddings once per image, cache them, then compare

Don't call the model once per *pair* — extract each image's embedding once,
cache it, then do pairwise comparisons on the cached vectors. For N images
that's N forward passes instead of N² forward passes, and it's a small
rewrite of `arciris.py`'s loop:

```python
from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
from PIL import Image
import pickle

cfg = get_cfg("cfg_baseline.yaml")
irisRec = irisRecognition(cfg)

embeddings = {}
quality_flags = {}

for image_id, filepath in manifest[["image_id", "filepath"]].itertuples(index=False):
    im = Image.open(filepath).convert("RGB").split()[0]
    im = irisRec.fix_image(im)

    pupil_xyr, iris_xyr = irisRec.circApprox(im)
    ok = irisRec.checkQuality(pupil_xyr, iris_xyr)
    quality_flags[image_id] = ok

    if ok:
        im_polar = irisRec.cartToPol_torch(im, pupil_xyr, iris_xyr)
        embeddings[image_id] = irisRec.extractVector(im_polar)

pickle.dump(embeddings, open("embeddings.pkl", "wb"))
```

Keep `quality_flags` around — you'll want it later to tell whether a false
positive involved a borderline-quality image or two clean ones (a real
embedding failure is a different, more interesting story than a segmentation
failure).

---

## 5. Build the pairwise comparisons and log scores + identifiers

One row per compared pair, labeled genuine/impostor using the manifest:

```python
import itertools, pandas as pd

records = []
for (id_a, vec_a), (id_b, vec_b) in itertools.combinations(embeddings.items(), 2):
    score = irisRec.matchVectors(vec_a, vec_b)

    row_a = manifest.loc[manifest.image_id == id_a].iloc[0]
    row_b = manifest.loc[manifest.image_id == id_b].iloc[0]
    label = "genuine" if (row_a.subject_id == row_b.subject_id and row_a.eye == row_b.eye) else "impostor"

    records.append({
        "image_id_a": id_a, "image_id_b": id_b,
        "subject_a": row_a.subject_id, "subject_b": row_b.subject_id,
        "eye_a": row_a.eye, "eye_b": row_b.eye,
        "label": label, "score": score,
    })

scores_df = pd.DataFrame(records)
scores_df.to_csv("pairwise_scores.csv", index=False)
```

If the dataset is large, all-vs-all (N(N-1)/2 pairs) may be overkill —
genuine pairs are usually scarce enough to keep in full, but you can sample
impostor pairs rather than computing every single one.

---

## 6. Plot the images

Two useful image plots, both keyed off `image_id`:

1. **Raw image → polar unwrap, side by side**, for a handful of samples — the
   segmentation circles overlaid on the original, `im_polar` below it. This is
   what demonstrates the pipeline is working correctly.
2. **The specific pairs flagged as false positives** (step 8) — both images in
   the pair, ideally with their polar unwraps too, so it's visually clear
   whether the false match is a hard case (poor quality, occlusion) or a real
   embedding failure.

```python
import matplotlib.pyplot as plt, cv2

def plot_pair(id_a, id_b, manifest, polar_a=None, polar_b=None):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax_row, img_id, polar in [(axes[0], id_a, polar_a), (axes[1], id_b, polar_b)]:
        filepath = manifest.loc[manifest.image_id == img_id, "filepath"].iloc[0]
        ax_row[0].imshow(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax_row[0].set_title(img_id); ax_row[0].axis("off")
        if polar is not None:
            ax_row[1].imshow(polar, cmap="gray"); ax_row[1].set_title("polar unwrap"); ax_row[1].axis("off")
    return fig
```

---

## 7. Plot the "tails and peaks" — genuine vs. impostor score distributions

```python
genuine = scores_df.loc[scores_df.label == "genuine", "score"]
impostor = scores_df.loc[scores_df.label == "impostor", "score"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(genuine, bins=40, alpha=0.6, density=True, label=f"Genuine (n={len(genuine)})", color="tab:green")
ax.hist(impostor, bins=40, alpha=0.6, density=True, label=f"Impostor (n={len(impostor)})", color="tab:red")
ax.set_xlabel("ArcIris match score (acos of cosine similarity) — lower = more similar")
ax.set_ylabel("Density")
ax.set_title("Genuine vs. Impostor Score Distributions")
ax.legend()
```

To reason about separability, not just eyeball it:
- **d′ (decidability index)**: `abs(genuine.mean() - impostor.mean()) / sqrt(0.5*(genuine.var() + impostor.var()))`.
- **A threshold line**: draw a vertical line at whatever score you're treating
  as the accept boundary (remember: lower score = accept as same eye), so the
  plot shows how much impostor mass sits below it.
- **Shade the overlap region** between the two curves — that's the population
  of pairs producing errors.

An ROC/DET curve (false match rate vs. false non-match rate swept across
thresholds) is a good complementary second plot — that's what IREX-style
FNIR@FPIR numbers actually come from.

---

## 8. Identify the false positives from the log

An impostor pair whose score is *low* enough to fall on the genuine side of
your threshold:

```python
threshold = 0.35  # pick from the distribution plot / a target FPIR — don't trust this default

false_positives = scores_df[
    (scores_df.label == "impostor") & (scores_df.score <= threshold)
].sort_values("score")

false_positives.to_csv("false_positives.csv", index=False)
```

- Sort ascending — the lowest scores are the worst false matches (most
  confidently wrong), and the most worth rendering with `plot_pair`.
- Cross-reference against `quality_flags` from step 4 — a false positive where
  both images passed quality cleanly is more concerning than one where an
  image was borderline.
- Flip the comparison (`label == "genuine"` and `score >= threshold`) for
  false negatives, if the task wants both sides.

---

## 9. What to hand off

- `embeddings.pkl`, `pairwise_scores.csv`, `false_positives.csv` — all keyed by
  `image_id`, so any plot can be reproduced from the log alone.
- The distribution plot from step 7, with d′ reported.
- `plot_pair` renders for the worst false positives.
- One line connecting it back to the reading: where these false positives sit
  relative to the impostor tail is exactly the "test of statistical
  independence" Daugman was measuring — just with ArcIris's continuous
  embedding distance instead of his 2,048-bit Hamming distance.
