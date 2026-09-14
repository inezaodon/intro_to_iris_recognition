# Iris recognition study chat

Notes from Odon’s overnight study session with Cursor (Sep 10–11, 2026). Source: Adam Czajka’s reading list, plus Q&A on Daugman, ArcIris, and the CVRL open-source repo (with Siamul Karim Khan).

**Site:** https://inezaodon.github.io/intro_to_iris_recognition/  
**Repo:** https://github.com/inezaodon/intro_to_iris_recognition  
**Daugman slides:** https://inezaodon.github.io/intro_to_iris_recognition/#daugman/1  
**CVRL code:** https://github.com/CVRL/OpenSourceIrisRecognition

---

## Adam’s reading list (the assignment)

Study in this order:

1. **Daugman, “How iris recognition works”** — https://www.cl.cam.ac.uk/~jgd1000/ (PDF: https://www.cl.cam.ac.uk/~jgd1000/irisrecog.pdf). Father of iris recognition; sets the stage.
2. **Modern DL survey (ACM)** — https://dl.acm.org/doi/10.1145/3651306. Where deep learning actually helps (Daugman was not a fan).
3. **Notre Dame open-source IREX paper** — https://arxiv.org/abs/2605.20735. ArcIris and TripletIris on the NIST IREX leaderboard: https://pages.nist.gov/IREX10/
4. **Open-source repo** — https://github.com/CVRL/OpenSourceIrisRecognition. Methods plus tools (e.g. eye canthi detector). You will work with **ArcIris**.

Adam: weekly 30-minute meetings once the semester grid is set. Send preferred days/times.

---

## What to own vs skip

Own: **the pipeline, Daugman’s independence test, ArcIris, and the IREX metric alphabet.**

Skip tonight: CRYPTS (Earth Mover’s Distance; failed IREX’s 25-second search cap), super-resolution / synthetic irises / privacy sections of the survey, C++ LibTorch `fork()` / threading unless submitting to IREX, cover-to-cover of 200 survey papers.

---

## The pipeline (every paper)

NIR capture → find pupil / iris / eyelids → **rubber-sheet unwrap to polar** → **encode** → match → threshold.

Daugman invented steps 2–5. ArcIris keeps 2–3 and replaces 4–5 with a neural net.

### Rubber-sheet unwrap (normalize)

Take the iris **ring** in the photo and **flatten it into a rectangle**.

The iris is an annulus: pupil in the middle, sclera outside. Different photos have different pupil sizes, camera distances, and the pupil is often off-center. If you compared those raw rings, the same texture would sit in different pixel places.

The rubber-sheet model treats the iris like a stretchy doughnut. Every point gets polar coordinates:

- **θ** (angle) goes around the eye, 0 to 360°
- **r** (radius) is not in pixels. It is a fraction from **0 at the pupil edge to 1 at the outer iris edge**

Resample into a rectangle: angle along one axis, radius along the other. **ArcIris uses 512×64.** Pupil dilation becomes a uniform stretch that this mapping undoes. Distance to the camera drops out because r is relative, not absolute.

It does **not** fix head tilt. Rotation is handled later by sliding the code/embedding.

Plain picture: ring → unrolled rectangle of fixed size, so encoding always looks at “the same place” on the iris.

### Encode

Turns the unwrapped iris rectangle into a compact ID, so two photos can be compared without looking at pixels.

**Classic Daugman (IrisCode).** Slide small 2-D Gabor wavelets (oriented ripples) across the polar image. Each patch produces a complex number. Throw away how strong the response is (amplitude — that just tracks lighting) and keep only **which quadrant** the angle landed in. Four quadrants → **2 bits**. Do that all over the iris, many sizes, and you get **2,048 phase bits**, plus a matching **mask** that marks lashes, lids, and glare so those bits are ignored later.

That bit string is the template. Matching is XOR + Hamming distance, not “do these pictures look similar.”

**ArcIris.** Same polar rectangle (512×64), but a **ResNet100** reads it and outputs a **float vector** trained with ArcFace so same-eye images cluster and different eyes separate. Matching is cosine/Euclidean distance on that vector, not XOR.

Same job either way: **texture in, short signature out.** Daugman signs with phase bits; ArcIris signs with a neural embedding.

---

## Daugman 2004 — what to memorize

Thesis in one sentence: **recognition is the unique failure of a test of statistical independence on iris phase.**

- 2,048 phase bits + a mask for lashes / lids / glare
- Impostor Hamming distance ≈ 0.50 (fair coin)
- Only ~**249** independent bits, not 2,048 (bits are correlated)
- Accept around **HD ≤ 0.32** (~1 in 26 million false match after rotations)
- Left and right eyes of the *same person* are as different as two strangers — the pattern is **epigenetic**, not DNA
- Amplitude is thrown away (lighting). Phase is kept. XOR is why this still searches national databases in real time
- Rubber sheet does **not** fix rotation; code shifts do
- 1:N search is much harder than 1:1 verification
- Why faces fail 1:N: too few degrees of freedom, too correlated

The GitHub Pages deck (19 slides) walks: independence-test thesis, why faces fail 1:N, NIR and biology, left ≠ right, circular edge detector, rubber sheet, Gabor phase quadrants, masked Hamming distance, 249 DoF, best-of-7 rotation, HD threshold table, verification vs identification, d-prime, 2004 speed numbers, cheat sheet.

---

## ArcIris (`ndcvrl_002`) — what you will work on

Notre Dame’s modern iris matcher. On NIST IREX it is **`ndcvrl_002`**.

**What it is.** A **ResNet100** (from the ArcFace face-recognition codebase / InsightFace `iresnet100`) that reads a **polar 512×64** iris image — the rubber-sheet unwrap of a 640×480 NIR eye photo — and outputs a **float embedding**. Same-eye photos should land nearby; different eyes should not. Matching is cosine or Euclidean / `acos(cosine)` on that vector, not Hamming distance.

**How it is trained.** **ArcFace loss**: embeddings live on a hypersphere, and a fixed **angular margin** is added so the model is not allowed to just barely separate identities. More stable than triplet loss, which needs hard-example mining.

That is why ArcIris crushed their first IREX entry, **TripletIris** (`ndcvrl_001`, ConvNeXt-tiny + batch-hard triplets).

| | TripletIris (`ndcvrl_001`) | **ArcIris (`ndcvrl_002`)** |
|---|---|---|
| Backbone | ConvNeXt-tiny | **ResNet100** |
| Loss | batch-hard triplet | **ArcFace** (angular margin) |
| Input | polar 512×64 | same |
| IREX FNIR @ 1% FPIR | 32.6% | **4.8%** |
| Template | — | **8 kB**, 6th-smallest on the board |

**What sits in front of it.** Shared ND front-end, not Daugman’s 1990s edge detector:

- pixel mask (nested U-Net) — **defined in code, not loaded at inference**
- circle regressor (ResNet18 → pupil/iris center and radius)
- cheap quality gates only (tiny iris, inverted radii, almost no iris visible). They **do not** aggressively throw away hard images.

**IREX numbers.** Template **8 kB**. Rank-1 **97.62%**, rank-10 **98.22%**, accuracy rank **27/35**. Search of 1M irises in about **2.5 s**. Failure-to-enroll near zero by design.

**What it is not.** Not an IrisCode. Not closed-set softmax (“which of these 100 people?”). It is **open-set 1:N**: enroll a new person without retraining, then search with vector distance.

**Repo layout.** `methods/ArcIris/Python` for research; `methods/ArcIris/C++-IREX-X-submission` is the NIST wrapper. Python vs C++ scores are essentially the same.

Training weights are **not** in GitHub — Google Drive `ResNet100_154000.pt`. Trained on ~287k ND bona fide images; checkpoint chosen by max **d′**.

**Meeting line:** commercial VeriEye looks unbeatable on clean data because it **refuses** hard images. ArcIris almost never refuses (FTE ~0), so it stays useful for forensics, where you cannot throw the sample away.

**If Adam asks “what is ArcIris?”:** Daugman’s rubber sheet + a face-style ArcFace embedding instead of Gabor bits.

**Key files:**

- `methods/ArcIris/Python/arciris.py` — demo pipeline
- `methods/ArcIris/Python/cfg_baseline.yaml` — polar 64×512, circle ResNet18, `nn_model_path: ./models/ResNet100_154000.pt`
- `methods/ArcIris/Python/modules/irisRecognition.py` — inference class (`extractVector`, `circApprox`, `matchVectors`, `checkQuality`)
- `methods/ArcIris/Python/modules/network.py` — `iresnet100` **and** unused `NestedSharedAtrousResUNet`
- `methods/ArcIris/Python/1N_matching.py`, `find_1N_polar.py` — 1:N helpers; latter still calls mask APIs **not present** in current `irisRecognition.py`

---

## ACM survey (skim only)

- Segmentation is where DL clearly wins
- Softmax nets are closed-set and useless for IREX
- PAD and post-mortem are the other DL wins
- IrisCode still owns billion-scale XOR search

That is why Daugman was not a fan, and it is a fair take.

---

## Fine-tunes that fit *their* code

Looked at ArcIris Python (`irisRecognition.py`, `network.py`, `cfg_baseline.yaml`) plus HDBIF, the IREX C++ wrapper, and `iris-fm-tools`. Training code for the ResNet100 is **not** in the public repo — only inference.

### 1. Mask the polar image (or train with a mask channel)

The U-Net (`NestedSharedAtrousResUNet`) is sitting in `network.py` and is **never loaded**. Inference only does ResNet18 circles → rubber sheet → `iresnet100`. Lashes, lids, and glare go into the embedding. `find_1N_polar.py` still calls a mask API that the current class does not implement.

Why it works: Daugman’s whole point of the mask is “don’t let occlusions be identity.” On Q-FIRE / VII-Q-R2 / post-mortem that is most of the error. Either zero out non-iris pixels before `extractVector`, or fine-tune with a 4th mask channel. Highest-leverage change in the repo.

### 2. Circular shifts at match time (no retraining)

`matchVectors` is `acos(cosine)` on one alignment. Polar θ is cyclic; Daugman and HDBIF already shift. Head tilt / torsion is a horizontal roll of the 512-wide polar.

Why it works: a few extra distance calls, still trivial vs IREX’s 25 s budget, same rotation invariance IrisCodes have had since 2004.

### 3. Wire in CornerNet (canthi) before the rubber sheet

`iris-fm-tools` already trains `cornernet` on DINOv3 features. ArcIris never uses it.

Why it works: rubber sheet assumes you already know which way “up” is. A few degrees of in-plane rotation currently pollutes the embedding. Affine-align canthi, then unwrap — cheap, already in-house.

### 4. Fine-tune ArcFace on the failure mode you care about

Weights `ResNet100_154000.pt` were chosen by max d′ on ND bona fide. Public eval sets (CASIA-Lamp dilation, WBPMI decay, VII-Q junk) are mostly **not** that training distribution.

Why it works: ArcFace is already open-set; a short fine-tune with a small angular-margin LR on one target domain usually moves FNMR at 0.1% FMR more than swapping backbones. Freeze early layers; don’t train from scratch.

### 5. Stop faking RGB

`extractVector` takes a 1-channel NIR polar and `.repeat(1,3,1,1)`. The network was born as a 3-channel face net.

Why it works: extra channels are copies, so early filters waste capacity on a lie. Fine-tune with a 1-channel `conv1`, or keep 3-ch but apply **channel-wise NIR augmentations** (gamma, contrast) so the copies at least aren’t identical.

**Where it happens** — two places plus the conv that requires 3 channels.

**Encoder** — `extractVector` in [`methods/ArcIris/Python/modules/irisRecognition.py`](https://github.com/CVRL/OpenSourceIrisRecognition/blob/main/methods/ArcIris/Python/modules/irisRecognition.py):

```python
def extractVector(self, polar):
    im_polar = Image.fromarray(polar, "L")          # 1-channel NIR polar
    im_tensor = self.input_transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1)
    vector = self.nn_model(im_tensor)
```

`"L"` is grayscale. `repeat(1, 3, 1, 1)` copies that one channel into R, G, and B so `iresnet100` will accept it.

**Circle model** — `circApprox` in the same file:

```python
self.circle_model(
    self.input_transform(image).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
)
```

**Why they have to** — `IResNet` in [`methods/ArcIris/Python/modules/network.py`](https://github.com/CVRL/OpenSourceIrisRecognition/blob/main/methods/ArcIris/Python/modules/network.py):

```python
self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
```

First conv expects 3 input channels because this is the InsightFace face backbone. NIR iris is 1 channel; they fake RGB to match the weight shapes.

`input_transform` is `ToTensor()` + `Normalize(mean=(0.5,), std=(0.5,))` on the single channel, then replicated.

### 6. Polar roll as training augmentation

Match-time currently has no shift, so the model never saw θ-rolled genuines as the same class.

Why it works: rolling the 512-wide polar during ArcFace training is the right augmentation for iris (not ImageNet flips). Pair with (2) at test time.

### 7. Use the better circle model off the IREX clock

They shipped ResNet18 circles because IREX extract is 1.5 s. Their own table: ConvNeXt-tiny had higher IoU (0.938 vs 0.936) at 77 ms vs 30 ms.

Why it works: a few pixels of pupil/iris error smear the rubber sheet. For a Python research path (not a new IREX submit), swap in the better regressor or the `circlenet` head from `iris-fm-tools`. Segmentation error is still the #1 classic iris killer.

### 8. Occlusion-aware quality, not just radii

Python `checkQuality` only checks min pupil/iris radius, α ∈ [0.1, 0.8], and center distance. The paper listed **usable iris area**; that check is missing here.

Why it works: keep FTE near zero (their IREX philosophy) but **down-weight or flag** comparisons with little visible texture instead of pretending a 10% visible iris is a full embedding. Soft quality beats binary reject for forensics.

### 9. Fuse HDBIF bits with the ArcIris vector

HDBIF is already in the same repo: human-saliency filters + fractional HD + Daugman `HD_norm`. ArcIris is a global embedding. They fail on different images.

Why it works: score-level fusion (weighted sum / logistic) is a classic Notre Dame move. Don’t start with CRYPTS — it misses IREX timing for a reason.

### What not to pitch first

- Training a new backbone from scratch (they already compared ResNet50/100/200 for speed)
- Softmax closed-set classifiers (useless for IREX 1:N)
- Heavier ISO quality gating if the use case is forensic — they left that loose on purpose
- CRYPTS / EMD as the matcher

**One sentence to Adam:** “ArcIris currently embeds the whole polar, including occlusions, with no θ-shift; I’d mask with the U-Net you already wrote, add polar roll, and fine-tune ArcFace on the domain we care about.”

---

## Ask Adam

1. Extending ArcIris, a new IREX variant, or a downstream app (forensics / PAD / canthi / quality)?
2. Which Notre Dame datasets can you actually request from https://cvrl.nd.edu/projects/data/ ?
3. Python research path or C++ IREX path first?

When the semester grid is set: send 3–4 weekly 30-minute slots and one honest sentence — you got through the Daugman pipeline, ArcIris vs TripletIris, and IREX metrics.

---

## Site / repo work from this chat

- Study canvas: Cursor canvas `iris-recognition-study.canvas.tsx` (tabs: Tonight / Pipeline / Daugman / Deep learning / ArcIris / IREX / Quiz)
- GitHub Pages site `intro_to_iris_recognition` with a 19-slide Daugman deck
- Two live bugs then fixed (commit `e343444`):
  1. Hash updates in `assets/app.js` retriggered `boot()` → page looked frozen. Fixed with `applyingHash` / `push=false` on hash boot.
  2. `.prose p, .slide p` used cream text on cream slides. Split selectors; Daugman slides now near-black on `#fffdf7`. Cache-bust `?v=2` on CSS/JS.
- Hard-refresh (`Cmd-Shift-R`) if the old script is cached.

Chat-only constraint: from “until I write **jjk**” onward, answers only — no file edits — until this Desktop markdown request.

---

## Chat timeline

| When | You asked | Outcome |
|---|---|---|
| Sep 10, 10:33 PM | Study Adam’s list in one night | Overnight pack + canvas: pipeline, Daugman, ArcIris, what to skip |
| Sep 10, 10:51 PM | Daugman slide tab + GitHub Pages repo `intro_to_iris_recognition` | 19-slide deck, live site |
| Sep 10, 11:35 PM | Site not responding; Daugman contrast | Hash-loop + cream-on-cream CSS fixed |
| Sep 11, 12:22 AM | Chat-only until `jjk` | No more file edits until this export |
| Sep 11, 12:23 AM | What rubber-sheet unwrap means | Polar (θ, r) rectangle; does not fix rotation |
| Sep 11, 12:32 AM | Encode | IrisCode Gabor phase vs ArcIris embedding |
| Sep 11, 12:35 AM | More on ArcIris | `ndcvrl_002`, ArcFace, IREX numbers, forensic FTE |
| Sep 11, 12:53 PM | Fine-tunes from the CVRL repo | Nine code-grounded projects |
| Sep 11, 1:05 PM | Locate “stop faking RGB” | `extractVector` + `circApprox` `.repeat(1,3,1,1)` |
| Sep 11, 2:04 PM | This markdown on Desktop | This file |

---

## Sources

- Daugman, J. *How iris recognition works.* IEEE Trans. CSVT, 2004. https://www.cl.cam.ac.uk/~jgd1000/irisrecog.pdf
- ACM survey: https://dl.acm.org/doi/10.1145/3651306
- ND open-source IREX paper: https://arxiv.org/abs/2605.20735
- NIST IREX 10: https://pages.nist.gov/IREX10/
- CVRL OpenSourceIrisRecognition: https://github.com/CVRL/OpenSourceIrisRecognition
- Study site: https://inezaodon.github.io/intro_to_iris_recognition/
