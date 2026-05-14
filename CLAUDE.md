# AI-Powered Handwriting Generation System

## What this project is

A university project exploring 5 different deep-learning approaches to generating handwriting.
All approaches live under `approaches/` so they can be compared side-by-side.
The dataset is ~2100 real handwriting PNGs from 6 writers stored in `Data/Writers_pngs/`.

## Environment

Python venv is at `D:\Semester_4\Programming for AI\.venv\`.
Activate: `.\.venv\Scripts\Activate.ps1`
Key packages: PyTorch, torchvision, Pillow, NumPy, SciPy, scikit-image.

## Project structure

```
approaches/
├── 01_GAN/                          — Style Encoder + Generator + AC-GAN Discriminator
├── 02_MultiTaskCNN/                 — Classifier only (char + writer heads), NOT a generator
├── 03_FontToHandwriting_CNN/        — U-Net translates font images → handwriting style
├── 04_BezierRegression_RealImages/  — CNN regresses Bézier curves from real reference images
├── 05_CharNet/                      — char index → Bézier curves (current best, simplest)
└── COMPARISON.md                    — side-by-side table of all approaches

Data/
├── Writers_pngs/   — 6 writer folders, ~2100 PNG handwriting samples
└── bezier_labels.npy  — cached Bézier curve labels (auto-built on first train run)

checkpoints/  — ROOT level: ignored by git (legacy location, do not use)
```

## How to run each approach

All commands run from **inside** the approach folder. Default = generate only (no training).

```powershell
cd approaches\01_GAN                       && python run.py
cd approaches\02_MultiTaskCNN              && python run.py
cd approaches\03_FontToHandwriting_CNN     && python run.py
cd approaches\04_BezierRegression_RealImages && python run.py
cd approaches\05_CharNet                   && python run.py   # works immediately
```

Output PNGs saved to `approaches/0X_NAME/outputs/`.
Files named `lc_a.png` (lowercase), `uc_A.png` (uppercase), `digit_0.png` to avoid Windows case-collision.

## Training (GPU required — use Google Colab)

Each approach has a Colab notebook inside its folder:

| Approach | Notebook | Checkpoint file |
|----------|----------|-----------------|
| 01_GAN | `Colab_GAN.ipynb` | `checkpoints/checkpoint.pt` |
| 02_MultiTaskCNN | `Colab_MultiTaskCNN.ipynb` | `checkpoints/checkpoint.pt` |
| 03_FontToHandwriting_CNN | `Colab_FontCNN.ipynb` | `checkpoints/style_net.pt` |
| 04_BezierRegression_RealImages | `Colab_BezierReg.ipynb` | `checkpoints/style_net.pt` |
| 05_CharNet | `Colab_CharNet.ipynb` | `checkpoints/style_net.pt` |

Colab workflow: Runtime → T4 GPU → run all cells → checkpoint auto-downloads.
Place downloaded checkpoint in `approaches/0X_NAME/checkpoints/` then `python run.py` locally.

Checkpoints committed to git — teammates get them on `git clone`, no retraining needed.
Approach 05_CharNet already has a trained checkpoint at `approaches/05_CharNet/checkpoints/style_net.pt`.

To force-train locally (slow on CPU): `python run.py --train --epochs N`

## What each model actually does

| # | Generates new handwriting? | How |
|---|---------------------------|-----|
| 01 GAN | Yes — pixel images | Encoder turns reference photo → style vector; Generator decodes style+char label → image |
| 02 MultiTaskCNN | **No** — classifier only | Takes real image, predicts char + writer label. Output is your own dataset photos with labels. |
| 03 Font CNN | Yes — pixel images | U-Net translates a PIL font-rendered glyph into a handwriting-style image per writer |
| 04 Bézier Regression | Yes — stroke curves | CNN takes real reference photo → predicts 6 Bézier curves → renders them |
| 05 CharNet | Yes — stroke curves | Takes only a char index (0–61) → predicts 6 Bézier curves → renders. No reference image needed. |

## Known issues and fixes applied this session

**GAN outputs all same image**
- Root cause: old generate() used `dataset[0]` as reference for ALL 62 chars. Since the model
  reconstructs (encode X → decode X), passing char A's style when generating B produces A's shape.
- Fix: generate() now finds a real image of EACH character and uses that as its own reference.
- Also fixed: inference must use clean transforms (no random augmentation).

**Approach 4 train crash: "too many values to unpack (expected 3)"**
- Root cause: `HandwritingDataset.__getitem__` returns 4 values
  `(img, char_idx, writer_idx, bezier_label)` but train loop only unpacked 3.
- Fix: unpack as `imgs, char_idxs, _, bezier_labels` (writer_idx discarded with `_`).

## Git / GitHub

Remote: https://github.com/Abdullahshaz70/AI-Powered-Handwriting-Generation-System.git
Branch: main

`.gitignore` rules relevant to this project:
- `/checkpoints/` — root checkpoints folder ignored (legacy)
- `outputs/` — generated PNGs ignored everywhere (don't commit)
- `approaches/*/checkpoints/*.pt` — NOT ignored; these push to GitHub so teammates get them

## Data

6 writers: writer_Abdullah, writer_abdullah_60, writer_Fatima, writer_Hamza, writer_Hashim, writer_Salman_24067
Characters: a-z (lc), A-Z (uc), 0-9 — ~33 samples per character per writer
Filename pattern: `WriterName_lc_a_r01.png` (lc=lowercase, uc=uppercase, n=digit)
