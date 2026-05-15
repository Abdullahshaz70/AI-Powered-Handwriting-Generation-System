# Stats — AI Handwriting Generation System

This folder contains all the evaluation code and results for our 5 handwriting
generation approaches. The scripts measure how well each model works, and the
outputs folder holds all the charts and tables they produce.

---

## How to Run Everything

```powershell
# Activate the project venv first
.\.venv\Scripts\Activate.ps1

# Install any missing stats dependencies
pip install scikit-learn seaborn pandas easyocr

# Run ALL stats (from the project root)
python stats/scripts/run_all.py

# Skip the slow OCR step
python stats/scripts/run_all.py --skip-ocr

# Run just one specific script
python stats/scripts/run_all.py --only 02
python stats/scripts/run_all.py --only 05
python stats/scripts/run_all.py --only quality
python stats/scripts/run_all.py --only ocr
```

> **Before running:** Make sure each approach has generated its output PNGs first.
> Run `python approaches/0X_.../run.py` for each approach you want to evaluate.

---

## What Each Script Does

### `scripts/eval_02_classifier.py` → `outputs/approach_02/`

This script evaluates **Approach 02 (MultiTaskCNN)**, which is the only model
that does classification (not generation). It takes a real handwriting image
and predicts which character it is, and which writer wrote it.

**What the outputs show:**

| File | What it shows |
|------|---------------|
| `classification_report.txt` | Precision, recall, F1 for every character. High F1 = model recognises that character well. |
| `char_accuracy_bar.png` | Bar chart of accuracy per character, sorted hardest → easiest. Short bars = the model struggles with that character. |
| `confusion_char.png` | 62×62 grid. Each row = true character, each column = predicted character. Dark squares on the diagonal = correct predictions. Off-diagonal dark squares = common mistakes. |
| `confusion_writer.png` | Same idea but for identifying which of the 6 writers wrote the sample. |
| `tsne_by_writer.png` | Each dot = one handwriting image. Dots of the same colour = same writer. Well-separated colour clusters = the model has learned distinct style features for each writer. |
| `summary.csv` | Overall accuracy numbers in one row. |

---

### `scripts/eval_05_charnet.py` → `outputs/approach_05/`

This script evaluates **Approach 05 (CharNet)**, our best and simplest model.
CharNet takes a character index (like "give me the letter A") and predicts the
Bézier curves (stroke paths) to draw it.

The ground truth is computed by processing real handwriting images through
skeletonisation to extract their actual stroke curves.

**What the outputs show:**

| File | What it shows |
|------|---------------|
| `per_char_mse_bar.png` | How far the model's predicted curves are from the ground-truth curves, per character. Low bar = accurate prediction. Bars are colour-coded: 🔴 uppercase, 🟢 lowercase, 🟡 digit. |
| `category_mse_bar.png` | Average MSE grouped by category (uppercase / lowercase / digit). Shows if the model struggles more with one group. Error bars show variation within the group. |
| `endpoint_deviation_bar.png` | How far off the start and end points of each stroke are from the real ones. Low = strokes are placed accurately on the canvas. |
| `per_char_metrics.csv` | Full numbers per character. |
| `summary.csv` | Overall MSE and endpoint deviation. |

**What MSE means here:** The curves are in normalised coordinates (0 to 1).
An MSE of 0.001 means the curves are off by about 1/1000th of the canvas width on average — very good. An MSE above 0.01 means noticeable stroke displacement.

---

### `scripts/eval_image_quality.py` → `outputs/image_quality/`

This script compares the **visual quality** of generated images across all four
generator approaches (01 GAN, 03 Font CNN, 04 Bézier Reg, 05 CharNet).
It doesn't care about model internals — it only looks at the output PNG files.

**What the outputs show:**

| File | What it shows |
|------|---------------|
| `ssim_comparison.png` | SSIM = Structural Similarity Index. Measures how similar the generated image looks to a real handwriting sample of the same character. 1.0 = identical, 0.0 = nothing in common. Higher is better. |
| `sharpness_comparison.png` | How clean and crisp the ink strokes are. Blurry GAN outputs will score low. Bézier-rendered outputs (clean black curves on white) will score high. |
| `ink_coverage.png` | What fraction of the image canvas is covered by ink. Too low = character is tiny or faint. Too high = the image is full of noise or blurry blobs. |
| `distinctiveness_bar.png` | Are all 62 generated characters visually different from each other? A model suffering from mode collapse (generating the same image for every character) will score very low here. |
| `distinctiveness_heatmaps.png` | One heatmap per approach. Each cell = similarity between two characters. A good model has a mostly green matrix (all characters are distinct). Mode collapse shows up as a red/orange matrix. |
| `metrics_table.csv` | All numbers per approach per character. |
| `approach_summary.csv` | Averages per approach. |

---

### `scripts/eval_ocr_benchmark.py` → `outputs/ocr/`

This is the **cross-model comparison** — the fairest way to compare all
generators on the same scale.

We use **EasyOCR** (an open-source text recognition system trained on millions
of real and synthetic text images) to "read" each generated character.
If it reads the right character, the generator passes the test.

This is better than using our own Approach 02 classifier because EasyOCR
is truly neutral — it wasn't trained on any of our model's outputs.

**What the outputs show:**

| File | What it shows |
|------|---------------|
| `evolution_chart.png` | The most important chart. Shows OCR accuracy going from Approach 01 → 03 → 04 → 05. An upward curve = our project improved over time. |
| `ocr_accuracy_bar.png` | Accuracy per approach — what % of generated characters were correctly read by EasyOCR. |
| `ocr_f1_bar.png` | Macro F1 score per approach. More rigorous than accuracy — accounts for class imbalance and per-character precision/recall. |
| `ocr_results_detail.csv` | Full breakdown: for each character in each approach, what did EasyOCR read vs. what was intended. |
| `ocr_summary.csv` | Accuracy, precision, recall, F1 per approach in one table. |

**Note:** The first run of this script downloads the EasyOCR model (~100 MB).
Subsequent runs use the cached model.

---

## What Each Metric Means (Plain English)

| Metric | Simple Explanation | Good Value |
|--------|--------------------|------------|
| **Accuracy** | Out of all predictions, how many were right? | > 80% |
| **Precision** | When the model says "this is A", how often is it correct? | > 0.80 |
| **Recall** | Of all the real A's, how many did the model find? | > 0.80 |
| **F1 Score** | Balance of precision and recall. Better than accuracy when classes are uneven. | > 0.80 |
| **SSIM** | Does the generated image look structurally like real handwriting? | > 0.50 |
| **Bézier MSE** | How far are the predicted stroke paths from the real ones? | < 0.005 |
| **Endpoint Deviation** | How far off are the stroke start/end points? | < 0.05 |
| **Sharpness** | Is the ink clean and crisp, not blurry? | > 0.70 |
| **Distinctiveness** | Do all 62 characters look different from each other? | > 0.60 |
| **OCR Accuracy** | Can a real-world text reader correctly identify the character? | > 50% |

---

## Folder Structure

```
stats/
├── STATS.md                       ← This file
├── scripts/
│   ├── run_all.py                 ← Master runner
│   ├── eval_02_classifier.py      ← Approach 02 evaluation
│   ├── eval_05_charnet.py         ← Approach 05 evaluation
│   ├── eval_image_quality.py      ← Visual quality across all generators
│   └── eval_ocr_benchmark.py      ← OCR cross-model benchmark
└── outputs/                       ← All generated charts and tables
    ├── approach_02/               ← Approach 02 results
    ├── approach_05/               ← Approach 05 results
    ├── image_quality/             ← Cross-approach visual metrics
    └── ocr/                       ← OCR benchmark results
```

---

## Required Python Packages

All packages below should be installed in the project venv:

```
torch          ← already installed
torchvision    ← already installed
scikit-image   ← already installed
Pillow         ← already installed
numpy          ← already installed
scikit-learn   ← pip install scikit-learn
seaborn        ← pip install seaborn
pandas         ← pip install pandas
easyocr        ← pip install easyocr  (OCR benchmark only)
```
