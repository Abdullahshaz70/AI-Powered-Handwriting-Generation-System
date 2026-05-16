# AI-Powered Handwriting Generation System

## What we have made:

This project explores **5 different deep learning approaches** to generate realistic handwriting. Instead of one solution, we compare multiple neural network architectures side-by-side to see which one works best.

Given a character (like the letter 'a' or digit '5'), each approach tries to generate that character in the style of a specific writer's handwriting. Some generate pixel images, others generate stroke curves.

---

## The 5 Approaches

| # | Name | What It Does | Output |
|----|------|-------------|--------|
| **01** | GAN | Uses style encoder + generator + discriminator | Pixel images |
| **02** | MultiTaskCNN | Classifier only (doesn't generate, just labels) | Character + writer labels |
| **03** | Font to Handwriting | Translates typed font letters into handwriting style | Pixel images |
| **04** | Bézier Regression | Predicts stroke curves from reference images | Vector curves |
| **05** | CharNet | Direct character index → stroke curves (simplest, fastest) | Vector curves |

**Best for quick results?** Approach 05 (CharNet) — works immediately, no training needed.

---

## Quick Start

### Prerequisites
- Python 3.8+ 
- PyTorch, torchvision, Pillow, NumPy
- Activate the venv: `.\.venv\Scripts\Activate.ps1` (Windows)

### Generate Handwriting (No Training)

```powershell
# Pick any approach folder and run:
cd approaches\05_CharNet
python run.py
```

Generated images appear in `outputs/` folder as PNG files:
- `lc_a.png` (lowercase 'a')
- `uc_A.png` (uppercase 'A')
- `digit_0.png` (digit '0')

### Train a Model (GPU Required)

1. Open the **Colab notebook** inside each approach folder (e.g., `Colab_CharNet.ipynb`)
2. Run on Google Colab with T4 GPU
3. Download the trained checkpoint file
4. Place it in `approaches/0X_NAME/checkpoints/`
5. Run `python run.py` locally

Or force training locally: `python run.py --train --epochs 10` (slow on CPU)

---

## Project Structure

```
approaches/
├── 01_GAN/                          — Style encoder + generator + AC-GAN
├── 02_MultiTaskCNN/                 — Classifier only (doesn't generate)
├── 03_FontToHandwriting_CNN/        — Font → handwriting translator
├── 04_BezierRegression_RealImages/  — Real image → Bézier curves
├── 05_CharNet/                      — Char index → Bézier curves
└── COMPARISON.md                    — Detailed side-by-side comparison

Data/
├── Writers_pngs/                    — ~2100 real handwriting samples (6 writers)
└── bezier_labels.npy                — Cached Bézier labels
```

---

## The Dataset

- **6 Writers**: Abdullah, Abdullah_60, Fatima, Hamza, Hashim, Salman_24067
- **Characters**: a–z (lowercase), A–Z (uppercase), 0–9
- **Images per character**: ~33 samples per writer
- **Total**: ~2100 PNG images
- **Format**: `WriterName_lc_a_r01.png` (lc=lowercase, uc=uppercase)

Each approach uses this real handwriting data to learn style patterns.

---

## How It Works (Simple Explanation)

### Pixel-Based (Approaches 01, 03)
1. Take a character as input
2. Neural network generates a realistic pixel image in the writer's style
3. Output: PNG image

### Curve-Based (Approaches 04, 05)
1. Take a character as input
2. Neural network predicts 6 **Bézier curves** (mathematical strokes)
3. Render the curves as an image
4. Output: PNG image (or raw curves)

The curve approach is more elegant — fewer parameters, smoother lines, easier to vectorize.

---

## Troubleshooting

**Q: All generated images look the same?**  
A: Make sure the checkpoint file is loaded. Check that `checkpoints/style_net.pt` exists in the approach folder.

**Q: Getting CUDA/GPU errors?**  
A: CPU mode is slower but works. Models default to CPU if CUDA unavailable.

**Q: How do I compare the approaches?**  
A: See `approaches/COMPARISON.md` for a detailed table of pros/cons.

---

## Git & Collaboration

- **Remote**: https://github.com/Abdullahshaz70/AI-Powered-Handwriting-Generation-System.git
- **Branch**: main
- **Checkpoints**: Committed to git (teammates auto-get them on clone)
- **Outputs**: Ignored (don't commit generated images)

---

## Next Steps

1. **Run Approach 05** → `cd approaches\05_CharNet && python run.py`
2. **Explore outputs** → check `approaches/05_CharNet/outputs/`
3. **Try other approaches** → repeat step 1 with different folder
4. **Train your own** → use the Colab notebooks (takes ~15–30 min with GPU)

---

**Questions?** Check the approach-specific READMEs inside each folder, or see `COMPARISON.md` for detailed comparisons.