# How to Run Each Approach

> All commands are run from the **project root** (AI-Powered-Handwriting-Generation-System/).
> Activate the venv first: `.\.venv\Scripts\Activate.ps1`

---

## Quick Status

| # | Approach | Has Checkpoint? | Can Generate Now? | Needs Training? |
|---|----------|-----------------|-------------------|-----------------|
| 1 | GAN | No | No | Yes (~100 epochs) |
| 2 | MultiTaskCNN | No | No (classifier, not generator) | Yes (~100 epochs) |
| 3 | Font-to-Handwriting CNN | No | No | Yes (~80 epochs) |
| 4 | Bézier Regression (Real Images) | No | No | Yes (~100 epochs) |
| 5 | CharNet | **Yes** (`checkpoints/style_net.pt`) | **Yes** | Already trained |

---

## Approach 1 — GAN

### Path fix before running
The scripts were originally in `Data/` and expect `Writers_pngs` in the same folder.
Edit `approaches/01_GAN/train.py` line 17–18:
```python
# Change these two lines:
DATA_FOLDER    = os.path.join(_HERE, "Writers_pngs")
CHECKPOINT_DIR = os.path.join(_HERE, "..", "checkpoints")

# To:
DATA_FOLDER    = os.path.join(_HERE, "..", "..", "Data", "Writers_pngs")
CHECKPOINT_DIR = os.path.join(_HERE, "..", "..", "checkpoints", "01_GAN")
```

### Train
```powershell
cd approaches\01_GAN
python train.py
```
Prints per-epoch: `G_loss`, `D_loss`, `val_loss`. Saves checkpoints every 5 epochs to `checkpoints/01_GAN/`.

### What to expect
- Epoch 1–20: outputs will look like noise
- Epoch 40–60: rough character shapes start forming
- Epoch 80–100: best result — but may still be blurry (known limitation)

### Check output
After training, `train.py` saves generated image grids. Look in `checkpoints/01_GAN/` for `.pt` files.
To visually inspect, add at the end of your training loop or run separately:
```python
# Quick visual check — paste into a script and run
import torch
from approaches._01_GAN.generator import CharacterGenerator
from approaches._01_GAN.encoder import StyleEncoder
# load checkpoint and call generator.forward(style_vec, char_idx)
```

---

## Approach 2 — MultiTaskCNN

### Path fix before running
Edit `approaches/02_MultiTaskCNN/train.py` line 14–15:
```python
# Change:
DATA_ROOT = os.path.join(os.path.dirname(__file__), "Writers_pngs")
CKPT_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

# To:
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "Writers_pngs")
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "02_MultiTaskCNN")
```

### Train
```powershell
cd approaches\02_MultiTaskCNN
python train.py
```
Prints per-epoch: `char_acc`, `writer_acc`, `val_loss`. Best model → `checkpoints/02_MultiTaskCNN/best_model.pt`.

### Evaluate (after training)
```powershell
python evaluate.py
```
Outputs two PNG files:
- `eval_output/tsne_by_writer.png` — T-SNE cluster plot (do writers separate visually?)
- `eval_output/writer_confusion.png` — confusion matrix per writer

### Demo — classify a single image
```powershell
python demo.py --image "Data\Writers_pngs\writer_Abdullah\Abdullah_lc_a_r01.png"
```
Prints predicted character and predicted writer.

> **Note:** This approach does NOT generate handwriting. It only classifies.

---

## Approach 3 — Font-to-Handwriting CNN

### Path fix before running
Edit `approaches/03_FontToHandwriting_CNN/src/train.py` lines 22–23:
```python
# Change:
DATA_ROOT = os.path.join(_HERE, '..', 'Data', 'Writers_pngs')
CKPT_DIR  = os.path.join(_HERE, '..', 'checkpoints')

# To:
DATA_ROOT = os.path.join(_HERE, '..', '..', '..', 'Data', 'Writers_pngs')
CKPT_DIR  = os.path.join(_HERE, '..', '..', '..', 'checkpoints', '03_FontCNN')
```
Apply the same fix in `src/generate.py` for `CKPT_PATH` and `DATA_ROOT`.

### Train
```powershell
cd approaches\03_FontToHandwriting_CNN
python src\train.py
```
Prints per-epoch: `Train L1`, `Val L1`, `LR`. Best model → `checkpoints/03_FontCNN/style_net.pt`.

### Generate handwriting (after training)
```powershell
python app.py
```
Opens Flask at `http://localhost:5000`. Type any text in the browser, click Generate — gets back a handwriting image rendered from font → U-Net → output.

### Or generate from command line
```python
# In a Python REPL from the approaches/03_FontToHandwriting_CNN directory:
import sys; sys.path.insert(0, 'src')
from generate import load_model, generate_word
model, ckpt, device = load_model()
img = generate_word("hello", model, device)
img.save("test_hello.png")
```

---

## Approach 4 — Bézier Regression (Real Images)

### Path fix before running
Edit `approaches/04_BezierRegression_RealImages/src/train.py` lines 23–25:
```python
# Change:
DATA_ROOT  = os.path.normpath(os.path.join(_HERE, '..', 'Data', 'Writers_pngs'))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, '..', 'Data', 'bezier_labels.npy'))
CKPT_DIR   = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints'))

# To:
DATA_ROOT  = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'Data', 'Writers_pngs'))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'Data', 'bezier_labels.npy'))
CKPT_DIR   = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'checkpoints', '04_BezierReg'))
```
Apply the same fix in `src/generate.py`.

### Train
```powershell
cd approaches\04_BezierRegression_RealImages
python src\train.py
```
**First run** builds `bezier_labels.npy` cache by skeletonising all 2000+ images — takes ~5–10 minutes.
After that, prints: `Train MSE`, `Val MSE`, `LR`. Best model → `checkpoints/04_BezierReg/style_net.pt`.

### Generate handwriting (after training)
```powershell
python app.py
```
Flask app at `http://localhost:5000`. Unlike approach 3, you select a **writer** from the UI — the model uses that writer's reference images for style conditioning.

---

## Approach 5 — CharNet (Current, works immediately)

No path fixes needed. Checkpoint already exists at `checkpoints/style_net.pt`.

### Generate samples right now
```powershell
# From project root:
python generate_samples.py
```
Generates 10 PNG pages into `outputs/samples/`. Each page is a ruled handwriting page with one sentence rendered in the learned Bézier style.

### Options
```powershell
python generate_samples.py --noise 0.01   # cleaner strokes
python generate_samples.py --noise 0.04   # more natural variation
python generate_samples.py --out my_output_folder
python generate_samples.py --ckpt checkpoints/style_net.pt
```

### Train from scratch (optional)
```powershell
python src\train.py
```
300 epochs, prints `Train MSE` / `Val MSE` per epoch. Saves best checkpoint to `checkpoints/style_net.pt` (overwrites existing).

### Check output
Open any file in `outputs/samples/` — they are full ruled-page PNGs showing:
- Full alphabet (a–z, A–Z)
- Digits 0–9
- Pangrams like "The quick brown fox jumps over the lazy dog"

---

## Side-by-Side Comparison Tips

| What to compare | How |
|-----------------|-----|
| Training stability | Watch loss curve — GAN oscillates; others decrease smoothly |
| Output quality | Put a character PNG from each approach side by side |
| Speed to first output | CharNet = instant; GAN = slowest (adversarial overhead) |
| Input required at inference | GAN: ref image + char; MultiTaskCNN: image (no generation); FontCNN: font render; BezierReg: ref image + writer; CharNet: just a char index |
