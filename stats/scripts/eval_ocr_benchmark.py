"""
eval_ocr_benchmark.py
=====================
Cross-model OCR accuracy benchmark using EasyOCR.

For every generator approach, we load its output PNGs (one per character),
run EasyOCR on each image, and check if the recognised text matches the
intended character.

This is the fairest cross-model comparison: EasyOCR was trained on millions
of diverse text images, so it is domain-agnostic — it will not favour any
single generator's style.

Outputs saved to:  stats/outputs/ocr/
  - ocr_results_detail.csv      per-approach per-char: intended, recognised, correct
  - ocr_summary.csv             accuracy / macro-F1 per approach
  - ocr_accuracy_bar.png        bar chart of accuracy per approach
  - ocr_f1_bar.png              bar chart of macro-F1
  - evolution_chart.png         accuracy + sharpness evolution 01->03->04->05

Run from ANY directory:
    python stats/scripts/eval_ocr_benchmark.py

Requirements:
    pip install easyocr
"""

import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("  WARNING: easyocr not installed. Run: pip install easyocr")
    print("  Continuing in DUMMY mode (all results will be NaN).")

from sklearn.metrics import f1_score, precision_score, recall_score

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATS_DIR    = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(STATS_DIR)
OUT_DIR      = os.path.join(STATS_DIR, "outputs", "ocr")

APPROACH_OUTPUT_DIRS = {
    "01_GAN":       os.path.join(PROJECT_ROOT, "approaches", "01_GAN",                        "outputs"),
    "03_FontCNN":   os.path.join(PROJECT_ROOT, "approaches", "03_FontToHandwriting_CNN",       "outputs"),
    "04_BezierReg": os.path.join(PROJECT_ROOT, "approaches", "04_BezierRegression_RealImages", "outputs"),
    "05_CharNet":   os.path.join(PROJECT_ROOT, "approaches", "05_CharNet",                     "outputs"),
}

APPROACH_COLORS = {
    "01_GAN":       "#e74c3c",
    "03_FontCNN":   "#f39c12",
    "04_BezierReg": "#3498db",
    "05_CharNet":   "#2ecc71",
}

CANVAS_SIZE = 128


# ── Filename -> character ───────────────────────────────────────────────────────

def parse_char_from_fname(fname):
    """
    Parse character from various naming formats:
      lc_a.png / uc_A.png / digit_0.png
      writer0_lc_a.png / writer0_uc_A.png / writer0_digit_0.png
    """
    base  = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    for i, part in enumerate(parts):
        if part in ("lc", "uc") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if len(candidate) == 1:
                return candidate
        if part == "digit" and i + 1 < len(parts):
            candidate = parts[i + 1]
            if len(candidate) == 1:
                return candidate
    return None


def load_approach_images(approach_dir):
    """
    Returns list of (char, pil_image) sorted by character.
    For multi-writer approaches (writer0_lc_a.png style), only writer0
    images are used to keep exactly 62 samples per approach.
    """
    results = {}   # char -> PIL image  (last writer wins; use writer0 when possible)
    for fpath in sorted(glob.glob(os.path.join(approach_dir, "*.png"))):
        bname = os.path.basename(fpath)
        if bname.startswith("grid"):
            continue
        ch = parse_char_from_fname(fpath)
        if ch is None:
            continue
        # Prefer writer0_ images; only overwrite if no image yet
        is_writer0 = bname.startswith("writer0_")
        if ch not in results or is_writer0:
            try:
                results[ch] = Image.open(fpath).convert("RGB")
            except Exception:
                pass
    return sorted(results.items())


# ── Pre-process for better OCR ────────────────────────────────────────────────

def preprocess_for_ocr(pil_img, upscale=3):
    """
    Upscale and ensure dark ink on white background.
    EasyOCR performs better on larger images with clear contrast.
    """
    # Convert to grayscale
    gray = pil_img.convert("L")
    # Resize up
    w, h = gray.size
    gray = gray.resize((w * upscale, h * upscale), Image.LANCZOS)
    # Ensure the image has dark ink on white background
    arr   = np.array(gray)
    # If image is inverted (white ink on black), flip it
    if arr.mean() < 128:
        arr = 255 - arr
        gray = Image.fromarray(arr)
    return gray


# ── OCR recognition ───────────────────────────────────────────────────────────

def recognize_char(reader, pil_img):
    """
    Run EasyOCR on a single character image.
    Returns the best single-character recognition result.
    """
    if reader is None:
        return None

    proc = preprocess_for_ocr(pil_img)
    arr  = np.array(proc)

    try:
        results = reader.readtext(arr, detail=1, paragraph=False,
                                  allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        if not results:
            return None

        # Pick highest confidence result
        best = max(results, key=lambda r: r[2])
        text = best[1].strip()

        if len(text) == 1:
            return text
        # If multi-char returned, take first char
        if text:
            return text[0]
        return None
    except Exception:
        return None


def char_match(intended, recognised):
    """
    Strict exact match. Returns True/False.
    """
    if recognised is None:
        return False
    return intended == recognised


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_approach(reader, approach_name, approach_dir):
    """
    Returns list of dicts: {approach, char, intended, recognised, correct}
    """
    images = load_approach_images(approach_dir)
    if not images:
        print(f"    No images found in {approach_dir}")
        return []

    rows = []
    correct_count = 0
    for i, (ch, img) in enumerate(images, 1):
        recognised = recognize_char(reader, img)
        is_correct  = char_match(ch, recognised)
        if is_correct:
            correct_count += 1

        rows.append({
            "approach":   approach_name,
            "char":       ch,
            "intended":   ch,
            "recognised": recognised if recognised else "",
            "correct":    int(is_correct),
        })

        if i % 10 == 0 or i == len(images):
            print(f"    {i}/{len(images)} chars processed  "
                  f"(acc so far: {correct_count/i*100:.1f}%)")

    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bar(data_dict, ylabel, title, path, higher_is_better=True):
    names  = list(data_dict.keys())
    values = [data_dict[n] if data_dict[n] is not None else 0.0 for n in names]
    colors = [APPROACH_COLORS.get(n, "#95a5a6") for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, [v * 100 for v in values], color=colors,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
    ax.set_xlabel(direction, fontsize=9, color="#555")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_evolution(summary_df, path):
    """Line chart showing how accuracy evolves across approaches."""
    order = ["01_GAN", "03_FontCNN", "04_BezierReg", "05_CharNet"]
    labels = {
        "01_GAN":       "01\nGAN",
        "03_FontCNN":   "03\nFont CNN",
        "04_BezierReg": "04\nBézier Reg",
        "05_CharNet":   "05\nCharNet",
    }
    present = [a for a in order if a in summary_df["approach"].values]
    if len(present) < 2:
        return

    accs = []
    for a in present:
        row = summary_df[summary_df["approach"] == a]
        accs.append(float(row["accuracy"].values[0]) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(present))
    ax.plot(x, accs, "o-", color="#2ecc71", linewidth=2.5, markersize=9, markerfacecolor="white",
            markeredgewidth=2.5, markeredgecolor="#2ecc71", label="OCR Accuracy")
    for xi, (a, acc) in enumerate(zip(present, accs)):
        ax.annotate(f"{acc:.1f}%", (xi, acc),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold", color="#2ecc71")
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(a, a) for a in present], fontsize=11)
    ax.set_ylabel("OCR Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Project Evolution — OCR Accuracy Across Approaches\n"
                 "(Readability of generated handwriting over time)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    # Shade area under curve
    ax.fill_between(list(x), accs, alpha=0.1, color="#2ecc71")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== OCR Cross-Model Benchmark ===\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Initialise EasyOCR reader once (downloads model on first run ~100MB)
    reader = None
    if EASYOCR_AVAILABLE:
        print("Initialising EasyOCR reader (first run downloads ~100MB model)…")
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("  EasyOCR ready.")
        except Exception as e:
            print(f"  EasyOCR init failed: {e}")
            reader = None

    all_rows = []

    for approach_name, out_dir in APPROACH_OUTPUT_DIRS.items():
        if not os.path.isdir(out_dir):
            print(f"\n  SKIP {approach_name} — outputs/ not found. "
                  f"Run approaches/{approach_name}*/run.py first.")
            continue

        print(f"\n  Evaluating {approach_name}…")
        rows = evaluate_approach(reader, approach_name, out_dir)
        all_rows.extend(rows)

        if rows:
            acc = np.mean([r["correct"] for r in rows])
            print(f"    -> Accuracy: {acc*100:.1f}%  ({sum(r['correct'] for r in rows)}/{len(rows)})")

    if not all_rows:
        print("\n  No results. Make sure approach outputs/ folders exist.")
        return

    # ── Save detail CSV ───────────────────────────────────────────────────────
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(os.path.join(OUT_DIR, "ocr_results_detail.csv"), index=False)
    print("\n  Saved: ocr_results_detail.csv")

    # ── Summary per approach ──────────────────────────────────────────────────
    summary_rows = []
    for approach_name in APPROACH_OUTPUT_DIRS.keys():
        sub = detail_df[detail_df["approach"] == approach_name]
        if sub.empty:
            continue
        y_true = sub["intended"].tolist()
        y_pred = [r if r else "__none__" for r in sub["recognised"].tolist()]
        labels = sorted(set(y_true))
        accuracy  = float(sub["correct"].mean())
        try:
            macro_f1  = f1_score(y_true, y_pred, labels=labels, average="macro",   zero_division=0)
            macro_pre = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            macro_rec = recall_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        except Exception:
            macro_f1 = macro_pre = macro_rec = 0.0

        summary_rows.append({
            "approach":       approach_name,
            "accuracy":       round(accuracy, 4),
            "macro_precision": round(macro_pre, 4),
            "macro_recall":   round(macro_rec, 4),
            "macro_f1":       round(macro_f1, 4),
            "correct":        int(sub["correct"].sum()),
            "total":          len(sub),
        })
        print(f"  {approach_name}: acc={accuracy*100:.1f}%  F1={macro_f1:.3f}  "
              f"P={macro_pre:.3f}  R={macro_rec:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "ocr_summary.csv"), index=False)
    print("  Saved: ocr_summary.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    acc_dict = {r["approach"]: r["accuracy"] for r in summary_rows}
    f1_dict  = {r["approach"]: r["macro_f1"] for r in summary_rows}

    if acc_dict:
        plot_bar(acc_dict, "Accuracy (%)",
                 "OCR Accuracy — How Often Generated Chars Are Correctly Read\n"
                 "(EasyOCR as domain-agnostic judge)",
                 os.path.join(OUT_DIR, "ocr_accuracy_bar.png"))
        plot_bar(f1_dict, "Macro F1 (%)",
                 "OCR Macro F1 Score Across 62 Character Classes",
                 os.path.join(OUT_DIR, "ocr_f1_bar.png"))
        plot_evolution(summary_df, os.path.join(OUT_DIR, "evolution_chart.png"))

    print(f"\nAll OCR benchmark outputs -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
