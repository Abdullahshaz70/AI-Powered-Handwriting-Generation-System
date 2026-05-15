"""
eval_image_quality.py
=====================
Shared visual quality metrics for ALL generator approaches (01, 03, 04, 05).
Works on the output PNGs each approach saves to its outputs/ folder.

Metrics computed per approach:
  - SSIM vs real sample   : structural similarity against a matched real image
  - Ink Coverage Ratio    : % of canvas that is "ink" (dark pixels)
  - Pixel Entropy         : how binary the image is (crisp = low entropy)
  - Character Distinctiveness : pairwise cosine similarity across all 62 chars
                                (low = all chars look different = good)
  - Stroke Sharpness      : bimodality of pixel histogram (1.0 = perfectly binary)

Outputs saved to:  stats/outputs/image_quality/
  - metrics_table.csv           full table per approach per character
  - approach_summary.csv        one row per approach with averages
  - ssim_comparison.png         grouped bar chart of SSIM per approach
  - ink_coverage.png            ink coverage per approach
  - sharpness_comparison.png    stroke sharpness per approach
  - distinctiveness.png         pairwise similarity heatmaps

Run from ANY directory:
    python stats/scripts/eval_image_quality.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATS_DIR    = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(STATS_DIR)
DATA_ROOT    = os.path.join(PROJECT_ROOT, "Data", "Writers_pngs")
OUT_DIR      = os.path.join(STATS_DIR, "outputs", "image_quality")

APPROACH_OUTPUT_DIRS = {
    "01_GAN":          os.path.join(PROJECT_ROOT, "approaches", "01_GAN",                        "outputs"),
    "03_FontCNN":      os.path.join(PROJECT_ROOT, "approaches", "03_FontToHandwriting_CNN",       "outputs"),
    "04_BezierReg":    os.path.join(PROJECT_ROOT, "approaches", "04_BezierRegression_RealImages", "outputs"),
    "05_CharNet":      os.path.join(PROJECT_ROOT, "approaches", "05_CharNet",                     "outputs"),
}

CANVAS_SIZE = 128
INK_THRESHOLD = 128   # pixels below this value = ink


# ── Filename -> character mapping ──────────────────────────────────────────────

def parse_char_from_fname(fname):
    """
    Parse character from various filename formats:
      lc_a.png           -> 'a'
      uc_A.png           -> 'A'
      digit_0.png        -> '0'
      writer0_lc_a.png   -> 'a'   (approaches 03 and 04)
      writer0_uc_A.png   -> 'A'
      writer0_digit_0.png -> '0'
    """
    base  = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # Scan all parts for a type marker (lc / uc / digit)
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


def load_outputs(approach_dir):
    """
    Load all PNG outputs. Returns dict: char -> uint8 (128,128) array.
    For multi-writer approaches (03, 04) that output writerN_lc_a.png,
    multiple images per character are averaged to get one representative image.
    Grid images (grid*.png) are skipped.
    """
    buckets = {}   # char -> list of arrays
    for fpath in sorted(glob.glob(os.path.join(approach_dir, "*.png"))):
        bname = os.path.basename(fpath)
        if bname.startswith("grid"):
            continue
        ch = parse_char_from_fname(fpath)
        if ch is None:
            continue
        try:
            arr = np.array(
                Image.open(fpath).convert("L").resize((CANVAS_SIZE, CANVAS_SIZE)),
                dtype=np.float32
            )
            buckets.setdefault(ch, []).append(arr)
        except Exception:
            pass
    # Average across writers if multiple images exist
    return {ch: np.clip(np.mean(arrs, axis=0), 0, 255).astype(np.uint8)
            for ch, arrs in buckets.items()}


# ── Build real-image lookup: char -> list of np arrays ────────────────────────

def build_real_index():
    """
    Scan Writers_pngs and build a dict: char -> [array, ...] (all writers).
    Used for SSIM reference.
    """
    import string
    CHAR_TO_LABEL = {c: i for i, c in enumerate(
        string.ascii_uppercase + string.ascii_lowercase + string.digits
    )}
    real = {}
    if not os.path.isdir(DATA_ROOT):
        print(f"  WARNING: Data folder not found: {DATA_ROOT}")
        return real

    skip = {"Writers_Zip", "output_preview", "__pycache__"}
    for writer_entry in sorted(os.scandir(DATA_ROOT), key=lambda e: e.name):
        if not writer_entry.is_dir() or writer_entry.name in skip:
            continue
        for fname in os.listdir(writer_entry.path):
            if not fname.lower().endswith(".png"):
                continue
            parts = os.path.splitext(fname)[0].split("_")
            ch = None
            for i, p in enumerate(parts):
                if p in ("lc", "uc") and i + 1 < len(parts):
                    c = parts[i + 1]
                    if len(c) == 1 and c in CHAR_TO_LABEL:
                        ch = c
                        break
            if ch is None and len(parts) > 2:
                c = parts[2]
                if len(c) == 1 and c in CHAR_TO_LABEL:
                    ch = c
            if ch is None:
                continue
            fpath = os.path.join(writer_entry.path, fname)
            try:
                arr = np.array(
                    Image.open(fpath).convert("L").resize((CANVAS_SIZE, CANVAS_SIZE)),
                    dtype=np.uint8
                )
                real.setdefault(ch, []).append(arr)
            except Exception:
                pass
    print(f"  Real index built: {len(real)} characters, "
          f"{sum(len(v) for v in real.values())} total images")
    return real


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_ssim_vs_real(gen_img, real_imgs):
    """Average SSIM of generated image against all real samples for that char."""
    if not real_imgs:
        return np.nan
    scores = []
    for ref in real_imgs:
        try:
            s = ssim_fn(gen_img.astype(float), ref.astype(float),
                        data_range=255.0)
            scores.append(s)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else np.nan


def compute_ink_coverage(img):
    """Fraction of pixels that are ink (below threshold)."""
    return float(np.mean(img < INK_THRESHOLD))


def compute_pixel_entropy(img):
    """
    Shannon entropy of the pixel histogram.
    Low = near-binary (crisp strokes).
    High = many grey levels (blurry).
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_stroke_sharpness(img):
    """
    Bimodality coefficient B = (skewness^2 + 1) / kurtosis.
    High B -> more bimodal -> sharper black/white strokes.
    Clipped to [0, 1] range using logistic transform for interpretability.
    """
    from scipy.stats import skew, kurtosis
    flat = img.flatten().astype(float)
    s = skew(flat)
    k = kurtosis(flat, fisher=False)  # Pearson kurtosis
    k = max(k, 1e-6)
    b = (s ** 2 + 1) / k
    # Logistic normalisation: typical bimodal images have B ~0.5–1.5
    return float(1 / (1 + np.exp(-2 * (b - 0.5))))


def compute_distinctiveness(images_dict):
    """
    Pairwise cosine similarity across all 62 generated images.
    Returns mean pairwise similarity (lower = more distinct = better).
    Also returns the full similarity matrix for heatmap plotting.
    """
    chars = sorted(images_dict.keys())
    vecs  = np.array([images_dict[c].flatten().astype(float) for c in chars])
    # L2 normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_n = vecs / norms
    sim_matrix = vecs_n @ vecs_n.T
    # Mean off-diagonal (exclude self-similarity)
    n = len(chars)
    mask = ~np.eye(n, dtype=bool)
    return float(sim_matrix[mask].mean()), sim_matrix, chars


# ── Plotting ──────────────────────────────────────────────────────────────────

APPROACH_COLORS = {
    "01_GAN":       "#e74c3c",
    "03_FontCNN":   "#f39c12",
    "04_BezierReg": "#3498db",
    "05_CharNet":   "#2ecc71",
}


def bar_comparison(data_dict, ylabel, title, path, higher_is_better=True):
    """
    data_dict: {approach_name: value}
    """
    names  = list(data_dict.keys())
    values = [data_dict[n] for n in names]
    colors = [APPROACH_COLORS.get(n, "#95a5a6") for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    arrow = "↑ higher is better" if higher_is_better else "↓ lower is better"
    ax.set_xlabel(arrow, fontsize=9, color="#555")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_distinctiveness_heatmaps(sim_data, path):
    """
    sim_data: dict approach -> (mean_sim, matrix, chars)
    """
    n_approaches = len(sim_data)
    if n_approaches == 0:
        return
    fig, axes = plt.subplots(1, n_approaches, figsize=(5 * n_approaches, 5))
    if n_approaches == 1:
        axes = [axes]
    for ax, (name, (mean_sim, matrix, chars)) in zip(axes, sim_data.items()):
        sns.heatmap(matrix, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False, cbar=True)
        ax.set_title(f"{name}\nmean sim={mean_sim:.3f}", fontsize=10, fontweight="bold")
    fig.suptitle("Pairwise Cosine Similarity of 62 Generated Characters\n"
                 "(Green = distinct chars, Red = mode collapse)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Image Quality Evaluation (Approaches 01, 03, 04, 05) ===\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building real image index (for SSIM reference)…")
    real_index = build_real_index()

    all_rows    = []
    summaries   = {}
    sim_data    = {}

    for approach_name, out_dir in APPROACH_OUTPUT_DIRS.items():
        if not os.path.isdir(out_dir):
            print(f"  SKIP {approach_name} — outputs/ folder not found: {out_dir}")
            print(f"       Run: python approaches/{approach_name.split('_')[0]}*/run.py first")
            continue

        images = load_outputs(out_dir)
        if not images:
            print(f"  SKIP {approach_name} — no PNG files found in {out_dir}")
            continue

        print(f"\n  {approach_name}: {len(images)} characters loaded")

        ssims, coverages, entropies, sharpnesses = [], [], [], []

        for ch, img in images.items():
            s = compute_ssim_vs_real(img, real_index.get(ch, []))
            c = compute_ink_coverage(img)
            e = compute_pixel_entropy(img)
            sh = compute_stroke_sharpness(img)

            ssims.append(s)
            coverages.append(c)
            entropies.append(e)
            sharpnesses.append(sh)

            all_rows.append({
                "approach":   approach_name,
                "char":       ch,
                "ssim":       round(s, 4) if not np.isnan(s) else None,
                "ink_coverage": round(c, 4),
                "pixel_entropy": round(e, 4),
                "sharpness":  round(sh, 4),
            })

        # Distinctiveness
        mean_sim, sim_matrix, chars = compute_distinctiveness(images)
        sim_data[approach_name] = (mean_sim, sim_matrix, chars)

        avg_ssim  = float(np.nanmean(ssims))
        avg_cov   = float(np.mean(coverages))
        avg_ent   = float(np.mean(entropies))
        avg_sharp = float(np.mean(sharpnesses))

        summaries[approach_name] = {
            "avg_ssim":            round(avg_ssim, 4),
            "avg_ink_coverage":    round(avg_cov,  4),
            "avg_pixel_entropy":   round(avg_ent,  4),
            "avg_sharpness":       round(avg_sharp, 4),
            "avg_distinctiveness": round(1.0 - mean_sim, 4),  # higher = more distinct
            "num_chars":           len(images),
        }
        print(f"    SSIM={avg_ssim:.3f}  Coverage={avg_cov:.3f}  "
              f"Entropy={avg_ent:.3f}  Sharpness={avg_sharp:.3f}  "
              f"Distinctiveness={1-mean_sim:.3f}")

    if not summaries:
        print("\n  No approach outputs found. Run the approaches first to generate PNGs.")
        return

    # ── Save tables ───────────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUT_DIR, "metrics_table.csv"), index=False
    )
    summary_df = pd.DataFrame([
        {"approach": k, **v} for k, v in summaries.items()
    ])
    summary_df.to_csv(os.path.join(OUT_DIR, "approach_summary.csv"), index=False)
    print("\n  Saved: metrics_table.csv, approach_summary.csv")

    # ── Bar charts ────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots…")
    if len(summaries) > 1:
        bar_comparison(
            {k: v["avg_ssim"] for k, v in summaries.items()},
            "Average SSIM vs Real Samples",
            "Structural Similarity — Generated vs Real Handwriting\n(higher = more similar to real writing)",
            os.path.join(OUT_DIR, "ssim_comparison.png"),
            higher_is_better=True
        )
        bar_comparison(
            {k: v["avg_ink_coverage"] for k, v in summaries.items()},
            "Average Ink Coverage (fraction of canvas)",
            "Ink Coverage — How Much of the Canvas is Filled\n(too low = faint, too high = bloated)",
            os.path.join(OUT_DIR, "ink_coverage.png"),
            higher_is_better=False
        )
        bar_comparison(
            {k: v["avg_sharpness"] for k, v in summaries.items()},
            "Stroke Sharpness Score (0–1)",
            "Stroke Sharpness — How Clean/Binary the Ink Is\n(higher = crisper strokes)",
            os.path.join(OUT_DIR, "sharpness_comparison.png"),
            higher_is_better=True
        )
        bar_comparison(
            {k: v["avg_distinctiveness"] for k, v in summaries.items()},
            "Character Distinctiveness (1 − mean cosine sim)",
            "Character Distinctiveness — Are All 62 Characters Different?\n(higher = less mode collapse)",
            os.path.join(OUT_DIR, "distinctiveness_bar.png"),
            higher_is_better=True
        )

    plot_distinctiveness_heatmaps(
        sim_data,
        os.path.join(OUT_DIR, "distinctiveness_heatmaps.png")
    )

    print(f"\nAll image quality outputs -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
