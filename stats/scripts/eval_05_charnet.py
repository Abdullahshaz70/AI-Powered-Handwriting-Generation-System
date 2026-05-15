"""
eval_05_charnet.py
==================
Evaluates the Approach 05 CharNet (MLP) model.

Metrics produced:
  - Per-character Bézier MSE (how accurately the model predicts each character's curves)
  - Per-category MSE  (uppercase vs lowercase vs digits)
  - Checkpoint info   (epoch, overall val_MSE)
  - Curve endpoint deviation (how far predicted curve endpoints drift from ground truth)

Outputs saved to:  stats/outputs/approach_05/
  - per_char_mse_bar.png        bar chart of MSE per character
  - category_mse_bar.png        grouped bar: uppercase / lowercase / digit
  - endpoint_deviation_bar.png  endpoint L2 error per character
  - summary.csv                 one-row summary

Run from ANY directory:
    python stats/scripts/eval_05_charnet.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATS_DIR    = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(STATS_DIR)
APPROACH_05  = os.path.join(PROJECT_ROOT, "approaches", "05_CharNet")
SRC_DIR      = os.path.join(APPROACH_05, "src")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "Data", "Writers_pngs")
CACHE_PATH   = os.path.join(PROJECT_ROOT, "Data", "bezier_labels.npy")
OUT_DIR      = os.path.join(STATS_DIR, "outputs", "approach_05")

CKPT_CANDIDATES = [
    os.path.join(APPROACH_05, "checkpoints", "style_net.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "style_net.pt"),
]

# ── Add CharNet src to sys.path ────────────────────────────────────────────────
sys.path.insert(0, SRC_DIR)
from model  import CharNet, char_idx_to_cat
from bezier import LABEL_DIM, label_to_curves, N_CURVES, CP_PER_CURVE
from data   import (CHAR_TO_IDX, IDX_TO_CHAR, NUM_CHARS,
                    CANVAS_SIZE, load_all_data, char_idx_to_cat as data_cat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_checkpoint():
    for p in CKPT_CANDIDATES:
        if os.path.isfile(p):
            print(f"  Checkpoint found: {p}")
            return p
    raise FileNotFoundError(
        "CharNet checkpoint not found. "
        "Train in Colab (Colab_CharNet.ipynb) and place style_net.pt "
        "in approaches/05_CharNet/checkpoints/"
    )


def predict_all_chars(model, device):
    """Run model for all 62 characters and return predictions (62, LABEL_DIM)."""
    preds = []
    model.eval()
    with torch.no_grad():
        for idx in range(NUM_CHARS):
            char_t = torch.tensor([idx], device=device)
            cat_t  = torch.tensor([data_cat(idx)], device=device)
            out    = model(char_t, cat_t).squeeze().cpu().numpy()
            preds.append(out)
    return np.array(preds)  # (62, 48)


def compute_gt_per_char(records):
    """
    Average all ground-truth Bézier labels per character.
    records: list of (img_path, char_idx, writer_idx, label_48)
    Returns: dict  char_idx -> mean_label (np array 48,)
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for _, char_idx, _, label in records:
        buckets[char_idx].append(label)
    return {c: np.mean(np.array(lbls), axis=0) for c, lbls in buckets.items()}


def endpoint_deviation(pred_label, gt_label):
    """
    Average L2 distance between predicted and GT curve endpoints.
    Each cubic Bézier has P0 (start) and P3 (end) — indices 0-1 and 6-7 within each 8-float block.
    """
    deviations = []
    for i in range(N_CURVES):
        base = i * CP_PER_CURVE * 2
        # P0
        p0_pred = pred_label[base:base+2]
        p0_gt   = gt_label[base:base+2]
        # P3
        p3_pred = pred_label[base+6:base+8]
        p3_gt   = gt_label[base+6:base+8]
        deviations.append(np.linalg.norm(p0_pred - p0_gt))
        deviations.append(np.linalg.norm(p3_pred - p3_gt))
    return float(np.mean(deviations))


# ── Plot helpers ──────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Uppercase": "#3498db",
    "Lowercase": "#2ecc71",
    "Digit":     "#e74c3c",
}

def char_category(char_idx):
    if char_idx < 26:  return "Uppercase"
    if char_idx < 52:  return "Lowercase"
    return "Digit"


def plot_per_char_mse(char_mses, path):
    chars   = [IDX_TO_CHAR[i] for i in range(NUM_CHARS) if i in char_mses]
    indices = [i for i in range(NUM_CHARS) if i in char_mses]
    mses    = [char_mses[i] for i in indices]

    # Sort by MSE descending (worst first)
    order  = np.argsort(mses)[::-1]
    chars  = [chars[i] for i in order]
    indices = [indices[i] for i in order]
    mses   = [mses[i] for i in order]
    colors = [
        "#e74c3c" if char_category(indices[i]) == "Uppercase"
        else "#2ecc71" if char_category(indices[i]) == "Lowercase"
        else "#f39c12"
        for i in range(len(indices))
    ]

    fig, ax = plt.subplots(figsize=(22, 5))
    ax.bar(chars, mses, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(np.mean(mses), color="#333", linestyle="--",
               linewidth=1.4, label=f"Mean MSE = {np.mean(mses):.4f}")
    ax.set_xlabel("Character", fontsize=11)
    ax.set_ylabel("Bézier MSE", fontsize=11)
    ax.set_title("Approach 05 CharNet — Per-Character Bézier MSE (worst -> best)", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_els = [Patch(color="#e74c3c", label="Uppercase"),
                  Patch(color="#2ecc71", label="Lowercase"),
                  Patch(color="#f39c12", label="Digit")]
    ax.legend(handles=legend_els + [
        plt.Line2D([0],[0], color="#333", linestyle="--", label=f"Mean {np.mean(mses):.4f}")
    ])
    plt.xticks(fontsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_category_mse(char_mses, path):
    cat_vals = {"Uppercase": [], "Lowercase": [], "Digit": []}
    for idx, mse in char_mses.items():
        cat_vals[char_category(idx)].append(mse)

    cats  = list(cat_vals.keys())
    means = [np.mean(cat_vals[c]) for c in cats]
    stds  = [np.std(cat_vals[c])  for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(cats, means, yerr=stds, color=colors,
                  capsize=6, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean Bézier MSE ± Std", fontsize=11)
    ax.set_title("Approach 05 — MSE by Character Category", fontsize=13, fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_endpoint_deviation(ep_devs, path):
    indices = sorted(ep_devs.keys())
    chars   = [IDX_TO_CHAR[i] for i in indices]
    devs    = [ep_devs[i] for i in indices]
    colors  = [
        "#e74c3c" if char_category(i) == "Uppercase"
        else "#2ecc71" if char_category(i) == "Lowercase"
        else "#f39c12"
        for i in indices
    ]

    fig, ax = plt.subplots(figsize=(22, 5))
    ax.bar(chars, devs, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(np.mean(devs), color="#333", linestyle="--",
               linewidth=1.4, label=f"Mean = {np.mean(devs):.4f}")
    ax.set_xlabel("Character", fontsize=11)
    ax.set_ylabel("Mean Endpoint L2 Deviation", fontsize=11)
    ax.set_title("Approach 05 — Curve Endpoint Deviation (lower = more accurate stroke placement)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.xticks(fontsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Approach 05 — CharNet Evaluation ===\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint()
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}  val_MSE={ckpt.get('val_loss', 0):.6f}")

    model = CharNet(num_chars=62).to(device)
    model.load_state_dict(ckpt["model_state"])

    print("\nLoading data and Bézier label cache…")
    records, writer_names = load_all_data(DATA_ROOT, cache_path=CACHE_PATH)
    print(f"  {len(records)} total records, {len(writer_names)} writers")

    gt_per_char   = compute_gt_per_char(records)
    preds         = predict_all_chars(model, device)   # (62, 48)

    # ── Compute metrics ───────────────────────────────────────────────────────
    char_mses = {}
    ep_devs   = {}
    for idx in range(NUM_CHARS):
        if idx not in gt_per_char:
            continue
        pred = preds[idx]
        gt   = gt_per_char[idx]
        char_mses[idx] = float(np.mean((pred - gt) ** 2))
        ep_devs[idx]   = endpoint_deviation(pred, gt)

    overall_mse = np.mean(list(char_mses.values()))
    overall_epd = np.mean(list(ep_devs.values()))
    print(f"\nOverall Bézier MSE          : {overall_mse:.6f}")
    print(f"Overall Endpoint Deviation  : {overall_epd:.6f}")

    for cat_name, indices in [
        ("Uppercase", range(0, 26)),
        ("Lowercase", range(26, 52)),
        ("Digits",    range(52, 62)),
    ]:
        vals = [char_mses[i] for i in indices if i in char_mses]
        if vals:
            print(f"  {cat_name:10s} MSE: {np.mean(vals):.6f}")

    print("\nGenerating plots…")
    plot_per_char_mse(char_mses, os.path.join(OUT_DIR, "per_char_mse_bar.png"))
    plot_category_mse(char_mses, os.path.join(OUT_DIR, "category_mse_bar.png"))
    plot_endpoint_deviation(ep_devs, os.path.join(OUT_DIR, "endpoint_deviation_bar.png"))

    # ── Summary CSV ───────────────────────────────────────────────────────────
    rows = []
    for idx in sorted(char_mses.keys()):
        rows.append({
            "char":     IDX_TO_CHAR[idx],
            "category": char_category(idx),
            "mse":      round(char_mses[idx], 6),
            "endpoint_deviation": round(ep_devs.get(idx, 0), 6),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_char_metrics.csv"), index=False)

    summary = pd.DataFrame([{
        "approach":       "05_CharNet",
        "overall_mse":    round(overall_mse, 6),
        "val_mse_ckpt":   round(ckpt.get("val_loss", 0), 6),
        "endpoint_dev":   round(overall_epd, 6),
        "epoch":          ckpt.get("epoch", "?"),
    }])
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("  Saved: per_char_metrics.csv, summary.csv")

    print(f"\nAll Approach 05 outputs -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
