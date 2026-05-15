"""
eval_02_classifier.py
=====================
Full evaluation of the Approach 02 MultiTaskCNN classifier.

Outputs saved to:  stats/outputs/approach_02/
  - classification_report.txt   (per-character precision, recall, F1)
  - char_accuracy_bar.png       (per-character accuracy bar chart)
  - confusion_char.png          (62x62 character confusion matrix)
  - confusion_writer.png        (6x6 writer confusion matrix)
  - tsne_by_writer.png          (t-SNE of 256-dim CNN features)
  - summary.csv                 (overall numbers in one row)

Run from ANY directory:
    python stats/scripts/eval_02_classifier.py

Note: Checkpoint format uses key 'model' (not 'model_state') as saved by run.py
"""

import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATS_DIR    = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(STATS_DIR)
APPROACH_02  = os.path.join(PROJECT_ROOT, "approaches", "02_MultiTaskCNN")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "Data", "Writers_pngs")
OUT_DIR      = os.path.join(STATS_DIR, "outputs", "approach_02")

# Checkpoint: try several likely locations
CKPT_CANDIDATES = [
    os.path.join(APPROACH_02,   "checkpoints", "best_model.pt"),
    os.path.join(APPROACH_02,   "checkpoints", "checkpoint.pt"),
    os.path.join(PROJECT_ROOT,  "approaches",  "checkpoints", "best_model.pt"),
    os.path.join(PROJECT_ROOT,  "checkpoints", "best_model.pt"),
]

VAL_SPLIT  = 0.15
BATCH_SIZE = 64
SKIP_DIRS  = {"Writers_Zip", "output_preview", "__pycache__"}

# ── Add approach to sys.path so we can import its modules ─────────────────────
sys.path.insert(0, APPROACH_02)
from dataset import CHAR_TO_LABEL, load_all_writers
from model   import MultiTaskCNN
import torchvision.transforms as T
from torch.utils.data import Dataset

LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}

# ── Inline dataset class — mirrors run.py's WriterDataset exactly ─────────────
_TF = T.Compose([
    T.Grayscale(1), T.Resize((128, 128)),
    T.ToTensor(), T.Normalize((0.5,), (0.5,))
])

class WriterDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        path, char_label, writer_id = self.records[i]
        return _TF(Image.open(path)), char_label, writer_id

def load_data_with_writers(data_root):
    """Load all writers + assign writer IDs. Mirrors run.py's load_data()."""
    raw         = load_all_writers(data_root)
    writer_dirs = sorted(
        e.name for e in os.scandir(data_root)
        if e.is_dir() and e.name not in SKIP_DIRS
    )
    writer_map  = {name: i for i, name in enumerate(writer_dirs)}
    records = [
        (p, cl, writer_map.get(os.path.basename(os.path.dirname(p)), 0))
        for p, cl in raw
    ]
    return records, writer_dirs


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_checkpoint():
    for p in CKPT_CANDIDATES:
        if os.path.isfile(p):
            print(f"  Checkpoint found: {p}")
            return p
    raise FileNotFoundError(
        "No checkpoint found. Train Approach 02 first (Colab_MultiTaskCNN.ipynb) "
        "then place checkpoint.pt in approaches/02_MultiTaskCNN/checkpoints/"
    )


def load_checkpoint(ckpt_path, device):
    """Load checkpoint — handles both 'model' (run.py) and 'model_state' (train.py) keys."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    writer_names = ckpt.get("writer_names", [])
    num_writers  = ckpt.get("num_writers", len(writer_names))
    # Support both key naming conventions
    state_dict   = ckpt.get("model_state", ckpt.get("model", None))
    if state_dict is None:
        raise KeyError(f"Checkpoint has no 'model' or 'model_state' key. Keys: {list(ckpt.keys())}")
    model = MultiTaskCNN(num_writers=num_writers).to(device)
    model.load_state_dict(state_dict)
    return model, writer_names, ckpt


def build_val_loader(seed=42):
    records, writer_names = load_data_with_writers(DATA_ROOT)
    dataset  = WriterDataset(records)
    val_size = int(len(dataset) * VAL_SPLIT)
    _, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0), writer_names


def collect_predictions(model, loader, device):
    model.eval()
    char_preds, char_trues   = [], []
    writer_preds, writer_trues = [], []
    features_list = []

    with torch.no_grad():
        for images, char_labels, writer_labels in loader:
            images = images.to(device)
            feats  = model.extract_features(images).cpu().numpy()
            c_log, w_log = model(images)
            char_preds.append(c_log.argmax(1).cpu())
            char_trues.append(char_labels)
            writer_preds.append(w_log.argmax(1).cpu())
            writer_trues.append(writer_labels)
            features_list.append(feats)

    return (
        torch.cat(char_preds).numpy(),
        torch.cat(char_trues).numpy(),
        torch.cat(writer_preds).numpy(),
        torch.cat(writer_trues).numpy(),
        np.vstack(features_list),
    )


def top_k_accuracy(logits_list, trues_list, model, loader, device, k=5):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, char_labels, _ in loader:
            images = images.to(device)
            c_log, _ = model(images)
            topk = c_log.topk(k, dim=1).indices.cpu()
            for pred_row, true in zip(topk, char_labels):
                if true.item() in pred_row.tolist():
                    correct += 1
                total += 1
    return correct / total


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, labels, title, path, figsize=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    n  = len(labels)
    fs = figsize or (max(8, n // 2), max(7, n // 2))
    fig, ax = plt.subplots(figsize=fs)
    sns.heatmap(cm, annot=(n <= 12), fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3 if n <= 20 else 0)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(fontsize=6 if n > 20 else 9, rotation=45, ha="right")
    plt.yticks(fontsize=6 if n > 20 else 9, rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_per_char_accuracy(char_pred, char_true, path):
    chars, accs = [], []
    for c in sorted(np.unique(char_true)):
        mask = char_true == c
        accs.append((char_pred[mask] == char_true[mask]).mean() * 100)
        chars.append(LABEL_TO_CHAR.get(c, str(c)))

    # Sort by accuracy ascending (hardest first)
    order  = np.argsort(accs)
    chars  = [chars[i] for i in order]
    accs   = [accs[i]  for i in order]

    colors = ["#e74c3c" if a < 50 else "#f39c12" if a < 80 else "#2ecc71" for a in accs]
    fig, ax = plt.subplots(figsize=(22, 5))
    bars = ax.bar(chars, accs, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(np.mean(accs), color="#3498db", linestyle="--", linewidth=1.5, label=f"Mean {np.mean(accs):.1f}%")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Character", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Approach 02 — Per-Character Accuracy (sorted hardest -> easiest)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.xticks(fontsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_tsne(features, writer_true, writer_names, path):
    print("  Running t-SNE (may take ~1 min)…")
    perp = min(30, len(features) - 1)
    emb  = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, len(writer_names)))
    for wid, (name, color) in enumerate(zip(writer_names, palette)):
        mask = writer_true == wid
        ax.scatter(emb[mask, 0], emb[mask, 1], label=name,
                   alpha=0.65, s=18, color=color)
    ax.legend(fontsize=9)
    ax.set_title("t-SNE of CNN Features — Coloured by Writer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Approach 02 — MultiTaskCNN Evaluation ===\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path             = find_checkpoint()
    model, writer_names, ckpt = load_checkpoint(ckpt_path, device)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"writers={len(writer_names)}")

    print("\nBuilding validation split…")
    loader, writer_names_live = build_val_loader()
    if not writer_names:
        writer_names = writer_names_live

    char_pred, char_true, writer_pred, writer_true, features = \
        collect_predictions(model, loader, device)

    # ── Overall numbers ───────────────────────────────────────────────────────
    char_acc   = (char_pred   == char_true).mean()
    writer_acc = (writer_pred == writer_true).mean()
    top5_acc   = top_k_accuracy(None, None, model, loader, device, k=5)

    print(f"\nCharacter Accuracy (Top-1) : {char_acc*100:.2f}%")
    print(f"Character Accuracy (Top-5) : {top5_acc*100:.2f}%")
    print(f"Writer Accuracy            : {writer_acc*100:.2f}%")

    # ── Classification report ─────────────────────────────────────────────────
    char_labels_str = [LABEL_TO_CHAR.get(i, str(i)) for i in sorted(np.unique(char_true))]
    report = classification_report(
        char_true, char_pred,
        labels=sorted(np.unique(char_true)),
        target_names=char_labels_str,
        zero_division=0
    )
    report_path = os.path.join(OUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Approach 02 — MultiTaskCNN Character Classification\n")
        f.write(f"Epoch: {ckpt.get('epoch','?')} | Val split: {VAL_SPLIT}\n\n")
        f.write(report)
    print(f"\n  Saved: classification_report.txt")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_per_char_accuracy(
        char_pred, char_true,
        os.path.join(OUT_DIR, "char_accuracy_bar.png")
    )

    # Full 62-char label list for confusion matrix
    all_char_labels = [LABEL_TO_CHAR.get(i, str(i)) for i in range(len(CHAR_TO_LABEL))]
    plot_confusion(
        char_true, char_pred, all_char_labels,
        "Character Confusion Matrix (62 classes)",
        os.path.join(OUT_DIR, "confusion_char.png")
    )
    plot_confusion(
        writer_true, writer_pred, writer_names,
        "Writer Identification Confusion Matrix",
        os.path.join(OUT_DIR, "confusion_writer.png"),
        figsize=(8, 7)
    )
    plot_tsne(features, writer_true, writer_names,
              os.path.join(OUT_DIR, "tsne_by_writer.png"))

    # ── Summary CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame([{
        "approach":     "02_MultiTaskCNN",
        "char_acc_top1": round(char_acc * 100, 2),
        "char_acc_top5": round(top5_acc * 100, 2),
        "writer_acc":   round(writer_acc * 100, 2),
        "checkpoint_epoch": ckpt.get("epoch", "?"),
    }])
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("  Saved: summary.csv")

    print(f"\nAll Approach 02 outputs -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
