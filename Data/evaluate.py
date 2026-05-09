import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CHAR_TO_LABEL
from model import MultiTaskCNN
from train import MultiTaskCharDataset, load_all_writers_with_id

CKPT_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_model.pt")
DATA_ROOT  = os.path.join(os.path.dirname(__file__), "Writers_pngs")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_output")
VAL_SPLIT  = 0.15
BATCH_SIZE = 64

LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}


def load_val_loader(all_data, seed=42):
    dataset  = MultiTaskCharDataset(all_data)
    val_size = int(len(dataset) * VAL_SPLIT)
    _, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def collect_predictions(model, loader, device):
    model.eval()
    all_char_pred, all_char_true = [], []
    all_writer_pred, all_writer_true = [], []
    all_features = []

    with torch.no_grad():
        for images, char_labels, writer_labels in loader:
            images = images.to(device)
            features = model.extract_features(images).cpu().numpy()
            char_logits, writer_logits = model(images)

            all_char_pred.append(char_logits.argmax(1).cpu())
            all_char_true.append(char_labels)
            all_writer_pred.append(writer_logits.argmax(1).cpu())
            all_writer_true.append(writer_labels)
            all_features.append(features)

    return (
        torch.cat(all_char_pred).numpy(),
        torch.cat(all_char_true).numpy(),
        torch.cat(all_writer_pred).numpy(),
        torch.cat(all_writer_true).numpy(),
        np.vstack(all_features),
    )


def plot_confusion_matrix(pred, true, labels, title, save_path):
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(features, writer_true, writer_names, save_path):
    print("Running t-SNE (this may take a minute)...")
    perp = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(writer_names)))
    for wid, (name, color) in enumerate(zip(writer_names, colors)):
        mask = writer_true == wid
        ax.scatter(emb[mask, 0], emb[mask, 1], label=name, alpha=0.6,
                   s=18, color=color)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("t-SNE of CNN features coloured by writer")
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    writer_names = ckpt["writer_names"]
    num_writers  = ckpt["num_writers"]
    print(f"Checkpoint: epoch={ckpt['epoch']}, "
          f"char_acc={ckpt['char_acc']*100:.2f}%, writer_acc={ckpt['writer_acc']*100:.2f}%")

    model = MultiTaskCNN(num_writers=num_writers).to(device)
    model.load_state_dict(ckpt["model_state"])

    print("\nLoading dataset...")
    all_data, _ = load_all_writers_with_id(DATA_ROOT)
    loader = load_val_loader(all_data)

    char_pred, char_true, writer_pred, writer_true, features = collect_predictions(model, loader, device)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1 & 2: overall accuracies ──────────────────────────────────────────────
    char_acc   = (char_pred   == char_true).mean()
    writer_acc = (writer_pred == writer_true).mean()
    print(f"\nOverall character accuracy : {char_acc*100:.2f}%")
    print(f"Overall writer accuracy    : {writer_acc*100:.2f}%")

    # ── 3: per-character accuracy ──────────────────────────────────────────────
    print("\nPer-character accuracy (sorted by difficulty):")
    char_accs = {}
    for c in np.unique(char_true):
        mask = char_true == c
        char_accs[LABEL_TO_CHAR[c]] = (char_pred[mask] == char_true[mask]).mean()
    for ch, acc in sorted(char_accs.items(), key=lambda x: x[1]):
        print(f"  '{ch}': {acc*100:.1f}%")

    # ── 4: per-writer accuracy ─────────────────────────────────────────────────
    print("\nPer-writer accuracy:")
    for wid, name in enumerate(writer_names):
        mask = writer_true == wid
        if mask.sum() == 0:
            continue
        acc = (writer_pred[mask] == writer_true[mask]).mean()
        print(f"  {name}: {acc*100:.1f}%  (n={mask.sum()})")

    # ── 5: writer confusion matrix ─────────────────────────────────────────────
    plot_confusion_matrix(
        writer_pred, writer_true, writer_names,
        "Writer Identification Confusion Matrix",
        os.path.join(OUT_DIR, "writer_confusion.png"),
    )

    # ── 6: t-SNE ──────────────────────────────────────────────────────────────
    plot_tsne(features, writer_true, writer_names,
              os.path.join(OUT_DIR, "tsne_by_writer.png"))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
