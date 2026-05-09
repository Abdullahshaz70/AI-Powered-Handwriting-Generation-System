import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CHAR_TO_LABEL
from model import MultiTaskCNN
from train import MultiTaskCharDataset

CKPT_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_model.pt")
LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = MultiTaskCNN(num_writers=ckpt["num_writers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["writer_names"]


def preprocess(image_path):
    dummy_dataset = MultiTaskCharDataset([(image_path, 0, 0)])
    tensor, _, _ = dummy_dataset[0]
    return tensor.unsqueeze(0)


def predict_single(model, tensor, writer_names, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        char_logits, writer_logits = model(tensor)

    char_probs   = torch.softmax(char_logits, dim=1)[0]
    writer_probs = torch.softmax(writer_logits, dim=1)[0]

    char_conf, char_idx     = char_probs.max(0)
    writer_conf, writer_idx = writer_probs.max(0)

    return (
        LABEL_TO_CHAR[char_idx.item()],   char_conf.item(),
        writer_names[writer_idx.item()],  writer_conf.item(),
    )


def main():
    parser = argparse.ArgumentParser(description="Handwriting multi-task CNN demo")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",         help="Path to a single character PNG")
    group.add_argument("--writer_folder", help="Path to a folder of character PNGs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, writer_names = load_model(device)

    if args.image:
        tensor = preprocess(args.image)
        char, c_conf, writer, w_conf = predict_single(model, tensor, writer_names, device)
        print(f"Predicted character : {char}  (confidence {c_conf*100:.1f}%)")
        print(f"Predicted writer    : {writer}  (confidence {w_conf*100:.1f}%)")

    else:
        folder = args.writer_folder
        images = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        if not images:
            print(f"No PNG files found in {folder}")
            return

        char_counts   = {}
        writer_counts = {}
        total = len(images)

        print(f"\nRunning inference on {total} images in {folder}\n")
        print(f"{'File':<40} {'Char':>5} {'CharConf':>9} {'Writer':<30} {'WriterConf':>10}")
        print("-" * 100)

        for fname in images:
            path = os.path.join(folder, fname)
            try:
                tensor = preprocess(path)
                char, c_conf, writer, w_conf = predict_single(model, tensor, writer_names, device)
                char_counts[char]     = char_counts.get(char, 0) + 1
                writer_counts[writer] = writer_counts.get(writer, 0) + 1
                print(f"{fname:<40} {char:>5} {c_conf*100:>8.1f}% {writer:<30} {w_conf*100:>9.1f}%")
            except Exception as e:
                print(f"{fname:<40}  ERROR: {e}")

        print("\n── Summary ──────────────────────────────────────────────────────")
        top_char   = max(char_counts,   key=char_counts.get)
        top_writer = max(writer_counts, key=writer_counts.get)
        print(f"Most predicted character : '{top_char}'  ({char_counts[top_char]}/{total})")
        print(f"Most predicted writer    : {top_writer}  ({writer_counts[top_writer]}/{total})")


if __name__ == "__main__":
    main()
