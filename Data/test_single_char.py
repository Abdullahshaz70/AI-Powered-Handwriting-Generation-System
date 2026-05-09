import os
import torch
import numpy as np
from PIL import Image
from encoder import StyleEncoder
from generator import CharacterGenerator
from dataset import load_dataset, CharDataset

DEVICE = torch.device("cpu")
_HERE = os.path.dirname(os.path.abspath(__file__))

encoder = StyleEncoder().to(DEVICE)
generator = CharacterGenerator().to(DEVICE)
ckpt = torch.load(os.path.join(_HERE, "..", "checkpoints", "best_model.pt"), map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder_state"])
generator.load_state_dict(ckpt["generator_state"])
encoder.eval()
generator.eval()

data_list = load_dataset(os.path.join(_HERE, "Writers_pngs", "writer_Abdullah"))
dataset = CharDataset(data_list)
imgs = torch.stack([dataset[i][0] for i in range(min(32, len(dataset)))])
with torch.no_grad():
    style = encoder(imgs).mean(dim=0, keepdim=True)

label = torch.tensor([0], dtype=torch.long)  # 'A'
with torch.no_grad():
    out = generator(style, label)

arr = out.squeeze().cpu().numpy()
print(f"Raw tanh output -> min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

out_dir = os.path.join(_HERE, "..", "char_test")
os.makedirs(out_dir, exist_ok=True)

# Version 1: raw model output scaled to 0-255, upscaled 4x so we can see pixels
raw = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
img_raw = Image.fromarray(raw, "L").resize((512, 512), Image.NEAREST)
img_raw.save(os.path.join(out_dir, "v1_raw.png"))
print("v1_raw.png  — raw output (paper=dark, ink=bright)")

# Version 2: inverted (flip polarity)
inv = (255 - raw)
img_inv = Image.fromarray(inv, "L").resize((512, 512), Image.NEAREST)
img_inv.save(os.path.join(out_dir, "v2_inverted.png"))
print("v2_inverted.png  — inverted (ink=dark, paper=bright)")

# Version 3: inverted + contrast stretch
lo, hi = inv.min(), inv.max()
print(f"After inversion: lo={lo}  hi={hi}")
if hi > lo:
    stretched = ((inv.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
else:
    stretched = inv
img_str = Image.fromarray(stretched, "L").resize((512, 512), Image.NEAREST)
img_str.save(os.path.join(out_dir, "v3_stretched.png"))
print("v3_stretched.png  — inverted + contrast stretched")

# Version 4: raw contrast stretch (no inversion)
lo2, hi2 = raw.min(), raw.max()
print(f"Raw range: lo={lo2}  hi={hi2}")
if hi2 > lo2:
    stretched2 = ((raw.astype(np.float32) - lo2) / (hi2 - lo2) * 255).astype(np.uint8)
else:
    stretched2 = raw
img_str2 = Image.fromarray(stretched2, "L").resize((512, 512), Image.NEAREST)
img_str2.save(os.path.join(out_dir, "v4_raw_stretched.png"))
print("v4_raw_stretched.png  — raw + contrast stretched (no inversion)")

# Version 5: inverted + contrast stretch + gamma (what renderer now uses, at 512x512)
lo3, hi3 = inv.min(), inv.max()
if hi3 > lo3:
    s3 = ((inv.astype(np.float32) - lo3) / (hi3 - lo3) * 255).astype(np.uint8)
else:
    s3 = inv
img5 = Image.fromarray(s3, "L").resize((512, 512), Image.LANCZOS)
arr5 = np.array(img5).astype(np.float32) / 255.0
arr5 = np.power(arr5, 4.0)
img5 = Image.fromarray((arr5 * 255).astype(np.uint8), "L")
img5.save(os.path.join(out_dir, "v5_gamma.png"))
print("v5_gamma.png  — inverted + stretched + gamma=4 (current renderer logic)")

print(f"\nAll images saved to: {out_dir}")
print("Check v5_gamma.png — if this shows a character, the renderer should work.")
