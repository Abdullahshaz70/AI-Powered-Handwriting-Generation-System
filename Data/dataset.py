import string
import os
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter

class RandomRotation:
    def __init__(self, max_angle=5):
        self.max_angle = max_angle

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.max_angle, self.max_angle)
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

class RandomScaling:
    def __init__(self, scale_range=(0.9, 1.1), size=128):
        self.scale_range = scale_range
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(*self.scale_range)
        new_size = int(self.size * scale)
        img = img.resize((new_size, new_size), Image.BILINEAR)

        canvas = Image.new("L", (self.size, self.size), 0)
        x = (self.size - new_size) // 2
        y = (self.size - new_size) // 2
        canvas.paste(img, (x, y))
        return canvas

class ElasticDistortion:
    def __init__(self, alpha=5, sigma=4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)

        shape = arr.shape
        dx = (np.random.rand(*shape) * 2 - 1) * self.alpha
        dy = (np.random.rand(*shape) * 2 - 1) * self.alpha

        dx = gaussian_filter(dx, sigma=self.sigma)
        dy = gaussian_filter(dy, sigma=self.sigma)

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x_new = np.clip((x + dx).astype(int), 0, shape[1] - 1)
        y_new = np.clip((y + dy).astype(int), 0, shape[0] - 1)

        distorted = arr[y_new, x_new]
        return Image.fromarray(distorted.astype(np.uint8))

class Binarize:
    def __init__(self , threshold = 128):
        self.threshold = threshold

    def __call__(self, img : Image.Image) -> Image.Image:

        img = img.convert("L")

        arr = np.array(img)

        bin_arr = np.where(arr < self.threshold, 255, 0).astype(np.uint8)

        return Image.fromarray(bin_arr)

class CenterCharacter:
    def __init__(self , target_size = 128):
        self.target_size = target_size

    def __call__(self, img : Image.Image) -> Image.Image:

        img = img.convert('L')

        arr = np.array(img)

        co_ordinates = np.column_stack(np.where (arr>0))

        if co_ordinates.size == 0:
            return img.resize((self.target_size, self.target_size))

        top_left = co_ordinates.min(axis=0)
        bottom_right = co_ordinates.max(axis=0)

        top, left = top_left
        bottom, right = bottom_right

        cropped = arr[top:bottom + 1, left:right + 1]

        h, w = cropped.shape

        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        y_offset = (self.target_size - h) // 2
        x_offset = (self.target_size - w) // 2


        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = cropped


        return Image.fromarray(canvas)

class CharDataset(Dataset):
    def __init__(self , data_list):
        self.data = data_list

        self.transform = transforms.Compose([
            T.Grayscale(num_output_channels=1),
            Binarize(),
            CenterCharacter(),
            RandomRotation(),
            RandomScaling(),
            ElasticDistortion(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path, label = self.data[i]

        image = Image.open(img_path)

        image = self.transform(image)

        return image, label

all_chars = string.ascii_uppercase + string.ascii_lowercase  + string.digits
CHAR_TO_LABEL = {}
index = 0

for char in all_chars:
    CHAR_TO_LABEL[char] = index
    index+=1



def load_dataset(folder_path):
    dataset = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            continue

        parts = os.path.splitext(filename)[0].split('_')

        # Robust char extraction: find 'lc'/'uc' case marker, take the next part.
        # This handles both "Name_lc_a_r01" and "Name_24067_lc_a_r01" patterns.
        char_in_file = None
        for i, part in enumerate(parts):
            if part in ('lc', 'uc') and i + 1 < len(parts):
                candidate = parts[i + 1]
                if len(candidate) == 1 and candidate in CHAR_TO_LABEL:
                    char_in_file = candidate
                    break

        # Fallback: original index-2 approach for any other naming convention
        if char_in_file is None and len(parts) > 2:
            candidate = parts[2]
            if len(candidate) == 1 and candidate in CHAR_TO_LABEL:
                char_in_file = candidate

        if char_in_file is None:
            continue

        full_path = os.path.join(folder_path, filename)
        label = CHAR_TO_LABEL[char_in_file]
        dataset.append((full_path, label))

    return dataset


def load_all_writers(root_folder):
    all_data = []
    skip_dirs = {'Writers_Zip', 'output_preview'}

    for entry in os.scandir(root_folder):
        if not entry.is_dir() or entry.name in skip_dirs:
            continue
        writer_data = load_dataset(entry.path)
        if writer_data:
            print(f"  Loaded {len(writer_data)} samples from {entry.name}")
            all_data.extend(writer_data)

    print(f"Total samples across all writers: {len(all_data)}")
    return all_data




if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder_path = "D:\\Semester_4\\Programming for AI\\Projects\\AI-Powered-Handwriting-Generation-System\\Data\\Writers_pngs\\writer_Abdullah"
    output_path = "D:\\Semester_4\\Programming for AI\\Projects\\AI-Powered-Handwriting-Generation-System\\Data\\output_preview"

    os.makedirs(output_path, exist_ok=True)

    data_list = load_dataset(folder_path)
    print(f"Total images found: {len(data_list)}")


    preview_list = data_list[:16]
    dataset = CharDataset(preview_list)

    LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}


    for idx in range(len(dataset)):
        tensor, label = dataset[idx]
        char = LABEL_TO_CHAR[label]

        img_arr = ((tensor.squeeze().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        orig_filename = os.path.basename(preview_list[idx][0])
        out_name = f"processed_{orig_filename}"
        img.save(os.path.join(output_path, out_name))

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        tensor, label = dataset[idx]
        char = LABEL_TO_CHAR[label]
        img_arr = ((tensor.squeeze().numpy() + 1) / 2).clip(0, 1)
        ax.imshow(img_arr, cmap="gray")
        ax.set_title(f"'{char}' (label={label})", fontsize=9)
        ax.axis("off")
    plt.suptitle("Preprocessed Samples — writer_Abdullah", fontsize=12)
    plt.tight_layout()
    grid_path = os.path.join(output_path, "preview_grid.png")
    plt.savefig(grid_path, dpi=120)
    plt.close()

    print(f"Saved {len(dataset)} individual images to: {output_path}")
    print(f"Saved preview grid to: {grid_path}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    images, labels = next(iter(dataloader))
    print(f"\nBatch tensor shape : {images.shape}")
    print(f"Pixel min / max    : {images.min():.4f} / {images.max():.4f}")
    print(f"Pixel mean / std   : {images.mean():.4f} / {images.std():.4f}")
    print(f"Labels in batch    : {[LABEL_TO_CHAR[l.item()] for l in labels]}")