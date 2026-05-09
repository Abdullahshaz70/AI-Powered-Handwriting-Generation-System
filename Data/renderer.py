import os
import string
import textwrap
import torch
import numpy as np
from PIL import Image, ImageDraw
from fpdf import FPDF

from encoder import StyleEncoder
from generator import CharacterGenerator
from dataset import load_dataset, CharDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHAR_SIZE = 128
CHAR_RENDER_PX = 32
LINE_SPACING = 48
PAGE_LEFT_MARGIN = 60
PAGE_TOP_MARGIN = 60
PAGE_RIGHT_MARGIN = 60
CHARS_PER_LINE = 60
WORD_SPACE_PX = 12

all_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_LABEL = {c: i for i, c in enumerate(all_chars)}


def load_model(checkpoint_path):
    encoder = StyleEncoder().to(DEVICE)
    generator = CharacterGenerator().to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    generator.load_state_dict(ckpt["generator_state"])

    encoder.eval()
    generator.eval()

    return encoder, generator


def get_style_vector(encoder, writer_folder):
    data_list = load_dataset(writer_folder)
    dataset = CharDataset(data_list)

    images = []
    for i in range(min(32, len(dataset))):
        img, _ = dataset[i]
        images.append(img)

    batch = torch.stack(images).to(DEVICE)

    with torch.no_grad():
        style_vectors = encoder(batch)
        style = style_vectors.mean(dim=0, keepdim=True)

    return style


def generate_char_image(generator, style, char):
    if char not in CHAR_TO_LABEL:
        return None

    label = torch.tensor([CHAR_TO_LABEL[char]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        generated = generator(style, label)

    img_tensor = generated.squeeze().cpu()
    img_arr = ((img_tensor.numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    img_arr = 255 - img_arr  # flip: black ink on white paper

    # Stretch contrast
    lo, hi = img_arr.min(), img_arr.max()
    if hi > lo:
        img_arr = ((img_arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)

    # Soft threshold at 180 — keeps ink gradients, avoids wiping out faint strokes
    img_arr = np.where(img_arr < 180, img_arr, 255).astype(np.uint8)

    img = Image.fromarray(img_arr, mode="L")
    img = img.resize((CHAR_RENDER_PX, CHAR_RENDER_PX), Image.BILINEAR)

    return img


def render_text_to_image(generator, style, text, output_img_path):
    page_width = 794
    page_height = 1123

    canvas = Image.new("RGB", (page_width, page_height), color=(255, 255, 255))

    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=CHARS_PER_LINE)
            lines.extend(wrapped if wrapped else [""])

    x = PAGE_LEFT_MARGIN
    y = PAGE_TOP_MARGIN

    for line in lines:
        if line == "":
            y += LINE_SPACING
            if y + LINE_SPACING > page_height - PAGE_TOP_MARGIN:
                break
            continue

        x = PAGE_LEFT_MARGIN

        words = line.split(" ")
        for word_idx, word in enumerate(words):
            word_width = len(word) * CHAR_RENDER_PX

            if x + word_width > page_width - PAGE_RIGHT_MARGIN and x != PAGE_LEFT_MARGIN:
                x = PAGE_LEFT_MARGIN
                y += LINE_SPACING

            if y + CHAR_RENDER_PX > page_height - PAGE_TOP_MARGIN:
                break

            for char in word:
                if char in CHAR_TO_LABEL:
                    char_img = generate_char_image(generator, style, char)
                    if char_img is not None:
                        canvas.paste(char_img.convert("RGB"), (x, y))
                    x += CHAR_RENDER_PX

            if word_idx < len(words) - 1:
                x += WORD_SPACE_PX

        y += LINE_SPACING
        if y + LINE_SPACING > page_height - PAGE_TOP_MARGIN:
            break

    canvas.save(output_img_path)
    print(f"Page image saved -> {output_img_path}")
    return canvas


def render_text_to_pdf(generator, style, text, output_pdf_path, temp_dir="./temp_pages"):
    os.makedirs(temp_dir, exist_ok=True)

    page_width = 794
    page_height = 1123
    chars_per_line = CHARS_PER_LINE
    usable_height = page_height - PAGE_TOP_MARGIN * 2

    all_lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            all_lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=chars_per_line)
            all_lines.extend(wrapped if wrapped else [""])

    lines_per_page = usable_height // LINE_SPACING
    page_chunks = [all_lines[i:i + lines_per_page] for i in range(0, len(all_lines), lines_per_page)]

    pdf = FPDF(unit="pt", format="A4")
    pdf.set_margins(0, 0, 0)
    pdf.set_auto_page_break(False)

    for page_num, page_lines in enumerate(page_chunks):
        canvas = Image.new("RGB", (page_width, page_height), color=(255, 255, 255))

        x = PAGE_LEFT_MARGIN
        y = PAGE_TOP_MARGIN

        for line in page_lines:
            if line == "":
                y += LINE_SPACING
                continue

            x = PAGE_LEFT_MARGIN
            words = line.split(" ")

            for word_idx, word in enumerate(words):
                word_width = len(word) * CHAR_RENDER_PX

                if x + word_width > page_width - PAGE_RIGHT_MARGIN and x != PAGE_LEFT_MARGIN:
                    x = PAGE_LEFT_MARGIN
                    y += LINE_SPACING

                for char in word:
                    if char in CHAR_TO_LABEL:
                        char_img = generate_char_image(generator, style, char)
                        if char_img is not None:
                            char_rgb = Image.new("RGB", char_img.size, (255, 255, 255))
                            char_rgb.paste(char_img, mask=char_img)
                            canvas.paste(char_rgb, (x, y))
                        x += CHAR_RENDER_PX

                if word_idx < len(words) - 1:
                    x += WORD_SPACE_PX

            y += LINE_SPACING

        page_img_path = os.path.join(temp_dir, f"page_{page_num + 1:03d}.png")
        canvas.save(page_img_path)

        pdf.add_page()
        pdf.image(page_img_path, x=0, y=0, w=page_width, h=page_height)

        print(f"  Rendered page {page_num + 1}/{len(page_chunks)}")

    pdf.output(output_pdf_path)
    print(f"PDF saved -> {output_pdf_path}")

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(_HERE, "..", "checkpoints", "best_model.pt")
    WRITER_FOLDER = os.path.join(_HERE, "Writers_pngs", "writer_Abdullah")
    OUTPUT_PDF = os.path.join(_HERE, "..", "output_handwritten.pdf")

    TEXT = "The quick brown fox jumps over the lazy dog"

    print(f"Device: {DEVICE}")
    print("Loading model...")
    encoder, generator = load_model(CHECKPOINT_PATH)

    print("Extracting style vector from writer samples...")
    style = get_style_vector(encoder, WRITER_FOLDER)

    print("Rendering text to PDF...")
    render_text_to_pdf(generator, style, TEXT, OUTPUT_PDF)

    print("Done.")