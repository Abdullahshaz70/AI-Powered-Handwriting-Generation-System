# Approach 1: GAN (Generative Adversarial Network)

## Core Idea
Style Encoder CNN extracts a style vector from a reference handwriting image. Generator takes (character embedding + style vector) and produces a handwriting image. Discriminator tries to distinguish real vs. fake, plus classify the writer (AC-GAN variant added to fix mode collapse).

## Files
| File | Role |
|------|------|
| `encoder.py` | CNN encodes a reference PNG into a style latent vector |
| `generator.py` | Decoder/upsampling network — style vector + char embedding → image |
| `discriminator.py` | Judges real/fake + writer identity (AC-GAN) |
| `renderer.py` | Post-processing: gamma correction, inversion for ink visibility |
| `dataset.py` | Loads writer PNGs, pairs them by character |
| `train.py` | Adversarial training loop (G + D losses) |
| `index.html` | Canvas-based data collection UI |

## Inputs / Outputs
- **Input:** reference handwriting image + character index
- **Output:** generated handwriting image (pixel space)

## Key Problems Encountered
- Mode collapse — all outputs looked the same → fixed with AC-GAN auxiliary classifier
- Thin ink strokes invisible after downscale → fixed with gamma correction in renderer

## Why It Was Replaced
GAN training is unstable and hard to control. Results were blurry/inconsistent and the adversarial loop required careful tuning. Moved to a supervised approach instead.
