# Approach 3: Font-to-Handwriting CNN (HandwritingStyleNet U-Net)

## Core Idea
Image-to-image translation: feed a clean font-rendered character image in, get a handwriting-style image out. A U-Net style encoder-decoder (HandwritingStyleNet) learns the mapping from crisp font glyphs to handwritten strokes using real handwriting samples as targets.

## Files
| File | Role |
|------|------|
| `src/model.py` | HandwritingStyleNet — U-Net encoder-decoder |
| `src/data.py` | Pairs font-rendered images with real handwriting PNGs |
| `src/font_render.py` | Renders a character to an image using a system font (PIL) |
| `src/train.py` | Supervised pixel-level training (L1 + SSIM loss) |
| `src/generate.py` | Generate handwriting from a text string via font → model |
| `src/bezier.py` | Bézier curve utilities (carried over, not central here) |
| `src/skeleton.py` | Stroke skeleton utilities |
| `app.py` | Flask web app (port 5000) for interactive generation |
| `templates/index.html` | Web UI — type text, get handwriting image back |

## Inputs / Outputs
- **Input:** font-rendered character image (from PIL/FreeType)
- **Output:** handwriting-style image in pixel space

## Key Design Choices
- U-Net skip connections preserve stroke detail
- L1 + SSIM composite loss (pixel accuracy + structural similarity)
- Flask app for live demo

## Why It Was Replaced
Requires a font as intermediate — introduces font artifacts. The model learns font-specific quirks rather than true handwriting style. Replaced by directly regressing from real handwriting images.
