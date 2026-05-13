# Approach 5: CharNet (Current — Char Index → Bézier Curves)

## Core Idea
Maximally simplified: given only a character index (0–61 for a–z, A–Z, 0–9), predict 6 cubic Bézier curves. No writer ID, no reference image, no font. A small MLP/CNN (CharNet) with a character category embedding maps char_idx → 48 Bézier control point values. Curves are rendered into a handwriting image.

## Files
| File | Role |
|------|------|
| `src/model.py` | CharNet — character embedding + MLP → 6 × 4 × 2 Bézier points |
| `src/data.py` | Loads handwriting PNGs + bezier_labels.npy, char index only |
| `src/bezier.py` | Cubic Bézier evaluation and rasterisation |
| `src/skeleton.py` | Skeletonisation for label extraction |
| `src/train.py` | MSE training; N_CURVES=6, category embedding |
| `src/generate.py` | Given a text string, generate each character and assemble |
| `generate_samples.py` | Standalone script to batch-generate sample images |

## Inputs / Outputs
- **Input:** character index (integer 0–61)
- **Output:** 6 Bézier curves → rendered handwriting image

## Key Design Choices
- Category embedding (not one-hot) for character conditioning
- N_CURVES increased from 3 → 6 for more expressive strokes
- Flask removed — pure script-based generation
- Locally trained checkpoint included

## Current Status
Active approach. Checkpoint: `checkpoints/style_net.pt`
