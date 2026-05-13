# Approach 4: Bézier Curve Regression from Real Images

## Core Idea
Instead of generating pixel images, predict Bézier control points that describe handwriting strokes. A CNN takes a real handwriting image (plus writer ID + reference image) and regresses 6 cubic Bézier curves (6 × 4 × 2 = 48 values). Curves are then rendered to produce the final image.

## Files
| File | Role |
|------|------|
| `src/model.py` | CNN regressor: image + writer embedding → Bézier control points |
| `src/data.py` | Loads real handwriting PNGs + pre-computed bezier_labels.npy |
| `src/bezier.py` | Cubic Bézier evaluation and rendering |
| `src/skeleton.py` | Stroke skeletonisation (used to extract Bézier labels from images) |
| `src/train.py` | MSE regression training loop |
| `src/generate.py` | Given writer + reference image, predict and render curves |
| `app.py` | Flask web app for generation |
| `templates/index.html` | Web UI |

## Inputs / Outputs
- **Input:** real handwriting image + writer ID + reference handwriting image
- **Output:** Bézier control points → rendered stroke image

## Key Design Choices
- Bézier representation is compact and smooth (no pixel noise)
- Writer embedding injected into the model for style conditioning
- Labels extracted from real images via skeletonisation

## Why It Was Replaced
Requiring a reference image at inference time is inconvenient. Writer-conditioning added complexity. Simplified to CharNet which only needs a character index.
