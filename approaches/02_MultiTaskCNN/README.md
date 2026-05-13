# Approach 2: MultiTaskCNN (Classification, not Generation)

## Core Idea
A CNN trained with two heads: one classifies the character (a–z, 0–9), the other classifies the writer identity. Not a generative model — goal was to learn a shared representation of handwriting style that could later feed a generator. Trained on all 6 writers (2108 samples).

## Files
| File | Role |
|------|------|
| `model.py` | Shared CNN backbone + two classification heads |
| `dataset.py` | Loads writer PNGs with both char and writer labels |
| `train.py` | Multi-task loss (char CE + writer CE), writer-balanced sampler, CosineAnnealingLR, label smoothing |
| `evaluate.py` | T-SNE embedding visualization, confusion matrix per writer |
| `demo.py` | Run inference on a single image |
| `index.html` | Canvas data collection UI (same as GAN) |

## Inputs / Outputs
- **Input:** handwriting image (64×64 PNG)
- **Output:** predicted character label + predicted writer label

## Key Design Choices
- Writer loss reweighted to 0.6 to balance char vs writer objectives
- Writer-balanced sampler so no writer dominates training
- Label smoothing to reduce overconfidence

## Why It Was Replaced
This approach only classifies — it cannot generate new handwriting. It was a stepping stone to understand style representations but was discarded in favour of a true generative pipeline.
