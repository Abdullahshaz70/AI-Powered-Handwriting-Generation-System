# Approaches Comparison

| # | Approach | Input | Output | Architecture | Generative? | Replaced Because |
|---|----------|-------|--------|--------------|-------------|-----------------|
| 1 | **GAN** | ref image + char index | pixel image | Encoder + Generator + Discriminator (AC-GAN) | Yes | Unstable training, mode collapse, blurry outputs |
| 2 | **MultiTaskCNN** | handwriting image | char label + writer label | CNN + 2 classification heads | No (classifier only) | Can't generate — only classify |
| 3 | **Font-to-Handwriting CNN** | font-rendered image | pixel image | U-Net (HandwritingStyleNet) | Yes | Font artifacts; learns font quirks not true style |
| 4 | **Bézier Regression (Real Images)** | real image + writer ID + ref image | Bézier curves → image | CNN regressor + writer embedding | Yes | Too many inputs needed at inference time |
| 5 | **CharNet** *(current)* | char index only | Bézier curves → image | Char embedding + MLP | Yes | — (active) |

## Key Evolution

```
GAN (pixel, unstable)
  → MultiTaskCNN (understand style first, but no generation)
    → Font CNN (supervised pixel-to-pixel, but font dependency)
      → Bézier Regression (structured output, but complex inputs)
        → CharNet (simplest possible: 1 integer in, curves out)
```

## What Changed Each Time

- **GAN → MultiTaskCNN:** Gave up on adversarial training; focused on learning a good representation first
- **MultiTaskCNN → Font CNN:** Moved to actual generation using supervised image translation
- **Font CNN → Bézier Regression:** Dropped font dependency; switched output space from pixels to curves for cleaner strokes
- **Bézier Reg → CharNet:** Dropped writer ID and reference image; minimised inputs to just the character
