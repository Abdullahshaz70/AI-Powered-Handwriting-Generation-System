HandScript Dataset
==================
Writer ID : Salman_24067
Samples   : 310
Characters: 62 (A-Z, a-z, 0-9)
Rounds    : 5
Image size: 128 x 128 px
Format    : PNG (lossless, white bg, black ink)

Filename convention:
  {writerID}_{class}_{char}_{round}.png
  uc_ = uppercase, lc_ = lowercase, n_ = digit

Preprocessing still needed:
  1. Normalize pixel values to [0, 1]
  2. Optionally convert to grayscale
  3. Apply augmentation (rotation, elastic distortion)
