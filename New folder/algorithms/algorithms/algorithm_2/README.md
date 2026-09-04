# Algorithm 2 — DCT-based Contextual Filtering

Independent overlapping 2-D DCT fingerprint enhancement. Local Sobel
orientation, DCT radial frequency, confidence, and quality guide a contextual
coefficient filter. Each block is reconstructed by IDCT and raised-cosine
overlap-add; only then does the shared evaluation layer run.

The candidate ablation contains Basic DCT, Adaptive Frequency,
Confidence-Aware, and Proposed Full DCT. It does not use RHLT, Gabor, CLAHE,
wavelet/ridgelet, Wiener, FFT, or unsharp masking.
