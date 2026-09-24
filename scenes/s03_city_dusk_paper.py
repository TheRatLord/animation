"""Static paint / paper texture for s03_city_dusk: a fixed (never re-randomised) multiplicative value field
made of soft paper fibre + faint wash mottling, applied at very low amplitude, strongest in the mid-tones.
Being identical every frame it adds a hand-painted surface without any temporal flicker."""
import numpy as np
import cv2


def paper(W, H, seed=17, amt=0.01):
    rng = np.random.default_rng(seed)
    s = W / 1920.0
    # fine fibre: anisotropic blurred noise (slightly horizontal, like cold-press paper grain)
    n = rng.standard_normal((H, W)).astype(np.float32)
    fib = cv2.GaussianBlur(n, (0, 0), sigmaX=1.6 * s, sigmaY=0.9 * s)
    fib /= fib.std() + 1e-6
    # wash mottling: large soft blotches
    m = rng.standard_normal((max(H // 24, 4), max(W // 24, 4))).astype(np.float32)
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.5), (W, H), interpolation=cv2.INTER_CUBIC)
    m /= m.std() + 1e-6
    tex = 0.6 * fib + 0.4 * m
    return (1.0 + amt * tex).astype(np.float32)


def apply(img, tex):
    """mid-tone weighted multiply (keeps deep shadows and highlights clean); img float32 RGB 0..1"""
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.55 + img[..., 2] * 0.15
    w = np.clip(4.0 * lum * (1.2 - lum), 0.25, 1.0)
    return img * (1.0 + (tex - 1.0) * w)[..., None]
