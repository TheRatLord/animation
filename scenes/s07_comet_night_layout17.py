"""Round-17 cloud layout for s07_comet_night (yn_05): fewer, larger flat painted shapes.  Torn streaks are
thick bands with 1-3 big lobes (no bead chains), compact moonlit masses upper left / upper right, and
afterglow strata low on the horizon (violet tops, orange undersides)."""


def clusters(Wp, W, H, my):
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind, amt=1.0, warm=0.0, haze=0.0, heads=(1, 3)):
        x, y = fp(fx, fy)
        out.append(dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, amt=amt, warm=warm, haze=haze, heads=heads,
                        far=haze))
    A = 24.0   # torn streaks follow the comet's direction
    # ---- upper-left group (yn_05 left group): a big lit mass trailing a torn band down-right
    add(0.12, 0.085, 0.17, 0.15, 6, 'cu', 1.0)
    add(0.27, 0.19, 0.25, 0.17, A, 'st', 0.95, heads=(2, 3))
    add(0.03, 0.03, 0.14, 0.12, A - 4, 'st', 0.8, heads=(1, 2))
    add(0.33, 0.055, 0.13, 0.08, A + 2, 'st', 0.6, heads=(0, 1))
    add(0.38, 0.3, 0.07, 0.07, 3, 'cu', 0.95)
    # ---- mid-left: one torn band with a lit head + a small mass
    add(0.1, 0.47, 0.2, 0.15, A + 16, 'st', 0.85, heads=(1, 2))
    add(0.035, 0.38, 0.06, 0.07, 2, 'cu', 0.9)
    # ---- near the tail: a small lit mass and one torn band
    add(0.47, 0.32, 0.05, 0.06, 0, 'cu', 0.95)
    add(0.45, 0.555, 0.15, 0.12, A + 6, 'st', 0.8, heads=(1, 2))
    add(0.54, 0.37, 0.08, 0.08, A, 'st', 0.7, heads=(0, 1))
    # ---- upper right: broken bank + a band
    add(0.885, 0.125, 0.19, 0.1, 3, 'cu', 1.0)
    add(0.78, 0.06, 0.13, 0.09, A, 'st', 0.7, heads=(1, 2))
    add(0.985, 0.24, 0.08, 0.075, 0, 'cu', 0.95)
    add(0.82, 0.29, 0.1, 0.08, A, 'st', 0.6, heads=(0, 1))
    # ---- low clouds in the afterglow: violet tops, warm undersides, soft bases
    add(0.9, 0.815, 0.26, 0.1, 2, 'cu', 1.0, 0.6, 0.8)
    add(0.77, 0.895, 0.16, 0.06, 4, 'st', 0.65, 0.7, 0.85, heads=(0, 1))
    add(0.19, 0.78, 0.13, 0.07, 1, 'cu', 1.0, 0.75, 0.8)
    add(0.31, 0.84, 0.2, 0.06, 3, 'st', 0.65, 0.85, 0.85, heads=(0, 1))
    add(0.63, 0.855, 0.16, 0.045, 4, 'st', 0.5, 0.7, 0.9, heads=(0, 1))
    add(0.05, 0.95, 0.16, 0.06, 2, 'st', 0.55, 0.75, 0.9, heads=(0, 1))
    return out
