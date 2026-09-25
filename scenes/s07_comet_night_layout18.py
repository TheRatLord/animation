"""Round-18 cloud layout for s07_comet_night (yn_05 upper sky, yn_02 twilight banks).
Upper sky: lobed moonlit masses + broken streaks of puffs following the comet's direction (kept where the
round-17 composition worked).  Horizon: two violet-grey banks with rim-lit undersides + a band of small broken
wisps in the afterglow."""


def clusters(Wp, W, H, my):
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind, amt=1.0, warm=0.0, haze=0.0, heads=(1, 3)):
        x, y = fp(fx, fy)
        out.append(dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, amt=amt, warm=warm, haze=haze, heads=heads,
                        far=haze))
    A = 24.0   # torn streaks follow the comet's direction
    # ---- upper-left group: a big lit mass trailing a broken band down-right
    add(0.11, 0.08, 0.17, 0.15, 6, 'cu', 1.0)
    add(0.28, 0.2, 0.22, 0.15, A, 'st', 0.95, heads=(2, 3))
    add(0.03, 0.02, 0.12, 0.1, A - 4, 'st', 0.8, heads=(1, 2))
    add(0.34, 0.05, 0.12, 0.08, A + 2, 'st', 0.7, heads=(1, 2))
    add(0.4, 0.3, 0.06, 0.07, 3, 'cu', 0.95)
    # ---- mid-left
    add(0.1, 0.47, 0.17, 0.13, A + 16, 'st', 0.9, heads=(1, 2))
    add(0.04, 0.37, 0.07, 0.08, 2, 'cu', 0.95)
    # ---- near the tail (the tail shines through their thin parts)
    add(0.47, 0.31, 0.05, 0.065, 0, 'cu', 0.95)
    add(0.46, 0.55, 0.15, 0.11, A + 6, 'st', 0.85, heads=(1, 2))
    add(0.55, 0.37, 0.08, 0.08, A, 'st', 0.75, heads=(1, 1))
    # ---- upper right: broken bank + a band
    add(0.885, 0.12, 0.19, 0.11, 3, 'cu', 1.0)
    add(0.77, 0.06, 0.12, 0.09, A, 'st', 0.75, heads=(1, 2))
    add(0.985, 0.235, 0.08, 0.08, 0, 'cu', 0.95)
    add(0.82, 0.29, 0.1, 0.08, A, 'st', 0.65, heads=(1, 1))
    # ---- twilight banks (violet-grey bodies, hot rim underneath / sun side, lost bottoms)
    add(0.2, 0.785, 0.2, 0.1, 1, 'cu', 1.0, 0.95, 0.85)
    add(0.33, 0.815, 0.07, 0.05, 0, 'cu', 0.95, 0.95, 0.9)
    add(0.905, 0.815, 0.26, 0.11, 2, 'cu', 1.0, 0.6, 0.8)
    add(0.73, 0.87, 0.1, 0.05, 1, 'cu', 0.9, 0.6, 0.9)
    # ---- small broken wisps along the horizon band
    for (fx, fy, L, T, w) in ((0.06, 0.865, 0.14, 0.04, 0.9), (0.28, 0.88, 0.16, 0.035, 1.0),
                              (0.44, 0.9, 0.12, 0.03, 0.95), (0.55, 0.875, 0.1, 0.03, 0.8),
                              (0.64, 0.92, 0.14, 0.03, 0.75), (0.82, 0.9, 0.12, 0.035, 0.6),
                              (0.15, 0.93, 0.18, 0.03, 1.0), (0.37, 0.94, 0.1, 0.025, 1.0),
                              (0.97, 0.94, 0.12, 0.03, 0.55)):
        add(fx, fy, L, T, 1, 'st', 0.8, w, 0.9, heads=(0, 1))
    return out
