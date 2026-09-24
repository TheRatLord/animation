"""Round-16 cloud layout for s07_comet_night (yn_05): varied types - torn wisps along the comet's
direction, compact lit puffs, a broken bank upper right, hazy afterglow scraps low on the horizon."""


def clusters(Wp, W, H, my):
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind, amt=1.0, warm=0.0, haze=0.0, heads=(3, 7)):
        x, y = fp(fx, fy)
        out.append(dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, amt=amt, warm=warm, haze=haze, heads=heads,
                        far=haze))
    A = 27.0   # wisps follow the comet's direction
    # ---- upper-left: long torn wisps with a compact lit head (yn_05 left group)
    add(0.10, 0.08, 0.30, 0.10, A - 2, 'st', 0.9)
    add(0.24, 0.19, 0.26, 0.09, A + 2, 'st', 0.95)
    add(0.13, 0.13, 0.12, 0.13, 3, 'cu', 1.0)
    add(0.31, 0.25, 0.06, 0.075, 2, 'cu', 0.95)
    add(0.37, 0.31, 0.16, 0.06, A + 6, 'st', 0.8)
    add(0.03, 0.02, 0.14, 0.06, A, 'st', 0.7, heads=(1, 3))
    add(0.33, 0.06, 0.16, 0.04, A + 3, 'st', 0.5, heads=(0, 2))
    # ---- mid-left: stretched wisps + one small puff
    add(0.08, 0.44, 0.22, 0.07, A + 12, 'st', 0.8, heads=(0, 2))
    add(0.13, 0.54, 0.14, 0.05, A + 18, 'st', 0.65, heads=(0, 1))
    add(0.03, 0.38, 0.05, 0.06, 0, 'cu', 0.9)
    # ---- scraps near the tail
    add(0.47, 0.33, 0.045, 0.06, 0, 'cu', 0.95)
    add(0.54, 0.38, 0.09, 0.035, A, 'st', 0.7)
    add(0.40, 0.55, 0.12, 0.045, A + 8, 'st', 0.75, heads=(1, 2))
    add(0.52, 0.6, 0.08, 0.03, A, 'st', 0.55, heads=(0, 2))
    # ---- upper-right broken bank + small puffs
    add(0.89, 0.13, 0.15, 0.12, 4, 'cu', 1.0)
    add(0.79, 0.07, 0.14, 0.045, A, 'st', 0.65)
    add(0.985, 0.25, 0.08, 0.075, 0, 'cu', 0.95)
    add(0.83, 0.3, 0.1, 0.035, A, 'st', 0.55)
    # ---- low clouds in the afterglow: dimmed violet-grey tops, warm undersides, soft bases
    add(0.92, 0.82, 0.24, 0.1, 3, 'cu', 0.9, 0.55, 0.8)
    add(0.78, 0.9, 0.16, 0.04, 6, 'st', 0.6, 0.6, 0.85)
    add(0.19, 0.785, 0.12, 0.065, 2, 'cu', 0.8, 0.7, 0.8)
    add(0.31, 0.84, 0.2, 0.035, 4, 'st', 0.6, 0.8, 0.85)
    add(0.64, 0.855, 0.16, 0.025, 5, 'st', 0.45, 0.6, 0.9)
    add(0.05, 0.95, 0.16, 0.04, 3, 'st', 0.55, 0.7, 0.9)
    return out
