"""Round-19 cloud layout for s07_comet_night: fewer, bigger clouds, varied from dense cumulus to thin cirrus
(yn_05), plus two violet twilight banks and a few low wisps in the afterglow (yn_02)."""


def clusters(Wp, W, H, my):
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind, amt=1.0, warm=0.0, haze=0.0, heads=(1, 2)):
        x, y = fp(fx, fy)
        out.append(dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, amt=amt, warm=warm, haze=haze, heads=heads,
                        far=haze))
    A = 24.0
    # cirrus veils (thin, stretched, behind everything)
    add(0.17, 0.56, 0.36, 0.08, -6, 'ci', 0.7, haze=0.3)
    add(0.7, 0.07, 0.34, 0.08, 8, 'ci', 0.65, haze=0.3)
    add(0.7, 0.45, 0.22, 0.06, -4, 'ci', 0.5, haze=0.3)
    # upper-left: one big dense mass with a torn band trailing along the comet's direction
    add(0.13, 0.1, 0.3, 0.2, 5, 'cu', 1.0)
    add(0.36, 0.26, 0.3, 0.12, A, 'st', 0.95, heads=(1, 2))
    # a medium cumulus the tail shines through
    add(0.5, 0.5, 0.15, 0.11, 2, 'cu', 0.95)
    # upper right: a big broken bank
    add(0.88, 0.14, 0.28, 0.17, 3, 'cu', 1.0)
    add(0.84, 0.34, 0.18, 0.08, A - 6, 'st', 0.85, heads=(1, 1))
    # twilight banks
    add(0.21, 0.795, 0.3, 0.1, 1, 'cu', 1.0, 1.0, 0.45)
    add(0.9, 0.81, 0.26, 0.11, 2, 'cu', 1.0, 0.7, 0.4)
    add(0.66, 0.86, 0.12, 0.05, 1, 'st', 0.9, 0.8, 0.55, heads=(1, 1))
    for (fx, fy, L, T, w) in ((0.07, 0.88, 0.16, 0.04, 0.9), (0.4, 0.895, 0.16, 0.035, 1.0),
                              (0.52, 0.925, 0.14, 0.03, 0.9)):
        add(fx, fy, L, T, 1, 'st', 0.8, w, 0.6, heads=(0, 1))
    return out
