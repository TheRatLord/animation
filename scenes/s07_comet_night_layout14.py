"""Round-14 cloud layout for s07_comet_night: broken banks + diagonal streamers clustered like yn_05."""


def clusters(Wp, W, H, my):
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind, amt=1.0, warm=0.0, dens=1.0):
        x, y = fp(fx, fy)
        if kind == 'st':
            T = max(T * 1.5, 0.04)
        amt = 1.0 if amt >= 0.6 else amt + 0.2
        out.append(dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, H=H, amt=amt, warm=warm, dens=dens))
    A = 27.0   # streamers follow the comet's direction
    # ---- upper-left broken bank: torn streamers + a couple of puffy heads
    add(0.08, 0.1, 0.26, 0.065, A, 'st', 0.95)
    add(0.22, 0.2, 0.24, 0.075, A - 4, 'st', 1.0)
    add(0.14, 0.15, 0.13, 0.14, 6, 'cu', 1.0)
    add(0.3, 0.26, 0.08, 0.06, 5, 'cu', 0.95)
    add(0.36, 0.3, 0.12, 0.035, A + 5, 'st', 0.85)
    add(0.02, 0.03, 0.12, 0.04, A, 'st', 0.7)
    # thin high wisps (far, partly transparent)
    add(0.3, 0.07, 0.16, 0.025, A + 3, 'st', 0.45)
    add(0.47, 0.13, 0.1, 0.022, A, 'st', 0.4)
    # ---- mid-left: streamers + a scrap cluster
    add(0.07, 0.44, 0.17, 0.04, A + 12, 'st', 0.9)
    add(0.12, 0.52, 0.1, 0.03, A + 18, 'st', 0.7)
    add(0.03, 0.39, 0.07, 0.05, 0, 'cu', 0.9)
    # ---- small scraps near the tail (clustered, varied)
    add(0.47, 0.35, 0.05, 0.055, 0, 'cu', 0.9)
    add(0.53, 0.38, 0.05, 0.022, A, 'st', 0.7)
    add(0.39, 0.55, 0.09, 0.03, A + 8, 'st', 0.7, 0.1)
    add(0.51, 0.59, 0.06, 0.018, A, 'st', 0.55, 0.15)
    # ---- upper-right broken bank
    add(0.9, 0.14, 0.17, 0.11, 6, 'cu', 1.0)
    add(0.8, 0.08, 0.12, 0.03, A, 'st', 0.7)
    add(0.99, 0.26, 0.1, 0.06, 0, 'cu', 0.95)
    add(0.83, 0.3, 0.07, 0.02, A, 'st', 0.5)
    # ---- low twilight banks near the horizon (warm undersides), bigger and broken
    add(0.92, 0.83, 0.26, 0.12, 4, 'cu', 1.0, 0.35)
    add(0.78, 0.9, 0.14, 0.03, 8, 'st', 0.7, 0.5)
    add(0.19, 0.79, 0.12, 0.062, 3, 'cu', 0.8, 0.6)
    add(0.3, 0.84, 0.18, 0.022, 5, 'st', 0.55, 0.7)
    add(0.64, 0.855, 0.16, 0.016, 6, 'st', 0.45, 0.5)
    # clouds lower in the plate (seen after the tilt down, low over the ranges)
    add(0.05, 0.95, 0.14, 0.03, 4, 'st', 0.6, 0.6)
    return out
