"""s05_sakura round 23 railing (reviewer: 'rails are perfect uniform white tubes; add a warm specular glint only on
the sun-facing top edge, a cooler shadow underside and slight value falloff with distance').

draw_pipes23 = rail.draw_pipes with: a mid grey-blue painted body (not white), a darker cool underside, the warm
specular line moved up onto the sun-facing top edge (narrow), per-segment paint-tone variation (weathered steel),
and a per-segment distance fade toward the air colour (column 8: 0 near .. 1 far).
"""
import math
from numba import njit
from s05_sakura_rail import _ss, _h

@njit(cache=True, fastmath=True)
def draw_pipes23(img, C, dark, body, rim, spec, bounce, glint_ph, air):
    """C columns: x0 y0 r0 x1 y1 r1 spec(0..1) seg_id.  Far -> near order, AA."""
    H, W = img.shape[0], img.shape[1]
    for i in range(C.shape[0]):
        ax, ay, ar, bx, by, br, spk, sid = C[i, 0], C[i, 1], C[i, 2], C[i, 3], C[i, 4], C[i, 5], C[i, 6], C[i, 7]
        zf = C[i, 8]
        tone = 0.9 + 0.16 * _h(int(sid), 7)
        rm = max(ar, br) + 2
        x0 = max(0, int(min(ax, bx) - rm))
        x1 = min(W, int(max(ax, bx) + rm + 1))
        y0 = max(0, int(min(ay, by) - rm))
        y1 = min(H, int(max(ay, by) + rm + 1))
        ex = bx - ax
        ey = by - ay
        L2 = ex * ex + ey * ey + 1e-6
        ln = math.sqrt(L2)
        nx = -ey / ln
        ny = ex / ln
        if ny > 0:
            nx = -nx
            ny = -ny
        # glints: 1-2 hot spots per segment, sliding slowly with the camera
        g1 = _h(int(sid), 11)
        g2 = _h(int(sid), 12)
        gc1 = (g1 + glint_ph * (0.5 + g2)) % 1.0
        gc2 = (g2 * 0.7 + 0.3 + glint_ph * 0.8) % 1.0
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5 - ax
                py = y + 0.5 - ay
                t = (px * ex + py * ey) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = px - t * ex
                qy = py - t * ey
                rr = ar + (br - ar) * t
                d = math.sqrt(qx * qx + qy * qy)
                rv = max(rr, 0.5)
                cov = min(max(rv + 0.5 - d, 0.0), 1.0)
                if rr < 0.5:
                    cov *= rr / 0.5 * 0.9 + 0.1
                if cov <= 0.0:
                    continue
                sg = min(max((qx * nx + qy * ny) / rv, -1.0), 1.0)    # +1 = sun / upper side
                k = _ss(-0.45, 0.25, sg)
                r = (dark[0] + (body[0] - dark[0]) * k) * tone
                g = (dark[1] + (body[1] - dark[1]) * k) * tone
                b = (dark[2] + (body[2] - dark[2]) * k) * tone
                # warm bounce from the sunlit path on the underside
                bo = _ss(-0.6, -0.98, sg) * 0.35
                r += (bounce[0] - r) * bo
                g += (bounce[1] - g) * bo
                b += (bounce[2] - b) * bo
                # warm sun-side rim right at the upper edge
                ri = _ss(0.72, 0.95, sg) * min(1.0, rv / 1.2)
                r += (rim[0] - r) * ri
                g += (rim[1] - g) * ri
                b += (rim[2] - b) * ri
                # specular core line + hot glints
                gl = math.exp(-((t - gc1) / 0.05) ** 2) + 0.7 * math.exp(-((t - gc2) / 0.035) ** 2)
                sp = math.exp(-((sg - 0.8) / 0.1) ** 2) * spk * (0.25 + 1.5 * gl) * (1.0 - 0.6 * zf)
                r += spec[0] * sp
                g += spec[1] * sp
                b += spec[2] * sp
                r += (air[0] - r) * zf
                g += (air[1] - g) * zf
                b += (air[2] - b) * zf
                img[y, x, 0] = img[y, x, 0] * (1 - cov) + r * cov
                img[y, x, 1] = img[y, x, 1] * (1 - cov) + g * cov
                img[y, x, 2] = img[y, x, 2] * (1 - cov) + b * cov
