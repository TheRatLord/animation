"""s05_sakura round 4: painted steel railing pipes - cool blue shadow side, pale body, warm sun-side rim, crisp
specular core line whose brightness swells into hot glints along the rail, warm bounce from the path on the
underside."""
import math
from numba import njit


@njit(cache=True, fastmath=True)
def _ss(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True)
def _h(i, seed):
    n = (i * 374761393 + seed * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / 16777216.0


@njit(cache=True, fastmath=True)
def draw_pipes(img, C, dark, body, rim, spec, bounce, glint_ph):
    """C columns: x0 y0 r0 x1 y1 r1 spec(0..1) seg_id.  Far -> near order, AA."""
    H, W = img.shape[0], img.shape[1]
    for i in range(C.shape[0]):
        ax, ay, ar, bx, by, br, spk, sid = C[i, 0], C[i, 1], C[i, 2], C[i, 3], C[i, 4], C[i, 5], C[i, 6], C[i, 7]
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
                k = _ss(-0.75, 0.35, sg)
                r = dark[0] + (body[0] - dark[0]) * k
                g = dark[1] + (body[1] - dark[1]) * k
                b = dark[2] + (body[2] - dark[2]) * k
                # warm bounce from the sunlit path on the underside
                bo = _ss(-0.55, -0.95, sg) * 0.6
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
                sp = math.exp(-((sg - 0.45) / 0.14) ** 2) * spk * (0.45 + 1.4 * gl)
                r += spec[0] * sp
                g += spec[1] * sp
                b += spec[2] * sp
                img[y, x, 0] = img[y, x, 0] * (1 - cov) + r * cov
                img[y, x, 1] = img[y, x, 1] * (1 - cov) + g * cov
                img[y, x, 2] = img[y, x, 2] * (1 - cov) + b * cov
