"""dev harness: paint near tree cards with a given painter module on a sky bg (not a shot)"""
import sys, os, math, time, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import s05_sakura_env as E
import s05_sakura_tree2d as T2
modname = sys.argv[1]; which = [int(x) for x in sys.argv[2].split(',')]
M = importlib.import_module(modname)
TreeC = getattr(M, sys.argv[3] if len(sys.argv) > 3 else 'Tree22'); PainterC = getattr(M, sys.argv[4] if len(sys.argv) > 4 else 'Painter22')
W, H = 1920, 1080
cam = E.Cam(W, H); f = cam.f
sx, sy = 0.17 * W, 0.1 * H
rng = np.random.default_rng(55)
s_list = [11.5, 18.5, 25.5, 32.5, 40.0]
yy = np.linspace(0, 1, H)[:, None, None]
bg = (np.array([0.2, 0.45, 0.85]) * (1 - yy) + np.array([0.75, 0.87, 0.95]) * yy) * np.ones((H, W, 3))
out = bg.astype(np.float32).copy()
for i, s in enumerate(s_list):
    u = 2.5 + rng.uniform(-0.2, 0.3)
    sc = rng.uniform(0.95, 1.08)
    if i not in which:
        continue
    X, Y, Z = cam.to_cam(u, E.Y_PATH, s)
    bx, by = cam.proj(X, Y, Z)
    ss = 2; k = f * ss / Z
    tr_rng = np.random.default_rng(1000 + i)
    ldx, ldy = sx - bx, sy - by; ln = math.hypot(ldx, ldy)
    t0 = time.time()
    tree = TreeC(tr_rng, scale=1.15 * sc, lean=-1.0, spread=1.2, big=Z < 45, hmul=1.3, cl=max(1.0, Z / 14.0) ** 0.7, maxd=4 if Z < 40 else 3)
    pnt = PainterC(k, sun_dir=(ldx / ln, ldy / ln), px=ss * W / 1920.0, detail=1.0 if Z < 60 else 0.6)
    card, ox, oy = pnt.paint(tree, tr_rng)
    print(i, 'Z', round(Z, 1), 'n', len(tree.clusters), 'time', round(time.time() - t0, 1), getattr(pnt, 'stats', ''))
    rgba, x0, y0 = T2.card_to_screen(card, ox, oy, float(bx), float(by), ss)
    h, w = rgba.shape[:2]
    xa, ya = int(round(x0)), int(round(y0))
    X0, Y0, X1, Y1 = max(xa, 0), max(ya, 0), min(xa + w, W), min(ya + h, H)
    if X1 > X0 and Y1 > Y0:
        p = rgba[Y0 - ya:Y1 - ya, X0 - xa:X1 - xa]
        out[Y0:Y1, X0:X1] = out[Y0:Y1, X0:X1] * (1 - p[..., 3:4]) + p[..., :3] * p[..., 3:4]
cv2.imwrite(sys.argv[5] if len(sys.argv) > 5 else 'out/compare/tree_test.png', np.clip(out[..., ::-1] * 255, 0, 255).astype(np.uint8))
