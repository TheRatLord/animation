"""Shot runner.

  python run.py scenes/s01_sky.py --stills            # 960x540 stills at 5 times -> out/previews/<name>_tX.png + contact sheet
  python run.py scenes/s01_sky.py --stills --full     # same at 1920x1080
  python run.py scenes/s01_sky.py --video --half      # half-res preview mp4 -> out/shots/<name>_half.mp4
  python run.py scenes/s01_sky.py --video             # final 1920x1080 mp4 -> out/shots/<name>.mp4
  python run.py scenes/s01_sky.py --strip             # 6 consecutive frames (motion check) at 960x540
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from lib import core  # noqa: E402


def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return name, mod


def contact(paths, out):
    ims = [cv2.imread(p) for p in paths]
    h, w = ims[0].shape[:2]
    tw, th = w // 2, h // 2
    ims = [cv2.resize(i, (tw, th), interpolation=cv2.INTER_AREA) for i in ims]
    while len(ims) % 3:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]
    cv2.imwrite(out, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene')
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--strip', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--times', default='')
    a = ap.parse_args()
    name, mod = load(a.scene)
    D = mod.DURATION
    os.makedirs(os.path.join(ROOT, 'out', 'previews'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'out', 'shots'), exist_ok=True)

    if a.stills or a.strip:
        W, H = (1920, 1080) if a.full else (960, 540)
        t0 = time.time()
        sc = mod.Scene(W, H)
        print(f'setup {time.time() - t0:.1f}s')
        if a.strip:
            times = [D * 0.5 + i / core.FPS for i in range(6)]
        elif a.times:
            times = [float(x) for x in a.times.split(',')]
        else:
            times = [0.0, D * 0.25, D * 0.5, D * 0.75, D - 1 / core.FPS]
        paths = []
        for t in times:
            t1 = time.time()
            img = sc.frame(t)
            p = os.path.join(ROOT, 'out', 'previews', f'{name}_{"strip" if a.strip else "t"}{t:.2f}.png')
            core.save_png(p, img)
            paths.append(p)
            print(f't={t:.2f} {time.time() - t1:.2f}s -> {p}')
        cs = os.path.join(ROOT, 'out', 'previews', f'{name}_{"strip" if a.strip else "contact"}.png')
        contact(paths, cs)
        print('contact sheet ->', cs)

    if a.video:
        W, H = (960, 540) if a.half else (1920, 1080)
        sc = mod.Scene(W, H)
        out = os.path.join(ROOT, 'out', 'shots', f'{name}{"_half" if a.half else ""}.mp4')
        vw = core.VideoWriter(out, W, H)
        n = int(round(D * core.FPS))
        t0 = time.time()
        for i in range(n):
            vw.write(sc.frame(i / core.FPS))
            if i % 24 == 0:
                print(f'frame {i}/{n} {time.time() - t0:.0f}s', flush=True)
        vw.close()
        print('video ->', out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
