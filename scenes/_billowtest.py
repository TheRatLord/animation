import numpy as np
from lib import core as C
from lib import billow as B
DURATION = 3.0
class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.sky = C.vertical_gradient(W, H, [(0, '#0b3fa8'), (0.45, '#2a7fe0'), (0.8, '#8cc8f2'), (1, '#dcefff')])
        self.sun = (W * 0.80, H * 0.10)
        m = B.tower_masses2(W, H, W * 0.50, H * 0.97, H * 0.06, W * 0.34, seed=4)
        m += B.heap_masses(W, H, W * 0.10, H * 0.80, W * 0.22, H * 0.16, seed=5)
        self.cloud = B.paint_masses(W, H, m, self.sun, 'noon', ground_y=H, haze_amt=0.35)
    def frame(self, t):
        W, H = self.W, self.H
        img = C.over(self.sky, self.cloud[..., :3], self.cloud[..., 3])
        img = img + C.glow(W, H, *self.sun, W * 0.02, 2.2)[..., None] * np.array([1, .97, .9], np.float32) * 0.8
        img = C.bloom(img, 0.9, 0.25)
        return C.finish(img, t, ca=0, grain_amt=0.006)
