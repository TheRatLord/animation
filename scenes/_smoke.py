import numpy as np
from lib import core as C
DURATION = 1.0
class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.sky = C.vertical_gradient(W, H, [(0, '#1f5fbf'), (0.6, '#8fc7ef'), (1, '#f4e3c8')])
        n = C.fbm(W, H, 3, 6, seed=2)
        self.cl = C.smoothstep(0.55, 0.75, n)
    def frame(self, t):
        W, H = self.W, self.H
        img = C.over(self.sky, np.ones(3, np.float32), self.cl)
        img = img + C.god_rays(self.cl * 1.0, W * 0.7, H * 0.2, strength=0.5)
        img = img + C.lens_flare(W, H, W * 0.7, H * 0.2)
        img = C.bloom(img)
        return C.finish(img, t)
