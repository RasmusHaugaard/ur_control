import numpy as np


def sample_hyper_sphere_volume(r: float, d: int):
    while True:
        p = np.random.uniform(-r, r, d)
        if np.linalg.norm(p) <= r:
            return p
