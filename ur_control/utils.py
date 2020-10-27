import numpy as np


def sample_hyper_sphere_volume(r: float, d: int):
    while True:
        p = np.random.uniform(-r, r, d)
        if np.linalg.norm(p) <= r:
            return p


def translate_ft(ft, p):
    f_ = ft[:3]
    t_ = ft[3:] + np.cross(ft[:3], p)
    return (*f_, *t_)


