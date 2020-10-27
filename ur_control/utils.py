import time
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


def rate(hz):
    desired_loop_duration = 1 / hz
    last = time.time()
    while True:
        yield True
        now = time.time()
        duration = now - last
        sleep = max(0, desired_loop_duration - duration)
        time.sleep(sleep)
        last = now + sleep


def _rate_test():
    hz = 250
    print(1 / hz)
    now = time.time()
    for i, _ in zip(range(10), rate(250)):
        print(i, time.time() - now)
        time.sleep(0.1)


if __name__ == '__main__':
    _rate_test()
