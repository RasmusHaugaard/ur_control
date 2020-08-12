import numpy as np


# https://en.wikipedia.org/wiki/Archimedean_spiral

def get_spiral_params(spiral_width, start_radius=None):
    b = spiral_width / (2 * np.pi)
    # choosing a = b ensures the area with the given tolerance is covered
    # while avoiding unnecessary high-acceleration movement near the center
    a = b if start_radius is None else start_radius
    return a, b


def theta_constant_speed(a, b, max_r, speed=0.01, dt=1 / 250):
    theta = [0]
    while radius_from_theta(a, b, theta[-1]) < max_r:
        theta.append(next_theta_constant_speed(theta[-1], a, b, speed, dt))
    return np.array(theta)


def next_theta_constant_speed(last_theta, a, b, speed=0.01, dt=1 / 250):
    # TODO: max_acceleration
    return last_theta + speed / v_dtheta(a, b, last_theta) * dt


def radius_from_theta(a, b, theta):
    return a + b * theta


def dx_dtheta(a, b, theta):
    S, C = np.sin(theta), np.cos(theta)
    return - a * S + b * C - b * theta * S


def dy_dtheta(a, b, theta):
    S, C = np.sin(theta), np.cos(theta)
    return a * C + b * S + b * theta * C


def v_dtheta(a, b, theta):
    dxy_dtheta = np.stack((dx_dtheta(a, b, theta), dy_dtheta(a, b, theta)), axis=-1)
    return np.linalg.norm(dxy_dtheta, axis=-1)


def v_approx(a, b, theta):
    return radius_from_theta(a, b, theta)
