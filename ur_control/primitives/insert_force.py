import time

import numpy as np

from ..robot import Robot, TerminateReason, DeviatingMotionError


def insert_force(r: Robot, push_force=10., rot_compliant=True, nudge_scale=.5, nudge_frequency=1.,
                 limits=(1., 1., 1., 0.2, 0.2, 0.2), damping=0.01, gain_scaling=1.,
                 depth_stop: float = None, force_stop: float = None, stop_condition=lambda: False,
                 timeout=20., max_angle_diff=np.deg2rad(10.)):
    # TODO: take tcp_T_peg_tip
    start = time.time()
    r.ctrl.forceModeSetDamping(damping)
    r.ctrl.forceModeSetGainScaling(gain_scaling)
    compliance = [1, 1, 1] + [int(rot_compliant)] * 3
    base_t_tcp_init = r.base_t_tcp()
    try:
        while True:
            base_t_tcp_now = r.base_t_tcp()
            ft_tcp = r.ft_tcp()
            tcp_init_t_tcp_now = base_t_tcp_init.inv @ base_t_tcp_now
            cur_duration = time.time() - start
            if cur_duration > timeout:
                raise TimeoutError()
            if np.linalg.norm(tcp_init_t_tcp_now.rotvec) > max_angle_diff:
                raise DeviatingMotionError()
            if depth_stop is not None:
                if tcp_init_t_tcp_now.p[2] > depth_stop:
                    return TerminateReason.TRAVEL_LIMIT
            elif force_stop is not None:
                if abs(ft_tcp[2]) > force_stop:
                    return TerminateReason.FORCE_TORQUE_LIMIT
            if stop_condition():
                return TerminateReason.STOP_CONDITION

            theta = (time.time() - start) * np.pi * nudge_frequency
            phi = theta / 9.5  # TODO: hyper parameter
            xy = np.array((np.cos(phi), np.sin(phi))) * np.sin(theta) * nudge_scale
            r.ctrl.forceMode(
                base_t_tcp_now,
                compliance, [0, 0, push_force, *xy, 0],
                2, limits
            )
            time.sleep(r.ctrl_dt)
    finally:
        r.ctrl.forceModeStop()
