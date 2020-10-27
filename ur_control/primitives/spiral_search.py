import time
import numpy as np
from transform3d import Transform

from ..robot import Robot, DeviatingMotionError, TerminateReason
from . import spiral_utils


def spiral_search(r: Robot, tol: float, max_radius: float, push_force=5., speed=0.01,
                  ft_stop=(15., 15., np.inf, 2., 2., 2.), stop_condition=lambda: False,
                  lookahead_time=0.1, gain=600):
    # TODO: acc limit
    # TODO: insertion direction
    try:
        base_t_tcp_init = r.base_t_tcp()
        r.ctrl.forceMode(
            base_t_tcp_init,
            [0, 0, 1, 0, 0, 0], [0, 0, push_force, 0, 0, 0],
            2, [0.05] * 6
        )

        a, b = spiral_utils.get_spiral_params(tol)
        theta = 0
        while True:
            loop_start = time.time()
            theta = spiral_utils.next_theta_constant_speed(theta, a, b, speed=speed, dt=r.ctrl_dt)
            r = spiral_utils.radius_from_theta(a, b, theta)
            if r > max_radius:
                raise DeviatingMotionError()
            x, y = np.cos(theta) * r - a, np.sin(theta) * r
            base_t_tcp = Transform.from_xyz_rotvec(r.recv.getActualTCPPose())
            ft_tcp = base_t_tcp.inv.rotate(r.recv.getActualTCPForce())
            if any(np.abs(ft_tcp) > ft_stop):
                return TerminateReason.FORCE_TORQUE_LIMIT
            elif stop_condition():
                return TerminateReason.STOP_CONDITION
            tcp_start_t_tcp_next = Transform(p=(x, y, 0))
            base_t_tcp_next = base_t_tcp_init @ tcp_start_t_tcp_next
            r.ctrl.servoL(base_t_tcp_next, 0.5, 0.25, r.ctrl_dt, lookahead_time, gain)
            loop_duration = time.time() - loop_start
            time.sleep(max(0., r.ctrl_dt - loop_duration))
    finally:
        r.ctrl.servoStop()
        r.ctrl.forceModeStop()
