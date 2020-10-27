import numpy as np
from transform3d import Transform

from ..robot import Robot, DeviatingMotionError, TerminateReason
from ..utils import rate
from . import spiral_utils


def spiral_search(r: Robot, tol: float, max_radius: float, push_force=5., speed=0.01,
                  ft_stop=(15., 15., np.inf, 2., 2., 2.), stop_condition=lambda: False,
                  lookahead_time=0.1, servo_gain=600, damping=0.01, force_gain=1.):
    # TODO: acc limit
    # TODO: insertion direction
    base_t_tcp_init = r.base_t_tcp()
    r.ctrl.forceModeSetDamping(damping)
    r.ctrl.forceModeSetGainScaling(force_gain)
    try:
        r.ctrl.forceMode(
            base_t_tcp_init,
            [0, 0, 1, 0, 0, 0], [0, 0, push_force, 0, 0, 0],
            2, [tol, tol, 0.02] + [0.01] * 3
        )

        a, b = spiral_utils.get_spiral_params(tol)
        theta = 0
        while rate(r.control_frequency):
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
            r.ctrl.servoL(base_t_tcp_next, speed, r.acc_l, r.ctrl_dt, lookahead_time, servo_gain)
    finally:
        r.ctrl.servoStop()
        r.ctrl.forceModeStop()
