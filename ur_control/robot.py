import time
from typing import Union
from numbers import Number
from enum import Enum

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from transform3d import Transform

from . import utils

Frame = Union[str, Transform]


class DeviatingMotionError(RuntimeError):
    pass


class TerminateReason(Enum):
    """Different reasons for successfully terminating a control primitive"""
    STOP_CONDITION = 1
    FORCE_TORQUE_LIMIT = 2
    TRAVEL_LIMIT = 3


# TODO: stop robot if connection is lost (watchdog)
# TODO: context manager for setting the tcp offset (requires getTcp())
#       and then add a reference (linear) frame in moveL


class Robot:
    def __init__(self, recv: RTDEReceiveInterface, ctrl: RTDEControlInterface,
                 control_frequency=250., speed_l=0.25, acc_l=1.2, speed_j=1.05, acc_j=1.4):
        self.recv = recv
        self.ctrl = ctrl
        self.control_frequency = control_frequency
        self.speed_l, self.acc_l = speed_l, acc_l
        self.speed_j, self.acc_j = speed_j, acc_j

    @classmethod
    def from_ip(cls, ip: str, no_recv=False, no_ctrl=False,
                control_frequency=250., speed_l=0.25, acc_l=1.2, speed_j=1.05, acc_j=1.4):
        return cls(
            recv=None if no_recv else RTDEReceiveInterface(ip, []),
            ctrl=None if no_ctrl else RTDEControlInterface(ip),
            control_frequency=control_frequency,
            speed_l=speed_l, acc_l=acc_l,
            speed_j=speed_j, acc_j=acc_j
        )

    @property
    def ctrl_dt(self):
        return 1 / self.control_frequency

    def q(self):
        return np.array(self.recv.getActualQ())

    def base_t_tcp(self):
        """Returns the tcp frame seen in the base frame"""
        # caching is currently avoided because changing the tcp frame should invalidate cache
        return Transform.from_xyz_rotvec(self.recv.getActualTCPPose())

    def zero_ft_sensor(self, delay=0.25):
        time.sleep(delay)
        self.ctrl.zeroFtSensor()

    def base_ft_tcp(self):
        """Returns the force torque measured in the tcp frame but seen in the base frame"""
        return np.array(self.recv.getActualTCPForce())

    def a_ft_b(self, base_t_a: Transform, base_t_b: Transform,
               base_t_tcp: Transform = None):
        """Return the force torque measured in b but seen in a"""
        base_t_tcp = base_t_tcp or self.base_t_tcp()
        base_ft_b = utils.translate_ft(self.base_ft_tcp(), base_t_b.p - base_t_tcp.p)
        a_ft_b = base_t_a.inv.rotate(base_ft_b)
        return a_ft_b

    def ft_base(self, base_t_frame=Transform()):
        """Return the force torque measured in [frame]"""
        return self.a_ft_b(base_t_frame, base_t_frame)

    def ft_tcp(self, tcp_t_frame=Transform()):
        """Returns the force torque measured in [frame]"""
        base_t_tcp = self.base_t_tcp()
        base_t_frame = base_t_tcp @ tcp_t_frame
        return self.a_ft_b(base_t_frame, base_t_frame, base_t_tcp=base_t_tcp)

    def _process_path(self, path, space='joint', speed=None, acc=None, max_blend=0., max_blend_ratio=1.):
        """Processes a path before providing it to moveJ or moveL"""
        path = list(path)
        if space == 'joint':
            speed = self.speed_j if speed is None else speed
            acc = self.acc_j if acc is None else acc
            d = 6
        elif space == 'cartesian':
            speed = self.speed_l if speed is None else speed
            acc = self.acc_l if acc is None else acc
            d = 3
        else:
            raise ValueError()

        if isinstance(path[0], Number):
            path = [path]

        for i in range(len(path)):
            wp = path[i]
            if isinstance(wp, Waypoint):
                p = wp.p
                wp_speed = speed if wp.speed is None else wp.speed
                wp_acc = acc if wp.acc is None else wp.acc
                wp_max_blend = max_blend if wp.max_blend is None else wp.max_blend
                wp_max_blend_ratio = max_blend_ratio if wp.max_blend_ratio is None else wp.max_blend_ratio
            else:
                p = list(wp)
                wp_speed = wp[6] if len(wp) > 6 else speed
                wp_acc = wp[7] if len(wp) > 7 else acc
                wp_max_blend = wp[8] if len(wp) > 8 else max_blend
                wp_max_blend_ratio = max_blend_ratio
            path[i] = Waypoint(p, wp_speed, wp_acc, wp_max_blend, wp_max_blend_ratio)

        points = np.array([wp.p[:d] for wp in path])
        dists = [0, *np.linalg.norm(points[:-1] - points[1:], axis=-1), 0]
        ur_path = []
        for i in range(len(path)):
            wp = path[i]
            assert len(wp.p) == 6
            max_possible_blend = min(dists[i], dists[i + 1]) / 2
            blend = min(max_possible_blend * max(0, min(wp.max_blend_ratio, 0.999)), wp.max_blend)
            ur_path.append([*wp.p, wp.speed, wp.acc, blend])
        return ur_path

    def move_l(self, path, speed=None, acc=None, max_blend=0., max_blend_ratio=1.):
        ur_path = self._process_path(path, 'cartesian', speed, acc, max_blend, max_blend_ratio)
        assert self.ctrl.moveL(ur_path)

    def move_j(self, path, speed=None, acc=None, max_blend=0., max_blend_ratio=1.):
        ur_path = self._process_path(path, 'joint', speed, acc, max_blend, max_blend_ratio)
        assert self.ctrl.moveJ(ur_path)

    def move_l_qlim(self, base_t_tcp_desired: Transform, speed_l=None, acc_l=None, speed_j=None, acc_q=None,
                    lookahead_time=0.1, gain=600):
        """Moves linear in cartesian space but also limits the speed in joint space"""
        speed_l = self.speed_l if speed_l is None else speed_l
        acc_l = self.acc_l if acc_l is None else acc_l
        speed_j = self.speed_j if speed_j is None else speed_j
        acc_q = self.acc_q if acc_q is None else acc_q

        base_t_tcp_start = self.base_t_tcp()
        tcp_start_t_tcp_desired = base_t_tcp_start.inv @ base_t_tcp_desired
        s = 0
        while utils.rate(self.control_frequency) and s < 1:
            s_delta = 0.1  # TODO: find based on speed_l and acc_l
            base_t_tcp_next = base_t_tcp_start @ tcp_start_t_tcp_desired * (s + s_delta)
            q_now = self.q()
            q_next = self.ctrl.getInverseKinematics(base_t_tcp_next)
            q_delta = q_next - q_now
            q_speed = np.linalg.norm(q_delta) / self.ctrl_dt
            factor = min(1, speed_j / (q_speed + 1e-9))
            q_next = q_now + q_delta * factor
            s = s + s_delta * factor
            self.ctrl.servoJ(q_next)


class Waypoint:
    def __init__(self, p, speed=None, acc=None, max_blend=None, max_blend_ratio=None):
        self.p, self.speed, self.acc = p, speed, acc
        self.max_blend, self.max_blend_ratio = max_blend, max_blend_ratio
