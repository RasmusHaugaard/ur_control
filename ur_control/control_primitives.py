import time
from typing import Sequence, Union
from enum import Enum

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from transform3d import Transform

from . import spiral_utils
from . import utils

Frame = Union[str, Transform]


class DeviatingMotionError(RuntimeError):
    pass


class TerminateReason(Enum):
    """Different reasons for successfully terminating a control primitive"""
    STOP_CONDITION = 1
    FORCE_TORQUE_LIMIT = 2
    TRAVEL_LIMIT = 3


# TODO: stop robot if connection is lost


class ControlPrimitives:
    """
    Class containing simple control primitives

    """

    def __init__(self, recv: RTDEReceiveInterface,
                 ctrl: RTDEControlInterface,
                 control_frequency=250.):
        self.recv = recv
        self.ctrl = ctrl
        self.control_frequency = control_frequency
        self.dt = 1. / self.control_frequency

    @classmethod
    def from_ip(cls, ip: str, control_frequency=250.):
        return cls(
            recv=RTDEReceiveInterface(ip),
            ctrl=RTDEControlInterface(ip),
            control_frequency=control_frequency
        )

    def base_t_tcp(self):
        # TODO: possibly cache by default with a ttl ~≃ self.dt
        return Transform.from_xyz_rotvec(self.recv.getActualTCPPose())

    def ft_base(self):
        return self.recv.getActualTCPForce()

    def ft(self, base_t_frame: Transform):
        # TODO: enable moving the contact point for force-torque sensor values
        #  currently, only the frame is changed, the contact point is always the ft-sensor
        return base_t_frame.inv.rotate(self.ft_base())

    def ft_tcp(self):
        return self.ft(self.base_t_tcp())

    def _get_base_t_frame(self, base_t_frame: Frame):
        if isinstance(base_t_frame, str):
            if base_t_frame.lower() == 'base':
                base_t_frame = Transform()
            elif base_t_frame.lower() == 'tcp':
                base_t_frame = self.base_t_tcp()
        if not isinstance(base_t_frame, Transform):
            raise ValueError('{} not recognized as frame'.format(base_t_frame))
        return base_t_frame

    def zero_ft_sensor(self, delay=0.5):
        # TODO: what is the 'correct' delay?
        time.sleep(delay)
        self.ctrl.zeroFtSensor()

    def speed_until_ft(self, speed: Sequence, base_t_frame: Frame = 'tcp', acceleration=0.25,
                       stop_ft=(5., 5., 5., 2., 2., 2.), stop_travel=np.inf,
                       timeout=np.inf, max_travel=np.inf, stop_condition=lambda: False):
        if len(speed) == 3:
            speed = (*speed, 0, 0, 0)
        base_t_frame = self._get_base_t_frame(base_t_frame)
        start_time = time.time()
        p_init = self.base_t_tcp().t
        self.ctrl.speedL(base_t_frame.rotate(speed), acceleration)
        try:
            while True:
                loop_start = time.time()
                if loop_start - start_time > timeout:
                    raise TimeoutError()
                if np.any(np.abs(self.ft_tcp()) > stop_ft):
                    break
                p_now = self.base_t_tcp().t
                travel = np.linalg.norm(p_now - p_init)
                if travel > stop_travel:
                    return TerminateReason.TRAVEL_LIMIT
                if travel > max_travel:
                    raise DeviatingMotionError()
                if stop_condition():
                    return TerminateReason.STOP_CONDITION
                loop_duration = time.time() - loop_start
                time.sleep(max(0., self.dt - loop_duration))
        finally:
            self.ctrl.speedStop()

    def spiral_search(self, tol: float, max_radius: float, push_force=5., speed=0.01,
                      ft_stop=(15., 15., np.inf, 2., 2., 2.), stop_condition=lambda: False,
                      lookahead_time=0.1, gain=600):
        # TODO: acc limit
        # TODO: insertion direction
        try:
            base_t_tcp_init = self.base_t_tcp()
            self.ctrl.forceModeStart(
                base_t_tcp_init.as_xyz_rotvec(),
                [0, 0, 1, 0, 0, 0], [0, 0, push_force, 0, 0, 0],
                2, [0.05] * 6
            )

            a, b = spiral_utils.get_spiral_params(tol)
            theta = 0
            while True:
                loop_start = time.time()
                theta = spiral_utils.next_theta_constant_speed(theta, a, b, speed=speed, dt=self.dt)
                r = spiral_utils.radius_from_theta(a, b, theta)
                if r > max_radius:
                    raise DeviatingMotionError()
                x, y = np.cos(theta) * r - a, np.sin(theta) * r
                base_t_tcp = Transform.from_xyz_rotvec(self.recv.getActualTCPPose())
                ft_tcp = base_t_tcp.inv.rotate(self.recv.getActualTCPForce())
                if any(np.abs(ft_tcp) > ft_stop):
                    return TerminateReason.FORCE_TORQUE_LIMIT
                elif stop_condition():
                    return TerminateReason.STOP_CONDITION
                tcp_start_t_tcp_next = Transform(p=(x, y, 0))
                base_t_tcp_next = base_t_tcp_init @ tcp_start_t_tcp_next
                self.ctrl.servoL(base_t_tcp_next.as_xyz_rotvec(), 0.5, 0.25, self.dt, lookahead_time, gain)
                loop_duration = time.time() - loop_start
                time.sleep(max(0., self.dt - loop_duration))
        finally:
            self.ctrl.servoStop()
            self.ctrl.forceModeStop()

    def disc_sampling_search(self, radius: float, depth: float, poke_speed=0.03, move_speed=.25, move_acc=1.2,
                             stop_condition=lambda: False, timeout=20.):
        # TODO: insertion direction
        base_t_tcp_init = self.base_t_tcp()
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError()
            x, y = utils.sample_hyper_sphere_volume(r=radius, d=2)
            tcp_init_t_tcp_next = Transform(p=(x, y, 0))
            base_t_tcp_next = base_t_tcp_init @ tcp_init_t_tcp_next
            self.ctrl.moveL(base_t_tcp_next.as_xyz_rotvec(), move_speed, move_acc)
            try:
                self.speed_until_ft((0, 0, poke_speed), max_travel=depth)
            except DeviatingMotionError:
                return TerminateReason.TRAVEL_LIMIT
            if stop_condition():
                return TerminateReason.STOP_CONDITION
            self.ctrl.moveL(base_t_tcp_next.as_xyz_rotvec(), move_speed, move_acc)

    def insert_force(self, push_force=10., rot_compliant=True, nudge_scale=.5, nudge_frequency=1.,
                     limits=(1., 1., 1., 0.2, 0.2, 0.2), damping=0.01, gain_scaling=1.,
                     depth_stop: float = None, force_stop: float = None, stop_condition=lambda: False,
                     timeout=20., max_angle_diff=np.deg2rad(10.)):
        # TODO: take tcp_T_peg_tip. Nudging then might have to be rotational instead.
        start = time.time()
        self.ctrl.forceModeSetDamping(damping)
        self.ctrl.forceModeSetGainScaling(gain_scaling)
        compliance = np.ones(6)
        compliance[3:] = int(rot_compliant)
        base_t_tcp_init = self.base_t_tcp()
        try:
            while True:
                base_t_tcp_now = self.base_t_tcp()
                ft_tcp = self.ft_tcp()
                tcp_init_t_tcp_now = base_t_tcp_init.inv @ base_t_tcp_now
                cur_duration = time.time() - start
                if cur_duration > timeout:
                    raise TimeoutError()
                if np.linalg.norm(tcp_init_t_tcp_now.r.as_rotvec()) > max_angle_diff:
                    raise DeviatingMotionError()
                if depth_stop is not None:
                    if tcp_init_t_tcp_now.t[2] > depth_stop:
                        return TerminateReason.TRAVEL_LIMIT
                elif force_stop is not None:
                    if abs(ft_tcp[2]) > force_stop:
                        return TerminateReason.FORCE_TORQUE_LIMIT
                if stop_condition():
                    return TerminateReason.STOP_CONDITION

                theta = (time.time() - start) * np.pi * nudge_frequency
                phi = theta / 9.5  # TODO: hyper parameter
                xy = np.array((np.cos(phi), np.sin(phi))) * np.sin(theta) * nudge_scale
                self.ctrl.forceModeStart(
                    base_t_tcp_now.as_xyz_rotvec(),
                    compliance, [0, 0, push_force, *xy, 0],
                    2, limits
                )
                time.sleep(self.dt)
        finally:
            self.ctrl.forceModeStop()
