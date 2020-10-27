import time
from typing import Sequence

import numpy as np

from ..robot import Robot, Frame, TerminateReason, DeviatingMotionError


def speed_until_ft(r: Robot, speed: Sequence, base_t_frame: Frame = 'tcp', acceleration=0.25,
                   stop_ft=(5., 5., 5., 2., 2., 2.), stop_travel=np.inf,
                   timeout=np.inf, max_travel=np.inf, stop_condition=lambda: False):
    if len(speed) == 3:
        speed = (*speed, 0, 0, 0)
    base_t_frame = r.base_t_frame(base_t_frame)
    start_time = time.time()
    p_init = r.base_t_tcp().p
    r.ctrl.speedL(base_t_frame.rotate(speed), acceleration)
    try:
        while True:
            loop_start = time.time()
            if loop_start - start_time > timeout:
                raise TimeoutError()
            if np.any(np.abs(r.ft_tcp()) > stop_ft):
                break
            p_now = r.base_t_tcp().p
            travel = np.linalg.norm(p_now - p_init)
            if travel > stop_travel:
                return TerminateReason.TRAVEL_LIMIT
            if travel > max_travel:
                raise DeviatingMotionError()
            if stop_condition():
                return TerminateReason.STOP_CONDITION
            loop_duration = time.time() - loop_start
            time.sleep(max(0., r.ctrl_dt - loop_duration))
    finally:
        r.ctrl.speedStop()
