import time

from transform3d import Transform

from ..robot import Robot, DeviatingMotionError, TerminateReason
from .. import utils
from .speed_until_force import speed_until_ft


def disc_search(r: Robot, radius: float, depth: float, poke_speed=0.03, move_speed=.25, move_acc=1.2,
                stop_condition=lambda: False, timeout=20.):
    # TODO: insertion direction
    base_t_tcp_init = r.base_t_tcp()
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError()
        x, y = utils.sample_hyper_sphere_volume(r=radius, d=2)
        tcp_init_t_tcp_next = Transform(p=(x, y, 0))
        base_t_tcp_next = base_t_tcp_init @ tcp_init_t_tcp_next
        r.ctrl.moveL(base_t_tcp_next, move_speed, move_acc)
        try:
            speed_until_ft(r, (0, 0, poke_speed), max_travel=depth)
        except DeviatingMotionError:
            return TerminateReason.TRAVEL_LIMIT
        if stop_condition():
            return TerminateReason.STOP_CONDITION
        r.ctrl.moveL(base_t_tcp_next, move_speed, move_acc)
