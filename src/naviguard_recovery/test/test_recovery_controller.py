import pytest
from naviguard_recovery.recovery_controller import RecoveryController


def test_recovery_controller_stop():
    ctrl = RecoveryController()
    vx, wz = ctrl.compute_stop()
    assert vx == 0.0
    assert wz == 0.0


def test_recovery_controller_backtrack_and_reach():
    ctrl = RecoveryController()

    # Current pose (1.0, 0.0, 0.0), target waypoint (0.5, 0.0, 0.0) -> Reversing along x
    vx, wz, reached = ctrl.compute_backtrack_step(
        current_pose=(1.0, 0.0, 0.0),
        target_waypoint=(0.5, 0.0, 0.0),
    )
    assert vx < 0.0  # Reverse velocity
    assert reached is False

    # Within goal tolerance (< 0.12m)
    vx_done, wz_done, reached_done = ctrl.compute_backtrack_step(
        current_pose=(0.55, 0.0, 0.0),
        target_waypoint=(0.50, 0.0, 0.0),
    )
    assert reached_done is True
    assert vx_done == 0.0


def test_recovery_controller_rotation_and_reach():
    ctrl = RecoveryController()

    # Current yaw 0.0, target yaw 0.5 rad
    vx, wz, reached = ctrl.compute_rotation_step(0.0, 0.5)
    assert vx == 0.0
    assert wz > 0.0  # Positive CCW turn
    assert reached is False

    # Within tolerance (< 0.06 rad)
    vx_done, wz_done, reached_done = ctrl.compute_rotation_step(0.48, 0.50)
    assert reached_done is True
    assert wz_done == 0.0
