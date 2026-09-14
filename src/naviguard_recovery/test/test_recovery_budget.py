import pytest
from naviguard_recovery.recovery_state_machine import RecoveryBudget


def test_recovery_budget_tracking():
    budget = RecoveryBudget(
        max_attempts=3,
        max_total_duration_sec=30.0,
        max_backtrack_dist_m=3.0,
        max_rotation_deg=120.0,
    )

    assert not budget.is_exhausted()

    # Attempt 1
    budget.record_attempt("STOP_AND_RELOCALIZE", "VO_LOST")
    budget.record_motion(dist_m=1.0, rot_deg=45.0)
    budget.total_recovery_time_sec += 5.0
    assert not budget.is_exhausted()

    # Attempt 2
    budget.record_attempt("SHORT_BACKTRACK", "VO_LOST")
    budget.record_motion(dist_m=1.5, rot_deg=30.0)
    assert not budget.is_exhausted()

    # Exceed distance budget (1.0 + 1.5 + 1.0 = 3.5 > 3.0)
    budget.record_motion(dist_m=1.0, rot_deg=0.0)
    assert budget.is_exhausted()
