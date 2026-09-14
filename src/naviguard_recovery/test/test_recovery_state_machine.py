import pytest
from naviguard_recovery.recovery_state_machine import (
    RecoveryBudget,
    RecoveryState,
    RecoveryStateMachine,
    RecoveryStateMachineConfig,
)


def test_recovery_state_enum():
    assert RecoveryState.NORMAL == 0
    assert RecoveryState.SAFE_STOP == 2
    assert RecoveryState.RECOVER == 4
    assert RecoveryState.FAILED_SAFE == 9

    assert RecoveryState.NORMAL.to_string() == "NORMAL"
    assert RecoveryState.SAFE_STOP.to_string() == "SAFE_STOP"
    assert RecoveryState.FAILED_SAFE.to_string() == "FAILED_SAFE"

    assert RecoveryState.from_string("normal") == RecoveryState.NORMAL
    assert RecoveryState.from_string("SAFE_STOP") == RecoveryState.SAFE_STOP

    with pytest.raises(ValueError):
        RecoveryState.from_string("INVALID_STATE")


def test_state_machine_nominal_transitions():
    cfg = RecoveryStateMachineConfig(safe_stop_dwell_sec=0.5, verification_dwell_sec=1.0)
    fsm = RecoveryStateMachine(cfg)
    fsm.reset(0.0)

    assert fsm.state == RecoveryState.NORMAL

    # Phase 7 sends VERIFY
    fsm.update_phase7_input("VERIFY", "VO_LOW_INLIERS", 1.0)
    assert fsm.state == RecoveryState.VERIFY

    # Phase 7 clears back to CONTINUE
    fsm.update_phase7_input("CONTINUE", "NOMINAL", 2.0)
    assert fsm.state == RecoveryState.NORMAL

    # Phase 7 sends RECOVER -> triggers SAFE_STOP
    fsm.update_phase7_input("RECOVER", "SLAM_TRACKING_LOST", 3.0)
    assert fsm.state == RecoveryState.SAFE_STOP
    assert fsm.failure_reason == "SLAM_TRACKING_LOST"


def test_full_recovery_lifecycle():
    cfg = RecoveryStateMachineConfig(safe_stop_dwell_sec=0.5, verification_dwell_sec=1.0)
    fsm = RecoveryStateMachine(cfg)
    fsm.reset(0.0)

    # 1. Trigger SAFE_STOP
    fsm.update_phase7_input("RECOVER", "SLAM_TRACKING_LOST", 1.0)
    assert fsm.state == RecoveryState.SAFE_STOP

    # 2. Dwell in SAFE_STOP for 0.6s (> 0.5s) -> transitions to SELECT_CHECKPOINT
    fsm.step_recovery_lifecycle(1.6, 0.6, False, False, False)
    assert fsm.state == RecoveryState.SELECT_CHECKPOINT

    # 3. SELECT_CHECKPOINT transitions to RECOVER
    fsm.transition_to(RecoveryState.RECOVER, 1.7)
    assert fsm.state == RecoveryState.RECOVER

    # 4. Action in progress -> stays in RECOVER
    fsm.step_recovery_lifecycle(2.0, 0.3, True, False, False)
    assert fsm.state == RecoveryState.RECOVER

    # 5. Action finished -> transitions to VISUAL_REACQUISITION_OBSERVATION
    fsm.step_recovery_lifecycle(2.5, 0.5, False, False, False)
    assert fsm.state == RecoveryState.VISUAL_REACQUISITION_OBSERVATION

    # 5b. Dwell in observation (>= 0.50s) -> transitions to RELOCALIZE
    fsm.step_recovery_lifecycle(3.1, 0.6, False, False, False)
    assert fsm.state == RecoveryState.RELOCALIZE

    # 6. Relocalize succeeded -> transitions to VERIFY_RECOVERY
    fsm.step_recovery_lifecycle(3.6, 0.5, False, True, False)
    assert fsm.state == RecoveryState.VERIFY_RECOVERY

    # 7. Verification dwell: 0.5s (< 1.0s) -> stays in VERIFY_RECOVERY
    fsm.step_recovery_lifecycle(4.1, 0.5, False, True, True)
    assert fsm.state == RecoveryState.VERIFY_RECOVERY

    # 8. Verification dwell: 1.1s (>= 1.0s) -> transitions to REPLAN
    fsm.step_recovery_lifecycle(4.8, 0.7, False, True, True)
    assert fsm.state == RecoveryState.REPLAN

    # 9. REPLAN -> transitions to RESUME
    fsm.step_recovery_lifecycle(4.9, 0.1, False, True, True)
    assert fsm.state == RecoveryState.RESUME

    # 10. RESUME -> transitions to NORMAL
    fsm.step_recovery_lifecycle(5.0, 0.1, False, True, True)
    assert fsm.state == RecoveryState.NORMAL


def test_state_machine_budget_exhaustion_failed_safe():
    cfg = RecoveryStateMachineConfig()
    fsm = RecoveryStateMachine(cfg)
    fsm.reset(0.0)
    fsm.budget.max_attempts = 2

    # Exhaust attempt budget
    fsm.budget.record_attempt("STOP_AND_RELOCALIZE", "VO_LOST")
    fsm.budget.record_attempt("SHORT_BACKTRACK", "VO_LOST")
    assert fsm.budget.is_exhausted()

    # Any step in recovery states immediately transitions to FAILED_SAFE
    fsm.transition_to(RecoveryState.SAFE_STOP, 1.0)
    fsm.step_recovery_lifecycle(1.1, 0.1, False, False, False)
    assert fsm.state == RecoveryState.FAILED_SAFE
