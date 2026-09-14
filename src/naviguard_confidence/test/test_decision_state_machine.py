import pytest
from naviguard_confidence.confidence_dimensions import ConfidenceScores, DecisionState
from naviguard_confidence.decision_state_machine import (
    DecisionStateMachine,
    StateMachineConfig,
)


def test_state_machine_nominal_continue():
    cfg = StateMachineConfig(
        continue_threshold=0.75,
        verify_threshold=0.65,
        recover_threshold=0.35,
    )
    fsm = DecisionStateMachine(cfg)
    scores = ConfidenceScores(overall=0.92)

    # Step at t = 1.0s
    res = fsm.update(scores, "NOMINAL", [], 1.0)
    assert res.state == DecisionState.CONTINUE
    assert not res.transition_occurred


def test_state_machine_debounce_prevents_false_alarm():
    cfg = StateMachineConfig(
        verify_threshold=0.65,
        verify_debounce_sec=0.30,
    )
    fsm = DecisionStateMachine(cfg)
    fsm.reset(DecisionState.CONTINUE, 1.0)

    # Temporary dip for 0.15s (< 0.30s debounce)
    scores_dip = ConfidenceScores(overall=0.55, visual=0.38)
    res1 = fsm.update(scores_dip, "VO_LOW_INLIERS", [], 1.10)
    assert res1.state == DecisionState.CONTINUE  # Debounce holds!

    res2 = fsm.update(scores_dip, "VO_LOW_INLIERS", [], 1.25)
    assert res2.state == DecisionState.CONTINUE  # Still holding!

    # Quick recovery before debounce fires
    scores_ok = ConfidenceScores(overall=0.90)
    res3 = fsm.update(scores_ok, "NOMINAL", [], 1.30)
    assert res3.state == DecisionState.CONTINUE
    assert not res3.transition_occurred


def test_state_machine_sustained_degradation_transitions_to_verify():
    cfg = StateMachineConfig(
        verify_threshold=0.65,
        verify_debounce_sec=0.30,
    )
    fsm = DecisionStateMachine(cfg)
    fsm.reset(DecisionState.CONTINUE, 1.0)

    scores_deg = ConfidenceScores(overall=0.55, visual=0.35)

    # Sustained degradation exceeding 0.30s
    fsm.update(scores_deg, "VO_LOW_INLIERS", [], 1.10)
    fsm.update(scores_deg, "VO_LOW_INLIERS", [], 1.25)
    res = fsm.update(scores_deg, "VO_LOW_INLIERS", [], 1.45)

    assert res.state == DecisionState.VERIFY
    assert res.transition_occurred is True
    assert res.previous_state == DecisionState.CONTINUE
    assert "VO_LOW_INLIERS" in res.primary_reason


def test_state_machine_persistent_failure_transitions_to_recover():
    cfg = StateMachineConfig(
        verify_threshold=0.65,
        recover_threshold=0.35,
        verify_debounce_sec=0.20,
        recover_debounce_sec=0.50,
    )
    fsm = DecisionStateMachine(cfg)
    fsm.reset(DecisionState.VERIFY, 1.0)

    scores_lost = ConfidenceScores(overall=0.20, localization=0.10)

    # Degradation starting at 1.1s
    fsm.update(scores_lost, "SLAM_LOST", [], 1.10)
    fsm.update(scores_lost, "SLAM_LOST", [], 1.40)
    res = fsm.update(scores_lost, "SLAM_LOST", [], 1.65)  # 0.55s > 0.50s

    assert res.state == DecisionState.RECOVER
    assert res.transition_occurred is True
    assert res.previous_state == DecisionState.VERIFY


def test_state_machine_emergency_bypass():
    cfg = StateMachineConfig()
    fsm = DecisionStateMachine(cfg)
    fsm.reset(DecisionState.CONTINUE, 1.0)

    # Imminent obstacle collision (< 0.15 score) -> immediate RECOVER with zero delay
    scores_emerg = ConfidenceScores(overall=0.15, map=0.10)
    res = fsm.update(scores_emerg, "OBSTACLE_COLLISION_CRITICAL_0.20M", [], 1.05)

    assert res.state == DecisionState.RECOVER
    assert res.transition_occurred is True
    assert res.previous_state == DecisionState.CONTINUE


def test_state_machine_recovery_hysteresis():
    cfg = StateMachineConfig(
        recover_to_verify_threshold=0.50,
        continue_threshold=0.75,
        recover_clear_sec=1.00,
        continue_clear_sec=0.80,
    )
    fsm = DecisionStateMachine(cfg)
    fsm.reset(DecisionState.RECOVER, 1.0)

    scores_improved = ConfidenceScores(overall=0.60, localization=0.60, imu=1.0, wheel=1.0, cross_sensor=1.0)

    # Must dwell in acceptable condition for 1.00s before leaving RECOVER (from 1.5s to 2.55s)
    fsm.update(scores_improved, "IMPROVING", [], 1.5)
    assert fsm.current_state == DecisionState.RECOVER

    fsm.update(scores_improved, "IMPROVING", [], 2.0)
    assert fsm.current_state == DecisionState.RECOVER

    res_verify = fsm.update(scores_improved, "IMPROVING", [], 2.55)
    assert res_verify.state == DecisionState.VERIFY
    assert res_verify.transition_occurred is True

    # Now dwell in VERIFY for 0.80s at high confidence to return to CONTINUE
    scores_nominal = ConfidenceScores(overall=0.88, visual=0.85, localization=0.85, imu=1.0, wheel=1.0, temporal=1.0, cross_sensor=1.0, map=1.0)
    fsm.update(scores_nominal, "NOMINAL", [], 2.6)
    assert fsm.current_state == DecisionState.VERIFY

    fsm.update(scores_nominal, "NOMINAL", [], 3.0)
    assert fsm.current_state == DecisionState.VERIFY

    res_continue = fsm.update(scores_nominal, "NOMINAL", [], 3.45)
    assert res_continue.state == DecisionState.CONTINUE
    assert res_continue.transition_occurred is True
