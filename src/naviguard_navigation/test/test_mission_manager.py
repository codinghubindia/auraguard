import pytest
from naviguard_navigation.mission_manager import MissionManager, MissionState


def test_mission_manager_nominal_lifecycle():
    mgr = MissionManager(max_replan_retries=3)

    assert mgr.state == MissionState.IDLE

    # Goal received -> GOAL_SET
    assert mgr.transition_to(MissionState.GOAL_SET, now_sec=1.0) is True
    assert mgr.state == MissionState.GOAL_SET

    # Start planning -> PLANNING
    assert mgr.transition_to(MissionState.PLANNING, now_sec=1.5) is True
    assert mgr.state == MissionState.PLANNING

    # Path computed -> NAVIGATING
    assert mgr.transition_to(MissionState.NAVIGATING, now_sec=2.0) is True
    assert mgr.state == MissionState.NAVIGATING

    # Arrived at destination -> GOAL_REACHED
    assert mgr.transition_to(MissionState.GOAL_REACHED, now_sec=15.0) is True
    assert mgr.state == MissionState.GOAL_REACHED


def test_mission_manager_recovery_and_resume():
    mgr = MissionManager(max_replan_retries=3)

    mgr.state = MissionState.NAVIGATING

    # Confidence degrades -> RECOVERY_WAIT
    assert mgr.transition_to(MissionState.RECOVERY_WAIT, now_sec=5.0, reason="CONFIDENCE_RECOVER") is True
    assert mgr.state == MissionState.RECOVERY_WAIT

    # Recovery succeeded -> REPLANNING
    assert mgr.transition_to(MissionState.REPLANNING, now_sec=10.0, reason="RECOVERY_RESUME") is True
    assert mgr.state == MissionState.REPLANNING

    # Path re-planned -> NAVIGATING
    assert mgr.transition_to(MissionState.NAVIGATING, now_sec=10.5) is True
    assert mgr.state == MissionState.NAVIGATING


def test_mission_manager_retry_budget_exhaustion():
    mgr = MissionManager(max_replan_retries=2)

    # First replan failure
    can_retry = mgr.record_planning_failure(now_sec=1.0, reason="BLOCKED")
    assert can_retry is True
    assert mgr.replan_count == 1

    # Second failure: hits limit (2) -> transitions to MISSION_FAILED
    can_retry = mgr.record_planning_failure(now_sec=2.0, reason="BLOCKED_AGAIN")
    assert can_retry is False
    assert mgr.state == MissionState.MISSION_FAILED
