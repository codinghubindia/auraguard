import pytest
from naviguard_navigation.mission_manager import MissionManager, MissionState, FailureCode, FAILURE_TAXONOMY


def test_failure_taxonomy_completeness():
    """Ensure all defined FailureCodes have complete taxonomy metadata."""
    for attr in dir(FailureCode):
        if attr.isupper() and not attr.startswith("_"):
            code_val = getattr(FailureCode, attr)
            assert code_val in FAILURE_TAXONOMY, f"Missing taxonomy entry for {code_val}"

            meta = FAILURE_TAXONOMY[code_val]
            assert "category" in meta
            assert "human_reason" in meta
            assert "detail" in meta
            assert len(meta["human_reason"]) > 5
            assert len(meta["detail"]) > 10


def test_mission_manager_trigger_failure_structured_record():
    """Verify trigger_failure generates compliant structured record."""
    mgr = MissionManager()
    mgr.start_mission(now_sec=100.0, initial_path_length=25.0)

    context = {
        "recovery_state": "BACKTRACKING",
        "confidence": 0.85,
        "visual_confidence": 0.90,
        "localization_confidence": 0.82,
        "goal": [15.0, 10.0],
        "robot_pose": [5.0, 4.0],
        "distance_to_goal": 11.2,
        "recovery_attempts": 3,
        "recovery_max_attempts": 3,
        "blocked_regions": 2,
    }

    record = mgr.trigger_failure(
        failure_code=FailureCode.NO_SAFE_PATH,
        now_sec=115.5,
        context=context,
    )

    assert mgr.state == MissionState.MISSION_FAILED
    assert record["failure_code"] == FailureCode.NO_SAFE_PATH
    assert record["failure_category"] == "NAVIGATION"
    assert "No safe collision-free path" in record["human_reason"]
    assert record["recovery_attempts"] == 3
    assert record["distance_to_goal"] == 11.2
    assert record["timestamp"] == 115.5

    # Latest record stored on manager
    assert mgr.failure_record == record


def test_mission_manager_trigger_success_structured_record():
    """Verify trigger_success produces structured mission completion report."""
    mgr = MissionManager()
    mgr.transition_to(MissionState.GOAL_SET, 100.0)
    mgr.transition_to(MissionState.PLANNING, 100.1)
    mgr.transition_to(MissionState.NAVIGATING, 100.2)
    mgr.start_mission(now_sec=100.2, initial_path_length=20.0)
    # Simulate movement
    mgr.travel_distance_m = 21.4
    mgr.replan_count = 1

    record = mgr.trigger_success(now_sec=150.0, final_error_m=0.12, path_length_m=20.0)

    assert mgr.state == MissionState.GOAL_REACHED
    assert record["mission_state"] == "GOAL_REACHED"
    assert record["travel_distance_m"] == 21.4
    assert record["duration_sec"] == 49.8
    assert record["final_error_m"] == 0.12
    assert record["replan_count"] == 1
    assert mgr.success_record == record


def test_mission_manager_reset_clears_records():
    """Verify reset clears records and returns to IDLE state."""
    mgr = MissionManager()
    mgr.start_mission(now_sec=10.0, initial_path_length=15.0)
    mgr.trigger_failure(failure_code=FailureCode.PATH_BLOCKED, now_sec=15.0)
    assert mgr.failure_record is not None

    mgr.reset(now_sec=20.0)
    assert mgr.state == MissionState.IDLE
    assert mgr.failure_record is None
    assert mgr.success_record is None
