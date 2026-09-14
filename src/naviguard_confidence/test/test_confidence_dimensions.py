import pytest
from naviguard_confidence.confidence_dimensions import (
    ConfidenceScores,
    DecisionResult,
    DecisionState,
)


def test_decision_state_enum():
    assert DecisionState.CONTINUE == 0
    assert DecisionState.VERIFY == 1
    assert DecisionState.RECOVER == 2

    assert DecisionState.CONTINUE.to_string() == "CONTINUE"
    assert DecisionState.VERIFY.to_string() == "VERIFY"
    assert DecisionState.RECOVER.to_string() == "RECOVER"

    assert DecisionState.from_string("continue") == DecisionState.CONTINUE
    assert DecisionState.from_string("VERIFY") == DecisionState.VERIFY
    assert DecisionState.from_string(" Recover ") == DecisionState.RECOVER

    with pytest.raises(ValueError):
        DecisionState.from_string("INVALID_STATE")


def test_confidence_scores_clamping():
    scores = ConfidenceScores(
        visual=1.5,
        localization=-0.5,
        imu=0.8,
        wheel=2.0,
        temporal=-1.0,
        cross_sensor=0.9,
        map=0.5,
        overall=1.2,
    )
    scores.clamp()

    assert scores.visual == 1.0
    assert scores.localization == 0.0
    assert scores.imu == 0.8
    assert scores.wheel == 1.0
    assert scores.temporal == 0.0
    assert scores.cross_sensor == 0.9
    assert scores.map == 0.5
    assert scores.overall == 1.0


def test_decision_result_serialization():
    scores = ConfidenceScores()
    res = DecisionResult(
        state=DecisionState.VERIFY,
        scores=scores,
        primary_reason="VO_LOW_INLIERS",
        secondary_reasons=["SLAM_DEGRADED"],
        timestamp_sec=123.456,
        dwell_time_sec=2.5,
        transition_occurred=True,
        previous_state=DecisionState.CONTINUE,
    )
    d = res.to_dict()

    assert d["state"] == "VERIFY"
    assert d["state_code"] == 1
    assert d["primary_reason"] == "VO_LOW_INLIERS"
    assert d["secondary_reasons"] == ["SLAM_DEGRADED"]
    assert d["timestamp_sec"] == 123.456
    assert d["dwell_time_sec"] == 2.5
    assert d["transition_occurred"] is True
    assert d["previous_state"] == "CONTINUE"
    assert d["scores"]["visual"] == 1.0
