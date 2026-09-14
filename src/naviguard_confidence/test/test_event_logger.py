import json
import os
import tempfile
import pytest

from naviguard_confidence.confidence_dimensions import (
    ConfidenceScores,
    DecisionResult,
    DecisionState,
)
from naviguard_confidence.event_logger import ConfidenceEventLogger


def test_confidence_event_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test_events.csv")
        json_file = os.path.join(tmpdir, "test_summary.json")

        logger = ConfidenceEventLogger(csv_path=csv_file, json_path=json_file)

        # Step 1: CONTINUE for 1.0s
        res1 = DecisionResult(
            state=DecisionState.CONTINUE,
            scores=ConfidenceScores(overall=0.9),
            timestamp_sec=10.0,
        )
        logger.log_step(res1)

        res2 = DecisionResult(
            state=DecisionState.CONTINUE,
            scores=ConfidenceScores(overall=0.9),
            timestamp_sec=11.0,
        )
        logger.log_step(res2)

        # Step 2: Transition to VERIFY
        res3 = DecisionResult(
            state=DecisionState.VERIFY,
            scores=ConfidenceScores(overall=0.6, visual=0.4),
            primary_reason="VO_LOW_INLIERS",
            timestamp_sec=11.5,
            transition_occurred=True,
            previous_state=DecisionState.CONTINUE,
        )
        logger.log_step(res3)

        # Step 3: Transition to RECOVER
        res4 = DecisionResult(
            state=DecisionState.RECOVER,
            scores=ConfidenceScores(overall=0.2, localization=0.1),
            primary_reason="SLAM_LOST",
            timestamp_sec=13.0,
            transition_occurred=True,
            previous_state=DecisionState.VERIFY,
        )
        logger.log_step(res4)

        stats = logger.get_summary_statistics()
        assert stats["total_transitions"] == 2
        assert stats["total_monitoring_time_sec"] == 3.0
        assert stats["state_dwell_times_sec"]["CONTINUE"] == 1.0
        assert stats["state_dwell_times_sec"]["VERIFY"] == 0.5

        logger.save_json_summary()
        assert os.path.exists(json_file)
        assert os.path.exists(csv_file)

        with open(json_file, 'r') as f:
            data = json.load(f)
            assert data["total_transitions"] == 2
