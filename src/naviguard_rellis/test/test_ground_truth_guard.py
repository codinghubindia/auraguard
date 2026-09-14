import pytest
from naviguard_rellis.ground_truth_guard import GroundTruthGuard, GroundTruthLeakageError, RESTRICTED_RUNTIME_TOPICS, ALLOWED_EVALUATION_TOPICS


def test_guard_blocks_runtime_topics():
    for restricted in RESTRICTED_RUNTIME_TOPICS:
        with pytest.raises(GroundTruthLeakageError):
            GroundTruthGuard.assert_topic_allowed(restricted)


def test_guard_allows_evaluation_topics():
    for allowed in ALLOWED_EVALUATION_TOPICS:
        GroundTruthGuard.assert_topic_allowed(allowed)


def test_is_evaluation_topic():
    assert GroundTruthGuard.is_evaluation_topic('/evaluation/ground_truth_pose') is True
    assert GroundTruthGuard.is_evaluation_topic('/evaluation/metrics') is True
    assert GroundTruthGuard.is_evaluation_topic('/slam/pose') is False
    assert GroundTruthGuard.is_evaluation_topic('/cmd_vel') is False


def test_filter_safe_topics():
    raw = ['/slam/pose', '/evaluation/metrics', '/cmd_vel', '/evaluation/test']
    safe = GroundTruthGuard.filter_safe_topics(raw)
    assert '/slam/pose' not in safe
    assert '/cmd_vel' not in safe
    assert '/evaluation/metrics' in safe
    assert '/evaluation/test' in safe
