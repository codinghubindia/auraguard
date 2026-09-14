"""
NAVIGUARD Ground Truth Isolation Guard.

Enforces strict architectural boundaries between evaluation/ground-truth data
and the runtime navigation/control/estimation pipeline.

Any attempt to route ground-truth state or dataset labels into runtime navigation
topics (e.g. /slam/pose, /odom, /cmd_vel, /plan) raises an immediate exception.
"""

from typing import Iterable, Set

# Topics that must NEVER receive ground truth data
RESTRICTED_RUNTIME_TOPICS: Set[str] = {
    '/slam/pose',
    '/slam/trajectory',
    '/slam/landmarks',
    '/odom',
    '/cmd_vel',
    '/odometry/filtered',
    '/navigation/plan',
    '/navigation/goal',
    '/confidence/metrics',
    '/confidence/decision',
    '/recovery/command',
}

# Permitted topics for ground truth data in evaluation/benchmarking mode
ALLOWED_EVALUATION_TOPICS: Set[str] = {
    '/evaluation/ground_truth_pose',
    '/evaluation/ground_truth_path',
    '/evaluation/ground_truth_trajectory',
    '/evaluation/estimated_trajectory',
    '/evaluation/metrics',
    '/evaluation/pointcloud',
    '/evaluation/status',
}


class GroundTruthLeakageError(RuntimeError):
    """Raised when ground truth data is directed to a runtime navigation topic."""
    pass


class GroundTruthGuard:
    """Architectural gatekeeper ensuring ground truth isolation."""

    @staticmethod
    def assert_topic_allowed(topic_name: str) -> None:
        """
        Verify that a topic is permitted to receive evaluation/ground-truth data.

        Raises GroundTruthLeakageError if the topic is a restricted runtime topic.
        """
        clean_topic = topic_name.strip()
        if not clean_topic.startswith('/'):
            clean_topic = '/' + clean_topic

        if clean_topic in RESTRICTED_RUNTIME_TOPICS:
            raise GroundTruthLeakageError(
                f"CRITICAL VIOLATION: Attempted to publish ground truth data to "
                f"runtime navigation topic '{clean_topic}'! Ground truth must remain "
                f"isolated to /evaluation/* topics."
            )

    @staticmethod
    def is_evaluation_topic(topic_name: str) -> bool:
        """Return True if topic is under the /evaluation namespace."""
        clean_topic = topic_name.strip()
        if not clean_topic.startswith('/'):
            clean_topic = '/' + clean_topic
        return clean_topic.startswith('/evaluation/')

    @staticmethod
    def filter_safe_topics(topics: Iterable[str]) -> Set[str]:
        """Filter an iterable of topics, removing any restricted topics."""
        safe = set()
        for t in topics:
            clean = t.strip()
            if not clean.startswith('/'):
                clean = '/' + clean
            if clean not in RESTRICTED_RUNTIME_TOPICS:
                safe.add(clean)
        return safe
