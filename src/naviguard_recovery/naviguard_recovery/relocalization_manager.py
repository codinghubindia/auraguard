"""Relocalization Verification and State Consistency Validator.

Enforces multi-frame dwell criteria and sensor confidence validation before
declaring successful recovery.
"""

from typing import Optional, Tuple


class RelocalizationManager:
    """Validates relocalization and confidence recovery."""

    def __init__(self, min_dwell_sec: float = 1.00) -> None:
        self.min_dwell_sec = min_dwell_sec
        self.dwell_start_time: Optional[float] = None
        self.min_overall_conf = 0.75
        self.min_loc_conf = 0.70
        self.min_vis_conf = 0.55

    def reset(self) -> None:
        self.dwell_start_time = None

    def evaluate_verification(
        self,
        current_time: float,
        conf_overall: float,
        conf_loc: float,
        conf_vis: float,
        slam_tracking_state: str,
    ) -> Tuple[bool, float]:
        """Check if recovery verification criteria are satisfied across a continuous dwell window."""
        is_healthy = (
            conf_overall >= self.min_overall_conf
            and conf_loc >= self.min_loc_conf
            and conf_vis >= self.min_vis_conf
            and slam_tracking_state.upper() == "OK"
        )

        if not is_healthy:
            self.dwell_start_time = None
            return False, 0.0

        if self.dwell_start_time is None:
            self.dwell_start_time = current_time
            return False, 0.0

        dwell = current_time - self.dwell_start_time
        if dwell >= self.min_dwell_sec:
            return True, dwell

        return False, dwell
