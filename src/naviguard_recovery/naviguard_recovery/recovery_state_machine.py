"""Deterministic Finite State Machine and Budget Tracking for Autonomous Recovery.

Defines the 10 operational recovery states, transition logic, event processing,
and bounded recovery resource tracking.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class RecoveryState(IntEnum):
    """Operational recovery lifecycle states."""
    NORMAL = 0              # Normal mission execution (Phase 7: CONTINUE)
    VERIFY = 1              # Cautionary dwell/slowdown (Phase 7: VERIFY)
    SAFE_STOP = 2           # Immediate active zero-velocity command & stop dwell
    SELECT_CHECKPOINT = 3   # Evaluate trusted checkpoint history and score candidates
    RECOVER = 4             # Executing selected physical recovery strategy
    RELOCALIZE = 5          # Triggering / awaiting SLAM localization reacquisition
    VERIFY_RECOVERY = 6     # Sustained multi-frame confidence verification dwell
    REPLAN = 7              # Trusted state updated; request path replan to goal
    RESUME = 8              # Transitioning back to mission execution
    FAILED_SAFE = 9         # Terminal safe stop on exhausted budget or unresolvable fault
    VISUAL_REACQUISITION_ROTATION = 10      # Controlled rotation scan
    VISUAL_REACQUISITION_OBSERVATION = 11   # Stationary dwell for feature settlement
    TRANSLATIONAL_RELOCALIZATION = 12       # Controlled translation check

    def to_string(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "RecoveryState":
        clean_name = name.strip().upper()
        if clean_name in cls.__members__:
            return cls.__members__[clean_name]
        raise ValueError(f"Unknown RecoveryState: {name}")


@dataclass
class RecoveryAttemptRecord:
    """Diagnostic record for a single recovery attempt."""
    attempt_id: int = 1
    start_timestamp: float = 0.0
    strategy: str = "NONE"
    rotation_amount_deg: float = 0.0
    displacement_m: float = 0.0
    localization_result: str = "PENDING"
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    success: bool = False
    termination_reason: str = "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": int(self.attempt_id),
            "start_timestamp": round(float(self.start_timestamp), 3),
            "strategy": str(self.strategy),
            "rotation_amount_deg": round(float(self.rotation_amount_deg), 2),
            "displacement_m": round(float(self.displacement_m), 3),
            "localization_result": str(self.localization_result),
            "confidence_before": round(float(self.confidence_before), 3),
            "confidence_after": round(float(self.confidence_after), 3),
            "success": bool(self.success),
            "termination_reason": str(self.termination_reason),
        }


@dataclass
class RecoveryBudget:
    """Bounded resource budget per recovery mission episode."""
    max_attempts: int = 3
    max_total_duration_sec: float = 40.0
    max_backtrack_dist_m: float = 3.5
    max_rotation_deg: float = 180.0

    attempt_count: int = 0
    total_recovery_time_sec: float = 0.0
    distance_backtracked_m: float = 0.0
    rotation_used_deg: float = 0.0
    last_strategy: str = "NONE"
    failure_reason: str = "NONE"
    current_attempt: Optional[RecoveryAttemptRecord] = None
    attempt_history: List[RecoveryAttemptRecord] = field(default_factory=list)

    def record_attempt(
        self,
        strategy_name: str = "",
        reason: str = "",
        timestamp_sec: float = 0.0,
        conf_before: float = 0.0,
        **kwargs,
    ) -> RecoveryAttemptRecord:
        if not strategy_name:
            strategy_name = kwargs.get("strategy", kwargs.get("strategy_name", "NONE"))
        if not reason:
            reason = kwargs.get("reason", "NONE")
        if conf_before == 0.0:
            conf_before = float(kwargs.get("confidence_before", kwargs.get("conf_before", 0.0)))
        if timestamp_sec == 0.0:
            timestamp_sec = float(kwargs.get("timestamp_sec", kwargs.get("timestamp", 0.0)))

        self.attempt_count += 1
        self.last_strategy = strategy_name
        self.failure_reason = reason
        rec = RecoveryAttemptRecord(
            attempt_id=self.attempt_count,
            start_timestamp=timestamp_sec,
            strategy=strategy_name,
            confidence_before=conf_before,
            termination_reason=reason,
        )
        self.current_attempt = rec
        self.attempt_history.append(rec)
        return rec

    def finish_current_attempt(
        self,
        success: bool,
        loc_result: str,
        conf_after: float = 0.0,
        termination_reason: str = "",
    ) -> None:
        if self.current_attempt is not None:
            self.current_attempt.success = bool(success)
            self.current_attempt.localization_result = str(loc_result)
            self.current_attempt.confidence_after = float(conf_after)
            if termination_reason:
                self.current_attempt.termination_reason = str(termination_reason)

    def record_motion(self, dist_m: float, rot_deg: float) -> None:
        self.distance_backtracked_m += abs(dist_m)
        self.rotation_used_deg += abs(rot_deg)
        if self.current_attempt is not None:
            self.current_attempt.displacement_m += abs(dist_m)
            self.current_attempt.rotation_amount_deg += abs(rot_deg)

    def is_exhausted(self) -> bool:
        if self.attempt_count >= self.max_attempts:
            return True
        if self.total_recovery_time_sec >= self.max_total_duration_sec:
            return True
        if self.distance_backtracked_m >= self.max_backtrack_dist_m:
            return True
        if self.rotation_used_deg >= self.max_rotation_deg:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "attempt_count": self.attempt_count,
            "attempts": self.attempt_count,
            "max_total_duration_sec": self.max_total_duration_sec,
            "total_recovery_time_sec": float(self.total_recovery_time_sec),
            "max_backtrack_dist_m": self.max_backtrack_dist_m,
            "distance_backtracked_m": float(self.distance_backtracked_m),
            "max_rotation_deg": self.max_rotation_deg,
            "rotation_used_deg": float(self.rotation_used_deg),
            "last_strategy": self.last_strategy,
            "failure_reason": self.failure_reason,
            "is_exhausted": bool(self.is_exhausted()),
            "current_attempt": self.current_attempt.to_dict() if self.current_attempt else None,
            "attempt_history": [a.to_dict() for a in self.attempt_history],
        }



@dataclass
class RecoveryStateMachineConfig:
    safe_stop_dwell_sec: float = 0.50
    verification_dwell_sec: float = 1.00
    action_timeout_sec: float = 10.0


class RecoveryStateMachine:
    """Finite State Machine managing the complete autonomous recovery lifecycle."""

    def __init__(self, config: Optional[RecoveryStateMachineConfig] = None) -> None:
        self.cfg = config or RecoveryStateMachineConfig()
        self.state: RecoveryState = RecoveryState.NORMAL
        self.budget: RecoveryBudget = RecoveryBudget()
        self.state_entry_time: float = 0.0
        self.failure_reason: str = "NONE"
        self.active_strategy: str = "NONE"
        self.transition_occurred: bool = False
        self.previous_state: Optional[RecoveryState] = None

    def reset(self, stamp_sec: float = 0.0) -> None:
        self.state = RecoveryState.NORMAL
        self.budget = RecoveryBudget()
        self.state_entry_time = stamp_sec
        self.failure_reason = "NONE"
        self.active_strategy = "NONE"
        self.transition_occurred = False
        self.previous_state = None

    def transition_to(self, new_state: RecoveryState, stamp_sec: float, reason: str = "") -> None:
        if new_state != self.state:
            self.previous_state = self.state
            self.state = new_state
            self.state_entry_time = stamp_sec
            self.transition_occurred = True
            if reason:
                self.failure_reason = reason
        else:
            self.transition_occurred = False

    def update_phase7_input(self, phase7_state_str: str, primary_reason: str, stamp_sec: float) -> None:
        """Handle incoming decision from Phase 7 Confidence Node."""
        phase7_state_str = phase7_state_str.upper()

        if self.state == RecoveryState.NORMAL:
            if phase7_state_str == "VERIFY":
                self.transition_to(RecoveryState.VERIFY, stamp_sec, primary_reason)
            elif phase7_state_str == "RECOVER":
                self.transition_to(RecoveryState.SAFE_STOP, stamp_sec, primary_reason)

        elif self.state == RecoveryState.VERIFY:
            if phase7_state_str == "RECOVER":
                self.transition_to(RecoveryState.SAFE_STOP, stamp_sec, primary_reason)
            elif phase7_state_str == "CONTINUE":
                self.transition_to(RecoveryState.NORMAL, stamp_sec, "CONFIDENCE_RESTORED")

    def step_recovery_lifecycle(
        self,
        stamp_sec: float,
        dt: float,
        action_in_progress: bool,
        relocalization_ok: bool,
        verification_ok: bool,
    ) -> RecoveryState:
        """Step the internal recovery states deterministically."""
        # Accumulate time while in non-normal recovery states
        if self.state not in (RecoveryState.NORMAL, RecoveryState.VERIFY):
            self.budget.total_recovery_time_sec += dt

        # Global budget exhaustion check
        if self.state not in (RecoveryState.NORMAL, RecoveryState.FAILED_SAFE):
            if self.budget.is_exhausted():
                self.transition_to(RecoveryState.FAILED_SAFE, stamp_sec, "RECOVERY_BUDGET_EXHAUSTED")
                return self.state

        dwell = stamp_sec - self.state_entry_time

        if self.state == RecoveryState.SAFE_STOP:
            if dwell >= self.cfg.safe_stop_dwell_sec:
                self.transition_to(RecoveryState.SELECT_CHECKPOINT, stamp_sec)

        elif self.state == RecoveryState.SELECT_CHECKPOINT:
            # Action dispatcher will set active_strategy and transition to RECOVER
            pass

        elif self.state == RecoveryState.RECOVER:
            if not action_in_progress:
                self.transition_to(RecoveryState.VISUAL_REACQUISITION_OBSERVATION, stamp_sec)
            elif dwell >= self.cfg.action_timeout_sec:
                # Action timed out
                if self.budget.attempt_count < self.budget.max_attempts:
                    self.budget.finish_current_attempt(False, "ACTION_TIMED_OUT", 0.0, "ACTION_TIMED_OUT")
                    self.transition_to(RecoveryState.SAFE_STOP, stamp_sec, "ACTION_TIMED_OUT")
                else:
                    self.budget.finish_current_attempt(False, "ACTION_TIMED_OUT", 0.0, "ACTION_TIMED_OUT_BUDGET_EXHAUSTED")
                    self.transition_to(RecoveryState.FAILED_SAFE, stamp_sec, "ACTION_TIMED_OUT_BUDGET_EXHAUSTED")

        elif self.state == RecoveryState.VISUAL_REACQUISITION_OBSERVATION:
            # Settle dwell: 0.50s with zero velocity to collect clean visual features
            if dwell >= 0.50:
                self.transition_to(RecoveryState.RELOCALIZE, stamp_sec)

        elif self.state == RecoveryState.RELOCALIZE:
            if relocalization_ok:
                self.transition_to(RecoveryState.VERIFY_RECOVERY, stamp_sec)
            elif dwell >= 3.0:
                # Relocalization timed out
                if self.budget.attempt_count < self.budget.max_attempts:
                    self.budget.finish_current_attempt(False, "RELOCALIZE_TIMEOUT", 0.0, "RELOCALIZE_TIMEOUT")
                    self.transition_to(RecoveryState.SAFE_STOP, stamp_sec, "RELOCALIZE_TIMEOUT")
                else:
                    self.budget.finish_current_attempt(False, "RELOCALIZE_TIMEOUT", 0.0, "RELOCALIZE_FAILED_SAFE")
                    self.transition_to(RecoveryState.FAILED_SAFE, stamp_sec, "RELOCALIZE_FAILED_SAFE")

        elif self.state == RecoveryState.VERIFY_RECOVERY:
            if verification_ok and dwell >= self.cfg.verification_dwell_sec:
                self.budget.finish_current_attempt(True, "SUCCESS", 1.0, "VERIFICATION_OK")
                self.transition_to(RecoveryState.REPLAN, stamp_sec)
            elif not verification_ok and dwell >= 2.0:
                if self.budget.attempt_count < self.budget.max_attempts:
                    self.budget.finish_current_attempt(False, "VERIFICATION_FAILED", 0.0, "VERIFICATION_FAILED")
                    self.transition_to(RecoveryState.SAFE_STOP, stamp_sec, "VERIFICATION_FAILED")
                else:
                    self.budget.finish_current_attempt(False, "VERIFICATION_FAILED", 0.0, "VERIFICATION_BUDGET_EXHAUSTED")
                    self.transition_to(RecoveryState.FAILED_SAFE, stamp_sec, "VERIFICATION_BUDGET_EXHAUSTED")


        elif self.state == RecoveryState.REPLAN:
            # Replan request acknowledged
            self.transition_to(RecoveryState.RESUME, stamp_sec)

        elif self.state == RecoveryState.RESUME:
            # Mission resumed back to normal
            self.transition_to(RecoveryState.NORMAL, stamp_sec, "MISSION_RESUMED")

        return self.state
