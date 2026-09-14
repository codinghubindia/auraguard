"""Deterministic finite state machine with hysteresis and debounce for NAVIGUARD decisions.

Governs transitions between CONTINUE, VERIFY, and RECOVER states based on
multi-dimensional confidence metrics and persistence windows.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from naviguard_confidence.confidence_dimensions import (
    ConfidenceScores,
    DecisionResult,
    DecisionState,
)


@dataclass
class StateMachineConfig:
    # Thresholds
    continue_threshold: float = 0.75          # Minimum overall confidence to return to CONTINUE
    verify_threshold: float = 0.65            # Confidence below which VERIFY candidate is triggered
    recover_threshold: float = 0.35           # Confidence below which RECOVER candidate is triggered
    recover_to_verify_threshold: float = 0.50 # Confidence required to exit RECOVER to VERIFY

    # Minimum individual dimension thresholds
    min_dim_continue: float = 0.50            # All dims must exceed this to transition to CONTINUE
    min_dim_verify_trigger: float = 0.50      # Any dim <= this triggers VERIFY candidate
    critical_dim_recover_trigger: float = 0.20 # Any critical dim <= this triggers RECOVER candidate

    # Debounce / persistence durations (seconds)
    verify_debounce_sec: float = 0.30
    recover_debounce_sec: float = 0.80
    recover_clear_sec: float = 1.50           # Time in acceptable condition before leaving RECOVER
    continue_clear_sec: float = 1.00          # Time in acceptable condition before returning to CONTINUE

    # Startup grace period
    startup_grace_period_sec: float = 1.0     # Dwell in initial state before strict transitions


class DecisionStateMachine:
    """Finite State Machine enforcing persistence and hysteresis for autonomous decisions."""

    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.cfg = config or StateMachineConfig()
        self.current_state: DecisionState = DecisionState.CONTINUE
        self.state_entry_time: float = 0.0
        self.candidate_state: Optional[DecisionState] = None
        self.candidate_start_time: float = 0.0
        self.is_initialized: bool = False
        self.start_time: float = 0.0

    def reset(self, initial_state: DecisionState = DecisionState.CONTINUE, stamp_sec: float = 0.0) -> None:
        """Reset state machine to known configuration."""
        self.current_state = initial_state
        self.state_entry_time = stamp_sec
        self.candidate_state = None
        self.candidate_start_time = stamp_sec
        self.is_initialized = True
        self.start_time = stamp_sec

    def update(
        self,
        scores: ConfidenceScores,
        primary_reason: str,
        secondary_reasons: list,
        current_time: float,
    ) -> DecisionResult:
        """Step state machine with latest scores and compute deterministic decision."""
        if not self.is_initialized:
            self.reset(DecisionState.CONTINUE, current_time)

        # Emergency override check: immediate RECOVER with zero debounce
        is_emergency = (
            scores.map < 0.15
            or scores.imu == 0.0
            or scores.wheel == 0.0
            or "NAN_OR_INF" in primary_reason
            or "OBSTACLE_COLLISION_CRITICAL" in primary_reason
        )

        if is_emergency and self.current_state != DecisionState.RECOVER:
            prev_state = self.current_state
            self.current_state = DecisionState.RECOVER
            self.state_entry_time = current_time
            self.candidate_state = None
            return DecisionResult(
                state=self.current_state,
                scores=scores,
                primary_reason=primary_reason,
                secondary_reasons=secondary_reasons,
                timestamp_sec=current_time,
                dwell_time_sec=0.0,
                transition_occurred=True,
                previous_state=prev_state,
            )

        # Determine candidate target state based on scores
        target_candidate = self._evaluate_target_candidate(scores)

        # Transition candidate tracking
        transition_occurred = False
        previous_state: Optional[DecisionState] = None

        if target_candidate != self.current_state:
            if self.candidate_state != target_candidate:
                # Start tracking new candidate
                self.candidate_state = target_candidate
                self.candidate_start_time = current_time
            else:
                # Check persistence window
                dwell_candidate = current_time - self.candidate_start_time
                required_duration = self._get_required_persistence(self.current_state, target_candidate)

                if dwell_candidate >= required_duration:
                    # Execute transition
                    previous_state = self.current_state
                    self.current_state = target_candidate
                    self.state_entry_time = current_time
                    self.candidate_state = None
                    transition_occurred = True
        else:
            # Current state matches target candidate; reset candidate tracker
            self.candidate_state = None
            self.candidate_start_time = current_time

        dwell_time = current_time - self.state_entry_time

        return DecisionResult(
            state=self.current_state,
            scores=scores,
            primary_reason=primary_reason if self.current_state != DecisionState.CONTINUE else "NOMINAL_OPERATION",
            secondary_reasons=secondary_reasons,
            timestamp_sec=current_time,
            dwell_time_sec=dwell_time,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
        )

    def _evaluate_target_candidate(self, scores: ConfidenceScores) -> DecisionState:
        """Evaluate which state the instantaneous confidence levels correspond to."""
        min_dim = min(
            scores.visual,
            scores.localization,
            scores.imu,
            scores.wheel,
            scores.temporal,
            scores.cross_sensor,
            scores.map,
        )
        crit_dim = min(scores.imu, scores.wheel, scores.localization, scores.cross_sensor)

        # Condition for RECOVER
        if scores.overall < self.cfg.recover_threshold or crit_dim <= self.cfg.critical_dim_recover_trigger:
            return DecisionState.RECOVER

        # Condition for VERIFY
        if scores.overall < self.cfg.verify_threshold or min_dim <= self.cfg.min_dim_verify_trigger:
            return DecisionState.VERIFY

        # Condition for CONTINUE (requires high overall confidence and healthy all dims)
        if self.current_state == DecisionState.RECOVER:
            # To exit RECOVER, must first satisfy recover_to_verify threshold
            if scores.overall >= self.cfg.recover_to_verify_threshold and crit_dim >= 0.30:
                return DecisionState.VERIFY
            else:
                return DecisionState.RECOVER

        if self.current_state == DecisionState.VERIFY:
            if scores.overall >= self.cfg.continue_threshold and min_dim >= self.cfg.min_dim_continue:
                return DecisionState.CONTINUE
            else:
                return DecisionState.VERIFY

        return DecisionState.CONTINUE

    def _get_required_persistence(self, current: DecisionState, target: DecisionState) -> float:
        """Retrieve required continuous duration for state transition."""
        if current == DecisionState.CONTINUE and target == DecisionState.VERIFY:
            return self.cfg.verify_debounce_sec
        elif current == DecisionState.VERIFY and target == DecisionState.RECOVER:
            return self.cfg.recover_debounce_sec
        elif current == DecisionState.CONTINUE and target == DecisionState.RECOVER:
            return self.cfg.verify_debounce_sec + self.cfg.recover_debounce_sec
        elif current == DecisionState.RECOVER and target == DecisionState.VERIFY:
            return self.cfg.recover_clear_sec
        elif current == DecisionState.VERIFY and target == DecisionState.CONTINUE:
            return self.cfg.continue_clear_sec
        return 0.50
