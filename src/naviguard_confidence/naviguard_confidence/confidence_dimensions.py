"""Confidence dimensions and decision data structures for NAVIGUARD.

Defines the multi-dimensional confidence metric container, decision states,
and decision results for the NAVIGUARD autonomy engine.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np


class DecisionState(IntEnum):
    """Autonomy decision operating state."""
    CONTINUE = 0   # Normal autonomous operation permitted
    VERIFY = 1     # Cautious state: reduced speed, cross-validation, sensor check
    RECOVER = 2    # Critical state: immediate stop, initiate recovery procedure

    def to_string(self) -> str:
        """Convert state enum to uppercase string."""
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "DecisionState":
        """Parse state enum from string."""
        name_clean = name.strip().upper()
        if name_clean in cls.__members__:
            return cls.__members__[name_clean]
        raise ValueError(f"Unknown DecisionState: {name}")


@dataclass
class ConfidenceScores:
    """Multi-dimensional confidence scores bounded strictly in [0.0, 1.0]."""
    visual: float = 1.0
    localization: float = 1.0
    imu: float = 1.0
    wheel: float = 1.0
    temporal: float = 1.0
    cross_sensor: float = 1.0
    map: float = 1.0
    overall: float = 1.0

    def clamp(self) -> None:
        """Clamp all dimensions to [0.0, 1.0]."""
        self.visual = float(np.clip(self.visual, 0.0, 1.0))
        self.localization = float(np.clip(self.localization, 0.0, 1.0))
        self.imu = float(np.clip(self.imu, 0.0, 1.0))
        self.wheel = float(np.clip(self.wheel, 0.0, 1.0))
        self.temporal = float(np.clip(self.temporal, 0.0, 1.0))
        self.cross_sensor = float(np.clip(self.cross_sensor, 0.0, 1.0))
        self.map = float(np.clip(self.map, 0.0, 1.0))
        self.overall = float(np.clip(self.overall, 0.0, 1.0))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "visual": float(self.visual),
            "localization": float(self.localization),
            "imu": float(self.imu),
            "wheel": float(self.wheel),
            "temporal": float(self.temporal),
            "cross_sensor": float(self.cross_sensor),
            "map": float(self.map),
            "overall": float(self.overall),
        }


@dataclass
class DecisionResult:
    """Consolidated autonomy decision and multi-sensor confidence breakdown."""
    state: DecisionState = DecisionState.CONTINUE
    scores: ConfidenceScores = field(default_factory=ConfidenceScores)
    primary_reason: str = "NOMINAL_OPERATION"
    secondary_reasons: List[str] = field(default_factory=list)
    timestamp_sec: float = 0.0
    dwell_time_sec: float = 0.0
    transition_occurred: bool = False
    previous_state: Optional[DecisionState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "state": self.state.to_string(),
            "state_code": int(self.state),
            "scores": self.scores.to_dict(),
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "timestamp_sec": float(self.timestamp_sec),
            "dwell_time_sec": float(self.dwell_time_sec),
            "transition_occurred": bool(self.transition_occurred),
            "previous_state": self.previous_state.to_string() if self.previous_state is not None else None,
        }
