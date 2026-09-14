"""Goal management and validation for NAVIGUARD."""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from geometry_msgs.msg import PoseStamped


@dataclass
class NavigationGoal:
    x: float
    y: float
    yaw: Optional[float]
    frame_id: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "yaw": round(self.yaw, 3) if self.yaw is not None else None,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


class GoalManager:
    """Manages active navigation goals and validation."""

    def __init__(self, default_frame: str = "map") -> None:
        self.current_goal: Optional[NavigationGoal] = None
        self.default_frame = default_frame

    def set_goal_from_msg(self, msg: PoseStamped, now_sec: float) -> Optional[NavigationGoal]:
        """Parse geometry_msgs/msg/PoseStamped into a NavigationGoal."""
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        frame_id = msg.header.frame_id if msg.header.frame_id else self.default_frame

        # Extract yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        # Check if quaternion is valid/non-zero
        norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
        yaw = math.atan2(siny_cosp, cosy_cosp) if norm_sq > 1e-4 else None

        self.current_goal = NavigationGoal(
            x=x,
            y=y,
            yaw=yaw,
            frame_id=frame_id,
            timestamp=now_sec,
        )
        return self.current_goal

    def set_goal(self, x: float, y: float, yaw: Optional[float] = None, now_sec: float = 0.0) -> NavigationGoal:
        """Explicitly set navigation goal."""
        self.current_goal = NavigationGoal(
            x=float(x),
            y=float(y),
            yaw=float(yaw) if yaw is not None else None,
            frame_id=self.default_frame,
            timestamp=now_sec,
        )
        return self.current_goal

    def clear_goal(self) -> None:
        """Clear active navigation goal."""
        self.current_goal = None

    def has_goal(self) -> bool:
        """Check if an active goal exists."""
        return self.current_goal is not None

    def get_goal(self) -> Optional[NavigationGoal]:
        """Get active goal."""
        return self.current_goal
