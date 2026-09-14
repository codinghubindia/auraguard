"""Diagnostics builder for NAVIGUARD mission and navigation status."""

import json
from typing import Optional, Dict, Any
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import Header


class NavigationDiagnostics:
    """Builds standard DiagnosticArray and JSON status strings for navigation telemetry."""

    @staticmethod
    def build_diagnostic_array(
        stamp,
        mission_state: str,
        goal_dict: Optional[Dict[str, Any]],
        dist_to_goal: float,
        current_wp_idx: int,
        total_wps: int,
        path_length_m: float,
        replan_count: int,
        cmd_vx: float,
        cmd_wz: float,
        confidence_decision: str,
        recovery_state: str,
        ownership: str,
    ) -> DiagnosticArray:
        """Create diagnostic_msgs/msg/DiagnosticArray message."""
        diag = DiagnosticArray()
        diag.header.stamp = stamp

        status = DiagnosticStatus()
        status.name = "naviguard:mission_navigation"
        status.hardware_id = "naviguard_ugv"

        if mission_state in ("NAVIGATING", "GOAL_REACHED", "IDLE"):
            status.level = DiagnosticStatus.OK
            status.message = f"Navigation {mission_state}"
        elif mission_state in ("PLANNING", "REPLANNING", "GOAL_SET"):
            status.level = DiagnosticStatus.WARN
            status.message = f"Navigation {mission_state}"
        else:  # RECOVERY_WAIT, MISSION_FAILED
            status.level = DiagnosticStatus.ERROR
            status.message = f"Navigation {mission_state}"

        def kv(k: str, v: Any) -> KeyValue:
            val = KeyValue()
            val.key = str(k)
            val.value = str(v)
            return val

        status.values = [
            kv("mission_state", mission_state),
            kv("goal_active", goal_dict is not None),
            kv("goal_target", json.dumps(goal_dict) if goal_dict else "None"),
            kv("distance_to_goal_m", f"{dist_to_goal:.2f}"),
            kv("current_waypoint", f"{current_wp_idx}/{total_wps}"),
            kv("path_length_m", f"{path_length_m:.2f}"),
            kv("replan_count", replan_count),
            kv("command_linear_velocity", f"{cmd_vx:.3f}"),
            kv("command_angular_velocity", f"{cmd_wz:.3f}"),
            kv("confidence_decision", confidence_decision),
            kv("recovery_state", recovery_state),
            kv("cmd_vel_ownership", ownership),
        ]

        diag.status.append(status)
        return diag
