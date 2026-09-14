"""Diagnostics and RViz Marker visualization builder for NAVIGUARD recovery."""

from typing import List, Optional, Tuple
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from naviguard_recovery.recovery_state_machine import RecoveryBudget, RecoveryState
from naviguard_recovery.trusted_state_manager import TrustedCheckpoint


class RecoveryDiagnosticsBuilder:
    """Builds structured ROS 2 diagnostics and visual markers for recovery monitoring."""

    def __init__(self, hardware_id: str = "naviguard_ugv_recovery") -> None:
        self.hardware_id = hardware_id

    def build_diagnostic_array(
        self,
        stamp_msg,
        state: RecoveryState,
        strategy: str,
        budget: RecoveryBudget,
        failure_reason: str,
        dwell_sec: float,
        selected_checkpoint: Optional[TrustedCheckpoint],
        checkpoints_count: int,
    ) -> DiagnosticArray:
        msg = DiagnosticArray()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = "map"

        ds = DiagnosticStatus()
        ds.name = "naviguard_recovery: state"
        ds.hardware_id = self.hardware_id

        if state == RecoveryState.NORMAL:
            ds.level = DiagnosticStatus.OK
            ds.message = "Nominal mission execution"
        elif state in (RecoveryState.VERIFY, RecoveryState.REPLAN, RecoveryState.RESUME):
            ds.level = DiagnosticStatus.WARN
            ds.message = f"State: {state.to_string()} ({failure_reason})"
        elif state == RecoveryState.FAILED_SAFE:
            ds.level = DiagnosticStatus.ERROR
            ds.message = f"Terminal FAILED_SAFE: {failure_reason}"
        else:
            ds.level = DiagnosticStatus.WARN
            ds.message = f"Active Recovery: {state.to_string()} | Strategy: {strategy}"

        ds.values = [
            KeyValue(key="recovery_state", value=state.to_string()),
            KeyValue(key="recovery_state_code", value=str(int(state))),
            KeyValue(key="active_strategy", value=strategy),
            KeyValue(key="failure_reason", value=failure_reason),
            KeyValue(key="attempt_count", value=str(budget.attempt_count)),
            KeyValue(key="max_attempts", value=str(budget.max_attempts)),
            KeyValue(key="total_recovery_time_sec", value=f"{budget.total_recovery_time_sec:.2f}"),
            KeyValue(key="distance_backtracked_m", value=f"{budget.distance_backtracked_m:.2f}"),
            KeyValue(key="rotation_used_deg", value=f"{budget.rotation_used_deg:.1f}"),
            KeyValue(key="state_dwell_sec", value=f"{dwell_sec:.2f}"),
            KeyValue(key="buffered_checkpoints", value=str(checkpoints_count)),
            KeyValue(
                key="selected_checkpoint_id",
                value=str(selected_checkpoint.checkpoint_id) if selected_checkpoint else "NONE",
            ),
        ]
        msg.status.append(ds)
        return msg

    def build_markers(
        self,
        stamp_msg,
        state: RecoveryState,
        strategy: str,
        checkpoints: List[TrustedCheckpoint],
        selected_checkpoint: Optional[TrustedCheckpoint],
        backtrack_path: List[Tuple[float, float, float]],
        robot_pose: Tuple[float, float, float],
    ) -> MarkerArray:
        marker_arr = MarkerArray()

        # 1. Buffered Checkpoints (Cyan spheres)
        m_ckpts = Marker()
        m_ckpts.header.stamp = stamp_msg
        m_ckpts.header.frame_id = "map"
        m_ckpts.ns = "recovery_checkpoints"
        m_ckpts.id = 0
        m_ckpts.type = Marker.SPHERE_LIST
        m_ckpts.action = Marker.ADD
        m_ckpts.scale.x = 0.15
        m_ckpts.scale.y = 0.15
        m_ckpts.scale.z = 0.15
        m_ckpts.color.r = 0.0
        m_ckpts.color.g = 0.8
        m_ckpts.color.b = 1.0
        m_ckpts.color.a = 0.7

        for ckpt in checkpoints:
            p = Point()
            p.x = float(ckpt.map_pose[0])
            p.y = float(ckpt.map_pose[1])
            p.z = 0.05
            m_ckpts.points.append(p)
        marker_arr.markers.append(m_ckpts)

        # 2. Selected Recovery Checkpoint (Gold diamond)
        if selected_checkpoint:
            m_sel = Marker()
            m_sel.header.stamp = stamp_msg
            m_sel.header.frame_id = "map"
            m_sel.ns = "selected_checkpoint"
            m_sel.id = 1
            m_sel.type = Marker.CYLINDER
            m_sel.action = Marker.ADD
            m_sel.pose.position.x = float(selected_checkpoint.map_pose[0])
            m_sel.pose.position.y = float(selected_checkpoint.map_pose[1])
            m_sel.pose.position.z = 0.10
            m_sel.pose.orientation.w = 1.0
            m_sel.scale.x = 0.35
            m_sel.scale.y = 0.35
            m_sel.scale.z = 0.10
            m_sel.color.r = 1.0
            m_sel.color.g = 0.85
            m_sel.color.b = 0.0
            m_sel.color.a = 0.90
            marker_arr.markers.append(m_sel)

        # 3. Backtrack Path (Orange line strip)
        if backtrack_path:
            m_path = Marker()
            m_path.header.stamp = stamp_msg
            m_path.header.frame_id = "map"
            m_path.ns = "recovery_backtrack_path"
            m_path.id = 2
            m_path.type = Marker.LINE_STRIP
            m_path.action = Marker.ADD
            m_path.scale.x = 0.05
            m_path.color.r = 1.0
            m_path.color.g = 0.4
            m_path.color.b = 0.0
            m_path.color.a = 0.85
            for wp in backtrack_path:
                p = Point()
                p.x = float(wp[0])
                p.y = float(wp[1])
                p.z = 0.05
                m_path.points.append(p)
            marker_arr.markers.append(m_path)

        # 4. Status Billboard floating above UGV
        m_text = Marker()
        m_text.header.stamp = stamp_msg
        m_text.header.frame_id = "map"
        m_text.ns = "recovery_status_text"
        m_text.id = 3
        m_text.type = Marker.TEXT_VIEW_FACING
        m_text.action = Marker.ADD
        m_text.pose.position.x = float(robot_pose[0])
        m_text.pose.position.y = float(robot_pose[1])
        m_text.pose.position.z = 0.75
        m_text.pose.orientation.w = 1.0
        m_text.scale.z = 0.14
        m_text.color.r = 1.0
        m_text.color.g = 1.0
        m_text.color.b = 1.0
        m_text.color.a = 0.95
        m_text.text = f"RECOVERY: {state.to_string()}\nStrategy: {strategy}"
        marker_arr.markers.append(m_text)

        return marker_arr
