"""Sensor and pipeline monitors for multi-dimensional confidence estimation.

Monitors visual odometry, SLAM localization, IMU, wheel odometry,
sensor synchronization, cross-sensor consistency, and obstacle clearance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from naviguard_confidence.confidence_dimensions import ConfidenceScores


@dataclass
class VisualMonitorConfig:
    timeout_sec: float = 1.0
    min_inliers_continue: int = 35
    min_inliers_verify: int = 15
    min_inlier_ratio: float = 0.30
    max_disp_std_px: float = 40.0


class VisualMonitor:
    """Monitors VO feature tracking, RANSAC inliers, and geometric validity."""

    def __init__(self, config: Optional[VisualMonitorConfig] = None) -> None:
        self.cfg = config or VisualMonitorConfig()

    def evaluate(
        self,
        telemetry: Optional[Dict[str, str]],
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if telemetry is None or last_msg_time <= 0.0:
            return 0.2, ["VO_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.0, [f"VO_DATA_TIMEOUT_{dt:.1f}S"]

        score = 1.0

        # Inliers count
        num_inliers = int(telemetry.get("geom_num_inliers", telemetry.get("num_inliers", "0")))
        if num_inliers >= self.cfg.min_inliers_continue:
            pass
        elif num_inliers >= self.cfg.min_inliers_verify:
            ratio = (num_inliers - self.cfg.min_inliers_verify) / max(1, self.cfg.min_inliers_continue - self.cfg.min_inliers_verify)
            score = min(score, 0.35 + 0.40 * ratio)
            reasons.append(f"VO_DEGRADED_INLIERS_{num_inliers}")
        else:
            score = min(score, 0.15)
            reasons.append(f"VO_INSUFFICIENT_INLIERS_{num_inliers}")

        # Geometric validity
        geom_valid_str = telemetry.get("geom_is_valid", "false").lower()
        geom_status = telemetry.get("geom_status", "").upper()
        if geom_valid_str not in ("true", "1"):
            if geom_status in ("DEGENERATE_SMALL_BASELINE", "ROTATION_ONLY_MOTION", "ROTATIONAL_REACQUISITION", "STATIONARY") and num_inliers >= self.cfg.min_inliers_continue:
                score = min(score, 0.85)
                reasons.append(f"VO_ROTATION_STABLE_STRUCTURE_{geom_status}")
            elif geom_status in ("DEGENERATE_SMALL_BASELINE", "ROTATION_ONLY_MOTION", "ROTATIONAL_REACQUISITION", "STATIONARY") and num_inliers >= self.cfg.min_inliers_verify:
                score = min(score, 0.65)
                reasons.append(f"VO_ROTATION_LOW_INLIERS_{geom_status}")
            else:
                score = min(score, 0.35)
                reasons.append("VO_GEOMETRIC_ESTIMATE_INVALID")

        # Inlier ratio
        try:
            inlier_ratio = float(telemetry.get("geom_inlier_ratio", telemetry.get("tracking_ratio", "1.0")))
            if inlier_ratio < self.cfg.min_inlier_ratio:
                score *= 0.85
                reasons.append(f"VO_LOW_INLIER_RATIO_{inlier_ratio:.2f}")
        except ValueError:
            pass

        # Flow dispersion
        try:
            disp_std = float(telemetry.get("disp_std_px", "0.0"))
            if disp_std > self.cfg.max_disp_std_px:
                score *= 0.85
                reasons.append(f"VO_HIGH_DISPERSION_{disp_std:.1f}PX")
        except ValueError:
            pass

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class LocalizationMonitorConfig:
    timeout_sec: float = 1.5
    max_reproj_error_px: float = 4.0
    min_landmarks: int = 8


class LocalizationMonitor:
    """Monitors SLAM tracking state, keyframe density, and reprojection error."""

    def __init__(self, config: Optional[LocalizationMonitorConfig] = None) -> None:
        self.cfg = config or LocalizationMonitorConfig()

    def evaluate(
        self,
        slam_diag: Optional[Dict[str, str]],
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if slam_diag is None or last_msg_time <= 0.0:
            return 0.3, ["SLAM_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.1, [f"SLAM_DIAG_TIMEOUT_{dt:.1f}S"]

        score = 1.0
        tracking_state = slam_diag.get("tracking_state", "INITIALIZING").upper()

        if tracking_state == "OK":
            pass
        elif tracking_state == "DEGRADED":
            score = 0.50
            fail_reason = slam_diag.get("failure_reason", "DEGRADED")
            reasons.append(f"SLAM_STATE_DEGRADED_{fail_reason}")
        elif tracking_state == "INITIALIZING":
            score = 0.45
            reasons.append("SLAM_STATE_INITIALIZING")
        else:  # TRACKING_LOST or other
            score = 0.05
            fail_reason = slam_diag.get("failure_reason", "LOST")
            reasons.append(f"SLAM_TRACKING_LOST_{fail_reason}")

        # Reprojection error
        try:
            reproj_err = float(slam_diag.get("reprojection_error_px", "0.0"))
            if reproj_err > self.cfg.max_reproj_error_px:
                score *= max(0.5, 1.0 - 0.1 * (reproj_err - self.cfg.max_reproj_error_px))
                reasons.append(f"SLAM_HIGH_REPROJECTION_ERR_{reproj_err:.1f}PX")
        except ValueError:
            pass

        # Landmarks check
        try:
            num_lm = int(slam_diag.get("num_landmarks", "10"))
            if num_lm < self.cfg.min_landmarks and tracking_state == "OK":
                score *= 0.85
                reasons.append(f"SLAM_SPARSE_LANDMARKS_{num_lm}")
        except ValueError:
            pass

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class ImuMonitorConfig:
    timeout_sec: float = 0.5
    min_accel_norm: float = 4.0
    max_accel_norm: float = 22.0
    max_gyro_norm: float = 4.0


class ImuMonitor:
    """Monitors IMU freshness, acceleration bounds, angular velocity, and NaN/Inf."""

    def __init__(self, config: Optional[ImuMonitorConfig] = None) -> None:
        self.cfg = config or ImuMonitorConfig()

    def evaluate(
        self,
        accel: Tuple[float, float, float],
        gyro: Tuple[float, float, float],
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if last_msg_time <= 0.0:
            return 0.1, ["IMU_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.0, [f"IMU_TIMEOUT_{dt:.2f}S"]

        # NaN / Inf checks
        all_vals = list(accel) + list(gyro)
        if any(np.isnan(v) or np.isinf(v) for v in all_vals):
            return 0.0, ["IMU_NAN_OR_INF_DETECTED"]

        score = 1.0
        accel_norm = float(np.linalg.norm(accel))
        gyro_norm = float(np.linalg.norm(gyro))

        if accel_norm < self.cfg.min_accel_norm:
            score = 0.15
            reasons.append(f"IMU_LOW_GRAVITY_ANOMALY_{accel_norm:.1f}")
        elif accel_norm > self.cfg.max_accel_norm:
            score = min(score, 0.20)
            reasons.append(f"IMU_ACCEL_SHOCK_OR_IMPACT_{accel_norm:.1f}")

        if gyro_norm > self.cfg.max_gyro_norm:
            score = min(score, 0.25)
            reasons.append(f"IMU_GYRO_EXCEEDED_LIMIT_{gyro_norm:.1f}")

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class WheelMonitorConfig:
    timeout_sec: float = 0.5
    max_linear_speed_mps: float = 2.0
    max_lateral_speed_mps: float = 0.25
    max_accel_mps2: float = 5.0


class WheelMonitor:
    """Monitors wheel odometry rate, kinematic limits, slip, and encoder jumps."""

    def __init__(self, config: Optional[WheelMonitorConfig] = None) -> None:
        self.cfg = config or WheelMonitorConfig()
        self.prev_vx: Optional[float] = None
        self.prev_time: Optional[float] = None

    def evaluate(
        self,
        vx: float,
        vy: float,
        wz: float,
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if last_msg_time <= 0.0:
            return 0.1, ["WHEEL_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.0, [f"WHEEL_ODOM_TIMEOUT_{dt:.2f}S"]

        if np.isnan(vx) or np.isnan(vy) or np.isnan(wz) or np.isinf(vx) or np.isinf(vy) or np.isinf(wz):
            return 0.0, ["WHEEL_NAN_OR_INF_DETECTED"]

        score = 1.0

        # Linear speed limits
        if abs(vx) > self.cfg.max_linear_speed_mps:
            score = min(score, 0.20)
            reasons.append(f"WHEEL_OVERSPEED_{abs(vx):.2f}MPS")

        # Lateral slip (non-holonomic constraint for diff drive)
        if abs(vy) > self.cfg.max_lateral_speed_mps:
            score = min(score, 0.40)
            reasons.append(f"WHEEL_LATERAL_SLIP_{abs(vy):.2f}MPS")

        # Acceleration jump check
        if self.prev_vx is not None and self.prev_time is not None:
            time_delta = current_time - self.prev_time
            if 0.001 < time_delta < 0.5:
                accel = abs(vx - self.prev_vx) / time_delta
                if accel > self.cfg.max_accel_mps2:
                    score = min(score, 0.45)
                    reasons.append(f"WHEEL_ACCEL_JUMP_{accel:.1f}MPS2")

        self.prev_vx = vx
        self.prev_time = current_time

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class TemporalMonitorConfig:
    timeout_sec: float = 2.0
    max_median_diff_ms: float = 60.0


class TemporalMonitor:
    """Monitors synchronization quality, period jitter, and cross-sensor latency."""

    def __init__(self, config: Optional[TemporalMonitorConfig] = None) -> None:
        self.cfg = config or TemporalMonitorConfig()

    def evaluate(
        self,
        sync_diag: Optional[Dict[str, str]],
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if sync_diag is None or last_msg_time <= 0.0:
            return 0.5, ["SYNC_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.35, [f"SYNC_DIAG_TIMEOUT_{dt:.1f}S"]

        score = 1.0
        # Check overall status
        status = sync_diag.get("overall_status", "PASS").upper()
        if status == "PASS":
            pass
        elif status == "WARNING":
            score = 0.65
            reasons.append("SENSOR_SYNC_JITTER_WARNING")
        else:  # FAIL
            score = 0.20
            reasons.append("SENSOR_SYNC_CRITICAL_FAIL")

        # Check cam-imu median latency
        try:
            cam_imu_ms = float(sync_diag.get("cam_imu_median_ms", "10.0"))
            if cam_imu_ms > self.cfg.max_median_diff_ms:
                score *= 0.75
                reasons.append(f"CAM_IMU_LATENCY_{cam_imu_ms:.1f}MS")
        except ValueError:
            pass

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class CrossSensorMonitorConfig:
    timeout_sec: float = 1.5
    max_yaw_rate_residual_radps: float = 0.40
    max_rejections_allowed: int = 8


class CrossSensorMonitor:
    """Monitors cross-sensor consistency residuals and EKF gating rejections."""

    def __init__(self, config: Optional[CrossSensorMonitorConfig] = None) -> None:
        self.cfg = config or CrossSensorMonitorConfig()

    def evaluate(
        self,
        state_diag: Optional[Dict[str, str]],
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if state_diag is None or last_msg_time <= 0.0:
            return 0.5, ["STATE_EST_NO_DATA"]

        dt = current_time - last_msg_time
        if dt > self.cfg.timeout_sec:
            return 0.30, [f"STATE_EST_TIMEOUT_{dt:.1f}S"]

        score = 1.0

        # Consistency flags
        vis_wheel_ok = state_diag.get("is_vis_wheel_consistent", "true").lower() in ("true", "1")
        vis_imu_ok = state_diag.get("is_vis_imu_consistent", "true").lower() in ("true", "1")
        wheel_imu_ok = state_diag.get("is_wheel_imu_consistent", "true").lower() in ("true", "1")

        if not wheel_imu_ok:
            score = min(score, 0.40)
            reasons.append("WHEEL_IMU_DISAGREEMENT")
        if not vis_wheel_ok:
            score = min(score, 0.55)
            reasons.append("VISUAL_WHEEL_DISAGREEMENT")
        if not vis_imu_ok:
            score = min(score, 0.55)
            reasons.append("VISUAL_IMU_DISAGREEMENT")

        # Yaw rate residual
        try:
            res_wz = abs(float(state_diag.get("res_wz_wheel_imu_radps", "0.0")))
            if res_wz > self.cfg.max_yaw_rate_residual_radps:
                score = min(score, 0.35)
                reasons.append(f"HIGH_WHEEL_IMU_RESIDUAL_{res_wz:.2f}RADPS")
        except ValueError:
            pass

        # Rejections count
        try:
            rej_w = int(state_diag.get("rejections_wheel", "0"))
            rej_i = int(state_diag.get("rejections_imu", "0"))
            rej_v = int(state_diag.get("rejections_visual", "0"))
            total_rej = rej_w + rej_i + rej_v
            if total_rej > self.cfg.max_rejections_allowed:
                score *= 0.70
                reasons.append(f"HIGH_EKF_REJECTIONS_{total_rej}")
        except ValueError:
            pass

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class MapMonitorConfig:
    critical_obstacle_dist_m: float = 0.35
    caution_obstacle_dist_m: float = 0.75
    obstacle_occupancy_threshold: int = 65


class MapMonitor:
    """Monitors 2D occupancy grid map for obstacle clearance around the robot."""

    def __init__(self, config: Optional[MapMonitorConfig] = None) -> None:
        self.cfg = config or MapMonitorConfig()

    def evaluate(
        self,
        grid_data: Optional[List[int]],
        grid_resolution: float,
        grid_width: int,
        grid_height: int,
        grid_origin_x: float,
        grid_origin_y: float,
        robot_x: float,
        robot_y: float,
        last_msg_time: float,
        current_time: float,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if grid_data is None or len(grid_data) == 0 or last_msg_time <= 0.0:
            return 0.85, []  # Nominal outdoor default if map not populated yet

        # Check cells within search radius
        search_radius_m = self.cfg.caution_obstacle_dist_m
        cell_radius = int(np.ceil(search_radius_m / max(0.01, grid_resolution)))

        # Robot cell indices
        rx_idx = int((robot_x - grid_origin_x) / grid_resolution)
        ry_idx = int((robot_y - grid_origin_y) / grid_resolution)

        min_obs_dist = 999.0

        for dy in range(-cell_radius, cell_radius + 1):
            cy = ry_idx + dy
            if cy < 0 or cy >= grid_height:
                continue
            for dx in range(-cell_radius, cell_radius + 1):
                cx = rx_idx + dx
                if cx < 0 or cx >= grid_width:
                    continue
                dist = np.hypot(dx * grid_resolution, dy * grid_resolution)
                if dist > search_radius_m:
                    continue

                idx = cy * grid_width + cx
                occ_val = grid_data[idx]
                if occ_val >= self.cfg.obstacle_occupancy_threshold:
                    if dist < min_obs_dist:
                        min_obs_dist = dist

        score = 1.0
        if min_obs_dist < self.cfg.critical_obstacle_dist_m:
            score = 0.10
            reasons.append(f"OBSTACLE_COLLISION_CRITICAL_{min_obs_dist:.2f}M")
        elif min_obs_dist < self.cfg.caution_obstacle_dist_m:
            score = 0.50
            reasons.append(f"OBSTACLE_PROXIMITY_CAUTION_{min_obs_dist:.2f}M")

        return float(np.clip(score, 0.0, 1.0)), reasons


@dataclass
class ConfidenceEngineWeights:
    visual: float = 0.20
    localization: float = 0.25
    imu: float = 0.15
    wheel: float = 0.15
    temporal: float = 0.10
    cross_sensor: float = 0.10
    map: float = 0.05


class CompositeConfidenceEngine:
    """Combines 7 sensor/pipeline dimensions into a bounded, safety-gated confidence score."""

    def __init__(self, weights: Optional[ConfidenceEngineWeights] = None) -> None:
        self.weights = weights or ConfidenceEngineWeights()
        self.visual_mon = VisualMonitor()
        self.loc_mon = LocalizationMonitor()
        self.imu_mon = ImuMonitor()
        self.wheel_mon = WheelMonitor()
        self.temporal_mon = TemporalMonitor()
        self.cross_mon = CrossSensorMonitor()
        self.map_mon = MapMonitor()

    def compute(
        self,
        current_time: float,
        vo_telemetry: Optional[Dict[str, str]],
        vo_last_time: float,
        slam_diag: Optional[Dict[str, str]],
        slam_last_time: float,
        imu_accel: Tuple[float, float, float],
        imu_gyro: Tuple[float, float, float],
        imu_last_time: float,
        wheel_twist: Tuple[float, float, float],  # (vx, vy, wz)
        wheel_last_time: float,
        sync_diag: Optional[Dict[str, str]],
        sync_last_time: float,
        state_diag: Optional[Dict[str, str]],
        state_last_time: float,
        map_data: Optional[List[int]],
        map_res: float,
        map_w: int,
        map_h: int,
        map_ox: float,
        map_oy: float,
        robot_pose: Tuple[float, float],  # (x, y)
        map_last_time: float,
    ) -> Tuple[ConfidenceScores, str, List[str]]:
        """Evaluate all dimensions and compute composite confidence with safety gating."""
        all_reasons: List[str] = []
        scores = ConfidenceScores()

        s_vis, r_vis = self.visual_mon.evaluate(vo_telemetry, vo_last_time, current_time)
        scores.visual = s_vis
        all_reasons.extend(r_vis)

        s_loc, r_loc = self.loc_mon.evaluate(slam_diag, slam_last_time, current_time)
        scores.localization = s_loc
        all_reasons.extend(r_loc)

        s_imu, r_imu = self.imu_mon.evaluate(imu_accel, imu_gyro, imu_last_time, current_time)
        scores.imu = s_imu
        all_reasons.extend(r_imu)

        s_wheel, r_wheel = self.wheel_mon.evaluate(
            wheel_twist[0], wheel_twist[1], wheel_twist[2], wheel_last_time, current_time
        )
        scores.wheel = s_wheel
        all_reasons.extend(r_wheel)

        s_temp, r_temp = self.temporal_mon.evaluate(sync_diag, sync_last_time, current_time)
        scores.temporal = s_temp
        all_reasons.extend(r_temp)

        s_cross, r_cross = self.cross_mon.evaluate(state_diag, state_last_time, current_time)
        scores.cross_sensor = s_cross
        all_reasons.extend(r_cross)

        s_map, r_map = self.map_mon.evaluate(
            map_data, map_res, map_w, map_h, map_ox, map_oy, robot_pose[0], robot_pose[1], map_last_time, current_time
        )
        scores.map = s_map
        all_reasons.extend(r_map)

        # Weighted sum
        w_sum = (
            self.weights.visual * scores.visual
            + self.weights.localization * scores.localization
            + self.weights.imu * scores.imu
            + self.weights.wheel * scores.wheel
            + self.weights.temporal * scores.temporal
            + self.weights.cross_sensor * scores.cross_sensor
            + self.weights.map * scores.map
        )
        w_total = (
            self.weights.visual + self.weights.localization + self.weights.imu
            + self.weights.wheel + self.weights.temporal + self.weights.cross_sensor
            + self.weights.map
        )
        composite = w_sum / max(1e-6, w_total)

        # Safety Gating:
        # 1. Critical sensor safety floor: if IMU, wheel, localization, or cross-sensor has collapsed
        critical_floor = min(scores.imu, scores.wheel, scores.localization, scores.cross_sensor)
        if critical_floor < 0.25:
            composite = min(composite, max(critical_floor * 1.5, 0.20))

        # 2. Obstacle collision critical
        if scores.map < 0.20:
            composite = min(composite, 0.15)

        # 3. Complete visual loss and localization loss
        if scores.visual < 0.20 and scores.localization < 0.30:
            composite = min(composite, 0.25)

        scores.overall = composite
        scores.clamp()

        # Primary and secondary reasons determination
        if not all_reasons:
            primary_reason = "NOMINAL_OPERATION"
            secondary_reasons = []
        else:
            # Rank reasons by lowest dimension score
            dim_rank = [
                (scores.map, r_map),
                (scores.imu, r_imu),
                (scores.wheel, r_wheel),
                (scores.cross_sensor, r_cross),
                (scores.localization, r_loc),
                (scores.visual, r_vis),
                (scores.temporal, r_temp),
            ]
            dim_rank.sort(key=lambda x: x[0])

            ordered_reasons: List[str] = []
            for _, rlist in dim_rank:
                for r in rlist:
                    if r not in ordered_reasons:
                        ordered_reasons.append(r)

            primary_reason = ordered_reasons[0] if ordered_reasons else "DEGRADED_CONFIDENCE"
            secondary_reasons = ordered_reasons[1:]

        return scores, primary_reason, secondary_reasons
