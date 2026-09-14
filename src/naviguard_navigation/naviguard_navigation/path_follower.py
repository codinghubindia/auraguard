"""Pure pursuit path following controller with environmental adaptive speed for NAVIGUARD."""

import math
from typing import List, Tuple, Optional, Dict, Any
from naviguard_navigation.waypoint_generator import Waypoint
from naviguard_navigation.vehicle_geometry import VehicleGeometry


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PathFollower:
    """Computes bounded linear and angular velocities with multi-factor adaptive speed scaling."""

    def __init__(
        self,
        lookahead_distance_m: float = 0.40,
        max_linear_velocity: float = 0.25,
        min_linear_velocity: float = 0.05,
        max_angular_velocity: float = 0.35,
        kp_heading: float = 1.2,
        kp_cross_track: float = 0.5,
        turn_in_place_angle_threshold: float = 1.30,  # ~75 deg entry threshold
        turn_in_place_exit_threshold: float = 0.65,   # ~37 deg exit threshold (hysteresis)
    ) -> None:
        self.lookahead_distance_m = lookahead_distance_m
        self.max_linear_velocity = max_linear_velocity
        self.min_linear_velocity = min_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.kp_heading = kp_heading
        self.kp_cross_track = kp_cross_track
        self.turn_in_place_angle_threshold = turn_in_place_angle_threshold
        self.turn_in_place_exit_threshold = turn_in_place_exit_threshold
        self.is_turning_in_place: bool = False

        self.current_waypoint_idx: int = 0

    def reset(self) -> None:
        """Reset path follower state."""
        self.current_waypoint_idx = 0
        self.is_turning_in_place = False

    def compute_commands(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        waypoints: List[Waypoint],
        speed_scale: float = 1.0,
        terrain_factor: float = 1.0,
        clearance_factor: float = 1.0,
    ) -> Tuple[float, float, Optional[Waypoint], float]:
        """Compute (vx, wz, lookahead_waypoint, cross_track_error) with adaptive environmental speed scaling."""
        if not waypoints:
            return 0.0, 0.0, None, 0.0

        # 1. Update current closest waypoint (search forward)
        best_idx = self.current_waypoint_idx
        min_dist_sq = float('inf')
        search_end = min(len(waypoints), self.current_waypoint_idx + 8)

        for i in range(self.current_waypoint_idx, search_end):
            wp = waypoints[i]
            d2 = (wp.x - robot_x) ** 2 + (wp.y - robot_y) ** 2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_idx = i

        self.current_waypoint_idx = best_idx

        # 2. Find lookahead point
        lookahead_wp = waypoints[-1]
        for i in range(self.current_waypoint_idx, len(waypoints)):
            wp = waypoints[i]
            d = math.hypot(wp.x - robot_x, wp.y - robot_y)
            if d >= self.lookahead_distance_m:
                lookahead_wp = wp
                break

        # 3. Calculate heading error to lookahead point
        target_heading = math.atan2(lookahead_wp.y - robot_y, lookahead_wp.x - robot_x)
        heading_error = normalize_angle(target_heading - robot_yaw)

        # 4. Calculate signed cross-track error relative to current path segment
        curr_wp = waypoints[self.current_waypoint_idx]
        dx = robot_x - curr_wp.x
        dy = robot_y - curr_wp.y
        path_yaw = curr_wp.yaw
        cross_track_error = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy

        # 5. Angular control
        wz = self.kp_heading * heading_error - self.kp_cross_track * cross_track_error
        wz = max(-self.max_angular_velocity, min(self.max_angular_velocity, wz))

        # 6. Adaptive Linear Speed Policy:
        # speed = base_speed * confidence_factor * terrain_factor * clearance_factor
        adaptive_scale = (
            max(0.2, min(1.0, speed_scale)) *
            max(0.35, min(1.0, terrain_factor)) *
            max(0.40, min(1.0, clearance_factor))
        )
        v_limit = self.max_linear_velocity * adaptive_scale

        dist_to_goal = math.hypot(waypoints[-1].x - robot_x, waypoints[-1].y - robot_y)

        # Heading hysteresis to prevent knife-edge chatter between stopping and driving forward
        if not self.is_turning_in_place:
            if abs(heading_error) > self.turn_in_place_angle_threshold:
                self.is_turning_in_place = True
        else:
            if abs(heading_error) < self.turn_in_place_exit_threshold:
                self.is_turning_in_place = False

        if self.is_turning_in_place:
            # Pure in-place turn to align with path
            vx = 0.0
        else:
            # Human-like continuous arc driving:
            # - For small heading error (< 0.60 rad / ~34 deg), cruise normally scaled by cos(heading_error)
            # - For moderate heading error (0.60 to 1.30 rad / ~34-75 deg), smoothly roll along arc at crawl speed
            #   (0.06 - 0.08 m/s) maintaining continuous momentum
            cos_factor = max(0.0, math.cos(heading_error))
            if abs(heading_error) > 0.60:
                arc_crawl = max(self.min_linear_velocity, min(0.08, v_limit * 0.40))
                vx = arc_crawl
            else:
                vx = v_limit * cos_factor

            # Smooth deceleration on final approach (within 1.0 m)
            if dist_to_goal < 1.0:
                slowdown = max(0.3, dist_to_goal / 1.0)
                vx *= slowdown

            # Enforce non-zero crawling speed above minimum velocity
            vx = max(self.min_linear_velocity, min(v_limit, vx))

        return float(vx), float(wz), lookahead_wp, float(abs(cross_track_error))

    def compute_commands_with_trajectory_adjustment(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        waypoints: List[Waypoint],
        occ_grid: Any = None,
        speed_scale: float = 1.0,
        terrain_factor: float = 1.0,
        clearance_factor: float = 1.0,
    ) -> Tuple[float, float, Optional[Waypoint], float, Dict[str, Any]]:
        """Compute commands with forward trajectory rollout, direction adjustment, and small-gap centering.
        
        Evaluates candidate steering trajectories against the occupancy grid:
        - If an obstacle encroaches on the nominal path, actively adjusts steering direction (wz)
          away from the hazard.
        - When passing through small gaps (0.58m <= width < 0.68m), centers the trajectory between
          obstacles and scales linear velocity down to precision crawl speed.
        - Identifies whether the vehicle can pass (can_pass=True/False) for operator visualization.
        """
        # 1. Base pure-pursuit calculation
        nom_vx, nom_wz, lookahead_wp, cross_err = self.compute_commands(
            robot_x, robot_y, robot_yaw, waypoints,
            speed_scale=speed_scale,
            terrain_factor=terrain_factor,
            clearance_factor=clearance_factor,
        )

        traj_info = {
            "can_pass": True,
            "status": "SAFE",
            "min_clearance_m": 1.0,
            "steering_adjustment_rad": 0.0,
            "in_small_gap": False,
            "corridor_width_m": VehicleGeometry.NOMINAL_PASSAGE_WIDTH_M,
        }

        # If vehicle is turning in place (vx == 0 or is_turning_in_place), keep pure in-place turn
        if nom_vx <= 0.01 or self.is_turning_in_place:
            traj_info["can_pass"] = True
            traj_info["status"] = "TURNING_IN_PLACE"
            traj_info["reason"] = "ALIGNING_HEADING"
            return nom_vx, nom_wz, lookahead_wp, cross_err, traj_info

        if occ_grid is None or not getattr(occ_grid, "is_initialized", False):
            return nom_vx, nom_wz, lookahead_wp, cross_err, traj_info

        # 2. Check current passage clearance at robot pose
        available_width = 1.0
        in_small_gap = False
        if hasattr(occ_grid, "evaluate_passage_at"):
            _, passage_status, diag = occ_grid.evaluate_passage_at(robot_x, robot_y, robot_yaw)
            available_width = float(diag.get("available_width_m", 1.0))
            in_small_gap = (VehicleGeometry.TIGHT_PASSAGE_LIMIT_M <= available_width < VehicleGeometry.NOMINAL_PASSAGE_WIDTH_M) or (passage_status == "TIGHT")

        traj_info["corridor_width_m"] = round(available_width, 3)
        traj_info["in_small_gap"] = in_small_gap

        # 3. Human-like Trajectory Rollout & Direction Adjustment
        # Separate immediate collision horizon from steering preview horizon:
        # - Immediate horizon (0.45m): Only obstacles right in front of the bumper require halting.
        # - Preview horizon (1.60m): Distant obstacles are used to preview and smoothly adjust steering curvature.
        candidate_offsets = [0.0, -0.06, +0.06, -0.12, +0.12, -0.18, +0.18, -0.24, +0.24, -0.30, +0.30]
        immediate_horizon_m = 0.45
        preview_horizon_m = 1.60
        rollout_step_m = 0.10
        num_rollout_steps = max(4, int(preview_horizon_m / rollout_step_m))

        best_wz = nom_wz
        best_score = -1e9
        best_clearance = 0.0
        best_free_dist = 0.0
        has_viable_candidate = False
        nominal_clearance = 0.50
        nominal_free_dist = 0.0

        for idx, offset in enumerate(candidate_offsets):
            cand_w = nom_wz + offset
            cand_w = max(-self.max_angular_velocity, min(self.max_angular_velocity, cand_w))

            # Simulate arc forward step-by-step
            arc_clearances = []
            d_free = 0.0
            cand_blocked_immediate = False

            eff_v = max(0.08, nom_vx)
            for st in range(1, num_rollout_steps + 1):
                s = st * rollout_step_m
                d_theta = cand_w * (s / eff_v)
                sx = robot_x + s * math.cos(robot_yaw + d_theta * 0.5)
                sy = robot_y + s * math.sin(robot_yaw + d_theta * 0.5)
                syaw = robot_yaw + d_theta

                # Check footprint collision against grid with physical 4cm safety cushion
                if not VehicleGeometry.is_footprint_collision_free(sx, sy, syaw, occ_grid, margin_m=0.04):
                    # Collision detected at distance s
                    if s <= immediate_horizon_m:
                        cand_blocked_immediate = True
                    break

                d_free = s

                # Query cell clearance
                mpt = occ_grid.world_to_map(sx, sy)
                if mpt and hasattr(occ_grid, "get_clearance"):
                    c_val = occ_grid.get_clearance(mpt[0], mpt[1])
                    arc_clearances.append(c_val)

            if cand_blocked_immediate:
                # Discard candidates that immediately hit an obstacle within critical horizon
                continue

            has_viable_candidate = True
            min_arc_clr = min(arc_clearances) if arc_clearances else 0.50
            if idx == 0:
                nominal_clearance = min_arc_clr
                nominal_free_dist = d_free

            # Human-like driving scoring:
            # 1. Heavily reward longer collision-free distance (look for the open corridor!)
            # 2. Reward lateral clearance margin
            # 3. Penalize excessive deviation from path follower unless avoiding obstacle
            # 4. Reward centering in small gaps
            free_dist_score = (d_free / preview_horizon_m) * 4.0
            clearance_score = min_arc_clr * 2.5
            path_deviation_penalty = abs(offset) * 1.5
            centering_bonus = min_arc_clr * 3.5 if in_small_gap else 0.0

            score = free_dist_score + clearance_score - path_deviation_penalty + centering_bonus

            if score > best_score:
                best_score = score
                best_wz = cand_w
                best_clearance = min_arc_clr
                best_free_dist = d_free

        # 4. Apply Direction Adjustment & Human-like Speed Modulation
        final_vx = nom_vx
        final_wz = nom_wz
        traj_info["min_clearance_m"] = round(best_clearance if has_viable_candidate else nominal_clearance, 2)
        traj_info["free_dist_m"] = round(best_free_dist if has_viable_candidate else nominal_free_dist, 2)

        if not has_viable_candidate or best_free_dist < 0.32:
            # HALT immediately when forward path is blocked in immediate critical zone (<0.32m)
            final_vx = 0.0
            traj_info["can_pass"] = False
            traj_info["status"] = "BLOCKED"
            traj_info["reason"] = "IMMEDIATE_COLLISION_HAZARD"
        else:
            traj_info["can_pass"] = True
            steering_delta = best_wz - nom_wz
            traj_info["steering_adjustment_rad"] = round(steering_delta, 3)

            # Smoothly adopt best adjusted steering whenever it improves free distance or clearance
            if abs(steering_delta) > 0.03:
                final_wz = best_wz

            # Active Centering in Small Gaps
            if in_small_gap and occ_grid is not None and hasattr(occ_grid, "world_to_map") and hasattr(occ_grid, "get_clearance"):
                # Query lateral clearances at robot flanks
                lx = robot_x - 0.26 * math.sin(robot_yaw)
                ly = robot_y + 0.26 * math.cos(robot_yaw)
                rx_p = robot_x + 0.26 * math.sin(robot_yaw)
                ry_p = robot_y - 0.26 * math.cos(robot_yaw)
                lp = occ_grid.world_to_map(lx, ly)
                rp = occ_grid.world_to_map(rx_p, ry_p)
                if lp and rp:
                    lc = occ_grid.get_clearance(lp[0], lp[1])
                    rc = occ_grid.get_clearance(rp[0], rp[1])
                    # If closer to left (lc < rc), steer right (negative wz nudge)
                    # If closer to right (rc < lc), steer left (positive wz nudge)
                    centering_nudge = max(-0.12, min(0.12, (lc - rc) * 0.9))
                    final_wz += centering_nudge

            # Speed modulation based on gap size and forward distance:
            if in_small_gap:
                traj_info["status"] = "TIGHT"
                # Precision crawl speed through tight passage
                final_vx = max(self.min_linear_velocity, min(0.10, nom_vx * 0.50))
            elif best_free_dist < 0.90:
                # Approaching an obstacle ahead: creep cautiously while steering away smoothly
                traj_info["status"] = "TIGHT"
                speed_factor = max(0.20, (best_free_dist - 0.35) / 0.55)
                final_vx = max(self.min_linear_velocity, min(nom_vx, nom_vx * speed_factor))
            elif best_clearance < VehicleGeometry.SAFETY_MARGIN_M:
                traj_info["status"] = "TIGHT"
                final_vx = max(self.min_linear_velocity, min(0.15, nom_vx * 0.70))
            else:
                traj_info["status"] = "SAFE"
                final_vx = nom_vx

        return float(final_vx), float(final_wz), lookahead_wp, cross_err, traj_info

