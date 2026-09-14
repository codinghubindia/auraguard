"""NAVIGUARD LiDAR and Radar Obstacle Detection and 2D/3D Metric Spatial Localization.

Processes real-time LaserScan returns:
- Filters chassis self-reflections and sensor noise
- Extracts Cartesian 2D body coordinates (x_forward, y_left)
- Performs spatial jump-distance Euclidean clustering
- Calculates metric object positions, bounding radii, distances, and corridor clearances
- Computes passage width and lateral centering offsets for collision-free tight-gap navigation
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sensor_msgs.msg import LaserScan


class LidarObstacleDetector:
    """Detects and localizes spatial obstacles from 2D LiDAR / Radar scan returns."""

    # Vehicle chassis envelope for self-hit filtering (meters)
    CHASSIS_X_MIN: float = -0.29
    CHASSIS_X_MAX: float = +0.29
    CHASSIS_Y_MIN: float = -0.25
    CHASSIS_Y_MAX: float = +0.25

    def __init__(
        self,
        cluster_threshold_m: float = 0.28,
        min_cluster_points: int = 2,
        max_sensing_range_m: float = 16.0,
        min_sensing_range_m: float = 0.15,
        forward_corridor_width_m: float = 0.58,  # Vehicle width (0.48m) + 0.10m margin
        critical_stop_distance_m: float = 0.38,
    ) -> None:
        self.cluster_threshold_m = cluster_threshold_m
        self.min_cluster_points = min_cluster_points
        self.max_sensing_range_m = max_sensing_range_m
        self.min_sensing_range_m = min_sensing_range_m
        self.forward_corridor_width_m = forward_corridor_width_m
        self.critical_stop_distance_m = critical_stop_distance_m

    def is_inside_chassis(self, x: float, y: float) -> bool:
        """Check if Cartesian point falls within the physical chassis footprint."""
        return (
            self.CHASSIS_X_MIN <= x <= self.CHASSIS_X_MAX and
            self.CHASSIS_Y_MIN <= y <= self.CHASSIS_Y_MAX
        )

    def process_scan(self, scan_msg: LaserScan) -> Dict[str, Any]:
        """Process a LaserScan message into metric Cartesian obstacle points and clusters.
        
        Returns a dictionary with:
        - 'valid_points': List of (x, y) coordinates relative to base_link
        - 'obstacles': List of clustered obstacle objects with metric (x, y, radius, distance)
        - 'closest_distance_m': Distance to nearest obstacle
        - 'critical_hazard': True if any obstacle is within critical stop envelope
        - 'corridor_clearance': Lateral clearances and centering recommendations
        """
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        n_points = len(ranges)
        if n_points == 0:
            return {
                "valid_points": [],
                "obstacles": [],
                "closest_distance_m": float('inf'),
                "critical_hazard": False,
                "corridor_clearance": {"left_clearance_m": 2.0, "right_clearance_m": 2.0, "centering_offset_m": 0.0},
            }

        angle_min = float(scan_msg.angle_min)
        angle_inc = float(scan_msg.angle_increment)
        r_min = max(self.min_sensing_range_m, float(scan_msg.range_min))
        r_max = min(self.max_sensing_range_m, float(scan_msg.range_max))

        # 1. Convert valid ranges to Cartesian body frame (x_forward, y_left)
        angles = angle_min + np.arange(n_points) * angle_inc
        valid_mask = (ranges >= r_min) & (ranges <= r_max) & np.isfinite(ranges)

        valid_r = ranges[valid_mask]
        valid_theta = angles[valid_mask]

        xs = valid_r * np.cos(valid_theta)
        ys = valid_r * np.sin(valid_theta)

        # Filter self chassis points
        cartesian_points: List[Tuple[float, float]] = []
        for x, y in zip(xs, ys):
            if not self.is_inside_chassis(x, y):
                cartesian_points.append((float(x), float(y)))

        if not cartesian_points:
            return {
                "valid_points": [],
                "obstacles": [],
                "closest_distance_m": float('inf'),
                "critical_hazard": False,
                "corridor_clearance": {"left_clearance_m": 2.0, "right_clearance_m": 2.0, "centering_offset_m": 0.0},
            }

        # 2. Euclidean Jump-Distance Spatial Clustering
        clusters: List[List[Tuple[float, float]]] = []
        current_cluster: List[Tuple[float, float]] = [cartesian_points[0]]

        for i in range(1, len(cartesian_points)):
            px, py = cartesian_points[i]
            prev_x, prev_y = cartesian_points[i - 1]
            dist = math.hypot(px - prev_x, py - prev_y)
            if dist <= self.cluster_threshold_m:
                current_cluster.append((px, py))
            else:
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [(px, py)]

        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)

        # Check wrap-around between first and last cluster if 360-degree scan
        if len(clusters) > 1 and abs(angle_min - (-math.pi)) < 0.1 and abs(angle_min + n_points * angle_inc - math.pi) < 0.1:
            first_pt = clusters[0][0]
            last_pt = clusters[-1][-1]
            if math.hypot(first_pt[0] - last_pt[0], first_pt[1] - last_pt[1]) <= self.cluster_threshold_m:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # 3. Analyze Clusters into Metric Obstacle Objects
        obstacles: List[Dict[str, Any]] = []
        closest_dist = float('inf')
        critical_hazard = False

        half_corridor = self.forward_corridor_width_m / 2.0

        for idx, cl in enumerate(clusters):
            cl_arr = np.array(cl, dtype=np.float32)
            xc = float(np.mean(cl_arr[:, 0]))
            yc = float(np.mean(cl_arr[:, 1]))

            # Metric distance to closest point in cluster
            dists = np.hypot(cl_arr[:, 0], cl_arr[:, 1])
            d_min = float(np.min(dists))
            if d_min < closest_dist:
                closest_dist = d_min

            # Estimated radius of bounding disc
            spreads = np.hypot(cl_arr[:, 0] - xc, cl_arr[:, 1] - yc)
            radius = float(max(0.12, min(0.65, np.max(spreads) + 0.05)))

            bearing_deg = math.degrees(math.atan2(yc, xc))

            # Directional zone
            if xc >= 0.28:
                if abs(yc) <= half_corridor:
                    zone = "FRONT_CENTER"
                elif yc > half_corridor:
                    zone = "FRONT_LEFT"
                else:
                    zone = "FRONT_RIGHT"
            elif xc >= -0.28:
                zone = "SIDE_LEFT" if yc > 0 else "SIDE_RIGHT"
            else:
                zone = "REAR"

            # Emergency condition: Obstacle directly in front path within critical stop envelope
            is_emergency = (
                xc > 0.26 and
                xc <= (0.28 + self.critical_stop_distance_m) and
                abs(yc) <= half_corridor
            )
            if is_emergency:
                critical_hazard = True

            obstacles.append({
                "id": idx + 1,
                "x_base": round(xc, 3),
                "y_base": round(yc, 3),
                "x_m": round(xc, 3),
                "y_m": round(yc, 3),
                "radius": round(radius, 3),
                "radius_m": round(radius, 3),
                "distance_m": round(d_min, 3),
                "bearing_deg": round(bearing_deg, 1),
                "zone": zone,
                "confirmed": True,
                "traversable": False,
                "emergency": is_emergency,
                "source": "LIDAR",
                "point_count": len(cl),
            })

        # 4. Corridor Clearance & Centering Calculation
        # Look at points in forward range [0.3m, 2.5m]
        left_clearances = [
            y for x, y in cartesian_points
            if 0.30 <= x <= 2.50 and y > 0.0
        ]
        right_clearances = [
            abs(y) for x, y in cartesian_points
            if 0.30 <= x <= 2.50 and y < 0.0
        ]

        min_left = min(left_clearances) if left_clearances else 2.0
        min_right = min(right_clearances) if right_clearances else 2.0
        # Positive centering_offset means robot is closer to right obstacle (needs to shift left)
        # Negative means robot is closer to left obstacle (needs to shift right)
        centering_offset = round((min_left - min_right) / 2.0, 3)

        corridor_info = {
            "left_clearance_m": round(min_left, 3),
            "right_clearance_m": round(min_right, 3),
            "available_gap_m": round(min_left + min_right, 3),
            "centering_offset_m": centering_offset,
        }

        return {
            "valid_points": cartesian_points,
            "obstacles": obstacles,
            "closest_distance_m": round(closest_dist, 3) if math.isfinite(closest_dist) else 99.0,
            "critical_hazard": critical_hazard,
            "corridor_clearance": corridor_info,
        }
