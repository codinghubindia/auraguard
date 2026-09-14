"""2D Occupancy Grid Mapper for NAVIGUARD UGV SLAM.

Generates a reusable 2D occupancy grid in the map frame from robot trajectory clearance
and projected 3D visual obstacle landmarks. Saves maps to standard ROS PGM/YAML formats.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose


class OccupancyGridMapper:
    """Maintains a discrete 2D grid map in the map coordinate frame."""

    def __init__(
        self,
        resolution: float = 0.05,
        width_m: float = 30.0,
        height_m: float = 30.0,
        origin_x: float = -15.0,
        origin_y: float = -15.0,
        clearance_radius_m: float = 0.35,
        min_obstacle_height: float = 0.08,
        max_obstacle_height: float = 1.8,
    ) -> None:
        self.resolution = resolution
        self.width_cells = int(np.round(width_m / resolution))
        self.height_cells = int(np.round(height_m / resolution))
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.clearance_radius_m = clearance_radius_m
        # Z is interpreted relative to the configured ground/base frame (ground plane at Z=0.0)
        # Any feature with Z < min_obstacle_height (such as ground texture or shadows) is rejected.
        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height

        # Values: -1 = unknown, 0 = free, 100 = occupied
        self.grid = np.full((self.height_cells, self.width_cells), -1, dtype=np.int8)

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert continuous map coordinates (meters) to grid cell indices (col, row)."""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        if 0 <= gx < self.width_cells and 0 <= gy < self.height_cells:
            return gx, gy
        return None

    def map_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell indices to continuous map coordinates."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def update_robot_clearance(self, x: float, y: float) -> None:
        """Mark free space in a disk of clearance_radius_m around robot pose."""
        pt = self.world_to_map(x, y)
        if pt is None:
            return

        cx, cy = pt
        rad_cells = int(np.ceil(self.clearance_radius_m / self.resolution))
        r2 = (self.clearance_radius_m / self.resolution) ** 2

        for dy in range(-rad_cells, rad_cells + 1):
            for dx in range(-rad_cells, rad_cells + 1):
                if dx * dx + dy * dy <= r2:
                    gx = cx + dx
                    gy = cy + dy
                    if 0 <= gx < self.width_cells and 0 <= gy < self.height_cells:
                        # Mark free only if not currently confirmed as high-confidence obstacle
                        if self.grid[gy, gx] != 100:
                            self.grid[gy, gx] = 0

    def add_obstacle_point(self, x: float, y: float, z: float) -> bool:
        """Mark occupied cell for a visual landmark obstacle if within ground obstacle height.

        Rejects ground-plane points, ground textures, shadows (Z < min_obstacle_height),
        and high overhead features (Z > max_obstacle_height). Returns True if marked.
        """
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return False

        if z < self.min_obstacle_height or z > self.max_obstacle_height:
            return False

        pt = self.world_to_map(x, y)
        if pt is not None:
            gx, gy = pt
            self.grid[gy, gx] = 100
            return True
        return False

    def clear_ray(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        max_range_m: float = 15.0,
    ) -> None:
        """Bresenham 2D line clearing from start to end (excluding end cell).

        Marks cells along the line of sight as free (0).
        Preserves existing confirmed obstacles (100) and stops ray if an obstacle is hit.
        """
        p0 = self.world_to_map(start_x, start_y)
        p1 = self.world_to_map(end_x, end_y)
        if p0 is None or p1 is None:
            return

        x0, y0 = p0
        x1, y1 = p1

        # Check maximum distance
        dist_cells = int(max_range_m / self.resolution)
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 > dist_cells ** 2:
            return

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        while (cx, cy) != (x1, y1):
            if 0 <= cx < self.width_cells and 0 <= cy < self.height_cells:
                # If an existing obstacle is in the line of sight, stop ray
                if self.grid[cy, cx] == 100 and (cx, cy) != (x0, y0):
                    break
                if self.grid[cy, cx] != 100:
                    self.grid[cy, cx] = 0

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def update_visual_observation(
        self,
        robot_x: float,
        robot_y: float,
        landmark_x: float,
        landmark_y: float,
        landmark_z: float,
    ) -> None:
        """Update occupancy grid from a 3D visual observation.

        Clears the traversed line-of-sight ray from robot to landmark as free space.
        If the landmark height qualifies as an obstacle (min_obstacle_height <= Z <= max_obstacle_height),
        marks the endpoint as an obstacle (100). Otherwise, if it is ground-level (Z < min_obstacle_height),
        marks the endpoint as traversable ground (0).
        """
        if not (np.isfinite(robot_x) and np.isfinite(robot_y) and
                np.isfinite(landmark_x) and np.isfinite(landmark_y) and np.isfinite(landmark_z)):
            return

        # 1. Clear free space along line of sight
        self.clear_ray(robot_x, robot_y, landmark_x, landmark_y)

        # 2. Process endpoint
        pt = self.world_to_map(landmark_x, landmark_y)
        if pt is None:
            return
        gx, gy = pt

        if self.min_obstacle_height <= landmark_z <= self.max_obstacle_height:
            self.grid[gy, gx] = 100
        elif landmark_z < self.min_obstacle_height:
            # Traversable ground feature (e.g. shadow or trail texture)
            if self.grid[gy, gx] != 100:
                self.grid[gy, gx] = 0

    def to_occupancy_grid_msg(self, stamp_sec: float) -> OccupancyGrid:
        """Construct standard nav_msgs/msg/OccupancyGrid message."""
        msg = OccupancyGrid()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"

        meta = MapMetaData()
        meta.map_load_time = msg.header.stamp
        meta.resolution = float(self.resolution)
        meta.width = int(self.width_cells)
        meta.height = int(self.height_cells)

        origin_pose = Pose()
        origin_pose.position.x = float(self.origin_x)
        origin_pose.position.y = float(self.origin_y)
        origin_pose.position.z = 0.0
        origin_pose.orientation.w = 1.0
        meta.origin = origin_pose

        msg.info = meta
        msg.data = self.grid.flatten().tolist()
        return msg

    def save_map_files(self, base_path: str) -> Tuple[str, str]:
        """Export map to standard ROS PGM image and YAML metadata files."""
        pgm_path = f"{base_path}.pgm"
        yaml_path = f"{base_path}.yaml"
        os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)

        # Convert grid values: 0 -> 254 (free), 100 -> 0 (occupied), -1 -> 205 (unknown)
        pgm_img = np.full(self.grid.shape, 205, dtype=np.uint8)
        pgm_img[self.grid == 0] = 254
        pgm_img[self.grid == 100] = 0

        # PGM files are stored with top-row first (image row 0 is top)
        # ROS occupancy grids have row 0 at bottom (origin_y)
        pgm_img_flipped = np.flipud(pgm_img)

        # Write PGM binary file (P5)
        with open(pgm_path, 'wb') as f:
            header = f"P5\n{self.width_cells} {self.height_cells}\n255\n".encode('ascii')
            f.write(header)
            f.write(pgm_img_flipped.tobytes())

        # Write YAML metadata file
        yaml_content = f"""image: {os.path.basename(pgm_path)}
mode: trinary
resolution: {self.resolution:.4f}
origin: [{self.origin_x:.4f}, {self.origin_y:.4f}, 0.0000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        return pgm_path, yaml_path

    def load_from_grid_data(self, data: List[int], width: int, height: int, resolution: float, origin_x: float, origin_y: float) -> None:
        """Load grid array directly from serialized map data."""
        self.width_cells = width
        self.height_cells = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.grid = np.array(data, dtype=np.int8).reshape((height, width))
