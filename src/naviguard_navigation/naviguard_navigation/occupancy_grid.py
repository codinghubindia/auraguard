"""Occupancy Grid representation, coordinate transforms, and obstacle inflation for NAVIGUARD."""

import math
from typing import Optional, Tuple, List
import numpy as np
from nav_msgs.msg import OccupancyGrid


class NavigationOccupancyGrid:
    """Processes SLAM occupancy grid with footprint inflation and proximity costs."""

    def __init__(
        self,
        inflation_radius_m: float = 0.50,
        proximity_radius_m: float = 1.0,
        allow_unknown: bool = True,
        unknown_cost_penalty: float = 5.0,
        obstacle_threshold: int = 50,
    ) -> None:
        self.inflation_radius_m = inflation_radius_m
        self.proximity_radius_m = proximity_radius_m
        self.allow_unknown = allow_unknown
        self.unknown_cost_penalty = unknown_cost_penalty
        self.obstacle_threshold = obstacle_threshold

        self.resolution: float = 0.05
        self.width_cells: int = 0
        self.height_cells: int = 0
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.frame_id: str = "map"

        # Raw grid: -1 = unknown, 0 = free, 100 = occupied
        self.base_raw_grid: np.ndarray = np.empty((0, 0), dtype=np.int8)
        self.raw_grid: np.ndarray = np.empty((0, 0), dtype=np.int8)
        # Cost grid: 0 = free, 100 = lethal obstacle/inflated, >0 = proximity cost
        self.cost_grid: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.persistent_blocked_regions: List[Tuple[float, float, float]] = []
        self.is_initialized: bool = False

    def update_from_msg(self, msg: OccupancyGrid) -> None:
        """Update map geometry and cost layers from nav_msgs/msg/OccupancyGrid."""
        self.resolution = float(msg.info.resolution)
        self.width_cells = int(msg.info.width)
        self.height_cells = int(msg.info.height)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)
        self.frame_id = msg.header.frame_id if msg.header.frame_id else "map"

        data = np.array(msg.data, dtype=np.int8).reshape((self.height_cells, self.width_cells))
        self.base_raw_grid = data.copy()
        self.raw_grid = data.copy()
        self.is_initialized = True
        self._compute_cost_grid()

    def _compute_cost_grid(self) -> None:
        """Compute inflated lethal obstacles and smooth proximity decay costs."""
        if self.base_raw_grid.size > 0:
            self.raw_grid = self.base_raw_grid.copy()
        for bx, by, br in self.persistent_blocked_regions:
            self._apply_raw_blockage(bx, by, br)

        cost = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)
        lethal_mask = self.raw_grid >= self.obstacle_threshold

        # If no obstacles, check unknown
        if not np.any(lethal_mask):
            if not self.allow_unknown:
                cost[self.raw_grid < 0] = 100.0
            else:
                cost[self.raw_grid < 0] = self.unknown_cost_penalty
            self.cost_grid = cost
            return

        # Distance transform to nearest obstacle
        try:
            from scipy.ndimage import distance_transform_edt
            dist_cells = distance_transform_edt(~lethal_mask)
            dist_m = dist_cells * self.resolution
        except ImportError:
            dist_m = self._manual_distance_transform(lethal_mask)

        # 1. Lethal obstacle inflation
        cost[dist_m <= self.inflation_radius_m] = 100.0

        # 2. Proximity decay cost for safe clearance
        prox_mask = (dist_m > self.inflation_radius_m) & (dist_m <= self.proximity_radius_m)
        if np.any(prox_mask):
            prox_norm = (self.proximity_radius_m - dist_m[prox_mask]) / (self.proximity_radius_m - self.inflation_radius_m)
            cost[prox_mask] = np.maximum(cost[prox_mask], prox_norm * 30.0)

        # 3. Unknown space cost handling
        if not self.allow_unknown:
            cost[self.raw_grid < 0] = 100.0  # Forbidden
        else:
            unknown_mask = (self.raw_grid < 0) & (cost < 100.0)
            cost[unknown_mask] += self.unknown_cost_penalty

        self.cost_grid = cost

    def _apply_raw_blockage(self, cx: float, cy: float, radius_m: float) -> int:
        if not self.is_initialized and self.width_cells == 0:
            return 0
        rad_cells = int(np.ceil(radius_m / max(0.01, self.resolution)))
        pt = self.world_to_map(cx, cy)
        if pt is None:
            return 0
        gx, gy = pt
        y_min, y_max = max(0, gy - rad_cells), min(self.height_cells, gy + rad_cells + 1)
        x_min, x_max = max(0, gx - rad_cells), min(self.width_cells, gx + rad_cells + 1)
        count = 0
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                d = math.hypot((x - gx) * self.resolution, (y - gy) * self.resolution)
                if d <= radius_m:
                    self.raw_grid[y, x] = 100
                    count += 1
        return count

    def mark_blocked_region(self, cx: float, cy: float, radius_m: float = 0.45) -> int:
        """Mark a circular patch around (cx, cy) as lethal obstacle cells."""
        self.persistent_blocked_regions.append((float(cx), float(cy), float(radius_m)))
        count = self._apply_raw_blockage(cx, cy, radius_m)
        self._compute_cost_grid()
        return count

    def clear_blocked_regions(self) -> None:
        """Clear all persistent blocked regions and recompute cost grid."""
        self.persistent_blocked_regions.clear()
        if self.is_initialized and self.raw_grid.size > 0:
            self._compute_cost_grid()

    def _manual_distance_transform(self, lethal_mask: np.ndarray) -> np.ndarray:
        """Fallback Euclidean distance transform using radial neighborhood."""
        dist = np.full((self.height_cells, self.width_cells), 999.0, dtype=np.float32)
        occ_y, occ_x = np.where(lethal_mask)
        if len(occ_x) == 0:
            return dist

        rad_cells = int(np.ceil(self.proximity_radius_m / self.resolution))
        for y, x in zip(occ_y, occ_x):
            y_min, y_max = max(0, y - rad_cells), min(self.height_cells, y + rad_cells + 1)
            x_min, x_max = max(0, x - rad_cells), min(self.width_cells, x + rad_cells + 1)
            for gy in range(y_min, y_max):
                for gx in range(x_min, x_max):
                    d = math.hypot(gx - x, gy - y) * self.resolution
                    if d < dist[gy, gx]:
                        dist[gy, gx] = d
        dist[lethal_mask] = 0.0
        return dist

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert continuous map coordinates (meters) to grid indices (gx, gy)."""
        if not self.is_initialized:
            return None
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        if 0 <= gx < self.width_cells and 0 <= gy < self.height_cells:
            return gx, gy
        return None

    def map_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to continuous map coordinates (cell centers)."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def is_lethal(self, gx: int, gy: int) -> bool:
        """Check if grid cell is lethal obstacle or out of bounds."""
        if not (0 <= gx < self.width_cells and 0 <= gy < self.height_cells):
            return True
        return bool(self.cost_grid[gy, gx] >= 100.0)

    def get_cost(self, gx: int, gy: int) -> float:
        """Get planning traversal cost of a grid cell."""
        if not (0 <= gx < self.width_cells and 0 <= gy < self.height_cells):
            return 100.0
        return float(self.cost_grid[gy, gx])

    def is_segment_collision_free(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Bresenham raycast to check if straight line segment is collision-free."""
        p1 = self.world_to_map(x1, y1)
        p2 = self.world_to_map(x2, y2)
        if p1 is None or p2 is None:
            return False

        gx1, gy1 = p1
        gx2, gy2 = p2

        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy

        cx, cy = gx1, gy1
        while True:
            if self.is_lethal(cx, cy):
                return False
            if cx == gx2 and cy == gy2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return True
