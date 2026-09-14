"""A* Global Path Planner with multi-factor cost model for NAVIGUARD."""

import heapq
import math
from typing import Optional, List, Tuple, Dict
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid


class GlobalPlannerAStar:
    """8-connected grid-based A* planner incorporating obstacle clearance, terrain difficulty, slope, and turning costs."""

    # 8-connected neighbor offsets: (dx, dy, step_cost_mult)
    NEIGHBORS = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        heuristic_weight: float = 1.0,
        turn_penalty_weight: float = 0.5,
        clearance_weight: float = 0.15,
        terrain_weight: float = 0.15,
        slope_weight: float = 0.20,
        max_iterations: int = 150000,
    ) -> None:
        self.heuristic_weight = heuristic_weight
        self.turn_penalty_weight = turn_penalty_weight
        self.clearance_weight = clearance_weight
        self.terrain_weight = terrain_weight
        self.slope_weight = slope_weight
        self.max_iterations = max_iterations

    def plan(
        self,
        grid: NavigationOccupancyGrid,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Compute collision-free, clearance-optimal global path from start to goal in world coordinates."""
        if not grid.is_initialized:
            return None

        start_cell = grid.world_to_map(start_world[0], start_world[1])
        goal_cell = grid.world_to_map(goal_world[0], goal_world[1])

        if start_cell is None or goal_cell is None:
            return None

        # Handle Start == Goal
        if start_cell == goal_cell:
            return [start_world, goal_world]

        # If start cell is in inflated obstacle (e.g. robot spawned near wall), find nearest free cell
        if grid.is_lethal(start_cell[0], start_cell[1]):
            start_cell = self._find_nearest_free_cell(grid, start_cell, max_radius_cells=8)
            if start_cell is None:
                return None

        # If goal cell is lethal, check if nearby free cell exists within clearance tolerance
        if grid.is_lethal(goal_cell[0], goal_cell[1]):
            goal_cell = self._find_nearest_free_cell(grid, goal_cell, max_radius_cells=8)
            if goal_cell is None:
                return None

        # A* Search
        start_node = (start_cell[0], start_cell[1])
        goal_node = (goal_cell[0], goal_cell[1])

        counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int], Optional[Tuple[int, int]]]] = []
        heapq.heappush(open_set, (0.0, counter, start_node, None))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_node: 0.0}
        closed_set = set()

        goal_reached = False
        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1
            current_f, _, current_node, parent_dir = heapq.heappop(open_set)

            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            if current_node == goal_node:
                goal_reached = True
                break

            cx, cy = current_node
            current_g = g_score[current_node]

            for dx, dy, step_mult in self.NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if neighbor in closed_set:
                    continue

                if grid.is_lethal(nx, ny):
                    continue

                # Diagonal safety: prevent cutting sharp corners through diagonal obstacles
                if dx != 0 and dy != 0:
                    if grid.is_lethal(cx + dx, cy) or grid.is_lethal(cx, cy + dy):
                        continue

                # 1. Base geometric step distance
                step_dist = step_mult * grid.resolution

                # 2. Multi-cost evaluation (clearance, terrain, slope, unknown)
                cell_cost = grid.get_cost(nx, ny)
                cost_penalty = cell_cost * self.clearance_weight

                # 3. Turning cost based on direction change
                turn_penalty = 0.0
                if parent_dir is not None and (dx != parent_dir[0] or dy != parent_dir[1]):
                    pdx, pdy = parent_dir
                    dot = (dx * pdx + dy * pdy) / (step_mult * math.hypot(pdx, pdy))
                    dot = max(-1.0, min(1.0, dot))
                    turn_angle = math.acos(dot)
                    turn_penalty = self.turn_penalty_weight * turn_angle * grid.resolution

                edge_cost = step_dist * (1.0 + cost_penalty) + turn_penalty
                tentative_g = current_g + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current_node
                    # Euclidean heuristic with weight
                    h_score = self.heuristic_weight * math.hypot(nx - goal_node[0], ny - goal_node[1]) * grid.resolution
                    f_score = tentative_g + h_score
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor, (dx, dy)))

        if not goal_reached:
            return None

        # Reconstruct path
        path_cells: List[Tuple[int, int]] = []
        curr: Optional[Tuple[int, int]] = goal_node
        while curr is not None:
            path_cells.append(curr)
            curr = came_from.get(curr)
        path_cells.reverse()

        # Convert to world coordinates
        path_world: List[Tuple[float, float]] = [start_world]
        for gx, gy in path_cells[1:-1]:
            wx, wy = grid.map_to_world(gx, gy)
            path_world.append((wx, wy))
        path_world.append(goal_world)

        return path_world

    def _find_nearest_free_cell(
        self,
        grid: NavigationOccupancyGrid,
        target_cell: Tuple[int, int],
        max_radius_cells: int = 8,
    ) -> Optional[Tuple[int, int]]:
        """Search radial neighborhood for nearest non-lethal cell with maximum clearance."""
        tx, ty = target_cell
        best_cell = None
        min_dist_sq = float('inf')

        for r in range(1, max_radius_cells + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    gx, gy = tx + dx, ty + dy
                    if 0 <= gx < grid.width_cells and 0 <= gy < grid.height_cells:
                        if not grid.is_lethal(gx, gy):
                            d2 = dx * dx + dy * dy
                            if d2 < min_dist_sq:
                                min_dist_sq = d2
                                best_cell = (gx, gy)
            if best_cell is not None:
                return best_cell

        return None
