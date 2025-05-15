"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq
from matplotlib import pyplot as plt

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.thick_map = None
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def create_node(self, position, g=float('inf'), h=0.0, parent=None):
        return {
            'position': position,
            'g': g,
            'h': h,
            'f': g + h,
            'parent': parent
        }

    def reconstruct_path(self, goal_node):
        path = []
        map_path = []
        current = goal_node

        while current is not None:
            path.append(np.array(self.grid.conv_map_to_world(*current['position'])))
            map_path.append(np.array(current['position']))
            current = current['parent']
        return np.transpose(np.array(path[::-1]))  # Reverse to get path from start to goal

    def thick_grid(self, occupancy_map):
        size = 12
        thick_map = np.zeros_like(occupancy_map)
        print(occupancy_map)
        for i in range(len(occupancy_map)):
            for j in range(len(occupancy_map[i])):
                if occupancy_map[i, j] > 0:
                    # pontos em torno de thick_map[i, j] são 40
                    for x in range(-size, size, 1):
                        for z in range(-size, size, 1):
                            if i + x > 0 and i + x < len(thick_map) and j + z > 0 and j + z < len(thick_map[0]):
                                thick_map[i + x, j + z] = 255
        return thick_map

    def display(self, mapa):
        plt.cla()
        print("PRINTEI")
        plt.imshow(mapa.T, origin='lower', extent=[self.grid.x_min_world, self.grid.x_max_world, self.grid.y_min_world, self.grid.y_max_world])

    def plan(self, start_world, goal_world):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        self.thick_map = self.thick_grid(self.grid.occupancy_map)
        # self.display(thick_map)
        # print(thick_map)
        # plt.cla()
        # plt.imsave("mapa.png", thick_map.T)
        start = self.grid.conv_world_to_map(start_world[0], start_world[1])
        goal = self.grid.conv_world_to_map(goal_world[0], goal_world[1])
        start_node = self.create_node(
            position=start,
            g=0,
            h=self.heuristic(start, goal)
        )
        open_list = [(start_node['f'], start)]
        open_dict = {start: start_node}
        closed_set = set()
        while open_list:
            _, current_pos = heapq.heappop(open_list)
            current_node = open_dict[current_pos]

            if current_pos == goal:
                return self.reconstruct_path(current_node)

            closed_set.add(current_pos)

            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue

                tentative_g = current_node['g'] + self.heuristic(current_pos, neighbor_pos)

                if neighbor_pos not in open_dict:
                    neighbor = self.create_node(
                        position=neighbor_pos,
                        g=tentative_g,
                        h=self.heuristic(neighbor_pos, goal),
                        parent=current_node
                    )
                    heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                    open_dict[neighbor_pos] = neighbor
                elif tentative_g < open_dict[neighbor_pos]['g']:
                    # Found a better path to the neighbor
                    neighbor = open_dict[neighbor_pos]
                    neighbor['g'] = tentative_g
                    neighbor['f'] = tentative_g + neighbor['h']
                    neighbor['parent'] = current_node
        return []  # No path found

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

    def valid_pos(self, cell):
        if cell[0] >= 0 and cell[0] < self.grid.x_max_map and cell[1] >= 0 and cell[1] < self.grid.y_max_map:
            return True
        return False

    def heuristic(self, cell_1, cell_2):
        x1, y1 = cell_1
        x2, y2 = cell_2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_neighbors(self, current_cell):
        x, y = current_cell
        rows, cols = self.grid.x_max_map, self.grid.y_max_map

        # All possible moves (including diagonals)
        possible_moves = [
            (x + 1, y), (x - 1, y),    # Right, Left
            (x, y + 1), (x, y - 1),    # Up, Down
            (x + 1, y + 1), (x - 1, y - 1),  # Diagonal moves
            (x + 1, y - 1), (x - 1, y + 1)
        ]
        return [
            (nx, ny) for nx, ny in possible_moves
            if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
            and self.thick_map[nx, ny] == 0   # Not an obstacle
        ]
