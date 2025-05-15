""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """

        score = 0
        indexes = np.where(lidar.get_sensor_values() < lidar.max_range)
        sensor = lidar.get_sensor_values()[indexes]
        angle = lidar.get_ray_angles()[indexes]
        laser_x = pose[0] + sensor * np.cos(pose[2] + angle)
        laser_y = pose[1] + sensor * np.sin(pose[2] + angle)
        laser_x, laser_y = self.grid.conv_world_to_map(laser_x, laser_y)
        valid = laser_x < self.grid.x_max_map
        laser_x, laser_y = laser_x[valid], laser_y[valid]
        valid = laser_y < self.grid.y_max_map
        laser_x, laser_y = laser_x[valid], laser_y[valid]
        return np.sum(self.grid.occupancy_map[laser_x, laser_y])

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom_pose : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        odom_pose_temp = odom_pose.copy()
        distance = np.linalg.norm(odom_pose_temp[:-1])
        alpha = np.arctan2(odom_pose_temp[1], odom_pose_temp[0])
        odom_pose_temp[0] = distance * np.cos(odom_pose_ref[2] + alpha)
        odom_pose_temp[1] = distance * np.sin(odom_pose_ref[2] + alpha)

        return odom_pose_ref + odom_pose_temp

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        i = 0
        N = 200
        sigma = 0.2
        best_random_pose = self.odom_pose_ref.copy()
        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, best_random_pose))
        while i < N:
            # offset = np.random.normal(0, sigma, 3)
            offset = np.array([np.random.normal(0.0, 1), np.random.normal(0.0, 1), np.random.normal(0.0, 0.15)])
            score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, best_random_pose + offset))
            if score > best_score:
                best_score = score
                best_random_pose = self.odom_pose_ref.copy() + offset
                i = 0
            else:
                i += 1
        self.odom_pose_ref = best_random_pose.copy()
        return best_score

    def unit_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm != 0:
            return vector / np.linalg.norm(vector)
        else:
            return np.zeros_like(vector)

    def update_map(self, lidar, pose, goal, odom=None, grad=None, traj=None, obsts=None):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        laser_dir_x = np.cos(pose[2] + lidar.get_ray_angles())
        laser_dir_y = np.sin(pose[2] + lidar.get_ray_angles())
        laser_x = pose[0] + lidar.get_sensor_values() * laser_dir_x
        laser_y = pose[1] + lidar.get_sensor_values() * laser_dir_y
        laser_x_minus = laser_x - self.unit_vector(laser_dir_x) * 100
        laser_x_plus = laser_x + self.unit_vector(laser_dir_x) * 100
        laser_y_minus = laser_y - self.unit_vector(laser_dir_y) * 100
        laser_y_plus = laser_y + self.unit_vector(laser_dir_y) * 100
        for x_1, x_2 in zip(laser_x, laser_y):
            self.grid.add_value_along_line(pose[0], pose[1], x_1, x_2, -1)
        for x_1, y_1, x_2, y_2 in zip(laser_x_minus, laser_y_minus, laser_x_plus, laser_y_plus):
            self.grid.add_value_along_line(x_1, y_1, x_2, y_2, 1)
        self.grid.add_map_points(laser_x, laser_y, 3)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)
        self.grid.display_cv(pose, goal=goal, odom=odom, traj=traj, grad=grad, obsts=obsts)

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1
        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
