"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, line_follower, spline_smooth, potential_field_control_multiobj
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.my_goal_i = 0
        self.goal_back_i = 0
        self.counter = 0
        self.stoppedCounter = 0
        self.grad_total = np.array([0, 0]).astype(np.float64)
        self.traj = None
        self.path = None
        self.obsts_vector = None

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp3()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        goal = [[0, -350, 0], [-200, -400, 0], [-220, -200, 0], [-400, -20, 0], [-800, -25, 0]]
        command, self.my_goal_i, self.grad_total = potential_field_control(self.lidar(), self.odometer_values(), goal, self.my_goal_i)
        self.counter += 1
        return command

    def control_tp3(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        command = {"forward": 0, "rotation": 0}
        goal = [[0, -500, 0], [-150, -400, 0], [-220, -200, 0], [-400, -20, 0], [-800, -40, 0]]
        # goal = [[0, -350, 0]]
        if self.my_goal_i < len(goal):
            if self.counter == 0:
                for _ in range(50):
                    self.tiny_slam.update_map(self.lidar(), self.corrected_pose, self.odometer_values())

            best_score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
            # print(f"best_score: {best_score}")
            if best_score > 4000:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
                self.tiny_slam.update_map(self.lidar(), self.corrected_pose, goal[self.my_goal_i], self.odometer_values(), self.grad_total, obsts=self.obsts_vector)
                command, self.my_goal_i, self.grad_total, self.obsts_vector = potential_field_control(self.lidar(), self.corrected_pose, goal, self.my_goal_i)
                # command, self.my_goal_i, self.grad_total, self.obsts_vector = potential_field_control_multiobj(self.lidar(), self.corrected_pose, goal, self.my_goal_i)
        else:
            goal_back = np.array([0, 0, 0])
            if self.my_goal_i == len(goal):
                self.traj = self.planner.plan(self.corrected_pose, goal_back)
                self.path = spline_smooth(np.transpose(self.traj).tolist())
                self.my_goal_i += 1
                self.counter = 0
            best_score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
            print(f"best_score: {best_score}")
            if best_score > 3000:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
                if self.path:
                    self.tiny_slam.update_map(self.lidar(), self.corrected_pose, self.path[0], self.odometer_values(), self.grad_total, self.traj)
                command, self.path = line_follower(self.corrected_pose, self.path)
            # if self.path:
                # self.occupancy_grid.display_cv(self.corrected_pose, goal=self.path[0], traj=self.traj, odom=self.corrected_pose)
        self.counter += 1
        return command
