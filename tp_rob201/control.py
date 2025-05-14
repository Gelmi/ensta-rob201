""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    speed = 0.0
    rotation_speed = 0.0
    
    dt = laser_dist[-1]*np.cos(np.arctan((laser_dist[300]*np.cos(angles[-1] - angles[300]) - laser_dist[-1])/laser_dist[300]*np.sin(angles[-1] - angles[300])))

    if np.average(laser_dist[170:190]) > 100:
        speed = 0.2
    else:
        if np.abs(np.average(laser_dist[140:170]) - np.average(laser_dist[190:210])) < 10:
            speed = -1
            rotation_speed = 0.9
        else:
            if np.average(laser_dist[140:170]) > np.average(laser_dist[190:210]):
                rotation_speed = -0.3
            else:
                rotation_speed = 0.2

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm != 0:
        return vector / np.linalg.norm(vector)
    else:
        return np.zeros_like(vector)


def potential_field_control(lidar, current_pose, goal_poses, current_goal_i):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    K = 1
    K_obst = 0.5
    K_angle = 1
    K_speed = 0.3
    safe_distance = 1000
    epsilon = 10
    d_lim = 5
    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    min_pos = np.argmin(laser_dist)
    laser_angle = angles[min_pos]
    min_dist = laser_dist[min_pos]

    goal_pose = goal_poses[current_goal_i]

    goal_vector = goal_pose[:-1] - current_pose[:-1]
    goal_distance = np.linalg.norm(goal_vector)

    if goal_distance > d_lim:
        grad_goal_vector = K * unit_vector(goal_vector)
    else:
        grad_goal_vector = K / d_lim * goal_vector

    obst_angle = laser_angle + current_pose[2]
    obst = [current_pose[0] + min_dist * np.cos(obst_angle), current_pose[1] + min_dist * np.sin(obst_angle)]
    obst_vector = obst - current_pose[:-1]

    grad_obst_vector = np.zeros_like(obst_vector)
    if min_dist <= safe_distance:
        grad_obst_vector = K_obst / (min_dist ** 3) * ((1 / min_dist) - (1 / safe_distance)) * obst_vector

    #print(current_pose[:-1], grad_obst_vector, min_dist, goal_distance, goal_pose)

    total_vector = goal_vector + obst_vector
    grad_total = grad_goal_vector + grad_obst_vector
    unit_grad_vector = unit_vector(grad_total)
    total_angle = np.arctan2(unit_grad_vector[1], unit_grad_vector[0])
    command_angle = K_angle * (total_angle - current_pose[2])

    if command_angle > np.pi / 2:
        command_speed = K_speed * np.linalg.norm(total_vector) * (np.pi / 2 * total_angle)
    else:
        command_speed = K_speed * np.linalg.norm(total_vector)

    command_angle = np.clip(command_angle, -1, 1)
    command_speed = np.clip(command_speed, -0.5, 0.5)

    if goal_distance < epsilon:
        command_speed = 0
        command_angle = 0
        if current_goal_i < len(goal_poses) - 1:
            current_goal_i += 1

    command = {"forward": command_speed, "rotation": command_angle}

    return command, current_goal_i
