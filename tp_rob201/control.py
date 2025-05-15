""" A set of robotics control functions """

import numpy as np
import math
from scipy.interpolate import splprep, splev

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


def prune_collinear(path, *, eps=1e-6):
    if len(path) < 3:
        return path[:]

    pruned = [path[0]]
    for i in range(1, len(path) - 1):
        x0, y0 = pruned[-1]
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        # área do triângulo = 0  →  colinear
        area = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
        if area > eps:            # só guarda se curva
            pruned.append(path[i])

    pruned.append(path[-1])
    return pruned


def spline_smooth(path, step=0.1):
    x, y = map(list, zip(*path))
    tck, _ = splprep([x, y], s=0)          # s=0 → passa por todos
    u = np.arange(0, 1.01, step)           # 0-1 param., amostra fina
    xs, ys = splev(u, tck)
    return list(zip(xs, ys))


def line_follower(current_pose, path, K_lin=0.03, K_ang=3, max_lin=0.2, max_ang=1.0, look_ahead=15):
    x, y, theta = current_pose
    while path and math.hypot(path[0][0] - x, path[0][1] - y) < look_ahead:
        path.pop(0)
    if not path:
        return {"forward": 0.0, "rotation": 0.0}, path

    goal_x, goal_y = path[0]

    dx = goal_x - x
    dy = goal_y - y

    distance_error = math.hypot(dx, dy)

    target_heading = math.atan2(dy, dx)

    heading_error = (target_heading - theta + math.pi) % (2 * math.pi) - math.pi

    v = K_lin * distance_error
    w = K_ang * heading_error

    v = np.clip(v, -max_lin, max_lin)
    w = np.clip(w, -max_ang, max_ang)

    if abs(heading_error) < 0.2:
        v = max(0.1, v)

    return {"forward": v, "rotation": w}, path


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
    K = 2
    K_obst = -1e4
    K_angle = 0.4
    K_speed = 0.1
    safe_distance = 100
    epsilon = 30
    d_lim = 50

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
    obst_vector = np.array([min_dist * np.cos(obst_angle), min_dist * np.sin(obst_angle)])

    grad_obst_vector = np.zeros_like(obst_vector)
    if min_dist <= safe_distance:
        grad_obst_vector = K_obst / (min_dist ** 3) * ((1 / min_dist) - (1 / safe_distance)) * obst_vector
    grad_total = grad_goal_vector + grad_obst_vector
    total_angle = np.arctan2(grad_total[1], grad_total[0])
    turn_angle = total_angle - current_pose[2]
    if turn_angle > np.pi:
        turn_angle = turn_angle - 2 * np.pi
    if turn_angle < -1 * np.pi:
        turn_angle = 2 * np.pi + turn_angle
    command_angle = K_angle * (turn_angle)

    if command_angle > np.pi / 2:
        command_speed = K_speed * np.linalg.norm(grad_total) * ((np.pi) / total_angle)
    else:
        command_speed = K_speed * np.linalg.norm(grad_total)

    command_angle = np.clip(command_angle, -1, 1)
    command_speed = np.clip(command_speed, -0.5, 0.5)

    if goal_distance < epsilon:
        command_speed = 0
        command_angle = 0
        if current_goal_i < len(goal_poses):
            #print("Cheguei")
            current_goal_i += 1

    command = {"forward": command_speed, "rotation": command_angle}

    return command, current_goal_i, grad_total, [obst_vector]


def potential_field_control_multiobj(lidar, current_pose, goal_poses, current_goal_i):
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
    K_obst = -1e5
    K_angle = 0.5
    K_speed = 0.3
    safe_distance = 3000
    epsilon = 30
    d_lim = 50
    n_sectors = 3

    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    min_pos = np.argmin(laser_dist)
    laser_angle = angles[min_pos]
    min_dist = laser_dist[min_pos]

    sector = (angles * n_sectors / (2 * np.pi)).astype(int)
    distance_dips = np.full(n_sectors, np.inf)
    np.minimum.at(distance_dips, sector, laser_dist)
    order = np.lexsort((laser_dist, sector))
    sorted_sector = sector[order]
    sorted_ang = angles[order]

    # índices da 1.ª ocorrência de cada setor na ordenação
    unique_sec, first_occ = np.unique(sorted_sector, return_index=True)

    angle_dips = np.full(n_sectors, np.nan)
    angle_dips[unique_sec] = sorted_ang[first_occ]

    K_obsts = -1e5 / len(distance_dips)

    goal_pose = goal_poses[current_goal_i]

    goal_vector = goal_pose[:-1] - current_pose[:-1]
    goal_distance = np.linalg.norm(goal_vector)

    if goal_distance > d_lim:
        grad_goal_vector = K * unit_vector(goal_vector)
    else:
        grad_goal_vector = K / d_lim * goal_vector

    obst_angle = laser_angle + current_pose[2]
    obsts_angle = angle_dips + current_pose[2]
    obst = [current_pose[0] + min_dist * np.cos(obst_angle), current_pose[1] + min_dist * np.sin(obst_angle)] 
    obsts = np.transpose([current_pose[0] + np.multiply(distance_dips, np.cos(obsts_angle)), current_pose[1] + np.multiply(distance_dips, np.sin(obsts_angle))])
    # print(obsts)
    obst_vector = obst - current_pose[:-1]
    obsts_vector = obsts - current_pose[:-1]
    # print(obsts_vector)

    grad_obst_vector = np.zeros_like(obst_vector)
    grad_obsts_vectors = np.zeros_like(obsts_vector)
    if min_dist <= safe_distance:
        grad_obst_vector = K_obst / (min_dist ** 3) * ((1 / min_dist) - (1 / safe_distance)) * obst_vector
        poids = (K_obsts / np.power(distance_dips, 3)) * ((1 / distance_dips) - (1 / safe_distance))
        grad_obsts_vectors = np.multiply(obsts_vector, np.array([[p, p] for p in poids]))
    # grad_total = grad_goal_vector + grad_obst_vector
    grad_total = grad_goal_vector + np.sum(grad_obsts_vectors)
    total_angle = np.arctan2(grad_total[1], grad_total[0])
    robot_angle = current_pose[2]
    turn_angle = total_angle - robot_angle
    if turn_angle > np.pi:
        turn_angle = turn_angle - 2 * np.pi
    if turn_angle < -1 * np.pi:
        turn_angle = 2 * np.pi + turn_angle
    command_angle = K_angle * (turn_angle)

    if command_angle > np.pi / 2:
        command_speed = K_speed * np.linalg.norm(grad_total) * (np.pi / 2 * total_angle)
    else:
        command_speed = K_speed * np.linalg.norm(grad_total)

    command_angle = np.clip(command_angle, -1, 1)
    command_speed = np.clip(command_speed, -0.5, 0.5)

    if goal_distance < epsilon:
        command_speed = 0
        command_angle = 0
        if current_goal_i < len(goal_poses):
            current_goal_i += 1

    command = {"forward": command_speed, "rotation": command_angle}

    return command, current_goal_i, grad_total, obsts_vector
