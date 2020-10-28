import gym
import d4rl
import cv2
from d4rl.kitchen.kitchen_envs import *
import numpy as np
import quaternion

env = KitchenMicrowaveV0(dense=True)
env.reset()
done = False
ctr = 0

max_path_length = 11
while not done:
    # kettle
    # if ctr % max_path_length == 0:
    #     env.reset()
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.2, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, -0.5]))
    # if ctr % max_path_length == 3:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.55, 0]))
    # if ctr % max_path_length == 4:
    #     env.close_gripper()
    # if ctr % max_path_length == 5:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 0.25]))
    # if ctr % max_path_length == 6:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.25, 1.1, 0.0]))
    # if ctr % max_path_length == 7:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, -0.25]))
    # if ctr % max_path_length == 8:
    #     env.open_gripper()

    # burner
    # if ctr % max_path_length == 0:
    #     env.reset()
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 0.55]))
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.5, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 1.25, 0]))
    # if ctr % max_path_length == 3:
    #     env.close_gripper()
    # if ctr % max_path_length == 4:
    #     rotation = env.quat_to_rpy(env.sim.data.body_xquat[10]) - np.array(
    #         [0, 0, -np.pi / 4]
    #     )
    #     env.rotate_ee(rotation)

    # switch
    # if ctr % max_path_length == 0:
    #     env.reset()
    #     env.close_gripper()
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 0.45]))
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.45, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 1.25, 0]))
    # if ctr % max_path_length == 3:
    #     env.goto_pose(env.get_ee_pose() + np.array([-0.45, 0, 0]))

    # slide cabinet
    # if ctr % max_path_length == 0:
    #     env.reset()
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 1]))
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.7, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 1, 0]))
    # if ctr % max_path_length == 3:
    #     env.close_gripper()
    # if ctr % max_path_length == 4:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.6, 0, 0]))

    # microwave
    if ctr % max_path_length == 0:
        env.reset()
        rotation = env.quat_to_rpy(env.sim.data.body_xquat[10]) - np.array(
            [-np.pi / 6, 0, 0]
        )
        env.rotate_ee(rotation)
        env.goto_pose(env.get_ee_pose() + np.array([-0.25, 0.0, 0]))
    if ctr % max_path_length == 1:
        env.goto_pose(env.get_ee_pose() + np.array([0, 0, -0.5]))
    if ctr % max_path_length == 2:
        env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.9, 0]))
    if ctr % max_path_length == 3:
        env.close_gripper()
    if ctr % max_path_length == 4:
        env.goto_pose(env.get_ee_pose() + np.array([0.0, -0.6, 0]))

    # hinge cabinet
    # if ctr % max_path_length == 0:
    #     env.reset()
    #     rotation = env.quat_to_rpy(env.sim.data.body_xquat[10]) - np.array(
    #         [-np.pi / 6, 0, 0]
    #     )
    #     env.rotate_ee(rotation)
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 1]))
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([-0.3, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 1.3, 0]))
    # if ctr % max_path_length == 3:
    #     env.close_gripper()
    #     env.close_gripper()
    # if ctr % max_path_length == 4:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.5, -0.5, 0]))
    # if ctr % max_path_length == 5:
    #     env.goto_pose(env.get_ee_pose() + np.array([0, -0.25, 0]))
    # if ctr % max_path_length == 6:
    #     rotation = env.quat_to_rpy(env.sim.data.body_xquat[10]) - np.array(
    #         [np.pi / 6, 0, 0]
    #     )
    #     env.rotate_ee(rotation)
    # if ctr % max_path_length == 7:
    #     env.goto_pose(env.get_ee_pose() + np.array([-0.35, 0, 0]))
    # if ctr % max_path_length == 8:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.2, 0]))
    # if ctr % max_path_length == 9:
    #     env.close_gripper()
    # if ctr % max_path_length == 10:
    #     env.goto_pose(env.get_ee_pose() + np.array([1, -0, 0]))

    for i in range(1000):
        env.render(mode="human")
    ctr += 1
    o, r, d, i = env.step(env.action_space.sample())
    print(r)
