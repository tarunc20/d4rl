import gym
import d4rl
import cv2
from d4rl.kitchen.kitchen_envs import *
import numpy as np
import quaternion

env = KitchenSlideCabinetV0(dense=True)
env.reset()
done = False
ctr = 0
# rotation = np.array([0, 0, 0, 0])
rotation = env.quat_to_rpy(env.sim.data.body_xquat[10]) - np.array([0, 0, np.pi / 2])
max_path_length = 9
while not done:
    # kettle
    # if ctr % max_path_length == 0:
    #     env.reset()
    # if ctr % max_path_length == 1:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.25, 0, 0]))
    # if ctr % max_path_length == 2:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, -0.75]))
    # if ctr % max_path_length == 3:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.75, 0]))
    # if ctr % max_path_length == 4:
    #     env.close_gripper()
    # if ctr % max_path_length == 5:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, 0.5]))
    # if ctr % max_path_length == 6:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 1.5, 0.0]))
    # if ctr % max_path_length == 7:
    #     env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, -0.5]))
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
    #     env.rotate_ee(rotation)
    #     env.rotate_ee(rotation)
    #     env.rotate_ee(rotation)
    #     env.rotate_ee(rotation)
    #     env.rotate_ee(rotation)

    # switch
    # if ctr % max_path_length == 0:
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
        env.goto_pose(env.get_ee_pose() + np.array([-0.65, 0.0, 0]))
    if ctr % max_path_length == 1:
        env.goto_pose(env.get_ee_pose() + np.array([0, 0, -0.5]))
    if ctr % max_path_length == 2:
        env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.8, 0]))
    if ctr % max_path_length == 3:
        env.close_gripper()
    if ctr % max_path_length == 4:
        env.goto_pose(env.get_ee_pose() + np.array([0.0, -0.2, 0]))

    # if ctr % max_path_length == 6:
    #     env.goto_pose(env.get_ee_pose() + np.array([0, -0.1, 0]))
    for i in range(1000):
        env.render(mode="human")
    # env.rotate_x_z(0)
    # for i in range(1000):
    #     env.render(mode="human")
    ctr += 1
    # env.close_gripper()
    # env.render(mode="human")
    # env.goto_pose(env.get_ee_pose() + np.array([0, -0.1, 0]))
    # env.render(mode="human")
    # if ctr % max_path_length == 3:
    #     env.rotate_quat_ee(rotation)
    # if ctr % 10 == 0:
    #     env.rotate_ee()
    o, r, d, i = env.step(env.action_space.sample())
    # print("microwave", i["microwave distance to goal"])
    # cv2.imshow("env", im)
    # cv2.waitKey(1)
    print(r)
