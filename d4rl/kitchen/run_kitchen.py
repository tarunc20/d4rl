import gym
import d4rl
import cv2
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveV0
import numpy as np
import quaternion

env = KitchenMicrowaveV0(dense=True)
env.reset()
done = False
ctr = 0
# rotation = np.array([0, 0, 0, 0])
rotation = env.sim.data.mocap_quat[0]
max_path_length = 7
while not done:

    if ctr % max_path_length == 0:
        env.reset()
    if ctr % max_path_length == 1:
        # env.goto_pose(env.get_ee_pose() + np.array([0, 0, 2]))
        env.goto_pose(env.get_ee_pose() + np.array([0.25, 0, 0]))
    if ctr % max_path_length == 2:
        # env.goto_pose(env.get_ee_pose() + np.array([-1, 0, 0]))
        env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.0, -0.2]))
    if ctr % max_path_length == 3:
        # env.goto_pose(env.get_ee_pose() + np.array([0, 1.5, 0]))
        env.goto_pose(env.get_ee_pose() + np.array([0.0, 0.5, 0]))
    if ctr % max_path_length == 4:
        env.close_gripper()
    if ctr % max_path_length == 5:
        env.close_gripper()
    # if ctr % max_path_length == 6:
    #     env.goto_pose(env.get_ee_pose() + np.array([0, -0.1, 0]))
    for i in range(1000):
        env.render(mode="human")
    ctr += 1
    # env.close_gripper()
    # env.render(mode="human")
    # env.goto_pose(env.get_ee_pose() + np.array([0, -0.1, 0]))
    # env.render(mode="human")
    # if ctr % max_path_length == 3:
    #     env.rotate_quat_ee(rotation)
    # if ctr % 10 == 0:
    #     env.rotate_ee()
    # o, r, d, i = env.step(env.action_space.sample())
    # print("microwave", i["microwave distance to goal"])
    # cv2.imshow("env", im)
    # cv2.waitKey(1)
    # print(r)
