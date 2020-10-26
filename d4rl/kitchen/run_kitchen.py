import gym
import d4rl
import cv2
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveV0

env = KitchenMicrowaveV0(dense=True)
env.reset()
done = False
while not done:
    o, r, d, i = env.step(env.action_space.sample())
    # print("microwave", i["microwave distance to goal"])
    im = env.render(mode="human")
    # cv2.imshow("env", im)
    # cv2.waitKey(1)
    print(r)
