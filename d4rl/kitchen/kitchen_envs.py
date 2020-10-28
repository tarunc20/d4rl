"""Environments using kitchen and Franka robot."""
import os
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

from d4rl.offline_env import OfflineEnv

from gym.spaces.box import Box

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3


@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = False
    TERMINATE_ON_TASK_COMPLETE = False

    def __init__(
        self,
        dataset_url=None,
        ref_max_score=None,
        ref_min_score=None,
        dense=True,
        **kwargs
    ):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.dense = dense
        super(KitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score,
        )

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.0
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = obs_dict["goal"]
        idx_offset = len(next_q_obs)
        completions = []
        dense = 0
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            dense += -1 * distance  # reward must be negative distance for RL
            complete = distance < BONUS_THRESH
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        if self.dense:
            reward_dict["r_total"] = dense
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        self.update_info(env_info)
        return obs, reward, done, env_info

    def update_info(self, info):
        next_q_obs = self.obs_dict["qp"]
        next_obj_obs = self.obs_dict["obj_qp"]
        next_goal = self.obs_dict["goal"]
        idx_offset = len(next_q_obs)
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            info[element + " distance to goal"] = distance
            info[element + " success"] = float(distance < BONUS_THRESH)
            info["success"] = float(distance < BONUS_THRESH)
        return info


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=3, **kwargs)
        self.step_to_primitive_name = {
            0: "drop",
            1: "angled_x_y_grasp",
            2: "move_backward",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                -np.pi / 6 - delta,
                -0.25 - delta,
                0.9 - delta,
                0,
                0,
                0,
                0,
                0,
                0.5 - delta,
                0,
                0,
                0,
                0.6 - delta,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                -np.pi / 6 + delta,
                -0.25 + delta,
                0.9 + delta,
                0,
                0,
                0,
                0,
                0,
                0.5 + delta,
                0,
                0,
                0,
                0.6 + delta,
            ]
        )
        self.action_space = Box(action_low, action_high)


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=5, **kwargs)
        self.step_to_primitive_name = {
            0: "drop",
            1: "angled_x_y_grasp",
            2: "move_delta_ee_pose",
            3: "drop",
            4: "open_gripper",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                0,
                0.2 - delta,
                0.65 - delta,
                0.25 - delta,
                1.1 - delta,
                0.25 - delta,
                0,
                0,
                0.25 - delta,
                0,
                0,
                0,
                0.0,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                0,
                0.2 + delta,
                0.65 + delta,
                0.25 + delta,
                1.1 + delta,
                0.25 + delta,
                0,
                0,
                0.5 + delta,
                0,
                0,
                0,
                0.0,
            ]
        )
        self.action_space = Box(action_low, action_high)


class KitchenBottomBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom burner"]

    def __init__(self, **kwargs):
        super().__init__(self, max_steps=3, **kwargs)


class KitchenTopBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top burner"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=3, **kwargs)

        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "rotate_about_y_axis",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                0,
                0.5 - delta,
                1.1 - delta,
                0.0,
                0.0,
                0.0,
                -np.pi / 4 - delta,
                0.55 - delta,
                0.0,
                0,
                0,
                0,
                0.0,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                0,
                0.5 + delta,
                1.1 + delta,
                0.0,
                0.0,
                0.0,
                -np.pi / 4 + delta,
                0.55 + delta,
                0.0,
                0,
                0,
                0,
                0.0,
            ]
        )
        self.action_space = Box(action_low, action_high)


class KitchenSlideCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=3, **kwargs)
        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "move_right",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.7 - delta,
                1.0 - delta,
                0.0,
                0.0,
                0.0,
                0.0,
                1 - delta,
                0.0,
                0.0,
                0.6 - delta,
                0.0,
                0.0,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.7 + delta,
                1.0 + delta,
                0.0,
                0.0,
                0.0,
                0.0,
                1 + delta,
                0.0,
                0.0,
                0.6 + delta,
                0.0,
                0.0,
            ]
        )
        self.action_space = Box(action_low, action_high)


class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=6, **kwargs)

        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "move_delta_ee_pose",
            3: "move_backward",
            4: "angled_x_y_grasp",
            5: "move_right",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                -np.pi / 6 - delta,
                -0.35 - delta,
                0.1 - delta,
                0.5 - delta,
                -0.5 - delta,
                0.0,
                0.0,
                1 - delta,
                0.0,
                0,
                1 - delta,
                0,
                0.25 - delta,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                np.pi / 6 + delta,
                -0.35 + delta,
                1.4 + delta,
                0.5 + delta,
                -0.5 + delta,
                0.0,
                0.0,
                1 + delta,
                0.0,
                0,
                1 + delta,
                0,
                0.25 + delta,
            ]
        )
        self.action_space = Box(action_low, action_high)


class KitchenLightSwitchV0(KitchenBase):
    TASK_ELEMENTS = ["light switch"]

    def __init__(self, delta=0, **kwargs):
        super().__init__(self, max_steps=5, **kwargs)
        self.step_to_primitive_name = {
            0: "close_gripper",
            1: "lift",
            2: "move_right",
            3: "move_forward",
            4: "move_left",
        }
        action_low = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.45 - delta,
                0.0,
                0.45 - delta,
                0.45 - delta,
                1.25 - delta,
                0.0,
            ]
        )

        action_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.45 + delta,
                0.0,
                0.45 + delta,
                0.45 + delta,
                1.25 + delta,
                0.0,
            ]
        )
        self.action_space = Box(action_low, action_high)
