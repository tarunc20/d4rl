"""Environments using kitchen and Franka robot."""
from gym import spaces
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
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
class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = False
    TERMINATE_ON_TASK_COMPLETE = False

    def __init__(self, dense=True, delta=0.0, multitask=False, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.dense = dense
        super(KitchenBase, self).__init__(**kwargs)
        self.multitask = multitask
        if self.multitask:
            self.task_to_step_to_primitive_name_dict = {
                "microwave": {
                    0: "drop",
                    1: "angled_x_y_grasp",
                    2: "move_backward",
                    3: "no_op",
                    4: "no_op",
                    5: "no_op",
                },
                "kettle": {
                    0: "drop",
                    1: "angled_x_y_grasp",
                    2: "move_delta_ee_pose",
                    3: "drop",
                    4: "open_gripper",
                    5: "no_op",
                },
                "top burner": {
                    0: "lift",
                    1: "angled_x_y_grasp",
                    2: "rotate_about_y_axis",
                    3: "no_op",
                    4: "no_op",
                    5: "no_op",
                },
                "slide cabinet": {
                    0: "lift",
                    1: "angled_x_y_grasp",
                    2: "move_right",
                    3: "no_op",
                    4: "no_op",
                    5: "no_op",
                },
                "hinge cabinet": {
                    0: "lift",
                    1: "angled_x_y_grasp",
                    2: "move_delta_ee_pose",
                    3: "move_backward",
                    4: "angled_x_y_grasp",
                    5: "move_right",
                },
                "light switch": {
                    0: "close_gripper",
                    1: "lift",
                    2: "move_right",
                    3: "move_forward",
                    4: "move_left",
                    5: "no_op",
                },
            }

            self.task_to_action_space = {
                "microwave": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
                "kettle": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
                "top burner": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
                "slide cabinet": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
                "hinge cabinet": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
                "light switch": Box(
                    np.array(
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
                    ),
                    np.array(
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
                    ),
                    dtype=np.float32,
                ),
            }
            self.num_tasks = len(self.TASK_ELEMENTS)
            self.one_hot_task = np.zeros(self.num_tasks)
            obs_upper_old = self.observation_space.high
            obs_lower_old = self.observation_space.low
            obs_upper_one_hot = np.ones(self.num_tasks)
            obs_lower_one_hot = np.zeros(self.num_tasks)
            obs_lower = np.concatenate((obs_lower_old, obs_lower_one_hot))
            obs_upper = np.concatenate((obs_upper_old, obs_upper_one_hot))
            self.observation_space = spaces.Box(
                obs_lower,
                obs_upper,
            )

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio
        )

        self.obs_dict = {}
        self.obs_dict["t"] = t
        self.obs_dict["qp"] = qp
        self.obs_dict["qv"] = qv
        self.obs_dict["obj_qp"] = obj_qp
        self.obs_dict["obj_qv"] = obj_qv
        self.obs_dict["goal"] = self.goal
        if not self.initializing and self.multitask:
            self.obs_dict["one_hot_task"] = self.one_hot_task
        if self.image_obs:
            img = self.render(mode="rgb_array")
            img = img.transpose(2, 0, 1).flatten()
            if not self.initializing and self.multitask:
                return np.concatenate((img, self.one_hot_task))
            else:
                return img
        else:
            return np.concatenate(
                [self.obs_dict["qp"], self.obs_dict["obj_qp"], self.obs_dict["goal"]]
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
        if self.multitask:
            idx = np.random.randint(0, len(self.TASK_ELEMENTS), 1)[0]
            task = self.TASK_ELEMENTS[idx]
            self.tasks_to_complete = [task]
            self.one_hot_task = np.zeros(len(self.TASK_ELEMENTS))
            self.one_hot_task[idx] = 1
            self.step_to_primitive_name = self.task_to_step_to_primitive_name_dict[task]
            self.action_space = self.task_to_action_space[task]
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate(
                    (self.action_space.low, act_lower_primitive)
                )
                action_high = np.concatenate(
                    (self.action_space.high, act_upper_primitive)
                )
                self.action_space = Box(action_low, action_high, dtype=np.float32)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
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
            complete = distance < BONUS_THRESH and self.step_count == self.max_steps - 1
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

    def step(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        obs, reward, done, env_info = super(KitchenBase, self).step(
            a,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        self.update_info(env_info)
        return obs, reward, done, env_info

    def update_info(self, info):
        next_q_obs = self.obs_dict["qp"]
        next_obj_obs = self.obs_dict["obj_qp"]
        next_goal = self.obs_dict["goal"]
        idx_offset = len(next_q_obs)
        if not self.initializing and self.multitask:
            tasks_to_log = self.TASK_ELEMENTS
        else:
            tasks_to_log = self.tasks_to_complete
        for element in tasks_to_log:
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
        super(KitchenMicrowaveV0, self).__init__(max_steps=3, **kwargs)
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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenKettleV0, self).__init__(max_steps=5, **kwargs)
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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenBottomBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom burner"]

    def __init__(self, **kwargs):
        super(KitchenBottomBurnerV0, self).__init__(max_steps=3, **kwargs)


class KitchenTopBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top burner"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenTopBurnerV0, self).__init__(max_steps=3, **kwargs)

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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenSlideCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenSlideCabinetV0, self).__init__(max_steps=3, **kwargs)
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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenHingeCabinetV0, self).__init__(max_steps=6, **kwargs)

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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenLightSwitchV0(KitchenBase):
    TASK_ELEMENTS = ["light switch"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenLightSwitchV0, self).__init__(max_steps=5, **kwargs)
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
        if not self.fixed_schema:
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            action_low = np.concatenate((action_low, act_lower_primitive))
            action_high = np.concatenate((action_high, act_upper_primitive))
        self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenMultitaskAllV0(KitchenBase):
    TASK_ELEMENTS = [
        "light switch",
        "top burner",
        "slide cabinet",
        "hinge cabinet",
        "kettle",
        "microwave",
    ]

    def __init__(self, delta=0, **kwargs):
        super(KitchenMultitaskAllV0, self).__init__(
            max_steps=6, delta=delta, multitask=True, **kwargs
        )
        self.reset_model()
