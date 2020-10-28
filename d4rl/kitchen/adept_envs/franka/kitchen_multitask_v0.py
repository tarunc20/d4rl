""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os

import mujoco_py
import numpy as np
from d4rl.kitchen.adept_envs import robot_env
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine
import quaternion

from robosuite.controllers import (
    EndEffectorImpedanceController,
    EndEffectorInverseKinematicsController,
)


@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        "default": os.path.join(os.path.dirname(__file__), "robot/franka_config.xml")
    }
    # Converted to velocity actuation
    ROBOTS = {"robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct"}
    MODEl = os.path.join(
        os.path.dirname(__file__), "../franka/assets/franka_kitchen_jntpos_act_ab.xml"
    )
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(self, robot_params={}, frame_skip=40):
        self.goal_concat = True
        self.obs_dict = {}
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.goal = np.zeros((30,))
        super().__init__(
            self.MODEl,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  # root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params
            ),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),
        )
        self.reset_mocap_welds(self.sim)
        self.sim.forward()
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + 0.01]
        ) + self.sim.data.get_site_xpos("end_effector")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("mocap", gripper_target)
        self.sim.data.set_mocap_quat("mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.init_qpos = self.sim.model.key_qpos[0].copy()
        # For the microwave kettle slide hinge
        self.init_qpos = np.array(
            [
                1.48388023e-01,
                -1.76848573e00,
                1.84390296e00,
                -2.47685760e00,
                2.60252026e-01,
                7.12533105e-01,
                1.59515394e00,
                4.79267505e-02,
                3.71350919e-02,
                -2.66279850e-04,
                -5.18043486e-05,
                3.12877220e-05,
                -4.51199853e-05,
                -3.90842156e-06,
                -4.22629655e-05,
                6.28065475e-05,
                4.04984708e-05,
                4.62730939e-04,
                -2.26906415e-04,
                -4.65501369e-04,
                -6.44129196e-03,
                -1.77048263e-03,
                1.08009684e-03,
                -2.69397440e-01,
                3.50383255e-01,
                1.61944683e00,
                1.00618764e00,
                4.06395120e-03,
                -6.62095997e-03,
                -2.68278933e-04,
            ]
        )

        self.init_qvel = self.sim.model.key_qvel[0].copy()

        self.act_mid = np.zeros(self.N_DOF_ROBOT)
        self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)

        act_lower = -1 * np.ones((self.N_DOF_ROBOT,))
        act_upper = 1 * np.ones((self.N_DOF_ROBOT,))
        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8.0 * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def ctrl_set_action(self, sim, action):
        self.data.ctrl[7] = action[-2]
        self.data.ctrl[8] = action[-1]

    def mocap_set_action(self, sim, action):
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7,))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            self.reset_mocap2body_xpos(sim)
            sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
            sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta

    def reset_mocap_welds(self, sim):
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        sim.forward()

    def reset_mocap2body_xpos(self, sim):
        if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
        ):
            return
        for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
        ):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def _set_action(self, action):
        assert action.shape == (9,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        pos_ctrl *= 0.05  # limit maximum change in position
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self.ctrl_set_action(self.sim, action)
        self.mocap_set_action(self.sim, action)

    def close_gripper(self):
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.sim.step()

    def open_gripper(self):
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]))
            self.sim.step()

    def get_ee_pose(self):
        return self.sim.data.get_site_xpos("end_effector")

    def goto_pose(self, pose):
        gripper = self.sim.data.qpos[7:9]
        for _ in range(300):
            self.reset_mocap2body_xpos(self.sim)
            delta = pose - self.get_ee_pose()
            self._set_action(
                np.array(
                    [
                        delta[0],
                        delta[1],
                        delta[2],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.sim.step()

    def rpy_to_quat(self, rpy):
        q = quaternion.from_euler_angles(rpy)
        return np.array([q.x, q.y, q.z, q.w])

    def quat_to_rpy(self, q):
        q = quaternion.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_euler_angles(q)

    def convert_xyzw_to_wxyz(self, q):
        return np.array([q[3], q[0], q[1], q[2]])

    def rotate_ee(self, rpy):
        gripper = self.sim.data.qpos[7:9]
        for _ in range(200):
            quat = self.rpy_to_quat(rpy)
            quat_delta = self.convert_xyzw_to_wxyz(quat) - self.sim.data.body_xquat[10]
            self._set_action(
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        quat_delta[0],
                        quat_delta[1],
                        quat_delta[2],
                        quat_delta[3],
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.sim.step()

    def step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init

        # TODO: setup primitive choice and run

        obs = self._get_obs()

        # rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            "time": self.obs_dict["t"],
            "score": score,
        }
        return obs, reward_dict["r_total"], done, env_info

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
        if self.goal_concat:
            return np.concatenate(
                [self.obs_dict["qp"], self.obs_dict["obj_qp"], self.obs_dict["goal"]]
            )

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  # sample a new goal on reset
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path["env_infos"]["score"])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path["env_infos"]["rewards"]["bonus"][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score)
        )

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs // 2,))

    def convert_to_active_observation(self, observation):
        return observation


class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self):
        super(KitchenTaskRelaxV1, self).__init__()

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict["true_reward"] = 0.0
        reward_dict["bonus"] = 0.0
        reward_dict["r_total"] = 0.0
        score = 0.0
        return reward_dict, score

    def render(self, mode="human"):
        if mode == "rgb_array":
            # camera = engine.MovableCamera(self.sim, 256, 256)
            # camera.set_pose(
            #     distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            # )
            # img = camera.render()
            img = self.sim_robot.renderer.render_offscreen(1000, 1000)
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()
