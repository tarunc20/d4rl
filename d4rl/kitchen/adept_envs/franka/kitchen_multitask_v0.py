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

from gym.spaces.box import Box

import mujoco_py
import numpy as np
import quaternion
from dm_control.mujoco import engine
from gym import spaces

from d4rl.kitchen.adept_envs import robot_env

OBS_TASK_INDICES = {
    "bottom left burner": np.array([11, 12]),
    "top left burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_TASK_GOALS = {
    "bottom left burner": np.array([-0.88, -0.01]),
    "top left burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

INIT_QPOS = np.array(
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


class KitchenV0(robot_env.RobotEnv):
    TASKS = []
    REMOVE_TASKS_WHEN_COMPLETE = False
    REWARD_THRESH = 0.3

    CALIBRATION_PATHS = {
        "default": os.path.join(os.path.dirname(__file__), "robot/franka_config.xml")
    }
    # Converted to velocity actuation
    ROBOTS = {"robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct"}
    EE_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__), "../franka/assets/franka_kitchen_ee_ctrl.xml"
    )
    JOINT_POSITION_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__),
        "../franka/assets/franka_kitchen_joint_position_ctrl.xml",
    )
    TORQUE_CTRL_MODEL = os.path.join(
        os.path.dirname(__file__),
        "../franka/assets/franka_kitchen_torque_ctrl.xml",
    )
    CTLR_MODES_DICT = dict(
        primitives=dict(
            model=EE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        end_effector=dict(
            model=EE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        torque=dict(
            model=TORQUE_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_Unconstrained"
            },
        ),
        joint_position=dict(
            model=JOINT_POSITION_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_PosAct"
            },
        ),
        joint_velocity=dict(
            model=JOINT_POSITION_CTRL_MODEL,
            robot={
                "robot": "d4rl.kitchen.adept_envs.franka.robot.franka_robot:Robot_VelAct"
            },
        ),
    )
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(
        self,
        frame_skip=40,
        use_image_obs=False,
        imwidth=64,
        imheight=64,
        action_scale=1,
        use_workspace_limits=True,
        control_mode="primitives",
        use_grasp_rewards=False,
        target_mode=False,
        use_six_dof_dummy=False,
        collect_primitives_info=False,
        render_intermediate_obs_to_info=False,
        num_low_level_actions_per_primitive=10,
        reward_type="sparse",
    ):
        self.target_mode = target_mode
        self.control_mode = control_mode
        self.MODEL = self.CTLR_MODES_DICT[self.control_mode]["model"]
        self.ROBOTS = self.CTLR_MODES_DICT[self.control_mode]["robot"]

        self.episodic_cumulative_reward = 0
        self.obs_dict = {}
        self.use_grasp_rewards = use_grasp_rewards

        self.use_image_obs = use_image_obs
        self.imwidth = imwidth
        self.imheight = imheight
        self.action_scale = action_scale
        self.render_intermediate_obs_to_info = render_intermediate_obs_to_info
        self.collect_primitives_info = collect_primitives_info
        self.primitives_info = {}
        self.combined_prev_action = np.zeros(7)
        self.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive

        self._num_low_level_steps_total = 0

        if use_six_dof_dummy:
            self.primitive_idx_to_name = {
                0: "angled_x_y_grasp",
                1: "six_dof_delta",
                2: "rotate_about_y_axis",
                3: "lift",
                4: "drop",
                5: "move_left",
                6: "move_right",
                7: "move_forward",
                8: "move_backward",
                9: "open_gripper",
                10: "close_gripper",
                11: "rotate_about_x_axis",
            }
            self.primitive_name_to_func = dict(
                angled_x_y_grasp=self.angled_x_y_grasp,
                six_dof_delta=self.six_dof_delta,
                rotate_about_y_axis=self.rotate_about_y_axis,
                lift=self.lift,
                drop=self.drop,
                move_left=self.move_left,
                move_right=self.move_right,
                move_forward=self.move_forward,
                move_backward=self.move_backward,
                open_gripper=self.open_gripper,
                close_gripper=self.close_gripper,
                rotate_about_x_axis=self.rotate_about_x_axis,
            )
            self.primitive_name_to_action_idx = dict(
                angled_x_y_grasp=[0, 1, 2, 3],
                six_dof_delta=[4, 5, 6, 7, 8, 9],
                rotate_about_y_axis=10,
                lift=11,
                drop=12,
                move_left=13,
                move_right=14,
                move_forward=15,
                move_backward=16,
                rotate_about_x_axis=17,
                open_gripper=18,
                close_gripper=19,
            )
            self.max_arg_len = 20
        else:
            self.primitive_idx_to_name = {
                0: "angled_x_y_grasp",
                1: "move_delta_ee_pose",
                2: "rotate_about_y_axis",
                3: "lift",
                4: "drop",
                5: "move_left",
                6: "move_right",
                7: "move_forward",
                8: "move_backward",
                9: "open_gripper",
                10: "close_gripper",
                11: "rotate_about_x_axis",
            }
            self.primitive_idx_to_num_low_level_steps = {
                0: 1000,
                1: 300,
                2: 200,
                3: 300,
                4: 300,
                5: 300,
                6: 300,
                7: 300,
                8: 300,
                9: 200,
                10: 200,
                11: 200,
            }
            self.primitive_name_to_func = dict(
                angled_x_y_grasp=self.angled_x_y_grasp,
                move_delta_ee_pose=self.move_delta_ee_pose,
                rotate_about_y_axis=self.rotate_about_y_axis,
                lift=self.lift,
                drop=self.drop,
                move_left=self.move_left,
                move_right=self.move_right,
                move_forward=self.move_forward,
                move_backward=self.move_backward,
                open_gripper=self.open_gripper,
                close_gripper=self.close_gripper,
                rotate_about_x_axis=self.rotate_about_x_axis,
            )
            self.primitive_name_to_action_idx = dict(
                angled_x_y_grasp=[0, 1, 2, 3],
                move_delta_ee_pose=[4, 5, 6],
                rotate_about_y_axis=7,
                lift=8,
                drop=9,
                move_left=10,
                move_right=11,
                move_forward=12,
                move_backward=13,
                rotate_about_x_axis=14,
                open_gripper=15,
                close_gripper=16,
            )
            self.max_arg_len = 17

        self.num_primitives = len(self.primitive_name_to_func)

        self.min_ee_pos = np.array([-0.9, 0, 1.5])
        self.max_ee_pos = np.array([0.7, 1.5, 3.25])
        self.use_workspace_limits = use_workspace_limits

        self.tasks_to_complete = set(self.TASKS)
        self.reward_type = reward_type

        super().__init__(
            self.MODEL,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  # root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
            ),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),
        )

        if self.use_image_obs:
            self.image_shape = (3, imheight, imwidth)

            self.observation_space = spaces.Box(
                0, 255, (np.prod(self.image_shape),), dtype=np.uint8
            )
        else:
            obs_upper = 8.0 * np.ones(self.obs_dim)
            obs_lower = -obs_upper
            self.observation_space = spaces.Box(obs_lower, obs_upper, dtype=np.float32)

        if self.control_mode in ["joint_position", "joint_velocity", "torque"]:
            self.act_mid = np.zeros(self.N_DOF_ROBOT)
            self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)

            act_lower = -1 * np.ones((self.N_DOF_ROBOT,))
            act_upper = 1 * np.ones((self.N_DOF_ROBOT,))
            self.action_space = spaces.Box(act_lower, act_upper)
        elif self.control_mode == "end_effector":
            # 3 for xyz, 3 for rpy, 1 for gripper.
            act_lower = -1 * np.ones((7,))
            act_upper = 1 * np.ones((7,))
            self.action_space = spaces.Box(act_lower, act_upper)
        elif self.control_mode == "primitives":
            action_space_low = -self.action_scale * np.ones(self.max_arg_len)
            action_space_high = self.action_scale * np.ones(self.max_arg_len)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)

        if self.control_mode in ["primitives", "end_effector"]:
            self.reset_mocap_welds(self.sim)
            self.sim.forward()
            gripper_target = (
                np.array([-0.498, 0.005, -0.431 + 0.01]) + self.get_ee_pose()
            )
            gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
            self.set_mocap_pos("mocap", gripper_target)
            self.set_mocap_quat("mocap", gripper_rotation)
            for _ in range(10):
                self.sim.step()

        self.init_qpos = INIT_QPOS
        self.init_qvel = self.sim.model.key_qvel[0].copy()

    def get_ee_pose(self):
        return self.get_site_xpos("end_effector")

    def get_ee_6d_pose(self):
        ee_pos = self.get_ee_pose()
        ee_quat = self.get_ee_quat()
        ee_rpy = self.quat_to_rpy(ee_quat)
        return np.concatenate((ee_pos, ee_rpy))

    def get_ee_quat(self):
        return self.sim.data.body_xquat[10]

    def rpy_to_quat(self, rpy):
        q = quaternion.from_euler_angles(rpy)
        return np.array([q.x, q.y, q.z, q.w])

    def quat_to_rpy(self, q):
        q = quaternion.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_euler_angles(q)

    def convert_xyzw_to_wxyz(self, q):
        return np.array([q[3], q[0], q[1], q[2]])

    def get_site_xpos(self, name):
        id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xpos[id]

    def get_mocap_pos(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_quat[mocap_id] = value

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx

    def ctrl_set_action(self, action):
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
                body_idx = obj2_id
            else:
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def _set_action(self, action):
        assert action.shape == (9,)

        action = action.copy()
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        if self.control_mode == "primitives":
            pos_ctrl *= 0.05
        elif self.control_mode == "end_effector":
            pos_ctrl *= 0.02
            rot_ctrl *= 0.05
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        self.combined_prev_action += np.concatenate((pos_ctrl, rot_ctrl))
        if (self.primitive_step_counter + 1) % (
            self.num_low_level_steps // self.num_low_level_actions_per_primitive
        ) == 0:
            self.primitives_info["low_level_action"].append(
                np.concatenate([self.combined_prev_action, gripper_ctrl])
            )
            self.combined_prev_action = np.zeros_like(self.combined_prev_action)

        # Apply action to simulation.
        self.ctrl_set_action(action)
        self.mocap_set_action(self.sim, action)

    def call_render_every_step(self):
        if self.render_intermediate_obs_to_info:
            if (self.primitive_step_counter + 1) % (
                self.num_low_level_steps // self.num_low_level_actions_per_primitive
            ) == 0:
                obs = (
                    self.render(
                        "rgb_array",
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                    .transpose(2, 0, 1)
                    .flatten()
                )
                self.primitives_info["low_level_obs"].append(obs.astype(np.uint8))
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def execute_primitive(self, compute_action, num_iterations):
        for _ in range(num_iterations):
            action = compute_action()
            self._set_action(action)
            self.sim.step()
            self.call_render_every_step()
            self.primitive_step_counter += 1
            self._num_low_level_steps_total += 1
        return action

    def close_gripper(self, d):
        d = np.abs(d) * (0.04 / self.action_scale)
        compute_action = lambda: np.array([*np.zeros(7), -d, -d])
        self.execute_primitive(compute_action, 200)

    def open_gripper(
        self,
        d,
    ):
        d = np.abs(d) * (0.04 / self.action_scale)
        compute_action = lambda: np.array([*np.zeros(7), d, d])
        self.execute_primitive(compute_action, 200)

    def rotate_ee(self, rpy):
        gripper = self.sim.data.qpos[7:9]

        def compute_action():
            quat = self.rpy_to_quat(rpy)
            quat_delta = self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
            action = np.array(
                [
                    *np.zeros(3),
                    *quat_delta,
                    *gripper,
                ]
            )
            return action

        self.execute_primitive(compute_action, 200)

    def goto_pose(self, pose):
        gripper = self.sim.data.qpos[7:9]
        if self.use_workspace_limits:
            pose = np.clip(pose, self.min_ee_pos, self.max_ee_pos)

        def compute_action():
            delta = pose - self.get_ee_pose()
            action = np.array([*delta, *np.zeros(4), *gripper])
            return action

        self.execute_primitive(compute_action, 300)

    def six_dof_delta(self, xyzrpy):
        gripper = self.sim.data.qpos[7:9]
        target_pos = xyzrpy[:3] + self.get_ee_pose()
        target_rpy = self.quat_to_rpy(self.get_ee_quat()) - xyzrpy[3:6]
        if self.use_workspace_limits:
            target_pos = np.clip(target_pos, self.min_ee_pos, self.max_ee_pos)

        def compute_action():
            pos_delta = target_pos - self.get_ee_pose()
            quat = self.rpy_to_quat(target_rpy)
            quat_delta = self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
            action = np.array([*pos_delta, *quat_delta, *gripper])
            return action

        self.execute_primitive(compute_action, 300)

    def rotate_about_x_axis(self, angle):
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([angle, 0, 0])
        self.rotate_ee(rotation)

    def angled_x_y_grasp(self, angle_and_xyd):
        angle, x_dist, y_dist, d_dist = angle_and_xyd
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([angle, 0, 0])
        self.rotate_ee(rotation)
        self.goto_pose(self.get_ee_pose() + np.array([x_dist, 0.0, 0]))
        self.goto_pose(self.get_ee_pose() + np.array([0.0, y_dist, 0]))
        self.close_gripper(d_dist)

    def move_delta_ee_pose(self, pose):
        self.goto_pose(self.get_ee_pose() + pose)

    def rotate_about_y_axis(self, angle):
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array([0, 0, angle])
        self.rotate_ee(rotation)

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, 0.0, z_dist]))

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, 0.0, -z_dist]))

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([-x_dist, 0.0, 0.0]))

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([x_dist, 0.0, 0.0]))

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, y_dist, 0.0]))

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(self.get_ee_pose() + np.array([0.0, -y_dist, 0.0]))

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def get_primitive_info_from_high_level_action(self, hl):
        primitive_idx, primitive_args = (
            np.argmax(hl[: self.num_primitives]),
            hl[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        return primitive_name, primitive_args, primitive_idx

    def act(self, a):
        if not self.initializing:
            a = a * self.action_scale
            a = np.clip(a, self.action_space.low, self.action_space.high)
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        self.num_low_level_steps = self.primitive_idx_to_num_low_level_steps[
            primitive_idx
        ]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        primitive(
            primitive_action,
        )

    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    def step(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.set_render_every_step(render_every_step, render_mode, render_im_shape)
        if not self.initializing:
            if self.control_mode in [
                "joint_position",
                "joint_velocity",
                "torque",
                "end_effector",
            ]:
                a = np.clip(a, -1.0, 1.0)
                if self.control_mode == "end_effector":
                    if self.target_mode:
                        target_pos = a[:3]
                        target_pos = np.clip(
                            target_pos, self.min_ee_pos, self.max_ee_pos
                        )
                        quat = self.rpy_to_quat(a[3:6])
                        target_gripper_pos = a[-1]
                        for _ in range(32):
                            quat_delta = (
                                self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
                            )
                            a[:3] = target_pos - self.get_ee_pose()
                            gripper_pos = self.sim.data.qpos[8]
                            gripper_ctrl = target_gripper_pos - gripper_pos
                            gripper_ctrl = a[-1]
                            self._set_action(
                                np.concatenate(
                                    [a[:3], quat_delta, [gripper_ctrl, -gripper_ctrl]]
                                )
                            )
                            self.sim.step()
                    else:
                        rotation = self.quat_to_rpy(self.get_ee_quat()) - np.array(
                            a[3:6]
                        )
                        target_pos = a[:3] + self.get_ee_pose()
                        target_pos = np.clip(
                            target_pos, self.min_ee_pos, self.max_ee_pos
                        )
                        a[:3] = target_pos - self.get_ee_pose()
                        for _ in range(32):
                            quat = self.rpy_to_quat(rotation)
                            quat_delta = (
                                self.convert_xyzw_to_wxyz(quat) - self.get_ee_quat()
                            )
                            self._set_action(
                                np.concatenate([a[:3], quat_delta, [a[-1], -a[-1]]])
                            )
                            self.sim.step()
                else:
                    if self.control_mode == "joint_velocity":
                        a = self.act_mid + a * self.act_amp  # mean center and scale
                    self.robot.step(
                        self, a, step_duration=self.skip * self.model.opt.timestep
                    )
            else:
                if render_every_step and render_mode == "rgb_array":
                    self.img_array = []
                self.img_array = []
                self.primitives_info = {}
                self.primitives_info["low_level_action"] = []
                self.primitives_info["low_level_obs"] = []
                self.primitive_step_counter = 0
                self.combined_prev_action = np.zeros_like(self.combined_prev_action)
                self.act(a)

        obs = self._get_obs()

        # rewards
        reward = self.get_reward(self.obs_dict)

        # termination
        done = not self.tasks_to_complete

        # finalize step
        env_info = {}
        self.unset_render_every_step()
        if self.collect_primitives_info and not self.initializing:
            self.primitives_info["low_level_obs"] = np.array(
                self.primitives_info["low_level_obs"]
            )
            self.primitives_info["low_level_action"] = np.array(
                self.primitives_info["low_level_action"]
            )
            env_info.update(self.primitives_info)

        self.update_info(env_info)
        return obs, reward, done, env_info

    def reset_model(self):
        self.tasks_to_complete = set(self.TASKS)
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        if self.control_mode in ["primitives", "end_effector"]:
            self.reset_mocap2body_xpos(self.sim)

        return self._get_obs()

    def close(self):
        self.robot.close()

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.set_mocap_pos("mocap", mocap_pos)
        self.set_mocap_quat("mocap", mocap_quat)
        self.sim.forward()

    def render(self, mode="human", imwidth=64, imheight=64):
        if mode == "rgb_array":
            if self.sim_robot._use_dm_backend:
                camera = engine.MovableCamera(self.sim, imwidth, imheight)
                camera.set_pose(
                    distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
                )
                img = camera.render()
            else:
                img = self.sim_robot.renderer.render_offscreen(
                    imwidth,
                    imheight,
                )
            return img
        else:
            super().render(mode=mode)

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(self, robot_noise_ratio=0.1)

        self.obs_dict = {}
        self.obs_dict["qp"] = qp
        self.obs_dict["qv"] = qv
        self.obs_dict["obj_qp"] = obj_qp
        self.obs_dict["obj_qv"] = obj_qv
        if self.use_image_obs:
            img = self.render(mode="rgb_array")
            img = img.transpose(2, 0, 1).flatten()
            return img
        else:
            if self.control_mode == "end_effector":
                return np.concatenate(
                    [
                        self.get_ee_6d_pose(),
                        qp[8:10],
                        self.obs_dict["obj_qp"],
                    ]
                )
            else:
                return np.concatenate(
                    [
                        self.obs_dict["qp"],
                        self.obs_dict["obj_qp"],
                    ]
                )

    def compute_grasp_rewards(self, task):
        if task == "slide cabinet":
            is_grasped = False
            for handle_idx in range(1, 6):
                obj_pos = self.get_site_xpos("schandle{}".format(handle_idx))
                left_pad = self.get_site_xpos("leftpad")
                right_pad = self.get_site_xpos("rightpad")
                within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.07
                within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.07
                right = right_pad[0] < obj_pos[0]
                left = obj_pos[0] < left_pad[0]
                if right and left and within_sphere_right and within_sphere_left:
                    is_grasped = True
        if task == "top left burner":
            is_grasped = False
            for handle_idx in range(1, 4):
                obj_pos = self.get_site_xpos("tlbhandle{}".format(handle_idx))
                left_pad = self.get_site_xpos("leftpad")
                right_pad = self.get_site_xpos("rightpad")
                within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
                within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
                right = right_pad[0] < obj_pos[0]
                left = obj_pos[0] < left_pad[0]
                if within_sphere_right and within_sphere_left and right and left:
                    is_grasped = True
        if task == "microwave":
            is_grasped = False
            for handle_idx in range(1, 6):
                obj_pos = self.get_site_xpos("mchandle{}".format(handle_idx))
                left_pad = self.get_site_xpos("leftpad")
                right_pad = self.get_site_xpos("rightpad")
                within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
                within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
                if (
                    right_pad[0] < obj_pos[0]
                    and obj_pos[0] < left_pad[0]
                    and within_sphere_right
                    and within_sphere_left
                ):
                    is_grasped = True
        if task == "hinge cabinet":
            is_grasped = False
            for handle_idx in range(1, 6):
                obj_pos = self.get_site_xpos("hchandle{}".format(handle_idx))
                left_pad = self.get_site_xpos("leftpad")
                right_pad = self.get_site_xpos("rightpad")
                within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
                within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
                if (
                    right_pad[0] < obj_pos[0]
                    and obj_pos[0] < left_pad[0]
                    and within_sphere_right
                ):
                    is_grasped = True
        if task == "light switch":
            is_grasped = False
            for handle_idx in range(1, 4):
                obj_pos = self.get_site_xpos("lshandle{}".format(handle_idx))
                left_pad = self.get_site_xpos("leftpad")
                right_pad = self.get_site_xpos("rightpad")
                within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
                within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
                if within_sphere_right and within_sphere_left:
                    is_grasped = True
        return is_grasped

    def get_reward(self, obs_dict):
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        completed_tasks = []
        dense_reward = 0
        for task in self.tasks_to_complete:
            task_idx = OBS_TASK_INDICES[task]
            distance = np.linalg.norm(
                next_obj_obs[..., task_idx - idx_offset] - OBS_TASK_GOALS[task]
            )
            dense_reward += -1 * distance  # Reward must be negative distance for RL.
            is_grasped = True
            if not self.initializing and self.use_grasp_rewards:
                is_grasped = self.compute_grasp_rewards()
            task_complete = distance < self.REWARD_THRESH and is_grasped
            if task_complete:
                completed_tasks.append(task)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(task) for task in completed_tasks]
        sparse_reward = float(len(completed_tasks))
        if self.reward_type == "dense":
            reward = dense_reward
        else:
            reward = sparse_reward
        return reward

    def update_info(self, info):
        next_q_obs = self.obs_dict["qp"]
        next_obj_obs = self.obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        for task in OBS_TASK_INDICES.keys():
            task_idx = OBS_TASK_INDICES[task]
            distance = np.linalg.norm(
                next_obj_obs[..., task_idx - idx_offset] - OBS_TASK_GOALS[task]
            )
            success = float(distance < self.REWARD_THRESH)
            if len(self.TASKS) == 1 and self.TASKS[0] == task:
                info["success"] = success
        info["num low level steps"] = self._num_low_level_steps_total // 32
        info["num low level steps true"] = self._num_low_level_steps_total
        self._num_low_level_steps_total = 0
        return info
