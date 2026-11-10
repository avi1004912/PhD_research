import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.75)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class AntCustomEnv(MujocoEnv, utils.EzPickle):
    """
    Custom Ant environment (stable walking, no vibration).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        healthy_z_range=(0.2, 1.0),
        terminate_when_unhealthy=True,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(-np.inf, np.inf, shape=(111,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            "ant.xml",
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[2:],   # remove x,y
                self.data.qvel.flat,
                self.data.cfrc_ext.flat,
            ]
        )

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        return self._healthy_z_range[0] < z < self._healthy_z_range[1]

    @property
    def terminated(self):
        return not self.is_healthy if self._terminate_when_unhealthy else False

    def step(self, action):
        xy_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_after = mass_center(self.model, self.data)

        velocity = (xy_after - xy_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        reward = forward_reward + self._healthy_reward - ctrl_cost

        obs = self._get_obs()
        terminated = self.terminated

        info = {
            "x_position": xy_after[0],
            "x_velocity": velocity[0],
            "reward_forward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
