import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class AntPositionHKEnv(MujocoEnv, utils.EzPickle):

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        obs = Box(-np.inf, np.inf, shape=(111,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            "./ant.xml",
            frame_skip=5,
            observation_space=obs,
            **kwargs
        )

        # Gait parameters
        self.freq = 1.0
        self.hip = 0.6
        self.knee = -0.9

        # Homeokinesis
        self.C = np.eye(8) * 0.02
        self.h = np.zeros(8)
        self.lr = 1e-5
        self.last_act = np.zeros(8)

        # Base area
        self.x_min, self.x_max = -5.0, 5.0
        self.y_min, self.y_max = -5.0, 5.0
        self.edge_margin = 0.5
        self.edge_push = 0.3

        # Random heading for roaming
        self.heading = 0.0
        self.heading_change = 0.02
        self.max_yaw_step = 0.005

        # Torso stabilization gains
        self.roll_gain = 1.0
        self.pitch_gain = 1.0

        # Slightly lift torso to prevent initial fall
        self.init_qpos[2] = 0.25

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            self.data.cfrc_ext.flat
        ])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        self.h[:] = 0
        self.C[:] = np.eye(8) * 0.02
        self.last_act[:] = 0
        self.heading = np.random.uniform(-np.pi, np.pi)
        return self._get_obs()

    # Diagonal trot gait
    def gait(self, t):
        s = np.sin(2 * np.pi * self.freq * t)
        a = np.zeros(8)
        # hips
        a[0] = self.hip * s
        a[3] = self.hip * s
        a[1] = -self.hip * s
        a[2] = -self.hip * s
        # knees
        a[4] = self.knee * np.maximum(0, s)
        a[7] = self.knee * np.maximum(0, s)
        a[5] = self.knee * np.maximum(0, -s)
        a[6] = self.knee * np.maximum(0, -s)
        return a

    # Homeokinesis
    def hk(self, obs, target):
        motor = self.last_act.copy()
        err = motor - target
        self.C += self.lr * np.outer(err, motor)
        self.C = np.clip(self.C, -0.05, 0.05)
        self.h += self.lr * err
        self.h = np.clip(self.h, -0.2, 0.2)
        return target + self.C @ motor + self.h

    def step(self, action=None):
        obs = self._get_obs()
        t = self.data.time

        # Base gait + HK
        base = self.gait(t)

        # --- persistent random gait variations ---
        base += np.random.uniform(-0.03, 0.03, size=8)

        act = self.hk(obs, base)

        # Torso stabilization + damping
        roll = self.data.qpos[3]
        pitch = self.data.qpos[4]
        roll_vel = self.data.qvel[3]
        pitch_vel = self.data.qvel[4]

        act[0] -= self.roll_gain * roll + 0.5 * roll_vel
        act[1] -= self.roll_gain * roll + 0.5 * roll_vel
        act[2] -= self.pitch_gain * pitch + 0.5 * pitch_vel
        act[3] -= self.pitch_gain * pitch + 0.5 * pitch_vel

        # --- small random torque kicks ---
        act += np.random.uniform(-0.01, 0.01, size=8)

        act = np.clip(act, -1.2, 1.2)
        self.last_act[:] = act

        self.do_simulation(act, self.frame_skip)

        # --- soft reflective walls ---
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # X-axis
        if qpos[0] > self.x_max - self.edge_margin:
            qvel[0] *= -self.edge_push
            self.heading = np.pi
        elif qpos[0] < self.x_min + self.edge_margin:
            qvel[0] *= -self.edge_push
            self.heading = 0
        # Y-axis
        if qpos[1] > self.y_max - self.edge_margin:
            qvel[1] *= -self.edge_push
            self.heading = -np.pi/2
        elif qpos[1] < self.y_min + self.edge_margin:
            qvel[1] *= -self.edge_push
            self.heading = np.pi/2

        self.set_state(qpos, qvel)

        # --- slow drifting heading ---
        self.heading += np.random.uniform(-self.heading_change, self.heading_change)

        # smooth yaw rotation
        current_yaw = self.data.qpos[5]
        yaw_diff = self.heading - current_yaw
        yaw_step = np.clip(yaw_diff, -self.max_yaw_step, self.max_yaw_step)
        qpos = self.data.qpos.copy()
        qpos[5] += yaw_step
        self.set_state(qpos, self.data.qvel)

        # --- periodic HK decay ---
        if int(t) % 120 == 0:  # every 2 minutes
            self.C *= 0.995
            self.h *= 0.995

        if self.render_mode == "human":
            self.render()

        return obs, 0, False, False, {}
