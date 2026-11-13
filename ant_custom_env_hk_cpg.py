import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
# Environment combining MuJoCo physics with Homeokinesis (HK) for Ant robot
class AntPositionHKEnv(MujocoEnv, utils.EzPickle):

    metadata = {"render_modes": ["human"], "render_fps": 20} # rendering settings

    def __init__(self, **kwargs):
        # Enables easy saving/loading of environment parameters
        utils.EzPickle.__init__(self, **kwargs)

        obs = Box(-np.inf, np.inf, shape=(111,), dtype=np.float64)  # Observation space: all robot positions, velocities, and external forces

        MujocoEnv.__init__( # Initialize MuJoCo environment
            self,
            "./ant.xml", # MuJoCo XML robot model
            frame_skip=5, # how many physics steps per control step
            observation_space=obs,
            **kwargs
        )

        # Gait parameters (diagonal trot)
        self.freq = 1.0 # gait frequency
        self.hip = 0.6 # hip joint amplitude
        self.knee = -0.9 # knee joint amplitude

        # Homeokinesis
        self.C = np.eye(8) * 0.05 # adaptive weight matrix for motor coordination #modification
        self.h = np.zeros(8) # bias term for HK
        self.lr = 1e-5  # learning rate for HK adaptation
        self.last_act = np.zeros(8) # store last motor commands for HK update

        # Base area for soft reflective walls
        self.x_min, self.x_max = -5.0, 5.0
        self.y_min, self.y_max = -5.0, 5.0
        self.edge_margin = 0.5 # distance before hitting wall
        self.edge_push = 0.3 # rebound factor at edges

        # Random heading for roaming
        self.heading = 0.0
        self.heading_change = 0.02
        self.max_yaw_step = 0.005

        # Torso stabilization gains
        self.roll_gain = 1.0
        self.pitch_gain = 1.0

        # Slightly lift torso to prevent initial fall
        self.init_qpos[2] = 0.25

    def _get_obs(self): # Construct the full observation vector
        return np.concatenate([
            self.data.qpos.flat[2:], # all positions except global x,y
            self.data.qvel.flat,  # joint velocities
            self.data.cfrc_ext.flat  # external contact forces
        ])

    def reset_model(self): # Reset robot to initial state
        self.set_state(self.init_qpos, self.init_qvel) # reset pose and velocity
        self.h[:] = 0 # reset HK bias
        self.C[:] = np.eye(8) * 0.02 # reset HK weights
        self.last_act[:] = 0 # reset last actions
        self.heading = np.random.uniform(-np.pi, np.pi) # random heading
        return self._get_obs()

    # Diagonal trot gait
    def gait(self, t):
        s = np.sin(2 * np.pi * self.freq * t) # oscillation signal
        a = np.zeros(8)   # 8 motors: 4 hips + 4 knees
        # hips : front-left & rear-right in phase, front-right & rear-left opposite
        a[0] = self.hip * s
        a[3] = self.hip * s
        a[1] = -self.hip * s
        a[2] = -self.hip * s
        # knees :  only move when leg is in stance/air (positive/negatives)
        a[4] = self.knee * np.maximum(0, s)
        a[7] = self.knee * np.maximum(0, s)
        a[5] = self.knee * np.maximum(0, -s)
        a[6] = self.knee * np.maximum(0, -s)
        return a

    # Homeokinesis motor adaptation
    def hk(self, obs, target):
        motor = self.last_act.copy() # current motor output
        err = motor - target # difference between last action and target
        self.C += self.lr * np.outer(err, motor) # update HK weight matrix
        self.C = np.clip(self.C, -0.05, 0.05) # constrain weights
        self.h += self.lr * err # update bias
        self.h = np.clip(self.h, -0.2, 0.2) # constrain bias
        return target + self.C @ motor + self.h # combined output: target + HK modulation

    def step(self, action=None): # --- Step function for environment ---
        obs = self._get_obs()
        t = self.data.time

        # Base gait + HK
        base = self.gait(t)

        # --- persistent random gait variations ---
        base += np.random.uniform(-0.03, 0.03, size=8)

        act = self.hk(obs, base) # --- Apply HK ---
 
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

        act = np.clip(act, -1.2, 1.2) # limit motor commands
        self.last_act[:] = act

        self.do_simulation(act, self.frame_skip) # --- Apply actions in MuJoCo ---

        # --- soft reflective walls ---
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # X-axis walls
        if qpos[0] > self.x_max - self.edge_margin:
            qvel[0] *= -self.edge_push
            self.heading = np.pi
        elif qpos[0] < self.x_min + self.edge_margin:
            qvel[0] *= -self.edge_push
            self.heading = 0
        # Y-axis walls
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
        # Important as Homeokinesis (HK) continuously adapts its weights C and bias h based on motor errors.
        # Over time, these values can grow too large, causing unstable or saturated motor outputs.
        # Periodic HK decay multiplies C and h by a factor 0.995 at every 2 minutes to slowly forget older adaptations, keeping the HK system stable while still allowing learning.
        if int(t) % 120 == 0:  # every 2 minutes,slowly reduce HK weights and bias to prevent runaway values
            self.C *= 0.995
            self.h *= 0.995

        if self.render_mode == "human":
            self.render()

        return obs, 0, False, False, {}
