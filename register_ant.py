import gymnasium as gym
from ant_custom_env import AntCustomEnv

gym.register(
    id="AntCustom-v0",
    entry_point="ant_custom_env:AntCustomEnv",
)
