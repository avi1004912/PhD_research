import gymnasium as gym
from ant_custom_env_hk_cpg import AntPositionHKEnv

gym.register(
    id="AntScriptedHK-v0",
    entry_point="ant_custom_env_hk_cpg:AntPositionHKEnv",
)
