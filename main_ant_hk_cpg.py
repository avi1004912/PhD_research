from ant_custom_env_hk_cpg import AntPositionHKEnv

env = AntPositionHKEnv(render_mode="human")

obs, _ = env.reset()

while True:
    obs, _, _, _, _ = env.step(None)
