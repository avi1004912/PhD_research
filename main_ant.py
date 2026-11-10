import register_ant 

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
import os
import argparse

 # IMPORTANT

model_dir = "models_ant"
log_dir = "logs_ant"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, algo):
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=8192,
            batch_size=256,
            ent_coef=0.05,
        )
    elif algo == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError("Unsupported algo")

    TIMESTEPS = 550_000
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"{model_dir}/{algo}_ant_550k")


def test(env, algo, model_path):
    model = PPO.load(model_path, env=env) if algo == "PPO" else SAC.load(model_path, env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gymenv")
    parser.add_argument("algo")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-s", "--test")
    args = parser.parse_args()

    if args.train:
        env = gym.make(args.gymenv)
        train(env, args.algo)

    if args.test:
        env = gym.make(args.gymenv, render_mode="human")
        test(env, args.algo, args.test)
