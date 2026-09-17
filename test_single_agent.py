"""
Minimal training script for the single-agent competition environment.

Run:
    python train_single.py train [total_timesteps]
    python train_single.py play [model_path]
"""

import sys

import gymnasium as gym
import bluesky_gym
import bluesky_gym.envs

from stable_baselines3 import PPO


bluesky_gym.register_envs()

ENV_NAME = "SingleAgentEnv-v0"
MODEL_PATH = "single_agent_model"


def train(total_timesteps=250_000):
    """
    Train a PPO model and save it.
    """

    env = gym.make(
        ENV_NAME,
        render_mode=None
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps
    )

    model.save(MODEL_PATH)

    env.close()


def play(model_path=MODEL_PATH, n_episodes=3):
    """
    Load the trained model and render its behaviour.
    """

    model = PPO.load(model_path)

    env = gym.make(
        ENV_NAME,
        render_mode="human"
    )

    for episode in range(n_episodes):

        obs, info = env.reset(seed=episode)

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"episode {episode}: "
            f"terminated={terminated} "
            f"truncated={truncated} "
            f"total Reward={reward} "
            f"info={info}"
        )

    env.close()


if __name__ == "__main__":

    args = sys.argv[1:]

    if args and args[0] == "play":

        model_path = (
            args[1]
            if len(args) > 1
            else MODEL_PATH
        )

        play(model_path)

    else:

        total_timesteps = (
            int(args[1])
            if len(args) > 1
            else 250_000
        )

        train(total_timesteps)