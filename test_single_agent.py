"""
Training script for the new continuous navigation MDP.

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

MODEL_PATH = (
    "single_agent_model_navigation"
)


def train(
    total_timesteps=50_000
):

    env = gym.make(
        ENV_NAME,
        render_mode=None,
    )

    model = PPO(
        "MultiInputPolicy",
        env,

        learning_rate=3e-4,

        n_steps=1024,
        batch_size=256,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.0,

        policy_kwargs=dict(

            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            ),

            log_std_init=-0.5,
        ),

        verbose=1,
    )

    model.learn(
        total_timesteps=
            total_timesteps
    )

    model.save(
        MODEL_PATH
    )

    env.close()


def play(
    model_path=MODEL_PATH,
    n_episodes=10,
):

    model = PPO.load(
        model_path
    )

    env = gym.make(
        ENV_NAME,
        render_mode="human",
    )

    raw_env = env.unwrapped

    for episode in range(
        n_episodes
    ):

        obs, info = env.reset(
            seed=episode
        )

        terminated = False
        truncated = False

        step = 0

        while not (
            terminated
            or truncated
        ):

            action, _ = (
                model.predict(
                    obs,
                    deterministic=True,
                )
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            print(
                f"step={step} "
                f"mode="
                f"{raw_env.last_navigation_mode} "
                f"action="
                f"[{action[0]:+.2f}, "
                f"{action[1]:+.2f}] "
                f"ref="
                f"{raw_env.last_reference_heading:+.1f}° "
                f"target_hdg="
                f"{raw_env.last_target_heading:+.1f}° "
                f"target_spd="
                f"{raw_env.last_target_speed:.1f}kt"
            )

            step += 1

        print(
            f"\nEpisode {episode}: "
            f"terminated="
            f"{terminated} "
            f"truncated="
            f"{truncated}\n"
            f"{info}\n"
        )

    env.close()


if __name__ == "__main__":

    args = sys.argv[1:]

    if (
        args
        and args[0]
        == "play"
    ):

        model_path = (
            args[1]
            if len(args) > 1
            else MODEL_PATH
        )

        play(
            model_path
        )

    else:

        total_timesteps = (
            int(args[1])
            if len(args) > 1
            else 50_000
        )

        train(
            total_timesteps
        )