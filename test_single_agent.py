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
import csv
import os

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3 import PPO


bluesky_gym.register_envs()


ENV_NAME = "SingleAgentEnv-v0"

MODEL_PATH = (
    "single_agent_model"
)

class CompetitionMetricsCallback(
    BaseCallback
):
    """
    Save final competition metrics for every training episode.
    """

    def __init__(
        self,
        csv_path="training_metrics.csv",
        verbose=0,
    ):

        super().__init__(
            verbose
        )

        self.csv_path = csv_path
        self.episode_index = 0

        self.fieldnames = [
            "episode",
            "total_timesteps",

            "waypoint_reached",
            "flight_time",

            "intrusion_events",
            "intrusion_time",

            "restricted_area_events",
            "time_in_restricted_area",

            "sector_exit_events",
            "time_outside_sector",

            "total_reward",
        ]

    def _on_training_start(
        self
    ):

        directory = os.path.dirname(
            os.path.abspath(
                self.csv_path
            )
        )

        os.makedirs(
            directory,
            exist_ok=True
        )

        with open(
            self.csv_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=
                    self.fieldnames
            )

            writer.writeheader()

    def _on_step(
        self
    ):

        infos = self.locals.get(
            "infos",
            []
        )

        dones = self.locals.get(
            "dones",
            []
        )

        for info, done in zip(
            infos,
            dones
        ):

            if not done:
                continue

            row = {
                "episode":
                    self.episode_index,

                "total_timesteps":
                    self.num_timesteps,

                "waypoint_reached":
                    info.get(
                        "waypoint_reached",
                        0
                    ),

                "flight_time":
                    info.get(
                        "flight_time",
                        0.0
                    ),

                "intrusion_events":
                    info.get(
                        "intrusion_events",
                        0
                    ),

                "intrusion_time":
                    info.get(
                        "intrusion_time",
                        0.0
                    ),

                "restricted_area_events":
                    info.get(
                        "restricted_area_events",
                        0
                    ),

                "time_in_restricted_area":
                    info.get(
                        "time_in_restricted_area",
                        0.0
                    ),

                "sector_exit_events":
                    info.get(
                        "sector_exit_events",
                        0
                    ),

                "time_outside_sector":
                    info.get(
                        "time_outside_sector",
                        0.0
                    ),

                "total_reward":
                    info.get(
                        "total_reward",
                        0.0
                    ),
            }

            with open(
                self.csv_path,
                "a",
                newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=
                        self.fieldnames
                )

                writer.writerow(
                    row
                )

            self.episode_index += 1

        return True


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

    metrics_callback = (
        CompetitionMetricsCallback(
            csv_path=
                "training_metrics.csv"
        )
    )

    model.learn(
        total_timesteps=
            total_timesteps,
        callback=
        metrics_callback
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