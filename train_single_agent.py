"""Train reproducible single-agent PPO runs, or play a saved model.

    python train_single_agent.py train 250000 --seed 0
    python train_single_agent.py curriculum --seed 0
    python train_single_agent.py play single_agent_model

Each training run gets its own model directory, manifest, logs and checkpoints.
"""

import argparse
import csv
from pathlib import Path
import time

import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from scripts.evaluate_competition import METRIC_KEYS
from scripts.experiment_support import (
    create_run, environment_config, failure_reasons, file_hash, provenance,
    validate_environment, write_json,
)

ENV_NAME = "SingleAgentEnv-v0"
MODEL_PATH = "single_agent_model"


class EpisodeMetricsCallback(BaseCallback):
    """Log objective metrics at episode ends, alongside the training return."""

    def __init__(self, path, append=False):
        super().__init__()
        self.path = path
        self.append = append
        self.stream = None

    def _on_training_start(self):
        self.close()
        path = Path(self.path)
        needs_header = not self.append or not path.exists() or path.stat().st_size == 0
        self.stream = path.open("a" if self.append else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=[
            "timesteps", "safe_goal_completion", *METRIC_KEYS,
        ])
        if needs_header:
            self.writer.writeheader()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.writer.writerow({
                    "timesteps": self.num_timesteps,
                    "safe_goal_completion": not failure_reasons(info),
                    **{key: info[key] for key in METRIC_KEYS},
                })
                self.stream.flush()
        return True

    def close(self):
        if self.stream is not None:
            self.stream.close()


def training_config(total_timesteps, seed, checkpoint_every, learning_rate,
                    n_steps, batch_size, gamma, gae_lambda, device):
    if seed < 0 or seed == 42:
        raise ValueError("Use a nonnegative training seed other than the official evaluation seed 42")
    if total_timesteps < 1 or checkpoint_every < 1 or n_steps < 2:
        raise ValueError("Timesteps/checkpoint interval must be positive and n_steps must be at least 2")
    if batch_size < 2 or batch_size > n_steps or n_steps % batch_size:
        raise ValueError("Batch size must be at least 2 and divide n_steps")
    if learning_rate <= 0 or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Invalid PPO learning rate, gamma or GAE lambda")
    return dict(learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                gamma=gamma, gae_lambda=gae_lambda, seed=seed, device=device)


def train(total_timesteps=250_000, seed=0, out_dir=None, checkpoint_every=50_000,
          learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.99,
          gae_lambda=0.95, device="cpu"):
    config = training_config(total_timesteps, seed, checkpoint_every, learning_rate,
                             n_steps, batch_size, gamma, gae_lambda, device)
    run_dir = create_run("models/single_agent", f"ppo_seed{seed}", out_dir)
    manifest = {**provenance(run_dir), "status": "running", "algorithm": "PPO",
                "policy": "MultiInputPolicy", "ppo": config,
                "requested_timesteps": total_timesteps, "checkpoint_every": checkpoint_every}
    write_json(run_dir / "manifest.json", manifest)
    env = None
    episode_log = EpisodeMetricsCallback(run_dir / "episodes.csv")
    model = None
    started = time.perf_counter()
    try:
        bluesky_gym.register_envs()
        env = gym.make(ENV_NAME, render_mode=None)
        validate_environment(env)
        manifest["environment"] = environment_config(env)
        write_json(run_dir / "manifest.json", manifest)
        env = Monitor(env, filename=str(run_dir / "monitor.csv"), info_keywords=tuple(METRIC_KEYS))
        model = PPO("MultiInputPolicy", env, verbose=1, **config)
        model.set_logger(configure(str(run_dir), ["stdout", "csv"]))
        checkpoint = CheckpointCallback(checkpoint_every, str(run_dir / "checkpoints"), "ppo")
        model.learn(total_timesteps=total_timesteps, callback=[episode_log, checkpoint])
        model.save(run_dir / "model.zip")
        manifest.update(status="completed", actual_timesteps=model.num_timesteps,
                        model_sha256=file_hash(run_dir / "model.zip"))
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        if model is not None:
            model.save(run_dir / "interrupted_model.zip")
            manifest["actual_timesteps"] = model.num_timesteps
        raise
    finally:
        episode_log.close()
        if env is not None:
            env.close()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(run_dir / "manifest.json", manifest)
        print(f"Training artifacts: {run_dir}", flush=True)
    return run_dir


def play(model_path=MODEL_PATH, n_episodes=3):
    model = PPO.load(model_path, device="cpu")
    bluesky_gym.register_envs()
    env = gym.make(ENV_NAME, render_mode="human")
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
            print(f"episode {episode}: terminated={terminated} truncated={truncated} "
                  f"total_reward={info['total_reward']} info={info}")
    finally:
        env.close()


def add_training_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", help="New run directory; defaults to models/single_agent")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("train")
    training.add_argument("total_timesteps", type=int, nargs="?", default=250_000)
    add_training_arguments(training)
    curriculum = commands.add_parser("curriculum", help="Train one PPO model through progressively harder scenarios")
    add_training_arguments(curriculum)
    curriculum.add_argument("--stage-budget", type=int, default=250_000, help="Maximum requested steps per stage")
    curriculum.add_argument("--chunk-steps", type=int, default=50_000, help="Steps between stage checks")
    curriculum.add_argument("--min-stage-steps", type=int, default=100_000)
    curriculum.add_argument("--eval-episodes", type=int, default=20)
    curriculum.add_argument("--eval-seed", type=int, default=1234)
    curriculum.add_argument("--goal-threshold", type=float, default=0.95)
    curriculum.add_argument("--safe-threshold", type=float, default=0.80)
    curriculum.add_argument("--passes-required", type=int, default=2, help="Consecutive checkpoint checks needed to advance")
    playback = commands.add_parser("play")
    playback.add_argument("model_path", nargs="?", default=MODEL_PATH)
    playback.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    options = vars(args)
    command = options.pop("command")
    if command == "play":
        play(options["model_path"], options["episodes"])
    elif command == "curriculum":
        from scripts.train_curriculum import train_curriculum
        train_curriculum(**options)
    else:
        train(**options)


if __name__ == "__main__":
    main()
