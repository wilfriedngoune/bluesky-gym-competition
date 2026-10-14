"""Train a single PPO model through gated, training-only curriculum stages."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from scripts.curriculum import STAGES, gate_passed, make_training_env, set_training_stage
from scripts.evaluate_competition import METRIC_KEYS
from scripts.experiment_support import (
    REPO_ROOT, create_run, environment_config, file_hash, provenance, write_json,
)
from train_single_agent import EpisodeMetricsCallback, training_config


def checkpoint_evaluation(model, output, episodes, seed, stage=None):
    """BlueSky is process-global: evaluate saved checkpoints in a child process."""
    if stage is None:
        args = ["-m", "scripts.evaluate_single_agent", "--models", str(model),
                "--episodes", str(episodes), "--seed", str(seed), "--out-dir", str(output)]
    else:
        args = ["-m", "scripts.evaluate_curriculum_stage", "--model", str(model),
                "--stage", stage.name, "--episodes", str(episodes),
                "--seed", str(seed), "--out-dir", str(output)]
    variables = os.environ.copy()
    variables.setdefault("SDL_VIDEODRIVER", "dummy")
    variables.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    log = output.with_suffix(".log")
    with log.open("w", encoding="utf-8") as stream:
        try:
            subprocess.run([sys.executable, *args], cwd=REPO_ROOT, env=variables,
                           stdout=stream, stderr=subprocess.STDOUT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Checkpoint evaluation failed; see {log}") from error
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise RuntimeError(f"Checkpoint evaluation did not complete: {output}")
    if stage is None:
        comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        return comparison[0]
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def train_curriculum(seed=0, out_dir=None, checkpoint_every=50_000,
                     learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.99,
                     gae_lambda=0.95, device="cpu", stage_budget=250_000,
                     chunk_steps=50_000, min_stage_steps=100_000, eval_episodes=20,
                     eval_seed=1234, goal_threshold=0.95, safe_threshold=0.80,
                     passes_required=2):
    config = training_config(stage_budget, seed, checkpoint_every, learning_rate,
                             n_steps, batch_size, gamma, gae_lambda, device)
    if chunk_steps < 1 or not 0 <= min_stage_steps <= stage_budget:
        raise ValueError("Chunk steps must be positive and minimum stage steps must fit its budget")
    if eval_episodes < 1 or eval_seed < 0 or eval_seed == 42 or eval_seed == seed:
        raise ValueError("Use positive validation episodes and a validation seed distinct from training and 42")
    if not 0 <= goal_threshold <= 1 or not 0 <= safe_threshold <= 1 or passes_required < 1:
        raise ValueError("Thresholds must be rates in [0, 1] and passes-required must be positive")
    folder = create_run("models/single_agent", f"curriculum_seed{seed}", out_dir)
    manifest = {
        **provenance(folder), "status": "running", "algorithm": "PPO",
        "policy": "MultiInputPolicy", "ppo": config, "curriculum_complete": False,
        "stage_budget": stage_budget, "chunk_steps": chunk_steps,
        "min_stage_steps": min_stage_steps, "checkpoint_every": checkpoint_every,
        "validation": {"episodes": eval_episodes, "seed": eval_seed,
                       "goal_threshold": goal_threshold, "safe_threshold": safe_threshold,
                       "passes_required": passes_required},
        "stage_plan": [asdict(stage) for stage in STAGES], "stages": [],
        "note": "Stage validation uses easier training scenarios. Full-distribution checks are development runs, not official reporting.",
    }
    write_json(folder / "manifest.json", manifest)
    env = model = episode_log = None
    started = time.perf_counter()
    try:
        env = make_training_env(STAGES[0])
        manifest["environment"] = environment_config(env)
        env = Monitor(env, info_keywords=tuple(METRIC_KEYS))
        model = PPO("MultiInputPolicy", env, verbose=1, **config)
        for index, stage in enumerate(STAGES):
            stage_dir = folder / f"{index:02d}_{stage.name}"
            stage_dir.mkdir()
            set_training_stage(env, stage)
            if index:
                # Force the next learn() to start a new episode on the new stage.
                # The policy, optimizer and cumulative step counter are retained.
                model.set_env(model.get_env(), force_reset=True)
                model.logger.close()
            model.set_logger(configure(str(stage_dir), ["stdout", "csv"]))
            episode_log = EpisodeMetricsCallback(stage_dir / "episodes.csv", append=True)
            checkpoints = CheckpointCallback(checkpoint_every, str(stage_dir / "checkpoints"), "ppo")
            stage_record = {**asdict(stage), "status": "training",
                            "start_timesteps": model.num_timesteps, "checks": []}
            manifest["stages"].append(stage_record)
            consecutive = 0
            stage_steps = 0
            mastered = False
            print(f"Stage {index + 1}/{len(STAGES)}: {stage.name}; obstacles={stage.obstacles}, intruders={stage.intruders}", flush=True)
            while stage_steps < stage_budget:
                requested = min(chunk_steps, stage_budget - stage_steps)
                model.learn(total_timesteps=requested, reset_num_timesteps=False,
                            callback=[episode_log, checkpoints])
                stage_steps = model.num_timesteps - stage_record["start_timesteps"]
                snapshot = stage_dir / f"evaluation_{model.num_timesteps}_steps.zip"
                model.save(snapshot)
                model.save(folder / "latest_model.zip")
                print(f"Validating {stage.name} after {stage_steps} stage steps", flush=True)
                output = stage_dir / f"validation_{model.num_timesteps}"
                summary = checkpoint_evaluation(snapshot, output, eval_episodes, eval_seed, stage)
                passed = gate_passed(summary, goal_threshold, safe_threshold)
                consecutive = consecutive + 1 if passed else 0
                mastered = stage_steps >= min_stage_steps and consecutive >= passes_required
                stage_record["checks"].append({
                    "timesteps": model.num_timesteps, "stage_steps": stage_steps,
                    "goal_completion_rate": summary["goal_completion_rate"],
                    "safe_goal_completion_rate": summary["safe_goal_completion_rate"],
                    "passed": passed, "consecutive_passes": consecutive,
                    "artifacts": str(output.relative_to(folder)),
                })
                stage_record["actual_steps"] = stage_steps
                manifest["actual_timesteps"] = model.num_timesteps
                write_json(folder / "manifest.json", manifest)
                print(f"Stage goal completion={summary['goal_completion_rate']:.1%}, safe completion={summary['safe_goal_completion_rate']:.1%}; consecutive passes={consecutive}/{passes_required}", flush=True)
                if mastered:
                    break
            episode_log.close()
            episode_log = None
            model.save(stage_dir / "model.zip")
            stage_record["status"] = "mastered" if mastered else "budget_exhausted"
            print("Evaluating the checkpoint on the full competition distribution (development seed)", flush=True)
            full_summary = checkpoint_evaluation(stage_dir / "model.zip", stage_dir / "full_evaluation",
                                                 eval_episodes, eval_seed)
            stage_record["full_distribution"] = {
                "goal_completion_rate": full_summary["goal_completion_rate"],
                "safe_goal_completion_rate": full_summary["safe_goal_completion_rate"],
                "artifacts": str((stage_dir / "full_evaluation").relative_to(folder)),
            }
            write_json(folder / "manifest.json", manifest)
            if not mastered:
                manifest["status"] = "stage_not_mastered"
                print(f"Stopped at {stage.name}: validation targets were not met within the budget. Checkpoints are preserved.", flush=True)
                break
        else:
            model.save(folder / "model.zip")
            manifest.update(status="completed", curriculum_complete=True,
                            model_sha256=file_hash(folder / "model.zip"))
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        if model is not None:
            model.save(folder / "interrupted_model.zip")
        raise
    finally:
        if episode_log is not None:
            episode_log.close()
        if model is not None:
            manifest["actual_timesteps"] = model.num_timesteps
            if model._custom_logger:
                model.logger.close()
        if env is not None:
            env.close()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(folder / "manifest.json", manifest)
        print(f"Curriculum artifacts: {folder}", flush=True)
    return folder
