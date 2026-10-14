"""Validate a checkpoint on an easier training stage, never for official reporting."""

import argparse
from dataclasses import asdict
import time

from scripts.curriculum import STAGES, get_stage, make_training_env
from scripts.evaluate_competition import load_policy
from scripts.evaluate_single_agent import run_development
from scripts.experiment_support import (
    create_run, environment_config, file_hash, metrics_summary, model_file,
    provenance, write_json,
)


def evaluate_stage(model_path, stage_name, episodes=20, seed=1234, out_dir=None):
    if episodes < 1 or seed < 0 or seed == 42:
        raise ValueError("Stage validation needs positive episodes and a nonnegative seed other than 42")
    stage = get_stage(stage_name)
    source = model_file(model_path)
    folder = create_run("results/single_agent", f"stage_{stage.name}", out_dir)
    model = folder / "model.zip"
    model.write_bytes(source.read_bytes())
    manifest = {**provenance(folder), "status": "running",
                "protocol": "curriculum_validation", "official_reporting": False,
                "stage": asdict(stage), "episodes": episodes, "seed": seed,
                "model": str(source), "model_sha256": file_hash(model),
                "deterministic_actions": True}
    write_json(folder / "manifest.json", manifest)
    env = None
    started = time.perf_counter()
    try:
        act = load_policy("sa", model)
        env = make_training_env(stage)
        manifest["environment"] = environment_config(env)
        records = run_development(env, act, episodes, seed, folder)
        summary = metrics_summary(records)
        summary["protocol"] = "curriculum_validation"
        summary["stage"] = asdict(stage)
        write_json(folder / "summary.json", summary)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if env is not None:
            env.close()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(folder / "manifest.json", manifest)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", choices=[stage.name for stage in STAGES], required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    evaluate_stage(args.model, args.stage, args.episodes, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
