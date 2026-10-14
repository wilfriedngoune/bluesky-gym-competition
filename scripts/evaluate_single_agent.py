"""Tracked single-agent evaluation on development or official scenarios.

Development: python -m scripts.evaluate_single_agent --models neutral single_agent_model
Official:    python -m scripts.evaluate_single_agent --protocol official --models single_agent_model
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from scripts import evaluate_competition as official
from scripts.experiment_support import (
    FIXED_CONFIG, create_run, environment_config, failure_reasons, file_hash, metrics_summary,
    model_file, provenance, validate_environment, write_json,
)

DEVELOPMENT_SEED = 1234


def run_development(env, act, n_episodes, seed, out_dir):
    """Seed only the first reset; persist each result before starting the next."""
    records = []
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as csv_file, \
         (out_dir / "scenarios.jsonl").open("w", encoding="utf-8") as scenarios, \
         (out_dir / "failures.jsonl").open("w", encoding="utf-8") as failures:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "episode_index", "scenario_sha256", "terminated", "truncated",
            "safe_goal_completion", *official.METRIC_KEYS,
        ])
        writer.writeheader()
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed if episode == 0 else None)
            scenario = asdict(env.unwrapped.scenario)
            digest = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
            scenarios.write(json.dumps({"episode_index": episode, "scenario_sha256": digest, "scenario": scenario}) + "\n")
            scenarios.flush()
            terminated = truncated = False
            while not (terminated or truncated):
                obs, _, terminated, truncated, info = env.step(act(obs))
            record = {key: float(info[key]) for key in official.METRIC_KEYS}
            reasons = failure_reasons(record)
            row = {
                "episode_index": episode, "scenario_sha256": digest,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "safe_goal_completion": not reasons, **record,
            }
            writer.writerow(row)
            csv_file.flush()
            if reasons:
                failures.write(json.dumps({**row, "reasons": reasons}) + "\n")
                failures.flush()
            records.append(record)
            if (episode + 1) % 10 == 0 or episode + 1 == n_episodes:
                print(f"  completed {episode + 1}/{n_episodes} episodes", flush=True)
    return records


def evaluate_models(models, protocol="development", seed=None, episodes=None, out_dir=None):
    if protocol == "official":
        if seed not in (None, official.SEED) or episodes not in (None, official.N_EPISODES):
            raise ValueError("Official evaluation requires 1000 episodes and first-reset seed 42")
        seed, episodes = official.SEED, official.N_EPISODES
    elif protocol == "development":
        seed = DEVELOPMENT_SEED if seed is None else seed
        episodes = 100 if episodes is None else episodes
        if seed == official.SEED:
            raise ValueError("Use a development seed other than the official seed 42")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    if episodes < 1 or seed < 0:
        raise ValueError("Episodes must be positive and seed must be nonnegative")

    # Resolve all inputs before creating any simulator or run artifacts.
    inputs = [None if name == "neutral" else model_file(name) for name in models]
    if not inputs:
        raise ValueError("Supply at least one policy")
    run_dir = create_run("results/single_agent", protocol, out_dir)
    manifest = {
        **provenance(run_dir), "status": "running", "protocol": protocol,
        "seed": seed, "episodes": episodes, "deterministic_actions": True,
        "fixed_evaluation_config": FIXED_CONFIG,
        "environment_builder": "scripts.evaluate_competition.make_env('sa'); see source snapshot",
        "policies": [{"model": str(path) if path else None, "sha256": file_hash(path) if path else None} for path in inputs],
    }
    write_json(run_dir / "manifest.json", manifest)
    comparisons = []
    scenario_sequence = None
    started = time.perf_counter()
    try:
        for index, path in enumerate(inputs):
            label = "neutral" if path is None else path.stem
            if path is not None and sum(item is not None and item.stem == path.stem for item in inputs) > 1:
                label = f"{path.parent.name}/{path.stem}"
            policy_dir = run_dir / f"{index:02d}_{'neutral' if path is None else path.stem}"
            policy_dir.mkdir()
            if path:
                # Keep the exact model used even if the input is later overwritten.
                saved_model = policy_dir / "model.zip"
                saved_model.write_bytes(path.read_bytes())
                if file_hash(saved_model) != manifest["policies"][index]["sha256"]:
                    raise RuntimeError(f"Model changed during evaluation setup: {path}")
            else:
                saved_model = None
            act = official.load_policy("sa", saved_model)
            print(f"Evaluating {label}: {protocol}, seed {seed}, {episodes} episodes", flush=True)
            if protocol == "official":
                # Preserve the canonical harness's loop, metrics, CSV and seed stream.
                records = official.run_single_agent(episodes, act)
                official.summarize(records, "Single-agent")
                official.write_csv(records, str(policy_dir / "episodes.csv"))
            else:
                env = official.make_env("sa")
                try:
                    validate_environment(env)
                    manifest["environment"] = environment_config(env)
                    write_json(run_dir / "manifest.json", manifest)
                    records = run_development(env, act, episodes, seed, policy_dir)
                finally:
                    env.close()
                sequence = [json.loads(line)["scenario_sha256"] for line in (policy_dir / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()]
                if scenario_sequence is None:
                    scenario_sequence = sequence
                elif sequence != scenario_sequence:
                    raise RuntimeError("Policies were not evaluated on identical scenarios")
            summary = metrics_summary(records)
            write_json(policy_dir / "summary.json", summary)
            comparisons.append({"policy": label, "directory": policy_dir.name, **summary})
            print(f"  goal completion: {summary['goal_completion_rate']:.1%}; safe goal completion: {summary['safe_goal_completion_rate']:.1%}", flush=True)
            write_json(run_dir / "comparison.json", comparisons)
        lines = [
            f"Protocol: {protocol}; first-reset seed: {seed}; episodes per policy: {episodes}.", "",
            "Safe completion is a supplementary diagnostic; reward is not used to rank policies.", "",
            "| Policy | Goal completion | Safe completion | Intrusion events | Restricted entries | Sector exits | Successful flight time [s] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in comparisons:
            stats = row["metrics"]
            successful_time = row["flight_time_success_mean_s"]
            time_text = "n/a" if successful_time is None else f"{successful_time:.1f}"
            lines.append(f"| {row['policy']} | {row['goal_completion_rate']:.1%} | {row['safe_goal_completion_rate']:.1%} | {stats['intrusion_events']['mean']:.2f} | {stats['restricted_area_events']['mean']:.2f} | {stats['sector_exit_events']['mean']:.2f} | {time_text} |")
        (run_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(run_dir / "manifest.json", manifest)
        print(f"Run artifacts: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["neutral", "single_agent_model"], help="PPO paths and/or neutral")
    parser.add_argument("--protocol", choices=["development", "official"], default="development")
    parser.add_argument("--seed", type=int, help="Development seed (default 1234); official is fixed at 42")
    parser.add_argument("--episodes", type=int, help="Development default 100; official fixed at 1000")
    parser.add_argument("--out-dir", help="New artifact directory; existing directories are never overwritten")
    args = parser.parse_args()
    evaluate_models(args.models, args.protocol, args.seed, args.episodes, args.out_dir)


if __name__ == "__main__":
    main()
