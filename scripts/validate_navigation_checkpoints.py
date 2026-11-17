"""Compare v5 checkpoints on held-out seed streams; never an official score.

Example:
python -m scripts.validate_navigation_checkpoints --models models/a/model.zip models/b/model.zip --seeds 101 102 --episodes 3 --out-dir models/validation_run

Each checkpoint receives the same scenarios for each seed. Saved actor/search
defaults are respected unless --no-lookahead or --with-lookahead is explicit.
Choose seeds independently of training; this tool does not infer whether a seed
was previously used. The canonical seed-42 competition evaluator is unchanged.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from scripts import evaluate_competition as evaluation
from scripts.competition_report import SAFETY_KEYS, TOTAL_KEYS


def aggregate_metrics(records):
    """Count clean arrivals using both official event counts and occupancy."""
    totals = {key: 0. for key in (*TOTAL_KEYS, "total_reward")}
    clean = clean_arrivals = 0
    for index, record in enumerate(records):
        values = {key: float(record[key]) for key in totals}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite metric in aircraft record {index}")
        if any(values[key] < 0 for key in TOTAL_KEYS):
            raise ValueError(f"Negative official metric in aircraft record {index}")
        if values["waypoint_reached"] not in (0, 1):
            raise ValueError("waypoint_reached must be zero or one")
        if any(not values[key].is_integer() for key in SAFETY_KEYS if key.endswith("_events")):
            raise ValueError("Event counts must be integers")
        for key in totals:
            totals[key] += values[key]
        is_clean = all(values[key] == 0 for key in SAFETY_KEYS)
        clean += int(is_clean)
        clean_arrivals += int(is_clean and values["waypoint_reached"] == 1)
    count = len(records)
    return dict(aircraft_records=count, totals=totals, clean_flights=clean,
                clean_arrivals=clean_arrivals,
                goal_completion_pct=100 * totals["waypoint_reached"] / count if count else None,
                clean_flight_pct=100 * clean / count if count else None,
                clean_arrival_pct=100 * clean_arrivals / count if count else None)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_summary(summary):
    lines = ["# Checkpoint validation — nonofficial", "",
             "The same requested validation scenario seed streams were used for every checkpoint. "
             "These development results are not the official seed-42 evaluation. "
             "Choose seeds not used for training; that separation is not inferred automatically. "
             "No single scalar winner is selected.", "",
             f"Seeds: {', '.join(map(str, summary['seeds']))}; "
             f"scenarios per seed: {summary['episodes_per_seed']}; aircraft per scenario: {summary['n_agents']}.", "",
             "| Model | Inference | Goals | Clean arrivals | Clean arrival % | Intrusion events | Intrusion seconds | Restricted events | Restricted seconds | Sector exits | Outside seconds | Flight seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary["models"]:
        metrics, totals = item["aggregate"], item["aggregate"]["totals"]
        values = (totals["waypoint_reached"], metrics["clean_arrivals"], metrics["clean_arrival_pct"],
                  totals["intrusion_events"], totals["intrusion_time"], totals["restricted_area_events"],
                  totals["time_in_restricted_area"], totals["sector_exit_events"],
                  totals["time_outside_sector"], totals["flight_time"])
        lines.append("| " + item["model"] + " | " + ("search" if item["lookahead_enabled"] else "actor only") + " | " + " | ".join(f"{value:g}" for value in values) + " |")
    lines.extend(["", "Event and time columns are totals across aircraft. Intrusions may count for both participants. "
                  "Clean requires zero events and zero occupancy time; clean arrivals additionally reach the goal. "
                  "Own reward totals are retained in JSON/CSV but are not used to rank checkpoints.", ""])
    return "\n".join(lines)


def validate_checkpoints(models, *, out_dir, seeds=(101, 102), episodes=3,
                         n_agents=10, disable_lookahead=None, progress=True):
    """Run standard scoring in separate validation seed streams without learning."""
    paths = [Path(path).resolve() for path in models]
    seeds = list(seeds)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Supply at least one distinct checkpoint")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Each checkpoint must name an existing model file")
    if (not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("Supply distinct nonnegative integer validation seeds")
    if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1
            or isinstance(n_agents, bool) or not isinstance(n_agents, int) or not 2 <= n_agents <= 10):
        raise ValueError("Use positive episodes and 2..10 aircraft")
    if disable_lookahead is not None and not isinstance(disable_lookahead, bool):
        raise ValueError("disable_lookahead must be boolean or None")
    output = Path(out_dir)
    # A new run directory owns all output names and protects earlier results.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = dict(phase="validation", official_protocol=False, mdp="navigation-v5",
                   seeds=seeds, episodes_per_seed=episodes, n_agents=n_agents,
                   inference_override=("saved" if disable_lookahead is None else
                                       "actor_only" if disable_lookahead else "with_lookahead"),
                   completed=False, models=[])
    _write_json(output / "summary.json", summary)
    try:
        for model_index, path in enumerate(paths):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            act = evaluation.load_policy("ma", str(path), mdp="navigation-v5")
            if disable_lookahead is not None:
                act.disable_lookahead = disable_lookahead
            item = dict(model=str(path), model_sha256=digest,
                        lookahead_enabled=not getattr(act, "disable_lookahead", False),
                        lookahead_config=asdict(act.lookahead_config),
                        saved_environment_settings=evaluation.policy_reward_kwargs(act, "navigation-v5"),
                        seeds=[])
            summary["models"].append(item)
            combined = []
            for seed in seeds:
                beginning = time.perf_counter()
                records = evaluation.run_multi_agent(episodes, act, n_agents,
                    mdp="navigation-v5", progress=progress, seed=seed)
                if len(records) != episodes * n_agents:
                    raise RuntimeError("Validation did not return one final record per aircraft episode")
                name = f"{model_index:02d}_{path.stem}_seed{seed}"
                evaluation.write_csv(records, str(output / (name + ".csv")))
                metrics = aggregate_metrics(records)
                seed_result = dict(seed=seed, source=name + ".csv", wall_seconds=time.perf_counter() - beginning,
                                   metrics=metrics)
                item["seeds"].append(seed_result)
                combined.extend(records)
                _write_json(output / (name + ".json"), dict(
                    phase="validation", official_protocol=False, model=str(path), model_sha256=digest,
                    n_agents=n_agents, n_scenarios=episodes, lookahead_enabled=item["lookahead_enabled"],
                    lookahead_config=item["lookahead_config"], **seed_result))
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Checkpoint changed during validation; comparison is incomplete")
            item["aggregate"] = aggregate_metrics(combined)
            _write_json(output / "summary.json", summary)
        summary["completed"] = True
        (output / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--models", nargs="+")
    source.add_argument("--checkpoint-dir")
    parser.add_argument("--pattern", default="*.zip", help="glob within --checkpoint-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-lookahead", dest="disable_lookahead", action="store_true")
    mode.add_argument("--with-lookahead", dest="disable_lookahead", action="store_false")
    parser.set_defaults(disable_lookahead=None)
    args = parser.parse_args(argv)
    models = args.models or sorted(Path(args.checkpoint_dir).glob(args.pattern))
    validate_checkpoints(models, out_dir=args.out_dir, seeds=args.seeds,
                         episodes=args.episodes, n_agents=args.n_agents,
                         disable_lookahead=args.disable_lookahead)


if __name__ == "__main__":
    main()
