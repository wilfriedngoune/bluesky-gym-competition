"""Inventory missed arrivals and optionally replay their scenarios for diagnosis.

The evaluation CSV has no aircraft IDs: its ordered blocks of ten completed
aircraft records identify scenarios only. Replay traces supply actual IDs.
Diagnostics never alter environment dynamics, rewards, termination or scoring.
"""

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time


DEFAULT_MODEL = "models/ma_navigation_v5_fresh_v6/model.zip"
DEFAULT_EVALUATION = "models/ma_navigation_v5_fresh_v6/model.evaluation.csv"
DEFAULT_OUTPUT = "models/ma_navigation_v5_fresh_v6/timeout_diagnostics"
SAFETY_FIELDS = (
    "intrusion_events", "intrusion_time", "restricted_area_events",
    "time_in_restricted_area", "sector_exit_events", "time_outside_sector",
)


def _number(row, key, default=None):
    value = row.get(key)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"Missing numeric field {key}")
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {key}: {value}")
    return result


def _clean(row):
    return all(_number(row, key) == 0 for key in SAFETY_FIELDS)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def inventory_failures(evaluation_csv, *, n_agents=10, seed=42):
    """Group complete, ordered evaluation records without guessing identities."""
    rows = _read_csv(evaluation_csv)
    if n_agents < 2 or not rows or len(rows) % n_agents:
        raise ValueError("Expected complete ordered scenario blocks of n_agents records")
    settings_path = Path(evaluation_csv).with_suffix(".settings.json")
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig")) if settings_path.exists() else None
    if settings is not None:
        for key, expected in (("seed", seed), ("n_agents", n_agents), ("n_scenarios", len(rows) // n_agents)):
            if key in settings and settings[key] != expected:
                raise ValueError(f"Evaluation settings {key}={settings[key]} disagree with requested {expected}")
        if settings.get("mdp", "navigation-v5") != "navigation-v5":
            raise ValueError("Timeout replay tool currently requires navigation-v5 evaluation")
    scenarios = []
    for start in range(0, len(rows), n_agents):
        failed = []
        for index, row in enumerate(rows[start:start + n_agents], start=start):
            reached = _number(row, "waypoint_reached")
            if reached not in (0., 1.):
                raise ValueError("waypoint_reached must be 0 or 1")
            if not reached:
                failed.append(dict(evaluation_row_index=index, clean=_clean(row),
                                   flight_time_seconds=_number(row, "flight_time"),
                                   metrics={name: _number(row, name) for name in SAFETY_FIELDS}))
        if failed:
            scenarios.append(dict(scenario=start // n_agents + 1, missed_arrivals=len(failed),
                                  clean_missed_arrivals=sum(row["clean"] for row in failed), records=failed))
    return dict(
        source=str(Path(evaluation_csv).resolve()), seed=seed, n_agents=n_agents,
        n_scenarios=len(rows) // n_agents, aircraft_records=len(rows),
        missed_arrivals=sum(row["missed_arrivals"] for row in scenarios),
        clean_missed_arrivals=sum(row["clean_missed_arrivals"] for row in scenarios), scenarios=scenarios,
        mapping_assumption="Ordered blocks of n_agents final records per scenario; aircraft IDs unavailable in this CSV.",
        evaluation_settings=settings,
    )


def _distance_nm(first, second):
    """Diagnostic great-circle path length, not an official scored metric."""
    lat1, lat2 = math.radians(_number(first, "lat_before")), math.radians(_number(second, "lat_before"))
    dlat = lat2 - lat1
    dlon = math.radians(_number(second, "lon_before") - _number(first, "lon_before"))
    chord = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3440.065 * 2 * math.asin(math.sqrt(min(1., max(0., chord))))


def _turn(first, second):
    return abs((_number(second, "heading_before_deg") - _number(first, "heading_before_deg") + 180.) % 360. - 180.)


def analyze_trace(trajectory_csv, *, time_limit=3000., window_seconds=300., minimum_progress_nm=.5):
    """Summarize observed flight behavior; circling labels are hypotheses only."""
    if (not math.isfinite(window_seconds) or window_seconds <= 0
            or not math.isfinite(time_limit) or time_limit <= 0
            or not math.isfinite(minimum_progress_nm) or minimum_progress_nm < 0):
        raise ValueError("Use a positive window and nonnegative progress threshold")
    groups = defaultdict(list)
    for row in _read_csv(trajectory_csv):
        groups[(int(row["episode_index"]), row["agent_id"])].append(row)
    if not groups:
        raise ValueError("Trajectory CSV contains no aircraft observations")
    aircraft = []
    for (episode, agent), rows in sorted(groups.items()):
        rows.sort(key=lambda row: _number(row, "time_before"))
        times = [_number(row, "time_before") for row in rows]
        distances = [_number(row, "goal_distance_before_nm") for row in rows]
        path, turning = [0.], [0.]
        for first, second in zip(rows, rows[1:]):
            path.append(path[-1] + _distance_nm(first, second))
            turning.append(turning[-1] + _turn(first, second))
        reached = bool(_number(rows[-1], "waypoint_reached"))
        final_time = _number(rows[-1], "time_after")
        status = "arrival" if reached else "timeout" if final_time >= time_limit - 1e-6 else "capped_or_incomplete"
        low_progress, circling = [], []
        first = 0
        while first < len(rows):
            last = bisect_left(times, times[first] + window_seconds, lo=first + 1)
            if last >= len(rows):
                break
            gain = distances[first] - distances[last]
            observed_path = path[last] - path[first]
            observed_turning = turning[last] - turning[first]
            if gain <= minimum_progress_nm:
                window = dict(start_seconds=times[first], end_seconds=times[last],
                              net_goal_distance_gain_nm=gain, observed_path_length_nm=observed_path,
                              absolute_heading_change_deg=observed_turning)
                low_progress.append(window)
                if observed_path >= 2. and observed_turning >= 270.:
                    circling.append(window)
            first = last
        metrics = {name: _number(rows[-1], name) for name in SAFETY_FIELDS}
        speeds = [_number(row, "speed_before_kt") for row in rows]
        aircraft.append(dict(
            scenario=episode + 1, agent_id=agent, status=status, clean=_clean(rows[-1]),
            last_record_time_after_seconds=final_time,
            first_observed_goal_distance_nm=distances[0], last_observed_goal_distance_nm=distances[-1],
            last_distance_observation_time_seconds=times[-1],
            distance_is_terminal_position=False,
            net_observed_goal_distance_gain_nm=distances[0] - distances[-1],
            observed_path_length_nm=path[-1], absolute_observed_heading_change_deg=turning[-1],
            observed_progress_per_path_nm=(distances[0] - distances[-1]) / path[-1] if path[-1] else None,
            mean_observed_speed_kt=sum(speeds) / len(speeds), minimum_observed_speed_kt=min(speeds),
            obstacle_occupied_samples=sum(str(row.get("inside_obstacle", "")).lower() == "true" for row in rows),
            outside_sector_samples=sum(str(row.get("inside_sector", "true")).lower() == "false" for row in rows),
            low_progress_windows=low_progress, possible_circling_windows=circling, metrics=metrics,
        ))
    return dict(source=str(Path(trajectory_csv).resolve()), aircraft=aircraft,
                summary=dict(arrivals=sum(row["status"] == "arrival" for row in aircraft),
                             timeouts=sum(row["status"] == "timeout" for row in aircraft),
                             capped_or_incomplete=sum(row["status"] == "capped_or_incomplete" for row in aircraft)),
                heuristics=dict(window_seconds=window_seconds, minimum_progress_nm=minimum_progress_nm,
                                possible_circling_minimum_path_nm=2., possible_circling_minimum_turn_deg=270.),
                caveats=["Distance and path endpoints are observations BEFORE each decision; the final post-step position is not logged.",
                         "Path length and turning are sampled diagnostics, not official competition metrics.",
                         "Low progress or turning can reflect legitimate avoidance; heuristic flags do not establish a cause."])


def markdown_report(report):
    inventory = report["inventory"]
    lines = ["# Navigation timeout diagnostics", "",
             f"Phase: {report['phase']}; operation completed: {report['completed']}.", "",
             f"Evaluation: {inventory['n_scenarios']} scenarios, {inventory['aircraft_records']} aircraft records, "
             f"seed {inventory['seed']}.",
             f"Missed arrivals: {inventory['missed_arrivals']}; clean missed arrivals: {inventory['clean_missed_arrivals']}.", "",
             "The source CSV identifies scenarios by ordered blocks; it does not identify individual aircraft.", "",
             "| Scenario | Missed arrivals | Clean missed arrivals |", "| --- | ---: | ---: |"]
    lines.extend(f"| {row['scenario']} | {row['missed_arrivals']} | {row['clean_missed_arrivals']} |" for row in inventory["scenarios"])
    for analysis in report["replays"]:
        lines.extend(["", f"## Scenario {analysis['scenario']} replay", "",
                      "| Aircraft | Status | Clean | Last observed goal distance [NM] | Low-progress windows | Possible circling windows |",
                      "| --- | --- | --- | ---: | ---: | ---: |"])
        for row in analysis["aircraft"]:
            lines.append(f"| {row['agent_id']} | {row['status']} | {row['clean']} | "
                         f"{row['last_observed_goal_distance_nm']:.3f} | {len(row['low_progress_windows'])} | {len(row['possible_circling_windows'])} |")
    lines.extend(["", "Trace goal distances are PRE-last-step observations, not terminal positions. "
                  "Low-progress/circling flags are hypotheses and may represent legitimate avoidance.",
                  "Capped/incomplete trajectories are not final evaluation outcomes. Official scoring and rewards are unchanged.", ""])
    if report.get("error"):
        lines.extend([f"Run error: {report['error']}", ""])
    return "\n".join(lines)


def run_diagnostics(*, model=DEFAULT_MODEL, evaluation=DEFAULT_EVALUATION, out_dir=DEFAULT_OUTPUT,
                    list_only=False, scenarios=None, max_steps=None, seed=42, n_agents=10,
                    lookahead_preset="saved", lookahead_simulations=None, lookahead_depth=None,
                    disable_lookahead=None, window_seconds=300.):
    if seed < 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("Use a nonnegative seed and positive optional max_steps")
    inventory = inventory_failures(evaluation, n_agents=n_agents, seed=seed)
    failed_scenarios = {row["scenario"] for row in inventory["scenarios"]}
    selected = sorted(failed_scenarios if scenarios is None else set(scenarios))
    if set(selected) - failed_scenarios:
        raise ValueError("--scenarios must select scenarios with missed arrivals in this evaluation")
    source = Path(model)
    if not source.is_file():
        source = Path(str(source) + ".zip")
    if not source.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {source}")
    source = source.resolve()
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output}; choose a new --out-dir")
    output.mkdir(parents=True)
    report = dict(phase="inventory_only" if list_only else "replayed", completed=False,
                  replay_coverage_complete=False, model=str(source), model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  evaluation_sha256=hashlib.sha256(Path(evaluation).read_bytes()).hexdigest(),
                  inventory=inventory, selected_scenarios=selected, replays=[],
                  requested_settings=dict(seed=seed, n_agents=n_agents, max_steps=max_steps,
                    lookahead_preset=lookahead_preset, lookahead_simulations=lookahead_simulations,
                    lookahead_depth=lookahead_depth, disable_lookahead=disable_lookahead,
                    window_seconds=window_seconds))
    def save():
        (output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (output / "diagnostics.md").write_text(markdown_report(report), encoding="utf-8")
    started = time.perf_counter()
    save()
    try:
        if not list_only:
            from scripts.evaluate_competition import load_policy, policy_reward_kwargs
            from bluesky_zoo.lookahead import resolve_inference_lookahead_config
            from train_zoo import replay
            policy = load_policy("ma", str(source), mdp="navigation-v5")
            resolved_config = resolve_inference_lookahead_config(
                policy.lookahead_config, preset=lookahead_preset, simulations=lookahead_simulations,
                max_depth=lookahead_depth)
            search_overridden = lookahead_preset not in (None, "saved") or any(
                value is not None for value in (lookahead_simulations, lookahead_depth))
            saved_disabled = bool(getattr(policy, "disable_lookahead", False))
            resolved_disabled = (saved_disabled and not search_overridden
                                 if disable_lookahead is None else bool(disable_lookahead))
            if resolved_disabled and search_overridden:
                raise ValueError("Lookahead settings require lookahead enabled")
            report["resolved_lookahead_config"] = asdict(resolved_config)
            report["lookahead_enabled"] = not resolved_disabled
            report["saved_environment_settings"] = policy_reward_kwargs(policy, "navigation-v5")
            report["inference_overridden"] = (resolved_config != policy.lookahead_config
                                               or resolved_disabled != saved_disabled)
            del policy
            for scenario in selected:
                trace = output / f"scenario{scenario:03d}.trajectory.csv"
                replay(str(source), n_episodes=1, seed=seed, n_agents=n_agents, mdp="navigation-v5",
                       headless=True, trajectory_csv=str(trace), scenario_start=scenario,
                       max_steps=max_steps, lookahead_preset=lookahead_preset,
                       lookahead_simulations=lookahead_simulations, lookahead_depth=lookahead_depth,
                       disable_lookahead=disable_lookahead)
                analysis = analyze_trace(trace, window_seconds=window_seconds)
                analysis["scenario"] = scenario
                report["replays"].append(analysis)
                save()
        report["completed"] = True
        report["replay_coverage_complete"] = (not list_only and len(report["replays"]) == len(selected)
            and all(not row["summary"]["capped_or_incomplete"] for row in report["replays"]))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        save()
    print(markdown_report(report))
    print(f"Reports saved to {output / 'diagnostics.md'} and {output / 'diagnostics.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", default=DEFAULT_EVALUATION)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--list-only", action="store_true", help="inventory missed arrivals without loading policy or simulator")
    parser.add_argument("--scenarios", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="cap replay decisions; unfinished flights remain diagnostic only")
    parser.add_argument("--window-seconds", type=float, default=300.)
    parser.add_argument("--lookahead-preset", choices=("saved", "fast"), default="saved")
    parser.add_argument("--lookahead-simulations", type=int, default=None)
    parser.add_argument("--lookahead-depth", type=int, default=None)
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--no-lookahead", dest="no_lookahead", action="store_true", default=None)
    inference.add_argument("--with-lookahead", dest="no_lookahead", action="store_false")
    args = parser.parse_args(argv)
    run_diagnostics(model=args.model, evaluation=args.evaluation, out_dir=args.out_dir,
                    list_only=args.list_only, scenarios=args.scenarios, max_steps=args.max_steps,
                    seed=args.seed, n_agents=args.n_agents, window_seconds=args.window_seconds,
                    lookahead_preset=args.lookahead_preset, lookahead_simulations=args.lookahead_simulations,
                    lookahead_depth=args.lookahead_depth, disable_lookahead=args.no_lookahead)


if __name__ == "__main__":
    main()
