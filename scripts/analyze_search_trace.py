"""Compare logged search continuations with matched actual decision samples."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def _read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def analyze(prefix):
    prefix = Path(prefix)
    settings = json.loads(Path(str(prefix) + ".settings.json").read_text(encoding="utf-8"))
    if not settings["completed"]:
        raise ValueError("Diagnostic replay is incomplete; wait for it to finish")
    candidates = _read(Path(str(prefix) + ".candidates.csv"))
    forecasts = _read(Path(str(prefix) + ".forecasts.csv"))
    actual = {(int(r["episode_index"]), float(r["time"]), r["own_agent"], r["other_agent"]): r
              for r in _read(Path(str(prefix) + ".actual.csv"))}
    commands = {(int(r["episode_index"]), float(r["time"]), r["agent"]): r
                for r in _read(Path(str(prefix) + ".commands.csv"))}
    selected = {(int(r["episode_index"]), float(r["time"]), r["root_agent"], int(r["candidate_index"])): r
                for r in candidates if r["selected"] == "True"}
    groups = {}
    for row in forecasts:
        key = (int(row["episode_index"]), float(row["decision_time"]), row["root_agent"], int(row["candidate_index"]))
        if key in selected:
            groups.setdefault((*key, row["other_agent"]), []).append(row)
    comparisons = []
    for key, rows in sorted(groups.items()):
        episode, decision, own, candidate_index, other = key
        chosen = selected[key[:4]]
        matched = [(r, actual[(episode, float(r["time"]), own, other)]) for r in rows
                   if (episode, float(r["time"]), own, other) in actual]
        if not matched:
            continue
        predicted_min = min(float(r["predicted_separation_nm"]) for r in rows)
        actual_min = min(float(a["separation_nm"]) for _, a in matched)
        errors = [abs(float(r["predicted_separation_nm"]) - float(a["separation_nm"])) for r, a in matched]
        own_errors = [math.hypot(float(r["own_north_nm"]) - float(a["own_north_nm"]),
                                 float(r["own_east_nm"]) - float(a["own_east_nm"])) for r, a in matched]
        other_errors = [math.hypot(float(r["other_north_nm"]) - float(a["other_north_nm"]),
                                   float(r["other_east_nm"]) - float(a["other_east_nm"])) for r, a in matched]
        first = min(rows, key=lambda r: float(r["time"]))
        actual_command = commands.get((episode, decision, other))
        heading_error = (abs((float(actual_command["target_heading_deg"]) - float(first["assumed_other_heading_deg"])
                              + 180) % 360 - 180) if actual_command else None)
        speed_error = (abs(float(actual_command["target_speed_kt"]) - float(first["assumed_other_speed_kt"]))
                       if actual_command else None)
        current = actual.get((episode, decision, own, other), {})
        first_decision = next(((r, a) for r, a in matched
                               if float(r["time"]) == decision + settings["decision_seconds"]), None)
        first_errors = (None, None, None)
        if first_decision is not None:
            r, a = first_decision
            first_errors = (
                abs(float(r["predicted_separation_nm"]) - float(a["separation_nm"])),
                math.hypot(float(r["own_north_nm"]) - float(a["own_north_nm"]),
                           float(r["own_east_nm"]) - float(a["own_east_nm"])),
                math.hypot(float(r["other_north_nm"]) - float(a["other_north_nm"]),
                           float(r["other_east_nm"]) - float(a["other_east_nm"])),
            )
        comparisons.append(dict(
            episode_index=episode, decision_time=decision, root_agent=own, other_agent=other,
            candidate_index=candidate_index, visits=int(chosen["visits"]), mean_return=float(chosen["mean_return"]),
            explored_depth=int(chosen["explored_depth"]), principal_path_depth=int(chosen["principal_path_depth"]),
            forecast_seconds=float(chosen["forecast_duration_seconds"]),
            current_separation_nm=current.get("separation_nm"), current_tcpa_seconds=current.get("tcpa_seconds"),
            current_dcpa_nm=current.get("dcpa_nm"), predicted_min_separation_nm=predicted_min,
            actual_min_sampled_separation_nm=actual_min, matched_samples=len(matched),
            forecast_end_observed=max(float(a["time"]) for _, a in matched) >= max(float(r["time"]) for r in rows),
            mean_abs_separation_error_nm=statistics.mean(errors), max_abs_separation_error_nm=max(errors),
            max_own_position_error_nm=max(own_errors), max_other_position_error_nm=max(other_errors),
            first_decision_separation_error_nm=first_errors[0],
            first_decision_own_position_error_nm=first_errors[1],
            first_decision_other_position_error_nm=first_errors[2],
            other_command_heading_difference_deg=heading_error, other_command_speed_difference_kt=speed_error,
            predicted_safe_actual_intrusion=predicted_min >= 5 and actual_min < 5,
            predicted_and_actual_intrusion=predicted_min < 5 and actual_min < 5,
        ))
    def mean(values):
        return statistics.mean(values) if values else None
    summary = dict(
        source=str(prefix), root_decisions=len(selected), compared_root_pair_windows=len(comparisons),
        fully_observed_forecast_windows=sum(r["forecast_end_observed"] for r in comparisons),
        recorded_until_seconds=settings.get("recorded_until_seconds"),
        requested_window_covered=settings.get("requested_window_covered"),
        mean_forecast_seconds=mean([r["forecast_seconds"] for r in comparisons]),
        mean_abs_separation_error_nm=mean([r["mean_abs_separation_error_nm"] for r in comparisons]),
        mean_first_decision_separation_error_nm=mean([r["first_decision_separation_error_nm"] for r in comparisons
                                                      if r["first_decision_separation_error_nm"] is not None]),
        mean_other_command_heading_difference_deg=mean([r["other_command_heading_difference_deg"] for r in comparisons
                                                        if r["other_command_heading_difference_deg"] is not None]),
        mean_other_command_speed_difference_kt=mean([r["other_command_speed_difference_kt"] for r in comparisons
                                                    if r["other_command_speed_difference_kt"] is not None]),
        predicted_safe_actual_intrusion_windows=sum(r["predicted_safe_actual_intrusion"] for r in comparisons),
        predicted_and_actual_intrusion_windows=sum(r["predicted_and_actual_intrusion"] for r in comparisons),
        principal_path_depth_counts={str(depth): sum(int(r["principal_path_depth"]) == depth for r in selected.values())
                                     for depth in sorted({int(r["principal_path_depth"]) for r in selected.values()})},
        caveats=("Windows overlap and can describe the same encounter repeatedly. Actual minima use decision-boundary "
                 "samples, not the official one-second metric sampler. Predicted minima use retained one-second "
                 "samples along most-visited searched continuations, which can have different horizons. "
                 "Beyond the next real decision, errors also include the own aircraft's subsequent replanning. "
                 "A capped replay can leave some forecast windows only partially observed; check coverage fields. "
                 "Command differences combine the actor-versus-search choices and small coordinate-projection differences. "
                 "Prediction error alone does not isolate command assumptions from dynamics approximation."),
    )
    return summary, comparisons


def write_analysis(prefix):
    prefix = Path(prefix)
    summary, comparisons = analyze(prefix)
    output = Path(str(prefix) + ".analysis")
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if comparisons:
        with Path(str(output) + ".csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    lines = ["# Search prediction diagnostics", "", f"Source: `{prefix}`.", "",
             f"Root decisions: {summary['root_decisions']}; compared root/pair windows: {summary['compared_root_pair_windows']}.", "",
             f"Predicted-safe / actual-intrusion windows: {summary['predicted_safe_actual_intrusion_windows']}.",
             f"Predicted-and-actual intrusion windows: {summary['predicted_and_actual_intrusion_windows']}.", "",
             "These are overlapping prediction windows, not counts of independent conflicts.", ""]
    for label, key in [("Mean forecast horizon [s]", "mean_forecast_seconds"),
                       ("Mean absolute separation error [NM]", "mean_abs_separation_error_nm"),
                       ("Mean next-real-decision separation error [NM]", "mean_first_decision_separation_error_nm"),
                       ("Mean root neighbour-command heading difference [deg]", "mean_other_command_heading_difference_deg"),
                       ("Mean root neighbour-command speed difference [kt]", "mean_other_command_speed_difference_kt")]:
        value = summary[key]
        lines.append(f"- {label}: {value:.3f}" if value is not None else f"- {label}: unavailable")
    examples = sorted((r for r in comparisons if r["predicted_safe_actual_intrusion"]),
                      key=lambda r: (r["decision_time"], r["root_agent"], r["other_agent"]))[:12]
    if examples:
        lines.extend(["", "## Examples of optimistic forecasts", "",
                      "| Decision [s] | Root | Other | Horizon [s] | Predicted min [NM] | Actual sampled min [NM] | Other command difference [deg] |",
                      "| ---: | --- | --- | ---: | ---: | ---: | ---: |"])
        for r in examples:
            delta = r["other_command_heading_difference_deg"]
            delta = f"{delta:.1f}" if delta is not None else "NA"
            lines.append(f"| {r['decision_time']:.0f} | {r['root_agent']} | {r['other_agent']} | "
                         f"{r['forecast_seconds']:.0f} | {r['predicted_min_separation_nm']:.2f} | "
                         f"{r['actual_min_sampled_separation_nm']:.2f} | {delta} |")
    lines.extend(["", summary["caveats"], ""])
    Path(str(output) + ".md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="prefix passed to replay --search-trace-prefix")
    args = parser.parse_args(argv)
    print(json.dumps(write_analysis(args.prefix), indent=2))


if __name__ == "__main__":
    main()
