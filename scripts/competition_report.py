"""Aggregate final aircraft records into the competition sharing template.

    python -m scripts.competition_report --ma models/ma_reward_v2/training_metrics.csv --training --out models/ma_reward_v2/training_snapshot
    python -m scripts.competition_report --sa sa_eval.csv --ma ma_eval.csv --episodes 1000 --seed 42 --out results/comparison

Imported CSVs do not establish official evaluation provenance. The evaluation
harness supplies protocol metadata directly after running the fixed scenario stream.
"""

import argparse
import csv
import json
import math
from pathlib import Path


TOTAL_KEYS = (
    "waypoint_reached", "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector", "flight_time",
)
SAFETY_KEYS = (
    "intrusion_events", "intrusion_time", "restricted_area_events",
    "time_in_restricted_area", "sector_exit_events", "time_outside_sector",
)
TABLE_ROWS = (
    ("waypoint_reached", "goals reached"), ("intrusion_events", "events"),
    ("intrusion_time", "seconds"), ("restricted_area_events", "events"),
    ("time_in_restricted_area", "seconds"), ("sector_exit_events", "events"),
    ("time_outside_sector", "seconds"), ("flight_time", "seconds"),
    ("violation-free", "clean flights"),
    ("violation-free_success", "clean arrivals"),
)


def build_report(records, *, kind, phase, n_agents=10, n_scenarios=None,
                 seed=None, mdp=None, model_path=None, source=None):
    if kind not in {"sa", "ma"} or phase not in {"training", "evaluation", "imported"}:
        raise ValueError("Invalid report kind or phase")
    agents = 1 if kind == "sa" else n_agents
    if agents < 1:
        raise ValueError("n_agents must be positive")
    totals = dict.fromkeys(TOTAL_KEYS, 0.0)
    clean = clean_success = 0
    groups = {}
    for index, record in enumerate(records):
        values = {key: float(record[key]) for key in TOTAL_KEYS}
        for key, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in record {index}")
            if key.endswith("_events") and not value.is_integer():
                raise ValueError(f"Event count must be integral: {key} in record {index}")
            totals[key] += value
        if values["waypoint_reached"] not in (0, 1):
            raise ValueError(f"waypoint_reached must be 0 or 1 in record {index}")
        is_clean = all(values[key] == 0 for key in SAFETY_KEYS)
        clean += int(is_clean)
        clean_success += int(is_clean and values["waypoint_reached"] == 1)
        if record.get("scenario_index") is not None:
            scenario = record["scenario_index"]
            groups[scenario] = groups.get(scenario, 0) + 1
    n_records = len(records)
    if phase == "evaluation" and n_scenarios is None:
        raise ValueError("Evaluation needs an explicit scenario count")
    if n_scenarios is not None and (n_scenarios < 1 or n_records != n_scenarios * agents):
        raise ValueError("Evaluation record count must equal scenarios times aircraft per scenario")
    complete = sum(count == agents for count in groups.values()) if groups else None
    partial = sum(count != agents for count in groups.values()) if groups else None
    totals.update({"violation-free": clean, "violation-free_success": clean_success})
    rate = lambda count: 100.0 * count / n_records if n_records else None
    return {
        "kind": kind, "phase": phase, "seed": seed, "mdp": mdp,
        "model_path": str(model_path) if model_path is not None else None,
        "source": str(source) if source is not None else None,
        "n_agents": agents, "n_scenarios": n_scenarios,
        "completed_training_scenarios": complete, "partial_training_scenarios": partial,
        "aircraft_records": n_records,
        "official_protocol": (phase == "evaluation" and n_scenarios == 1000
                              and seed == 42 and agents == (1 if kind == "sa" else 10)),
        "totals": totals,
        "rates": {"goal_completion_pct": rate(totals["waypoint_reached"]),
                  "violation_free_pct": rate(clean),
                  "violation_free_success_pct": rate(clean_success)},
    }


def _format(value):
    if value is None:
        return "NA"
    return f"{value:,.3f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"


def render_report(reports):
    by_kind = {report["kind"]: report for report in reports}
    lines = ["# Competition results", ""]
    for report in reports:
        track = "SINGLE-AGENT" if report["kind"] == "sa" else "MULTI-AGENT"
        status = ("OFFICIAL PROTOCOL (1000 scenarios, seed 42)" if report["official_protocol"]
                  else "TRAINING STATISTICS - NOT EVALUATION" if report["phase"] == "training"
                  else "NONOFFICIAL / VERIFY EVALUATION PROVENANCE")
        lines.extend([f"{track}: **{status}**.", ""])
        seed = report['seed'] if report['seed'] is not None else "not recorded"
        lines.append(f"Aircraft records: {report['aircraft_records']:,}; "
                     f"aircraft per scenario: {report['n_agents']}; seed: {seed}.")
        if report["completed_training_scenarios"] is not None:
            lines.append(f"Completed scenarios: {report['completed_training_scenarios']}; "
                         f"partially logged scenarios: {report['partial_training_scenarios']} "
                         "(their completed aircraft are included).")
        if report["model_path"]:
            lines.append(f"Model: `{report['model_path']}`; MDP: `{report['mdp']}`.")
        if report["source"]:
            lines.append(f"Records: `{report['source']}`.")
        lines.append("")
    lines.extend(["| Metric | SINGLE-AGENT | MULTI-AGENT | UNIT |", "| --- | ---: | ---: | --- |"])
    counts = []
    for kind in ("sa", "ma"):
        report = by_kind.get(kind)
        count = report["n_scenarios"] if report else None
        if report and report["phase"] == "training" and report["completed_training_scenarios"] is not None:
            counts.append(f"{report['completed_training_scenarios']:,} complete, "
                          f"{report['partial_training_scenarios']:,} partial")
        else:
            counts.append(f"{count:,} episodes" if count is not None else "NA")
    lines.append(f"| scenarios | {counts[0]} | {counts[1]} | episodes |")
    for key, unit in TABLE_ROWS:
        values = [_format(by_kind[kind]["totals"][key]) if kind in by_kind else "NA"
                  for kind in ("sa", "ma")]
        lines.append(f"| {key} | {values[0]} | {values[1]} | {unit} |")
    for key, label in (("goal_completion_pct", "goal completion"),
                       ("violation_free_pct", "violation-free rate"),
                       ("violation_free_success_pct", "violation-free arrival rate")):
        values = [_format(by_kind[kind]["rates"][key]) if kind in by_kind else "NA"
                  for kind in ("sa", "ma")]
        lines.append(f"| {label} | {values[0]} | {values[1]} | % |")
    lines.extend(["", "All metric values above are totals, not means. MA intrusions remain "
                  "per-aircraft counts; a conflict can count for both participants.", "",
                  "`violation-free` means zero safety events AND zero safety occupancy time; "
                  "a clean timeout counts. `violation-free_success` additionally requires reaching "
                  "the goal. These are derived statistics; the official scoring layer is unchanged.", ""])
    return "\n".join(lines)


def write_report(reports, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    markdown = Path(str(prefix) + ".md")
    machine_readable = Path(str(prefix) + ".json")
    rendered = render_report(reports)
    markdown.write_text(rendered, encoding="utf-8")
    machine_readable.write_text(json.dumps({"reports": reports}, indent=2) + "\n", encoding="utf-8")
    print(rendered)
    print(f"Reports saved to {markdown} and {machine_readable}")
    return markdown, machine_readable


def read_records(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sa", help="single-agent final-records CSV")
    parser.add_argument("--ma", help="multi-agent final-records CSV")
    parser.add_argument("--training", action="store_true", help="explicitly label training records")
    parser.add_argument("--episodes", type=int, help="scenario count for imported evaluation records")
    parser.add_argument("--seed", type=int, help="evaluation seed (must match CSV provenance)")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--out", required=True, help="report path prefix, without .md/.json")
    args = parser.parse_args()
    if not args.sa and not args.ma:
        parser.error("Provide --sa, --ma, or both")
    reports = [build_report(read_records(path), kind=kind,
                            phase="training" if args.training else "imported",
                            n_agents=args.n_agents, n_scenarios=args.episodes,
                            seed=args.seed, source=path)
               for kind, path in (("sa", args.sa), ("ma", args.ma)) if path]
    write_report(reports, args.out)


if __name__ == "__main__":
    main()
