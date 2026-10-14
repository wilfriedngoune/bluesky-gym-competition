"""Run artifacts and diagnostics; never modify competition metrics or scenarios."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess

from scripts.evaluate_competition import METRIC_KEYS

REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETY_KEYS = (
    "intrusion_events", "intrusion_time", "restricted_area_events",
    "time_in_restricted_area", "sector_exit_events", "time_outside_sector",
)
FIXED_CONFIG = {
    "n_intruders": 10, "n_obstacles": 5, "episode_time_limit": 3000.0,
    "intrusion_distance": 5.0, "distance_margin": 5.0, "sim_dt": 1,
    "ac_spd": 150.0, "altitude": 350, "center": (52.0, 4.0),
}
SOURCE_FILES = (
    "train_single_agent.py", "scripts/evaluate_competition.py",
    "scripts/evaluate_single_agent.py", "scripts/experiment_support.py",
    "scripts/replay_single_agent.py", "bluesky_gym/envs/single_agent_env.py",
    "bluesky_gym/envs/competition_env.py", "bluesky_gym/__init__.py",
    "bluesky_gym/envs/common/functions.py", "core/rendering.py",
    "core/scenario.py", "core/observations.py", "core/actions.py", "core/tools.py",
    "docs/competition/COMPETITION.md", "pyproject.toml", "uv.lock",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file(path):
    candidate = Path(path).resolve()
    if not candidate.is_file() and candidate.suffix != ".zip":
        candidate = Path(str(candidate) + ".zip")
    if not candidate.is_file():
        raise FileNotFoundError(f"Model does not exist: {candidate}")
    return candidate


def create_run(root, prefix, out_dir=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(out_dir) if out_dir else Path(root) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def provenance(run_dir):
    """Snapshot relevant source, including local changes not captured by a commit."""
    hashes = {}
    for name in SOURCE_FILES:
        source = REPO_ROOT / name
        if source.is_file():
            target = Path(run_dir) / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            hashes[name] = file_hash(target)
    packages = {}
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "bluesky-simulator", "shapely"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    git = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        try:
            result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
            git[key] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git[key] = None
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "git": git, "source_sha256": hashes,
    }


def validate_environment(env):
    """Fail early if this evaluation no longer uses the competition world."""
    from bluesky_gym.envs.competition_env import CompetitionEnv
    from core.scenario import ScenarioGenerator

    base = env.unwrapped
    for key, expected in FIXED_CONFIG.items():
        if getattr(base, key) != expected:
            raise ValueError(f"Competition evaluation requires {key}={expected!r}")
    for method in ("_update_metrics", "_get_info", "step"):
        if getattr(type(base), method) is not getattr(CompetitionEnv, method):
            raise ValueError(f"Competition evaluation must preserve {method}")
    if base._fixed_scenario is not None or type(base.scenario_generator) is not ScenarioGenerator:
        raise ValueError("Evaluation must use the original random scenario generator")


def environment_config(env):
    base = env.unwrapped
    return {
        "class": f"{type(base).__module__}.{type(base).__name__}",
        "fixed": {key: getattr(base, key) for key in FIXED_CONFIG},
        "actions": {key: getattr(base, key) for key in ("d_heading", "d_speed", "action_frequency")},
        "observation_space": str(base.observation_space),
        "action_space": str(base.action_space),
    }


def failure_reasons(record):
    reasons = []
    if not record["waypoint_reached"]:
        reasons.append("goal_not_reached")
    for label, keys in (
        ("loss_of_separation", ("intrusion_events", "intrusion_time")),
        ("restricted_area", ("restricted_area_events", "time_in_restricted_area")),
        ("sector_exit", ("sector_exit_events", "time_outside_sector")),
    ):
        if any(record[key] > 0 for key in keys):
            reasons.append(label)
    return reasons


def metrics_summary(records):
    import numpy as np

    if not records:
        raise ValueError("Evaluation requires at least one completed episode")
    columns = {key: np.asarray([record[key] for record in records], dtype=float) for key in METRIC_KEYS}
    reached = columns["waypoint_reached"] > 0
    safe = np.all(np.asarray([columns[key] == 0 for key in SAFETY_KEYS]), axis=0)
    categories = ("goal_not_reached", "loss_of_separation", "restricted_area", "sector_exit")
    return {
        "episodes": len(records),
        "goal_completion_rate": float(reached.mean()),
        "safe_goal_completion_rate": float((reached & safe).mean()),
        "violation_free_rate": float(safe.mean()),
        "flight_time_success_mean_s": float(columns["flight_time"][reached].mean()) if reached.any() else None,
        "metrics": {key: {"mean": float(values.mean()), "std": float(values.std())} for key, values in columns.items()},
        "failure_counts": {reason: sum(reason in failure_reasons(record) for record in records) for reason in categories},
        "diagnostic_note": "Safe-goal rate and failure counts are supplementary, not an official competition score. Reward is not comparable across designs.",
    }
