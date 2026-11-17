"""Train v5 with 5-second decisions and 1-degree/1-knot command deadbands.

Default: continue the saved 5-second baseline with ten aircraft. Use --fresh
for fresh weights and a 2/4/6/10-aircraft curriculum. Output uses a separate
experiment directory and existing model/metrics files are never overwritten.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "models/ma_navigation_v5_separation/model.zip"
OUTPUT = "models/ma_navigation_v5_responsive/model"


def training_command(*, timesteps=1_000_000, eval_episodes=20, seed=4,
                     model=OUTPUT, init_model=None, fresh=False, root=ROOT):
    """Build an explicit reproducible experiment; leave legacy defaults intact."""
    if timesteps < (4 if fresh else 1) or eval_episodes < 0 or seed < 0:
        raise ValueError("Use positive training timesteps, nonnegative evaluation count and seed; fresh needs at least 4 timesteps")
    if fresh and init_model is not None:
        raise ValueError("Use either fresh weights or an initialization checkpoint")
    root = Path(root).resolve()
    output = Path(model)
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    base = output.with_suffix("") if output.suffix == ".zip" else output
    metrics = base.parent / "training_metrics.csv"
    for path in (Path(str(base) + ".zip"), Path(str(base) + ".json"), metrics):
        if path.exists():
            raise FileExistsError(f"Experiment output already exists: {path}. Choose a new --model directory.")
    command = [sys.executable, "-u", str(root / "train_zoo.py"), "train", str(timesteps),
               "--mdp", "navigation-v5", "--decision-seconds", "5",
               "--heading-deadband-deg", "1", "--speed-deadband-kt", "1",
               "--aircraft-buffer-nm", "8", "--aircraft-buffer-cost", "0.05",
               "--seed", str(seed), "--model", str(output), "--metrics", str(metrics),
               "--checkpoint-dir", str(base.parent / "checkpoints"),
               "--eval-episodes", str(eval_episodes)]
    if fresh:
        command.extend(["--curriculum", "2", "4", "6", "10"])
    else:
        source = Path(init_model or BASELINE)
        source = source if source.is_absolute() else root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {source}")
        if source == Path(str(base) + ".zip"):
            raise ValueError("The output must be separate from the initialization checkpoint")
        command.extend(["--init-model", str(source), "--n-agents", "10"])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--model", default=OUTPUT, help="new experiment model path")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fresh", action="store_true", help="fresh weights with the traffic curriculum")
    source.add_argument("--init-model", help="matching v5 checkpoint (default: the saved 5-second baseline)")
    parser.add_argument("--dry-run", action="store_true", help="print the exact command without training")
    args = parser.parse_args(argv)
    try:
        command = training_command(timesteps=args.timesteps, eval_episodes=args.eval_episodes,
                                   seed=args.seed, model=args.model, init_model=args.init_model, fresh=args.fresh)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print("Responsive v5: 5-second decisions; heading/speed deadbands 1 degree/1 knot.", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
