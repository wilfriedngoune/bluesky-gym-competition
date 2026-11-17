"""Continue fresh-v6 at ten aircraft; optional training-only obstacle mixture.

Preserves the checkpoint's MDP, timing, action mapping and reward settings.
Weights transfer into a fresh PPO optimizer, matching train_zoo's existing
continuation behavior. Does not claim to resume optimizer/rollout state.
"""
import argparse
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE="models/ma_navigation_v5_fresh_v6/model.zip"
OUTPUT="models/ma_navigation_v6_continued/model"


def training_command(*,init_model=SOURCE,model=OUTPUT,timesteps=2_000_000,seed=20,
                     targeted_probability=0.,targeted_profile="mixed",eval_episodes=0,
                     checkpoint_freq=100_000,learning_rate=None,n_steps=None,batch_size=None,root=ROOT):
    if timesteps < 1 or seed < 0 or eval_episodes < 0 or checkpoint_freq < 0:
        raise ValueError("Use positive timesteps and nonnegative seeds/evaluation/checkpoint counts")
    if not math.isfinite(targeted_probability) or not 0 <= targeted_probability <= 1:
        raise ValueError("targeted_probability must be in [0,1]")
    if targeted_profile not in ("passage","mixed"):
        raise ValueError("targeted_profile must be passage or mixed")
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate<=0):
        raise ValueError("learning_rate must be positive and finite")
    if n_steps is not None and n_steps < 1 or batch_size is not None and batch_size < 2:
        raise ValueError("Use positive n_steps and batch_size >= 2")
    root=Path(root).resolve()
    source=Path(init_model);source=source if source.is_absolute() else root/source
    if not source.is_file():source=Path(str(source)+".zip")
    source=source.resolve()
    if not source.is_file():raise FileNotFoundError(source)
    output=Path(model);output=output if output.is_absolute() else root/output
    output=output.resolve();base=output.with_suffix("") if output.suffix==".zip" else output
    if Path(str(base)+".zip")==source:raise ValueError("Output must differ from initialization checkpoint")
    metrics=base.parent/"training_metrics.csv";checkpoints=base.parent/"checkpoints"
    if list(base.parent.glob(base.name+".*")) or metrics.exists() or checkpoints.exists() and any(checkpoints.iterdir()):
        raise FileExistsError(f"Experiment output already used: {base.parent}; choose a new --model directory")
    command=[sys.executable,"-u",str(root/"train_zoo.py"),"train",str(timesteps),
             "--mdp","navigation-v5","--n-agents","10","--seed",str(seed),
             "--init-model",str(source),"--model",str(base),"--metrics",str(metrics),
             "--checkpoint-freq",str(checkpoint_freq),"--checkpoint-dir",str(checkpoints),
             "--eval-episodes",str(eval_episodes),"--passage-training-probability",str(targeted_probability),
             "--passage-training-profile",targeted_profile]
    for flag,value in (("--learning-rate",learning_rate),("--n-steps",n_steps),("--batch-size",batch_size)):
        if value is not None:command.extend([flag,str(value)])
    return command


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-model",default=SOURCE)
    parser.add_argument("--model",default=OUTPUT)
    parser.add_argument("--timesteps",type=int,default=2_000_000)
    parser.add_argument("--seed",type=int,default=20)
    parser.add_argument("--targeted-probability",type=float,default=0.)
    parser.add_argument("--targeted-profile",choices=["passage","mixed"],default="mixed")
    parser.add_argument("--eval-episodes",type=int,default=0,help="separate final seed-42 evaluation; default skips it")
    parser.add_argument("--checkpoint-freq",type=int,default=100_000)
    parser.add_argument("--learning-rate",type=float,default=None)
    parser.add_argument("--n-steps",type=int,default=None)
    parser.add_argument("--batch-size",type=int,default=None)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    options=vars(args).copy();dry_run=options.pop("dry_run")
    try:command=training_command(**options)
    except (ValueError,FileNotFoundError,FileExistsError) as error:parser.error(str(error))
    print("Ten-aircraft continuation; saved rewards/timing retained; weights transfer with a new optimizer.",flush=True)
    print(subprocess.list2cmdline(command),flush=True)
    if not dry_run:subprocess.run(command,cwd=ROOT,check=True)


if __name__=="__main__":main()
