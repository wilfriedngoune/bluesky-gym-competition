"""Nonofficial parallel development check on the first N seed-42 scenarios.

Each worker advances the scenario RNG with resets only to its assigned index.
The competition environment consumes scenario RNG only on reset. Each flight
then uses the existing evaluation loop, simulator, scoring and saved settings.
Use the canonical evaluate_competition CLI for official reporting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time


def offset_reset(reset, scenario):
    """Return the assigned reset without changing the seed-42 scenario stream."""
    first = True
    def assigned_reset(*, seed=None, **kwargs):
        nonlocal first
        if first:
            first = False
            if seed != 42:
                raise ValueError("Development scenario offsets require seed 42")
            for index in range(scenario - 1):
                reset(seed=42 if index == 0 else None)
            return reset(seed=42 if scenario == 1 else None, **kwargs)
        return reset(seed=seed, **kwargs)
    return assigned_reset


def evaluate_scenario(task):
    scenario, model, prefix, buffer_nm, horizon = task
    import torch
    from scripts import evaluate_competition as evaluation
    torch.set_num_threads(1)
    started = time.perf_counter()
    act = evaluation.load_policy("ma", model, mdp="navigation-v5")
    # This dedicated experiment explicitly evaluates the search/filter hybrid.
    act.disable_lookahead = False
    act.lookahead_config = replace(act.lookahead_config, coordination_passes=1,
                                  safety_filter=True, safety_buffer_nm=buffer_nm,
                                  safety_horizon_seconds=horizon)
    factory = evaluation.make_env
    def assigned_env(*args, **kwargs):
        env = factory(*args, **kwargs)
        env.reset = offset_reset(env.reset, scenario)
        return env
    evaluation.make_env = assigned_env
    original_actions = evaluation.actions_for_agents
    decisions = 0
    def progress_actions(*args, **kwargs):
        nonlocal decisions
        actions = original_actions(*args, **kwargs)
        decisions += 1
        if decisions % 100 == 0:
            live_env = kwargs.get("env")
            print(f"Scenario {scenario}: {decisions} decisions, simulated time {live_env.sim_time:g}s",flush=True)
        return actions
    evaluation.actions_for_agents = progress_actions
    try:
        records = evaluation.run_multi_agent(1, act, 10, mdp="navigation-v5")
    finally:
        evaluation.make_env = factory
        evaluation.actions_for_agents = original_actions
    if len(records) != 10:
        raise RuntimeError(f"Scenario {scenario}: expected 10 completed aircraft records")
    result = dict(scenario=scenario, records=records, wall_seconds=time.perf_counter()-started,
                  lookahead_config=asdict(act.lookahead_config))
    Path(str(prefix)+f".scenario{scenario:02d}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def isolated_scenario(task):
    scenario, model, prefix, buffer_nm, horizon = task
    log_path = Path(str(prefix)+f".scenario{scenario:02d}.log")
    command = [sys.executable,"-u","-m","scripts.evaluate_safety_filter",
               "--model",model,"--out-prefix",prefix,"--worker-scenario",str(scenario),
               "--safety-buffer-nm",str(buffer_nm),"--safety-horizon-seconds",str(horizon)]
    with log_path.open("x") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                       creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    return json.loads(Path(str(prefix)+f".scenario{scenario:02d}.json").read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--out-prefix",required=True)
    parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--safety-buffer-nm",type=float,default=3.)
    parser.add_argument("--safety-horizon-seconds",type=float,default=60.)
    parser.add_argument("--worker-scenario",type=int,default=None,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1 <= args.episodes <= 1000 or not 1 <= args.workers <= 8:
        parser.error("Use 1..1000 scenarios and 1..8 workers")
    from bluesky_zoo.lookahead import LookaheadConfig
    LookaheadConfig(safety_filter=True,safety_buffer_nm=args.safety_buffer_nm,
                    safety_horizon_seconds=args.safety_horizon_seconds)
    if args.worker_scenario is not None:
        if not 1 <= args.worker_scenario <= 1000:
            parser.error("Invalid worker scenario")
        evaluate_scenario((args.worker_scenario,args.model,args.out_prefix,args.safety_buffer_nm,args.safety_horizon_seconds))
        return
    prefix=Path(args.out_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    if list(prefix.parent.glob(prefix.name+".*")):
        raise FileExistsError(f"Output prefix already used: {prefix}")
    started=time.perf_counter()
    settings=dict(model=str(Path(args.model).resolve()),model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  seed=42,mdp="navigation-v5",n_scenarios=args.episodes,n_agents=10,workers=args.workers,official=False,
                  requested_safety_buffer_nm=args.safety_buffer_nm,
                  requested_safety_horizon_seconds=args.safety_horizon_seconds,
                  completed=False,protocol="Independent processes replay reset-only seed-42 prefixes; original scoring loop per scenario.")
    settings_path=Path(str(prefix)+".settings.json")
    settings_path.write_text(json.dumps(settings,indent=2)+"\n")
    results=[]
    tasks=[(i,args.model,str(prefix),args.safety_buffer_nm,args.safety_horizon_seconds)
           for i in range(1,args.episodes+1)]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(isolated_scenario,task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                print(f"Completed {len(results)}/{args.episodes} (scenario {results[-1]['scenario']}); "
                      f"{time.perf_counter()-started:.1f}s elapsed",flush=True)
        records=[record for result in sorted(results,key=lambda x:x["scenario"]) for record in result["records"]]
        from scripts import evaluate_competition as evaluation
        evaluation.write_csv(records,str(prefix)+".csv")
        report=evaluation.build_report(records,kind="ma",phase="evaluation",n_agents=10,
            n_scenarios=args.episodes,seed=42,mdp="navigation-v5",model_path=args.model,source=str(prefix)+".csv")
        evaluation.write_report([report],prefix)
        settings.update(completed=True,lookahead_config=results[0]["lookahead_config"])
    finally:
        settings.update(wall_seconds=time.perf_counter()-started,completed_scenarios=sorted(r["scenario"] for r in results))
        settings_path.write_text(json.dumps(settings,indent=2)+"\n")


if __name__ == "__main__":
    main()
