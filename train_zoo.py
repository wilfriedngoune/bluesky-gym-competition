"""Train a shared PPO policy using the single-agent navigation MDP.

    python train_zoo.py train 1000000 --init-model single_agent_model.zip
    python train_zoo.py replay multi_agent_model
    python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model

Each aircraft is a vector slot, backed by one BlueSky simulation. Timesteps
count agent slots (including absorbing slots after arrival), not sim seconds.
See docs/competition/MULTI_AGENT_PIPELINE.md for the reporting protocol.
After training, totals reports are saved and a separate 1000-scenario evaluation
runs by default. Use --eval-episodes 20 for a quick check or 0 to skip evaluation.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import check_for_correct_spaces

from bluesky_zoo.sb3_vec_env import SharedPolicyVecEnv
from bluesky_zoo.active_rollout_buffer import ActiveAircraftCallback, ActiveAircraftRolloutBuffer
from bluesky_zoo.centralized_critic import (
    CENTRALIZED_KEY, CentralizedCriticPolicy, CentralizedObservationWrapper,
    is_centralized_observation_space,
)
from bluesky_zoo.navigation_reward import make_reward_config, REWARD_COMPONENT_KEYS
from bluesky_gym.envs.common.command_filter import resolve_command_config
from bluesky_gym.envs.common.decision_timing import (
    forecast_segment_seconds, rescale_discount, resolve_decision_seconds,
)
from bluesky_zoo.passage_navigation import resolve_passage_config
from bluesky_zoo.strategic_navigation import resolve_zone_config
from bluesky_zoo.lookahead import LookaheadConfig, resolve_inference_lookahead_config
from scripts.competition_report import build_report, read_records, write_report


SEED = 0
N_AGENTS = 10
MODEL_PATH = "multi_agent_model"
METRIC_KEYS = [
    "waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area", "sector_exit_events",
    "time_outside_sector", "total_reward",
]
ZONE_CONFIG_FIELDS = (
    "grace_seconds", "detour_allowance_nm", "closing_speed_kt", "arrival_time_factor",
    "minimum_shrink_seconds", "outside_second_cost", "excess_scale_nm",
)


def make_env(mdp="navigation", n_agents=N_AGENTS, render_mode=None, reward_config=None, action_config=None,
             passage_config=None, zone_config=None, decision_seconds=None):
    """Use the same MDP for training, replay and official evaluation."""
    timing = ({"action_frequency": resolve_decision_seconds(mdp, override=decision_seconds)}
              if decision_seconds is not None else {})
    if mdp == "navigation-v5":
        from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv
        return MultiAgentStrategicEnv(n_agents=n_agents, render_mode=render_mode, reward_config=reward_config,
                                      action_config=action_config, passage_config=passage_config,
                                      zone_config=zone_config, **timing)
    if zone_config is not None:
        raise ValueError("Zone settings require --mdp navigation-v5")
    if mdp == "navigation-v4":
        from bluesky_zoo.passage_navigation import MultiAgentPassageEnv
        return MultiAgentPassageEnv(n_agents=n_agents, render_mode=render_mode, reward_config=reward_config,
                                    action_config=action_config, passage_config=passage_config, **timing)
    if passage_config is not None:
        raise ValueError("Passage settings require --mdp navigation-v4 or navigation-v5")
    if mdp == "navigation-v3":
        from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv
        return MultiAgentTrafficEnv(n_agents=n_agents, render_mode=render_mode,
                                    reward_config=reward_config, action_config=action_config, **timing)
    if mdp == "navigation-v2":
        from bluesky_zoo.multi_agent_reward_env import MultiAgentRewardEnv
        return MultiAgentRewardEnv(n_agents=n_agents, render_mode=render_mode,
                                   reward_config=reward_config, action_config=action_config, **timing)
    if mdp == "navigation":
        from bluesky_zoo.multi_agent_env import MultiAgentEnv
        return MultiAgentEnv(n_agents=n_agents, render_mode=render_mode, action_config=action_config, **timing)
    if mdp == "base":
        config = resolve_command_config(action_config)
        if config.heading_deadband_deg or config.speed_deadband_kt:
            raise ValueError("Command deadbands require a navigation MDP")
        from bluesky_zoo.competition_v0 import CompetitionZooEnv
        return CompetitionZooEnv(n_agents=n_agents, render_mode=render_mode, **timing)
    raise ValueError(f"Unknown MDP: {mdp}")


def make_vec_env(n_agents=N_AGENTS, mdp="navigation", reward_config=None, action_config=None, passage_config=None,
                 centralized=False, zone_config=None, decision_seconds=None):
    env = make_env(mdp=mdp, n_agents=n_agents, reward_config=reward_config, action_config=action_config,
                   passage_config=passage_config, zone_config=zone_config, decision_seconds=decision_seconds)
    if centralized:
        env = CentralizedObservationWrapper(env)
    return SharedPolicyVecEnv(env)


class CompetitionMetricsCallback(BaseCallback):
    """Log each aircraft's fixed final metrics exactly once, on departure."""

    def __init__(self, csv_path="multi_agent_training_metrics.csv", verbose=0,
                 reward_component_keys=()):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.episode_index = 0
        self.reward_component_keys = tuple(reward_component_keys)
        self.fieldnames = [
            "episode", "scenario_index", "agent_id", "total_timesteps",
            *METRIC_KEYS, *(f"reward_{key}" for key in self.reward_component_keys),
        ]

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.fieldnames).writeheader()

    def _on_step(self):
        rows = []
        for info in self.locals.get("infos", []):
            # Absorbing slots have no final_metrics and must not create rows.
            if "final_metrics" not in info:
                continue
            rows.append({
                "episode": self.episode_index,
                "scenario_index": info["scenario_index"],
                "agent_id": info["agent_id"],
                "total_timesteps": self.num_timesteps,
                **{key: info["final_metrics"][key] for key in METRIC_KEYS},
                **{f"reward_{key}": info["reward_components"][key]
                   for key in self.reward_component_keys},
            })
            self.episode_index += 1
        if rows:
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.fieldnames).writerows(rows)
        return True


def train(total_timesteps=250_000, *, model_path=MODEL_PATH,
          csv_path="multi_agent_training_metrics.csv", init_model=None,
          n_agents=N_AGENTS, mdp="navigation", seed=SEED, n_steps=1024,
          batch_size=256, learning_rate=3e-4, n_epochs=10, device="auto",
          checkpoint_freq=100_000, checkpoint_dir="models/multi_agent/checkpoints",
          torch_threads=1, gamma=None, eval_episodes=1000,
          conflict_horizon_seconds=None, conflict_risk_second_cost=None,
          conflict_buffer_nm=None, mask_inactive=True,
          sector_buffer_second_cost=None, sector_buffer_nm=None,
          heading_deadband_deg=None, speed_deadband_kt=None,
          obstacle_buffer_nm=None, obstacle_release_nm=None, max_heading_offset_deg=None,
          lookahead_nm=None, passage_training_probability=0.0, centralized=False,
          zone_config=None, lookahead_simulations=None, lookahead_depth=None,
          aircraft_buffer_second_cost=None, aircraft_buffer_nm=None, decision_seconds=None,
          passage_training_profile="passage"):
    """Fine-tune SA policy weights, or train the same architecture from scratch."""
    if total_timesteps < 1 or n_steps < 1 or batch_size < 2 or n_epochs < 1:
        raise ValueError("Use positive timesteps, n_steps and n_epochs, and batch_size >= 2")
    if n_agents < 2 or n_steps * n_agents < 2 or torch_threads < 1:
        raise ValueError("Use at least two aircraft and positive torch_threads")
    if learning_rate <= 0 or checkpoint_freq < 0:
        raise ValueError("learning_rate must be positive and checkpoint_freq nonnegative")
    requested_gamma = gamma
    gamma = gamma if gamma is not None else (0.998 if mdp in {"navigation-v3", "navigation-v4", "navigation-v5"} else 0.99)
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if eval_episodes < 0:
        raise ValueError("eval_episodes must be nonnegative; use 0 to skip evaluation")
    if not np.isfinite(passage_training_probability) or not 0 <= passage_training_probability <= 1:
        raise ValueError("passage_training_probability must be in [0, 1]")
    if passage_training_profile not in ("passage", "mixed"):
        raise ValueError("passage_training_profile must be passage or mixed")
    passage_overrides = dict(obstacle_buffer_nm=obstacle_buffer_nm, obstacle_release_nm=obstacle_release_nm,
                             max_heading_offset_deg=max_heading_offset_deg, lookahead_nm=lookahead_nm)
    if mdp not in {"navigation-v4", "navigation-v5"} and (passage_training_probability or passage_training_profile != "passage" or any(v is not None for v in passage_overrides.values())):
        raise ValueError("Passage training/settings require --mdp navigation-v4 or navigation-v5")
    if mdp != "navigation-v5" and any(value is not None for value in (zone_config, lookahead_simulations, lookahead_depth)):
        raise ValueError("Zone/search settings require --mdp navigation-v5")
    if mdp == "navigation-v5" and centralized:
        raise ValueError("navigation-v5 lookahead currently supports a decentralized PPO critic; omit --centralized-critic")
    torch.set_num_threads(torch_threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pretrained = PPO.load(init_model, device=device) if init_model is not None else None
    saved_disable_lookahead = bool(getattr(pretrained, "competition_disable_lookahead", False))
    saved_distillation = getattr(pretrained, "competition_distillation", None)
    saved_seconds = getattr(pretrained, "competition_decision_seconds", None)
    previous_seconds = resolve_decision_seconds(mdp, saved=saved_seconds)
    decision_seconds = resolve_decision_seconds(mdp, saved=saved_seconds, override=decision_seconds)
    if requested_gamma is None and pretrained is not None:
        gamma = getattr(pretrained, "gamma", gamma)
    if requested_gamma is None:
        gamma = rescale_discount(gamma, previous_seconds, decision_seconds)
    reward_config = make_reward_config(
        mdp, gamma=gamma, conflict_horizon_seconds=conflict_horizon_seconds,
        conflict_risk_second_cost=conflict_risk_second_cost,
        conflict_buffer_nm=conflict_buffer_nm,
        sector_buffer_second_cost=sector_buffer_second_cost, sector_buffer_nm=sector_buffer_nm,
        aircraft_buffer_second_cost=aircraft_buffer_second_cost, aircraft_buffer_nm=aircraft_buffer_nm,
        saved=getattr(pretrained, "competition_reward_config", None),
    )
    action_config = resolve_command_config(
        getattr(pretrained, "competition_action_config", None),
        heading_deadband_deg=heading_deadband_deg, speed_deadband_kt=speed_deadband_kt,
    )
    passage_config = (resolve_passage_config(getattr(pretrained, "competition_passage_config", None),
                                             **passage_overrides) if mdp in {"navigation-v4", "navigation-v5"} else None)
    search_config = None
    if mdp == "navigation-v5":
        zone_overrides = zone_config if isinstance(zone_config, dict) else (asdict(zone_config) if zone_config is not None else {})
        zone_config = resolve_zone_config(getattr(pretrained, "competition_zone_config", None), **zone_overrides)
        saved_search = getattr(pretrained, "competition_lookahead_config", None)
        search_config = LookaheadConfig(**saved_search) if saved_search is not None else LookaheadConfig()
        search_config = replace(search_config, discount=gamma, **{
            key: value for key, value in {"simulations": lookahead_simulations, "max_depth": lookahead_depth}.items()
            if value is not None
        })
        search_config = replace(search_config, rollout_seconds=forecast_segment_seconds(
            search_config.rollout_seconds, decision_seconds))
    venv = make_vec_env(n_agents=n_agents, mdp=mdp, reward_config=reward_config, action_config=action_config,
                        passage_config=passage_config, centralized=centralized, zone_config=zone_config,
                        decision_seconds=decision_seconds)
    try:
        if passage_training_probability:
            from bluesky_zoo.passage_scenarios import PassageTrainingGenerator
            venv.env.scenario_generator = PassageTrainingGenerator(
                venv.env.scenario_generator, n_agents, passage_training_probability, venv.env.center,
                profile=passage_training_profile)
        if centralized:
            policy = CentralizedCriticPolicy
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5,
                global_key=CENTRALIZED_KEY,
            )
        else:
            policy = "MultiInputPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5,
            )
        if init_model is not None:
            if centralized and not is_centralized_observation_space(pretrained.observation_space):
                raise ValueError(
                    "--centralized-critic cannot initialize from a non-centralized checkpoint: its "
                    "observation space lacks the centralized critic key. Start fresh or use a "
                    "centralized checkpoint.")
            if not centralized and is_centralized_observation_space(pretrained.observation_space):
                raise ValueError(
                    "The checkpoint was trained with --centralized-critic; add the flag (and a "
                    "matching --mdp) or start fresh to initialize from it.")
            try:
                check_for_correct_spaces(venv, pretrained.observation_space, pretrained.action_space)
            except ValueError as error:
                raise ValueError(
                    "Checkpoint spaces do not match this MDP. navigation-v3 adds nine traffic "
                    "slots, masks and remaining time: start without --init-model, or use a "
                    "matching v3 checkpoint. V2/SA checkpoints cannot load directly into v3. "
                    "navigation-v4 adds clearance rays and changes heading semantics; start fresh or use matching v4 weights. "
                    "navigation-v5 adds shrinking-zone observations; start fresh or use matching v5 weights."
                ) from error
            policy_kwargs = pretrained.policy_kwargs

        model = PPO(
            policy, venv, learning_rate=learning_rate,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            policy_kwargs=policy_kwargs, seed=seed, device=device, verbose=1,
            rollout_buffer_class=ActiveAircraftRolloutBuffer if mask_inactive else None,
        )
        if pretrained is not None:
            model.policy.load_state_dict(pretrained.policy.state_dict())
            print(f"Initialized policy/value weights from {init_model}; starting a fresh optimizer.")
            del pretrained

        model.competition_mdp = mdp
        model.competition_decision_seconds = decision_seconds
        model.competition_action_config = asdict(action_config)
        model.competition_passage_config = asdict(passage_config) if passage_config is not None else None
        model.competition_zone_config = asdict(zone_config) if zone_config is not None else None
        model.competition_lookahead_config = asdict(search_config) if search_config is not None else None
        model.competition_disable_lookahead = saved_disable_lookahead
        if saved_distillation is not None:
            model.competition_distillation = saved_distillation
        model.competition_reward_config = (
            asdict(venv.get_attr("reward_config", indices=[0])[0])
            if reward_config is not None else None
        )
        callbacks = [ActiveAircraftCallback(), CompetitionMetricsCallback(
            csv_path, reward_component_keys=tuple(getattr(venv.env, "reward_component_keys", REWARD_COMPONENT_KEYS))
            if reward_config is not None else (),
        )]
        if checkpoint_freq:
            callbacks.append(CheckpointCallback(
                save_freq=max(checkpoint_freq // venv.num_envs, 1),
                save_path=checkpoint_dir, name_prefix=Path(model_path).stem,
            ))
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
        output = Path(model_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        metadata = {
            "mdp": mdp, "n_agents": n_agents, "n_obstacles": 5,
            "seed": seed, "requested_timesteps": total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "timestep_unit": "agent slots, including absorbing slots",
            "mask_inactive": mask_inactive,
            "centralized_critic": centralized,
            "action_config": model.competition_action_config,
            "passage_config": model.competition_passage_config,
            "zone_config": model.competition_zone_config,
            "lookahead_config": model.competition_lookahead_config,
            "inference_disable_lookahead": model.competition_disable_lookahead,
            "distillation": saved_distillation,
            "training_action_selection": "PPO actor only; inference follows saved lookahead mode"
            if search_config is not None else "PPO actor",
            "passage_training_probability": passage_training_probability,
            "passage_training_profile": passage_training_profile,
            "active_timesteps": model.competition_active_timesteps,
            "absorbing_timesteps": model.num_timesteps - model.competition_active_timesteps,
            "init_model": str(init_model) if init_model is not None else None,
            "n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs,
            "learning_rate": learning_rate, "gamma": model.gamma,
            "gae_lambda": model.gae_lambda, "policy_kwargs": model.policy_kwargs,
            "reward_config": model.competition_reward_config,
            "observation_shapes": {key: list(space.shape)
                                   for key, space in venv.observation_space.spaces.items()},
            "action_frequency": venv.get_attr("action_frequency", indices=[0])[0],
            "decision_seconds": decision_seconds,
            "max_decisions_per_aircraft": 3000 // decision_seconds,
            "gamma_rescaled_for_timing": requested_gamma is None and previous_seconds != decision_seconds,
            "metrics_path": str(csv_path), "device": str(model.device),
            "torch_threads": torch_threads,
            "versions": {name: version(name) for name in
                         ("torch", "stable-baselines3", "gymnasium", "pettingzoo", "bluesky-simulator")},
        }
        metadata_path = Path(str(output) + ".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Saved policy to {output} and configuration to {metadata_path}.")
        print("Evaluate with: python -m scripts.evaluate_competition --env ma "
              f"--mdp {mdp} --model {output}")
    finally:
        venv.close()
    # A fresh process prevents a second environment from sharing BlueSky state.
    finish_run_reports(output, csv_path, mdp=mdp, n_agents=n_agents,
                       eval_episodes=eval_episodes, metadata_path=metadata_path, training_seed=seed)
    return model


def finish_run_reports(model_path, csv_path, *, mdp, n_agents, eval_episodes,
                       metadata_path=None, training_seed=None):
    """Always report completed training flights, then evaluate the final policy."""
    output = Path(model_path)
    base = output.with_suffix("") if output.suffix == ".zip" else output
    training_prefix = Path(str(base) + ".training_report")
    training = build_report(read_records(csv_path), kind="ma", phase="training",
                            n_agents=n_agents, mdp=mdp, seed=training_seed,
                            model_path=output, source=csv_path)
    write_report([training], training_prefix)
    evaluation_csv = Path(str(base) + ".evaluation.csv").resolve()
    status = {
        "training_report": str(training_prefix) + ".md",
        "evaluation_csv": str(evaluation_csv) if eval_episodes else None,
        "evaluation_report": str(evaluation_csv.with_suffix(".md")) if eval_episodes else None,
        "evaluation_episodes": eval_episodes, "evaluation_seed": 42,
        "evaluation_n_agents": 10,
        "evaluation_status": "pending" if eval_episodes else "skipped",
    }

    def save_status():
        if metadata_path is not None:
            path = Path(metadata_path)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["reports"] = status
            path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_status()
    if not eval_episodes:
        print("Evaluation skipped (--eval-episodes 0); training statistics are not competition results.")
        return
    model_file = output if output.suffix == ".zip" else Path(str(output) + ".zip")
    print(f"Starting separate deterministic evaluation: {eval_episodes} scenarios, "
          "10 aircraft, seed 42. This adds evaluation time after training.", flush=True)
    command = [sys.executable, "-u", "-m", "scripts.evaluate_competition",
               "--env", "ma", "--mdp", mdp, "--model", str(model_file.resolve()),
               "--episodes", str(eval_episodes), "--n-agents", "10", "--out", str(evaluation_csv)]
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
    except (subprocess.CalledProcessError, OSError):
        status["evaluation_status"] = "failed"
        save_status()
        print("Training model and training report are saved; evaluation failed. "
              "Run scripts.evaluate_competition separately to retry.")
        raise
    status["evaluation_status"] = "complete"
    save_status()


def replay(model_path=MODEL_PATH, n_episodes=3, seed=SEED, *,
           n_agents=N_AGENTS, mdp="navigation", headless=False,
           heading_deadband_deg=None, speed_deadband_kt=None, passage_config=None, trajectory_csv=None,
           lookahead_simulations=None, lookahead_depth=None, lookahead_passes=None, disable_lookahead=None,
           safety_filter=None, safety_buffer_nm=None, safety_horizon_seconds=None,
           scenario_start=1, max_steps=None, search_trace_prefix=None,
           search_trace_start=40., search_trace_end=400., search_trace_agents=None,
           search_trace_neighbours=None, lookahead_preset="saved"):
    """Replay only live aircraft, using the same Dict policy as training."""
    from scripts.evaluate_competition import actions_for_agents, load_policy, policy_reward_kwargs
    if scenario_start < 1 or n_episodes < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("Use positive scenario_start, episode count and optional max_steps")
    if search_trace_prefix is not None and mdp != "navigation-v5":
        raise ValueError("Search diagnostics currently require --mdp navigation-v5")
    torch.set_num_threads(1)
    act = load_policy("ma", model_path, mdp=mdp)
    model = act.model
    search_overridden = (lookahead_preset not in (None, "saved")
                         or any(value is not None for value in
                                (lookahead_simulations, lookahead_depth, lookahead_passes)))
    if disable_lookahead is None:
        disable_lookahead = bool(getattr(act, "disable_lookahead", False)) and not search_overridden
    safety_options = {"safety_filter": safety_filter, "safety_buffer_nm": safety_buffer_nm,
                      "safety_horizon_seconds": safety_horizon_seconds}
    if any(value is not None for value in safety_options.values()) and (mdp != "navigation-v5" or disable_lookahead):
        raise ValueError("Safety filter settings require navigation-v5 with lookahead enabled")
    if mdp != "navigation-v5" and (disable_lookahead or search_overridden):
        raise ValueError("Lookahead settings require --mdp navigation-v5")
    if disable_lookahead and search_overridden:
        raise ValueError("Lookahead settings require lookahead enabled")
    if mdp == "navigation-v5":
        config = getattr(act, "lookahead_config", None)
        config = LookaheadConfig(**config) if isinstance(config, dict) else config
        if config is None:
            config = LookaheadConfig(discount=model.gamma)
        config = resolve_inference_lookahead_config(
            config, preset=lookahead_preset, simulations=lookahead_simulations,
            max_depth=lookahead_depth, coordination_passes=lookahead_passes)
        act.lookahead_config = replace(config, **{
            key: value for key, value in safety_options.items() if value is not None
        })
        act.disable_lookahead = disable_lookahead
        if act.lookahead_config.safety_filter and not disable_lookahead:
            print(f"Separation filter enabled: {5. + act.lookahead_config.safety_buffer_nm:g} NM target, "
                  f"{act.lookahead_config.safety_horizon_seconds:g}s held-command forecast; scoring remains 5 NM.")
        if disable_lookahead:
            print("Replay uses the PPO actor without lookahead; this differs from the saved search setup.")
        elif search_overridden:
            print(f"Replay lookahead override: {act.lookahead_config.simulations} simulations, "
                  f"depth {act.lookahead_config.max_depth}, {act.lookahead_config.coordination_passes} pass(es). "
                  "Behavior may differ from the saved search setup.")
    action_config = resolve_command_config(
        getattr(model, "competition_action_config", None),
        heading_deadband_deg=heading_deadband_deg, speed_deadband_kt=speed_deadband_kt,
    )
    if heading_deadband_deg is not None or speed_deadband_kt is not None:
        print("Replay command deadband override: behavior may differ from the trained policy setup.")
    saved_kwargs = policy_reward_kwargs(act, mdp)
    saved_kwargs["action_config"] = action_config
    saved_kwargs["passage_config"] = passage_config or getattr(model, "competition_passage_config", None)
    env = make_env(mdp=mdp, n_agents=n_agents, render_mode=None if headless else "human",
                   **saved_kwargs)
    if is_centralized_observation_space(model.observation_space):
        env = CentralizedObservationWrapper(env)
    trace = None
    search_trace = None
    completed = False
    try:
        from scripts.navigation_trace import NavigationTrace
        trace = NavigationTrace(trajectory_csv)
        if search_trace_prefix is not None:
            from scripts.search_trace import SearchTrace
            search_trace = SearchTrace(
                search_trace_prefix, env, act, model_path=model_path, seed=seed, scenario_start=scenario_start,
                start=search_trace_start, end=search_trace_end,
                agents=search_trace_agents, neighbours=search_trace_neighbours,
            )
            act.search_observer = search_trace
        saved_mdp = getattr(model, "competition_mdp", None)
        if saved_mdp is not None and saved_mdp != mdp:
            raise ValueError(f"Checkpoint was trained with --mdp {saved_mdp}; requested --mdp {mdp}")
        if (env.observation_space(env.possible_agents[0]) != model.observation_space
                or env.action_space(env.possible_agents[0]) != model.action_space):
            raise ValueError("Checkpoint spaces do not match this MDP; check --mdp.")
        # Scenario generation uses its own reset RNG. Advancing this stream
        # does not require replaying the preceding policies or modifying seeds.
        for skipped in range(scenario_start - 1):
            env.reset(seed=seed if skipped == 0 else None)
        for episode in range(scenario_start - 1, scenario_start - 1 + n_episodes):
            observations, _ = env.reset(seed=seed if episode == 0 else None)
            if search_trace is not None:
                search_trace.begin_episode(episode)
            steps = 0
            while env.agents and (max_steps is None or steps < max_steps):
                agents = list(env.agents)
                before_time = env.sim_time if search_trace is not None else None
                if search_trace is not None:
                    search_trace.capture_actual()
                before = trace.before(env, agents)
                actions = actions_for_agents(act, observations, agents, env=env)
                observations, _, terms, truncs, infos = env.step(actions)
                if search_trace is not None:
                    search_trace.record_commands(before_time, actions)
                    search_trace.capture_actual()
                trace.after(env, before, actions, infos, episode)
                steps += 1
                for agent, info in infos.items():
                    if terms[agent] or truncs[agent]:
                        print(f"episode={episode} agent={agent} metrics={info}")
            if env.agents:
                print(f"Diagnostic replay capped after {steps} decisions in scenario {episode + 1}; "
                      f"{len(env.agents)} unfinished flights have no final evaluation record.")
        completed = True
    finally:
        try:
            if trace is not None:
                trace.close()
            if search_trace is not None:
                search_trace.close(completed=completed)
        finally:
            env.close()


def train_curriculum(args):
    """Train fixed-shape v3 stages in isolated processes; evaluate the last one."""
    stages = args.curriculum
    if (args.mdp not in {"navigation-v3", "navigation-v4", "navigation-v5"} or not stages or stages[-1] != 10
            or stages != sorted(set(stages)) or any(not 2 <= n <= 10 for n in stages)):
        raise ValueError("Use navigation-v3, navigation-v4 or navigation-v5 with increasing curriculum stages from 2 to 10, ending at 10")
    if args.n_agents != 10:
        raise ValueError("Use --curriculum without a separate --n-agents setting")
    if args.timesteps < len(stages) or args.eval_episodes < 0:
        raise ValueError("Provide at least one timestep per stage and nonnegative evaluation episodes")
    resolve_decision_seconds(args.mdp, override=getattr(args, "decision_seconds", None))
    # Validate the complete profile before launching any potentially long stage.
    resolve_command_config(heading_deadband_deg=getattr(args, "heading_deadband_deg", None),
                           speed_deadband_kt=getattr(args, "speed_deadband_kt", None))
    passage_options = {name: getattr(args, name, None) for name in
                       ("obstacle_buffer_nm", "obstacle_release_nm", "max_heading_offset_deg", "lookahead_nm")}
    probability = getattr(args, "passage_training_probability", 0)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("passage_training_probability must be in [0, 1]")
    if args.mdp in {"navigation-v4", "navigation-v5"}:
        resolve_passage_config(**passage_options)
    elif probability or any(value is not None for value in passage_options.values()):
        raise ValueError("Passage training/settings require --mdp navigation-v4 or navigation-v5")
    zone_options = {name: getattr(args, "zone_" + name, None) for name in ZONE_CONFIG_FIELDS}
    search_options = {name: getattr(args, option, None) for name, option in
                      (("simulations", "lookahead_simulations"), ("max_depth", "lookahead_depth"))}
    if args.mdp == "navigation-v5":
        if getattr(args, "centralized_critic", False):
            raise ValueError("navigation-v5 lookahead currently supports a decentralized PPO critic; omit --centralized-critic")
        resolve_zone_config(**zone_options)
        LookaheadConfig(discount=args.gamma if args.gamma is not None else 0.998,
                        **{key: value for key, value in search_options.items() if value is not None})
    elif any(value is not None for value in (*zone_options.values(), *search_options.values())):
        raise ValueError("Zone/search settings require --mdp navigation-v5")
    make_reward_config(args.mdp, gamma=args.gamma, conflict_horizon_seconds=args.conflict_horizon,
                       conflict_risk_second_cost=args.conflict_risk_cost,
                       conflict_buffer_nm=args.conflict_buffer_nm,
                       sector_buffer_second_cost=getattr(args, "sector_buffer_cost", None),
                       sector_buffer_nm=getattr(args, "sector_buffer_nm", None),
                       aircraft_buffer_second_cost=getattr(args, "aircraft_buffer_cost", None),
                       aircraft_buffer_nm=getattr(args, "aircraft_buffer_nm", None))
    output = Path(args.model).resolve()
    base = output.with_suffix("") if output.suffix == ".zip" else output
    directory = base.parent / (base.name + "_curriculum")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "curriculum.json"
    manifest = {"mdp": args.mdp, "status": "running", "requested_total_timesteps": args.timesteps,
                "stages": [], "note": "Stage weights transfer with fresh optimizers; rollout budgets may overshoot."}
    source = Path(args.init_model).resolve() if args.init_model else None
    common = {
        "--n-steps": args.n_steps, "--batch-size": args.batch_size,
        "--learning-rate": args.learning_rate, "--n-epochs": args.n_epochs,
        "--device": args.device, "--torch-threads": args.torch_threads,
        "--checkpoint-freq": args.checkpoint_freq, "--gamma": args.gamma,
        "--conflict-horizon": args.conflict_horizon,
        "--conflict-risk-cost": args.conflict_risk_cost,
        "--conflict-buffer-nm": args.conflict_buffer_nm,
        "--sector-buffer-cost": getattr(args, "sector_buffer_cost", None),
        "--sector-buffer-nm": getattr(args, "sector_buffer_nm", None),
        "--aircraft-buffer-cost": getattr(args, "aircraft_buffer_cost", None),
        "--aircraft-buffer-nm": getattr(args, "aircraft_buffer_nm", None),
        "--decision-seconds": getattr(args, "decision_seconds", None),
        "--heading-deadband-deg": getattr(args, "heading_deadband_deg", None),
        "--speed-deadband-kt": getattr(args, "speed_deadband_kt", None),
        "--obstacle-buffer-nm": getattr(args, "obstacle_buffer_nm", None),
        "--obstacle-release-nm": getattr(args, "obstacle_release_nm", None),
        "--max-heading-offset-deg": getattr(args, "max_heading_offset_deg", None),
        "--lookahead-nm": getattr(args, "lookahead_nm", None),
        "--passage-training-probability": probability if probability else None,
        "--passage-training-profile": getattr(args, "passage_training_profile", "passage"),
        "--lookahead-simulations": getattr(args, "lookahead_simulations", None),
        "--lookahead-depth": getattr(args, "lookahead_depth", None),
        **{"--zone-" + name.replace("_", "-"): value for name, value in zone_options.items()},
    }
    for index, count in enumerate(stages):
        last = index == len(stages) - 1
        stage_dir = directory / f"{index:02d}_{count}_agents"
        target = output if last else stage_dir / "model"
        metrics = Path(args.metrics).resolve() if last else stage_dir / "training_metrics.csv"
        checkpoints = Path(args.checkpoint_dir).resolve() if last else stage_dir / "checkpoints"
        budget = args.timesteps // len(stages) + int(index < args.timesteps % len(stages))
        stage = {"n_agents": count, "requested_timesteps": budget,
                 "seed": args.seed + index, "model": str(target), "metrics": str(metrics),
                 "init_model": str(source) if source else None, "status": "running"}
        manifest["stages"].append(stage)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "train", str(budget),
                   "--mdp", args.mdp, "--n-agents", str(count), "--seed", str(args.seed + index),
                   "--model", str(target), "--metrics", str(metrics),
                   "--checkpoint-dir", str(checkpoints),
                   "--eval-episodes", str(args.eval_episodes if last else 0)]
        command.extend(item for flag, value in common.items() if value is not None
                       for item in (flag, str(value)))
        if source is not None:
            command.extend(["--init-model", str(source)])
        if getattr(args, "include_inactive_samples", False):
            command.append("--include-inactive-samples")
        if getattr(args, "centralized_critic", False):
            command.append("--centralized-critic")
        print(f"Curriculum stage {index + 1}/{len(stages)}: {count} aircraft, "
              f"{budget} requested timesteps.", flush=True)
        try:
            subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
        except (subprocess.CalledProcessError, OSError):
            stage["status"] = "failed"
            manifest["status"] = "failed"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            raise
        stage["status"] = "complete"
        metadata_path = Path(str(target) + ".json")
        if metadata_path.exists():
            stage["actual_timesteps"] = json.loads(metadata_path.read_text(encoding="utf-8"))["actual_timesteps"]
        source = target if target.suffix == ".zip" else Path(str(target) + ".zip")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    manifest["actual_total_timesteps"] = (
        sum(stage["actual_timesteps"] for stage in manifest["stages"])
        if all("actual_timesteps" in stage for stage in manifest["stages"]) else None
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["curriculum"] = {"stages": stages, "manifest": str(manifest_path),
                                  "requested_total_timesteps": args.timesteps,
                                  "actual_total_timesteps": manifest["actual_total_timesteps"]}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Curriculum complete. Final model: {source}. Stage manifest: {manifest_path}")


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original numeric timestep entry point.
    if not args_list or args_list[0].isdigit():
        args_list.insert(0, "train")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train or fine-tune a shared PPO policy")
    training.add_argument("timesteps", type=int, nargs="?", default=250_000)
    training.add_argument("--model", default=MODEL_PATH)
    training.add_argument("--metrics", default="multi_agent_training_metrics.csv")
    training.add_argument("--init-model", help="compatible SA or MA PPO checkpoint to initialize weights")
    training.add_argument("--include-inactive-samples", action="store_true",
                          help="legacy PPO updates including departed aircraft padding, for comparison")
    training.add_argument("--n-steps", type=int, default=1024)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--gamma", type=float, default=None,
                          help="PPO/shaping discount (v3/v4/v5 default 0.998; other MDPs 0.99)")
    training.add_argument("--conflict-horizon", type=float, default=None,
                          help="predictive warning horizon in seconds (v3: 180; v2: 120)")
    training.add_argument("--conflict-risk-cost", type=float, default=None,
                          help="maximum predictive warning cost per second (v3: 0.05; v2: 0.02)")
    training.add_argument("--conflict-buffer-nm", type=float, default=None,
                          help="predictive DCPA warning distance in NM (default: 8)")
    training.add_argument("--sector-buffer-cost", type=float, default=None,
                          help="optional boundary proximity cost per second (default: 0, disabled)")
    training.add_argument("--sector-buffer-nm", type=float, default=None,
                          help="reward-only warning distance inside the sector (default: 5 NM)")
    training.add_argument("--n-epochs", type=int, default=10)
    training.add_argument("--device", default="auto")
    training.add_argument("--torch-threads", type=int, default=1)
    training.add_argument("--checkpoint-freq", type=int, default=100_000,
                          help="agent-slot timesteps between backups; 0 disables backups")
    training.add_argument("--checkpoint-dir", default="models/multi_agent/checkpoints")
    training.add_argument("--eval-episodes", type=int, default=1000,
                          help="post-training evaluation scenarios (default: official 1000; 0 skips)")
    training.add_argument("--curriculum", type=int, nargs="+", default=None,
                          help="v3/v4/v5 traffic stages, e.g. 2 4 6 10; splits the timestep budget across stages")
    training.add_argument("--passage-training-probability", type=float, default=0.0,
                          help="v4/v5 only: fraction of custom 3..8 NM passage scenarios during training")
    training.add_argument("--passage-training-profile", choices=["passage", "mixed"], default="passage",
                          help="training-only scenario mixture: passages, or passages plus randomized detours")
    training.add_argument("--centralized-critic", action="store_true",
                          help="use a CTDE policy: decentralized actor + centralized critic "
                               "(adds a joint-state observation key consumed by the value net only)")
    training.add_argument("--aircraft-buffer-nm", type=float, default=None,
                          help="centre-to-centre aircraft warning radius, above 5 NM (default: 8)")
    training.add_argument("--aircraft-buffer-cost", type=float, default=None,
                          help="per-neighbour saturated proximity cost per second (fresh v5 default: 0.05)")
    training.add_argument("--decision-seconds", type=int, default=None,
                          help="hold heading/speed targets between decisions (e.g. 15); saved value or navigation default 5")
    playback = commands.add_parser("replay", aliases=["play"], help="render a saved policy")
    playback.add_argument("model_path", nargs="?", default=MODEL_PATH)
    playback.add_argument("--episodes", type=int, default=3)
    playback.add_argument("--headless", action="store_true")
    playback.add_argument("--trajectory-csv", help="log per-aircraft geometry, actions and metrics during replay")
    playback.add_argument("--scenario-start", type=int, default=1,
                          help="1-based scenario in the seeded reset stream for targeted diagnostic replay")
    playback.add_argument("--max-steps", type=int, default=None,
                          help="optional diagnostic decision cap per replay scenario")
    playback.add_argument("--search-trace-prefix", help="write candidate, forecast, actual encounter and command CSVs")
    playback.add_argument("--search-trace-start", type=float, default=40., help="first root decision time to diagnose")
    playback.add_argument("--search-trace-end", type=float, default=400., help="last root decision time to diagnose")
    playback.add_argument("--search-trace-agents", nargs="+", help="root aircraft to diagnose (default: all)")
    playback.add_argument("--search-trace-neighbours", nargs="+", help="neighbours to log (default: all)")
    inference = playback.add_mutually_exclusive_group()
    inference.add_argument("--no-lookahead", dest="no_lookahead", action="store_true", default=None,
                          help="replay actor only; default follows saved checkpoint mode")
    inference.add_argument("--with-lookahead", dest="no_lookahead", action="store_false",
                          help="explicitly enable v5 search, including for a distilled actor")
    for name in ZONE_CONFIG_FIELDS:
        training.add_argument("--zone-" + name.replace("_", "-"), type=float, default=None,
                              help="v5 only: override the saved/default shrinking-zone " + name.replace("_", " "))
    for command in (training, playback):
        command.add_argument("--obstacle-buffer-nm", type=float, default=None)
        command.add_argument("--obstacle-release-nm", type=float, default=None)
        command.add_argument("--max-heading-offset-deg", type=float, default=None)
        command.add_argument("--lookahead-nm", type=float, default=None)
        command.add_argument("--heading-deadband-deg", type=float, default=None,
                             help="retain heading commands below this change (default: saved setting or 0)")
        command.add_argument("--speed-deadband-kt", type=float, default=None,
                             help="retain speed commands below this change in knots (default: saved setting or 0)")
        command.add_argument("--lookahead-simulations", type=int, default=None,
                             help="v5 MCTS simulation budget per aircraft decision (default: saved setting or 24)")
        command.add_argument("--lookahead-depth", type=int, default=None,
                             help="v5 maximum search depth (default: saved setting or 6)")
        command.add_argument("--mdp", choices=["navigation", "navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5", "base"], default="navigation")
        command.add_argument("--n-agents", type=int, default=N_AGENTS)
        command.add_argument("--seed", type=int, default=SEED)
    playback.add_argument("--lookahead-preset", choices=["saved", "fast"], default="saved",
                          help="saved settings or fast: 12 simulations, depth 3, one pass; explicit overrides win")
    playback.add_argument("--lookahead-passes", type=int, choices=[1, 2], default=None,
                          help="v5 coordination passes sharing the total simulation budget")
    playback.add_argument("--safety-filter", action=argparse.BooleanOptionalAction, default=None,
                          help="v5 optional separation screen (default: saved setting or off)")
    playback.add_argument("--safety-buffer-nm", type=float, default=None,
                          help="extra NM above the official 5 NM minimum (default: saved setting or 3)")
    playback.add_argument("--safety-horizon-seconds", type=float, default=None,
                          help="held-command safety forecast horizon, at most 120s (default: saved setting or 60)")
    args = parser.parse_args(args_list)
    if args.command == "train":
        if args.curriculum is not None:
            train_curriculum(args)
            return
        train(
            args.timesteps, model_path=args.model, csv_path=args.metrics,
            init_model=args.init_model, n_agents=args.n_agents, mdp=args.mdp,
            seed=args.seed, n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, n_epochs=args.n_epochs, device=args.device,
            torch_threads=args.torch_threads, checkpoint_freq=args.checkpoint_freq,
            checkpoint_dir=args.checkpoint_dir,
            gamma=args.gamma,
            eval_episodes=args.eval_episodes,
            conflict_horizon_seconds=args.conflict_horizon,
            conflict_risk_second_cost=args.conflict_risk_cost,
            conflict_buffer_nm=args.conflict_buffer_nm,
            mask_inactive=not args.include_inactive_samples,
            sector_buffer_second_cost=args.sector_buffer_cost, sector_buffer_nm=args.sector_buffer_nm,
            heading_deadband_deg=args.heading_deadband_deg, speed_deadband_kt=args.speed_deadband_kt,
            obstacle_buffer_nm=args.obstacle_buffer_nm, obstacle_release_nm=args.obstacle_release_nm,
            max_heading_offset_deg=args.max_heading_offset_deg, lookahead_nm=args.lookahead_nm,
            passage_training_probability=args.passage_training_probability,
            passage_training_profile=args.passage_training_profile,
            centralized=args.centralized_critic,
            zone_config=({name: getattr(args, "zone_" + name) for name in ZONE_CONFIG_FIELDS
                          if getattr(args, "zone_" + name) is not None} or None),
            lookahead_simulations=args.lookahead_simulations, lookahead_depth=args.lookahead_depth,
            aircraft_buffer_second_cost=args.aircraft_buffer_cost, aircraft_buffer_nm=args.aircraft_buffer_nm,
            decision_seconds=args.decision_seconds,
        )
    else:
        if args.scenario_start < 1 or (args.max_steps is not None and args.max_steps < 1):
            parser.error("Use positive --scenario-start and --max-steps")
        if args.episodes < 1 or args.n_agents < 2:
            parser.error("Use at least one episode and two aircraft")
        options = {name: getattr(args, name) for name in
                   ("obstacle_buffer_nm", "obstacle_release_nm", "max_heading_offset_deg", "lookahead_nm")}
        passage_config = None
        if any(value is not None for value in options.values()):
            if args.mdp not in {"navigation-v4", "navigation-v5"}:
                parser.error("Passage settings require --mdp navigation-v4 or navigation-v5")
            saved = getattr(PPO.load(args.model_path), "competition_passage_config", None)
            passage_config = resolve_passage_config(saved, **options)
            print("Replay passage configuration override: behavior differs from the saved training setup.")
        replay(args.model_path, args.episodes, args.seed, n_agents=args.n_agents,
               mdp=args.mdp, headless=args.headless,
               heading_deadband_deg=args.heading_deadband_deg, speed_deadband_kt=args.speed_deadband_kt,
               passage_config=passage_config, trajectory_csv=args.trajectory_csv,
               lookahead_simulations=args.lookahead_simulations, lookahead_depth=args.lookahead_depth,
               lookahead_passes=args.lookahead_passes, disable_lookahead=args.no_lookahead,
               lookahead_preset=args.lookahead_preset,
               safety_filter=args.safety_filter, safety_buffer_nm=args.safety_buffer_nm,
               safety_horizon_seconds=args.safety_horizon_seconds,
               scenario_start=args.scenario_start, max_steps=args.max_steps,
               search_trace_prefix=args.search_trace_prefix, search_trace_start=args.search_trace_start,
               search_trace_end=args.search_trace_end, search_trace_agents=args.search_trace_agents,
               search_trace_neighbours=args.search_trace_neighbours)


if __name__ == "__main__":
    main()
