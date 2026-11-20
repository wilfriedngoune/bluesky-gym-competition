"""
Official evaluation harness for the competition.

Runs the canonical scoring protocol — the **first 1000 episodes of seed 42** —
and prints the standard metrics table (plus a per-episode CSV you can share so
judges can reproduce the numbers). Seed 42 is set on the first reset only;
subsequent resets continue the same RNG stream, giving a deterministic,
reproducible sequence of scenarios.

Run it as a module from the repo root:

    python -m scripts.evaluate_competition --env sa                 # required track
    python -m scripts.evaluate_competition --env ma                 # stretch track
    python -m scripts.evaluate_competition --env sa --episodes 20   # quick check
    python -m scripts.evaluate_competition --env ma --mdp navigation --model multi_agent_model

Use --mdp navigation for the custom SingleAgentEnv/MultiAgentEnv observation,
reward, and action mapping. The default base MDP preserves the original demo.
Both support PPO checkpoints with matching observation/action spaces. The
fixed scoring params, scenario distribution, and metrics stay as-is.
"""

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from bluesky_zoo.centralized_critic import CentralizedObservationWrapper, is_centralized_observation_space
from bluesky_gym.envs.common.decision_timing import resolve_decision_seconds
from scripts.competition_report import build_report, write_report
from bluesky_zoo.lookahead import resolve_inference_lookahead_config


SEED = 42            # official seed — do not change for a reported run
N_EPISODES = 1000    # official episode count
N_AGENTS_MA = 10     # fixed multi-agent competition config

# The nine scored metrics (produced by the fixed _update_metrics / _get_info).
METRIC_KEYS = [
    "waypoint_reached", "flight_time",
    "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector",
    "total_reward",
]


def make_env(kind, n_agents=N_AGENTS_MA, mdp="base", render_mode=None, reward_config=None, action_config=None,
             passage_config=None, zone_config=None, decision_seconds=None, transfer_config=None):
    """Return the environment to evaluate.

    Default = the base competition env; navigation selects the custom MDP.
    Do NOT change the fixed scoring params (intrusion_distance, episode_time_limit,
    distance_margin, n_intruders/n_agents, n_obstacles, ...).
    """
    if mdp == "navigation-transfer":
        if kind not in {"sa", "ma"}:
            raise ValueError(f"unknown environment kind: {kind!r}")
        from bluesky_zoo.transfer_navigation import SingleAgentTransferEnv, MultiAgentTransferEnv
        options = dict(render_mode=render_mode, reward_config=reward_config, action_config=action_config,
                       passage_config=passage_config, zone_config=zone_config, transfer_config=transfer_config)
        if decision_seconds is not None:
            options["action_frequency"] = resolve_decision_seconds(mdp, override=decision_seconds)
        return (SingleAgentTransferEnv(**options) if kind == "sa" else
                MultiAgentTransferEnv(n_agents=n_agents, **options))
    if transfer_config is not None:
        raise ValueError("Transfer configuration requires --mdp navigation-transfer")
    if kind not in {"sa", "ma"}:
        raise ValueError(f"unknown environment kind: {kind!r}")
    if mdp not in {"base", "navigation", "navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5"}:
        raise ValueError(f"unknown MDP: {mdp!r}")
    action_kwargs = {"action_config": action_config} if action_config is not None else {}
    if mdp == "base" and action_kwargs:
        from bluesky_gym.envs.common.command_filter import command_config
        config = command_config(action_config)
        if config.heading_deadband_deg or config.speed_deadband_kt:
            raise ValueError("Command deadbands require a navigation MDP")
    timing = ({"action_frequency": resolve_decision_seconds(mdp, override=decision_seconds)}
              if decision_seconds is not None else {})
    action_kwargs.update(timing)
    if passage_config is not None and mdp not in {"navigation-v4", "navigation-v5"}:
        raise ValueError("Passage settings require --mdp navigation-v4 or navigation-v5")
    if zone_config is not None and mdp != "navigation-v5":
        raise ValueError("Zone settings require --mdp navigation-v5")
    if mdp in {"navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5"}:
        if kind != "ma":
            raise ValueError(f"{mdp} is a multi-agent reward experiment; use --env ma")
        if mdp == "navigation-v5":
            from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv
            return MultiAgentStrategicEnv(n_agents=n_agents, render_mode=render_mode,
                                          reward_config=reward_config, passage_config=passage_config,
                                          zone_config=zone_config, **action_kwargs)
        if mdp == "navigation-v4":
            from bluesky_zoo.passage_navigation import MultiAgentPassageEnv
            return MultiAgentPassageEnv(n_agents=n_agents, render_mode=render_mode,
                                        reward_config=reward_config, passage_config=passage_config, **action_kwargs)
        if mdp == "navigation-v3":
            from bluesky_zoo.multi_agent_traffic_env import MultiAgentTrafficEnv
            return MultiAgentTrafficEnv(n_agents=n_agents, render_mode=render_mode,
                                        reward_config=reward_config, **action_kwargs)
        from bluesky_zoo.multi_agent_reward_env import MultiAgentRewardEnv
        return MultiAgentRewardEnv(n_agents=n_agents, render_mode=render_mode,
                                   reward_config=reward_config, **action_kwargs)
    if mdp == "navigation":
        if kind == "sa":
            from bluesky_gym.envs.single_agent_env import SingleAgentEnv
            return SingleAgentEnv(render_mode=render_mode, **action_kwargs)
        from bluesky_zoo.multi_agent_env import MultiAgentEnv
        return MultiAgentEnv(n_agents=n_agents, render_mode=render_mode, **action_kwargs)
    if kind == "sa":
        import gymnasium as gym
        import bluesky_gym
        bluesky_gym.register_envs()
        return gym.make("CompetitionEnv-v0", render_mode=render_mode, **timing)
    from bluesky_zoo.competition_v0 import CompetitionZooEnv
    return CompetitionZooEnv(n_agents=n_agents, render_mode=render_mode, **timing)


def load_policy(kind, model_path=None, mdp="base"):
    """Return an ``act(obs) -> action`` callable.

    Without a checkpoint, use the zero-action baseline. A checkpoint is loaded
    once and queried deterministically for each live aircraft's observation.
    """
    if model_path is None:
        if mdp == "navigation-v5":
            raise ValueError("navigation-v5 lookahead requires an RL checkpoint; pass --model")
        return lambda obs: np.zeros(2, dtype=np.float32)
    import torch
    from stable_baselines3 import PPO
    # These small MLPs are faster without a large CPU thread pool, especially
    # during inference at every five-second decision across ten aircraft.
    torch.set_num_threads(1)
    model = PPO.load(model_path)

    def act(obs):
        return model.predict(obs, deterministic=True)[0]

    # The rollout checks spaces before taking an action, so a base/navigation
    # mix-up fails with an actionable message instead of a shape error mid-run.
    act.model = model
    act.disable_lookahead = bool(getattr(model, "competition_disable_lookahead", False))
    if mdp == "navigation-transfer":
        act.disable_lookahead = True
    if mdp == "navigation-v5" or (mdp == "navigation-transfer" and kind == "sa"):
        from bluesky_zoo.lookahead import LookaheadConfig
        saved = getattr(model, "competition_lookahead_config", None)
        act.lookahead_config = LookaheadConfig(**saved) if saved is not None else LookaheadConfig(discount=model.gamma)
    return act


def action_for_single_agent(act, observation, env):
    """Use the SA copied-state predictor only when search is explicitly enabled."""
    config = getattr(act, "lookahead_config", None)
    if config is None or getattr(act, "disable_lookahead", False):
        return np.zeros(2, dtype=np.float32) if act is None else act(observation)
    from bluesky_zoo.single_agent_lookahead import SingleAgentLookaheadPolicy
    raw_env = getattr(env, "unwrapped", env)
    planner = getattr(act, "lookahead_planner", None)
    if planner is None or planner.env is not raw_env or planner.config != config:
        planner = SingleAgentLookaheadPolicy(act.model, raw_env, config)
        act.lookahead_planner = planner
    planner.observer = getattr(act, "search_observer", None)
    return planner.action(observation)


def configure_lookahead(act, kind, mdp, *, no_lookahead=None, lookahead_preset="saved",
                        lookahead_simulations=None, lookahead_depth=None, lookahead_passes=None):
    """Resolve explicit inference overrides without changing checkpoint defaults."""
    overridden = (lookahead_preset not in (None, "saved")
                  or any(value is not None for value in
                         (lookahead_simulations, lookahead_depth, lookahead_passes)))
    if no_lookahead is None:
        no_lookahead = bool(getattr(act, "disable_lookahead", mdp == "navigation-transfer")) and not overridden
    supported = mdp == "navigation-v5" or (mdp == "navigation-transfer" and kind == "sa")
    if mdp == "navigation-transfer" and kind != "sa" and not no_lookahead:
        raise ValueError("Multi-agent navigation-transfer remains actor-only; use --env sa for transfer lookahead")
    if overridden and (not supported or no_lookahead):
        raise ValueError("Lookahead settings require navigation-v5 or single-agent navigation-transfer with lookahead enabled")
    if supported:
        saved = getattr(act, "lookahead_config", None)
        if saved is None and not no_lookahead:
            from bluesky_zoo.lookahead import LookaheadConfig
            model = getattr(act, "model", None)
            if model is None:
                raise ValueError("Lookahead requires a learned RL checkpoint; pass --model")
            saved = LookaheadConfig(discount=model.gamma)
        if saved is not None:
            act.lookahead_config = resolve_inference_lookahead_config(
                saved, preset=lookahead_preset, simulations=lookahead_simulations,
                max_depth=lookahead_depth, coordination_passes=lookahead_passes)
            if mdp == "navigation-transfer" and not no_lookahead:
                if act.lookahead_config.safety_filter:
                    raise ValueError("Single-agent transfer lookahead does not support the optional separation filter")
                if act.lookahead_config.coordination_passes != 1:
                    raise ValueError("Single-agent transfer lookahead uses one search pass")
        if overridden:
            print(f"Evaluation lookahead override: {act.lookahead_config.simulations} simulations, "
                  f"depth {act.lookahead_config.max_depth}, {act.lookahead_config.coordination_passes} pass(es). "
                  "Behavior may differ from the saved search setup.")
    act.disable_lookahead = no_lookahead
    return no_lookahead


def actions_for_agents(act, observations, agents, env=None):
    """Batch a shared PPO prediction; preserve per-agent custom callables."""
    agents = list(agents)
    if not agents:
        return {}
    search_config = getattr(act, "lookahead_config", None)
    if search_config is not None and not getattr(act, "disable_lookahead", False):
        if env is None:
            raise ValueError("Policy-guided lookahead needs the live environment to take a read-only snapshot")
        from bluesky_zoo.lookahead_policy import LookaheadPolicy
        planner = getattr(act, "lookahead_planner", None)
        if planner is None or planner.env is not env or planner.config != search_config:
            planner = LookaheadPolicy(act.model, env, search_config)
            act.lookahead_planner = planner
        planner.observer = getattr(act, "search_observer", None)
        return planner.actions(observations, agents)
    model = getattr(act, "model", None)
    if model is None:
        return {a: (np.zeros(2, dtype=np.float32) if act is None else act(observations[a]))
                for a in agents}
    first_observation = observations[agents[0]]
    if isinstance(first_observation, dict):
        batch = {key: np.stack([observations[a][key] for a in agents])
                 for key in first_observation}
    else:
        batch = np.stack([observations[a] for a in agents])
    actions, _ = model.predict(batch, deterministic=True)
    return dict(zip(agents, actions))


def validate_policy_spaces(act, env, kind, mdp):
    """Check PPO checkpoint compatibility without creating another BlueSky env."""
    model = getattr(act, "model", None)
    if model is None:
        return
    saved_mdp = getattr(model, "competition_mdp", None)
    if saved_mdp is not None and saved_mdp != mdp:
        raise ValueError(f"Checkpoint was trained with --mdp {saved_mdp}; requested --mdp {mdp}")
    if kind == "sa":
        observation_space, action_space = env.observation_space, env.action_space
    else:
        agent = env.possible_agents[0]
        observation_space, action_space = env.observation_space(agent), env.action_space(agent)
    try:
        _check_spaces(model, observation_space, action_space)
    except ValueError as error:
        raise ValueError(
            f"PPO checkpoint is incompatible with --env {kind} --mdp {mdp}. "
            "Use the MDP used for training; navigation-v3 needs its nine-slot/mask/time inputs. "
            "navigation-v4 additionally needs clearance rays and its saved heading/buffer profile. "
            "navigation-v5 also needs shrinking-zone inputs and its saved search/zone profile. "
            "navigation-transfer needs the shared ten-slot/deadline/memory inputs; older models cannot load directly. "
            "Flattened base-policy checkpoints need their original observation wrapper. "
            f"{error}"
        ) from error


def _check_spaces(model, observation_space, action_space):
    if model.observation_space != observation_space:
        raise ValueError(f"Observation spaces differ: {model.observation_space} != {observation_space}")
    if model.action_space != action_space:
        raise ValueError(f"Action spaces differ: {model.action_space} != {action_space}")


def _evaluation_progress(completed, total, started):
    """Report real compute time separately from the unchanged simulated clock."""
    elapsed = max(time.perf_counter() - started, 0.0)
    remaining = elapsed / completed * (total - completed)
    print(f"Evaluation scenarios: {completed}/{total} "
          f"({elapsed:.1f}s elapsed; approx {remaining:.1f}s remaining)",
          end="\n" if completed == total else "\r", flush=True)


def run_single_agent(n_episodes, act, mdp="base", progress=False, seed=SEED):
    """One record (final info dict) per episode."""
    env = make_env("sa", mdp=mdp, **policy_reward_kwargs(act, mdp))
    records = []
    try:
        validate_policy_spaces(act, env, "sa", mdp)
        started = time.perf_counter()
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed if ep == 0 else None)
            terminated = truncated = False
            while not (terminated or truncated):
                obs, _, terminated, truncated, info = env.step(action_for_single_agent(act, obs, env))
            records.append(dict(info))
            if progress:
                _evaluation_progress(ep + 1, n_episodes, started)
    finally:
        env.close()
    return records


def run_multi_agent(n_episodes, act, n_agents, mdp="base", progress=False, seed=SEED):
    """One record per agent-episode (each agent's final info as it leaves)."""
    env = make_env("ma", n_agents=n_agents, mdp=mdp, **policy_reward_kwargs(act, mdp))
    model = getattr(act, "model", None)
    if model is not None and is_centralized_observation_space(model.observation_space):
        env = CentralizedObservationWrapper(env)
    records = []
    try:
        validate_policy_spaces(act, env, "ma", mdp)
        started = time.perf_counter()
        for ep in range(n_episodes):
            observations, infos = env.reset(seed=seed if ep == 0 else None)
            while env.agents:
                actions = actions_for_agents(act, observations, env.agents, env=env)
                observations, _, terminations, truncations, infos = env.step(actions)
                for a in infos:
                    if terminations[a] or truncations[a]:
                        records.append(dict(infos[a]))
            if progress:
                _evaluation_progress(ep + 1, n_episodes, started)
    finally:
        env.close()
    return records


def policy_reward_kwargs(act, mdp):
    """Reproduce saved reward and command settings, including PPO gamma."""
    model = getattr(act, "model", None)
    saved_actions = getattr(model, "competition_action_config", None)
    kwargs = {"action_config": saved_actions} if saved_actions is not None else {}
    if mdp == "navigation-transfer":
        kwargs["transfer_config"] = getattr(model, "competition_transfer_config", None)
        kwargs["passage_config"] = getattr(model, "competition_passage_config", None)
        kwargs["zone_config"] = getattr(model, "competition_zone_config", None)
        saved_reward = getattr(model, "competition_reward_config", None)
        if saved_reward is not None:
            kwargs["reward_config"] = saved_reward
            if saved_reward["gamma"] != model.gamma:
                raise ValueError("Checkpoint reward discount differs from PPO gamma")
    saved_seconds = getattr(model, "competition_decision_seconds", None)
    if saved_seconds is not None:
        kwargs["decision_seconds"] = resolve_decision_seconds(mdp, saved=saved_seconds)
    if mdp in {"navigation-v4", "navigation-v5"}:
        saved_passage = getattr(model, "competition_passage_config", None)
        if saved_passage is not None:
            kwargs["passage_config"] = saved_passage
    if mdp == "navigation-v5":
        saved_zone = getattr(model, "competition_zone_config", None)
        if saved_zone is not None:
            kwargs["zone_config"] = saved_zone
    if mdp not in {"navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5"}:
        return kwargs
    from dataclasses import asdict
    from bluesky_zoo.navigation_reward import make_reward_config
    model = getattr(act, "model", None)
    saved = getattr(model, "competition_reward_config", None)
    gamma = getattr(model, "gamma", 0.998 if mdp in {"navigation-v3", "navigation-v4", "navigation-v5"} else 0.99)
    if saved is not None and saved["gamma"] != gamma:
        raise ValueError("Checkpoint reward discount differs from PPO gamma")
    return {**kwargs, "reward_config": saved if saved is not None else asdict(make_reward_config(mdp, gamma=gamma))}


# Rows shown in the summary table: (label, metric key, transform).
SUMMARY_ROWS = [
    ("Goal completion rate [%]", "waypoint_reached", lambda x: 100.0 * x),
    ("Flight time [s]",          "flight_time",             None),
    ("Intrusion events",         "intrusion_events",        None),
    ("Intrusion time [s]",       "intrusion_time",          None),
    ("Restricted-area events",   "restricted_area_events",  None),
    ("Restricted-area time [s]", "time_in_restricted_area", None),
    ("Sector-exit events",       "sector_exit_events",      None),
    ("Time outside sector [s]",  "time_outside_sector",     None),
]


def summarize(records, label):
    """Print a Markdown metrics table. total_reward is reported separately
    because it comes from the competitor-defined reward, so it is not
    comparable across teams."""
    n = len(records)
    cols = {k: np.array([r[k] for r in records], dtype=float) for k in METRIC_KEYS}

    print(f"\n### {label} — mean over {n} {'episodes' if label.startswith('Single') else 'agent-episodes'} (seed {SEED})\n")
    print("| Metric | Mean | Std |")
    print("| --- | ---: | ---: |")
    for name, key, tf in SUMMARY_ROWS:
        vals = tf(cols[key]) if tf else cols[key]
        print(f"| {name} | {vals.mean():.3f} | {vals.std():.3f} |")
    print(f"\n_total_reward (own reward, not comparable across teams): "
          f"{cols['total_reward'].mean():.3f} ± {cols['total_reward'].std():.3f}_")


def write_csv(records, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index"] + METRIC_KEYS)
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow({"episode_index": i, **{k: r[k] for k in METRIC_KEYS}})
    print(f"\nwrote per-episode metrics to {out_path}")


def evaluate(kind, n_episodes, n_agents, model_path, out_path, mdp="base", no_lookahead=None, lookahead_passes=None,
             safety_filter=None, safety_buffer_nm=None, safety_horizon_seconds=None,
             lookahead_preset="saved", lookahead_simulations=None, lookahead_depth=None):
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if kind == "ma" and n_agents < 2:
        raise ValueError("multi-agent evaluation needs at least 2 agents")
    if n_episodes != N_EPISODES or (kind == "ma" and n_agents != N_AGENTS_MA):
        print(f"NONOFFICIAL CHECK: {n_episodes} scenarios, "
              f"{n_agents if kind == 'ma' else 1} aircraft; reportable results require "
              f"{N_EPISODES} scenarios and {'10 agents' if kind == 'ma' else 'the default SA configuration'}.")
    evaluation_started = time.perf_counter()
    act = load_policy(kind, model_path, mdp=mdp)
    no_lookahead = configure_lookahead(
        act, kind, mdp, no_lookahead=no_lookahead, lookahead_preset=lookahead_preset,
        lookahead_simulations=lookahead_simulations, lookahead_depth=lookahead_depth,
        lookahead_passes=lookahead_passes)
    safety_options = {"safety_filter": safety_filter, "safety_buffer_nm": safety_buffer_nm,
                      "safety_horizon_seconds": safety_horizon_seconds}
    if any(value is not None for value in safety_options.values()):
        if mdp != "navigation-v5" or no_lookahead:
            raise ValueError("Safety filter settings require navigation-v5 with lookahead enabled")
        from dataclasses import replace
        act.lookahead_config = replace(act.lookahead_config, **{
            key: value for key, value in safety_options.items() if value is not None})
    act.disable_lookahead = no_lookahead
    if kind == "sa":
        records = run_single_agent(n_episodes, act, mdp=mdp, progress=True)
        label = "Single-agent"
    else:
        records = run_multi_agent(n_episodes, act, n_agents, mdp=mdp, progress=True)
        label = "Multi-agent"
    label += f" ({mdp} MDP)"
    evaluation_seconds = time.perf_counter() - evaluation_started
    print(f"Evaluation wall time: {evaluation_seconds:.1f}s")
    summarize(records, label)
    suffix = "" if mdp == "base" else f"_{mdp}"
    csv_path = out_path or f"competition_eval_{kind}{suffix}.csv"
    write_csv(records, csv_path)
    if mdp in {"navigation-v5", "navigation-transfer"}:
        # Preserve the inference algorithm as well as the policy checkpoint so
        # actor-only ablations cannot be confused with search-enabled results.
        settings = {
            "mdp": mdp, "model": str(model_path), "seed": SEED,
            "n_scenarios": n_episodes, "n_agents": 1 if kind == "sa" else n_agents,
            "environment_kind": kind,
            "lookahead_enabled": not no_lookahead,
            "lookahead_preset": lookahead_preset or "saved",
            "evaluation_wall_seconds": evaluation_seconds,
            "lookahead_config": asdict(act.lookahead_config) if hasattr(act, "lookahead_config") and not no_lookahead else None,
            **policy_reward_kwargs(act, mdp),
        }
        Path(csv_path).with_suffix(".settings.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    report = build_report(
        records, kind=kind, phase="evaluation", n_agents=n_agents,
        n_scenarios=n_episodes, seed=SEED, mdp=mdp,
        model_path=model_path, source=csv_path,
    )
    write_report([report], Path(csv_path).with_suffix(""))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["sa", "ma", "both"], default="sa",
                        help="single-agent (required), multi-agent (stretch), or both")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help=f"number of episodes (official: {N_EPISODES})")
    parser.add_argument("--n-agents", type=int, default=N_AGENTS_MA,
                        help=f"agents for the multi-agent env (fixed: {N_AGENTS_MA})")
    parser.add_argument("--model", default=None, help="trained model path (see load_policy)")
    parser.add_argument("--mdp", choices=["base", "navigation", "navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5", "navigation-transfer"], default="base",
                        help="observation/reward/action design used to train the model")
    parser.add_argument("--out", default=None, help="per-episode CSV output path")
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--no-lookahead", dest="no_lookahead", action="store_true", default=None,
                           help="evaluate the learned actor alone; default follows checkpoint")
    inference.add_argument("--with-lookahead", dest="no_lookahead", action="store_false",
                           help="enable v5 search or single-agent navigation-transfer search")
    parser.add_argument("--lookahead-preset", choices=["saved", "fast"], default="saved",
                        help="saved settings or fast: 12 simulations, depth 3, one pass; explicit overrides win")
    parser.add_argument("--lookahead-simulations", type=int, default=None,
                        help="simulation budget per aircraft decision (overrides preset)")
    parser.add_argument("--lookahead-depth", type=int, default=None,
                        help="maximum tree depth (overrides preset)")
    parser.add_argument("--lookahead-passes", type=int, choices=[1, 2], default=None,
                        help="v5 passes sharing the saved total simulation budget")
    parser.add_argument("--safety-filter", action=argparse.BooleanOptionalAction, default=None,
                          help="v5 optional separation screen (default: saved setting or off)")
    parser.add_argument("--safety-buffer-nm", type=float, default=None,
                          help="extra NM above the official 5 NM minimum (default: saved setting or 3)")
    parser.add_argument("--safety-horizon-seconds", type=float, default=None,
                          help="held-command safety forecast horizon, at most 120s (default: saved setting or 60)")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.n_agents < 2:
        parser.error("--n-agents must be at least 2")

    if args.episodes != N_EPISODES:
        print(f"NOTE: running {args.episodes} episodes — the official protocol is "
              f"{N_EPISODES} episodes of seed {SEED}.")

    kinds = ["sa", "ma"] if args.env == "both" else [args.env]
    reports = []
    for kind in kinds:
        reports.append(evaluate(kind, args.episodes, args.n_agents, args.model,
                                args.out if args.env != "both" else None, mdp=args.mdp,
                                no_lookahead=args.no_lookahead, lookahead_passes=args.lookahead_passes,
                                safety_filter=args.safety_filter, safety_buffer_nm=args.safety_buffer_nm,
                                safety_horizon_seconds=args.safety_horizon_seconds,
                                lookahead_preset=args.lookahead_preset,
                                lookahead_simulations=args.lookahead_simulations,
                                lookahead_depth=args.lookahead_depth))
    if len(reports) == 2:
        write_report(reports, Path(args.out).with_suffix("") if args.out else "competition_comparison")


if __name__ == "__main__":
    main()
