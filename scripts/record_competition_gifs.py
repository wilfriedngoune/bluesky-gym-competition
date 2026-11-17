"""
Record GIFs of the competition environments.

Runs headless (no window) and captures rgb_array frames, so it works over SSH
or in CI. By default it regenerates the two GIFs embedded in the docs:

    docs/media/competition/single_agent.gif   (CompetitionEnv-v0)
    docs/media/competition/multi_agent.gif    (CompetitionZooEnv, n_agents=10)

with a neutral (fly-straight) policy — the initial heading points at the goal,
so this still shows the task, the other aircraft/intruders, the obstacles and
the sector.

It also produces submission reels: choose seeds, --mdp navigation and --model
to record the same custom MDP and PPO checkpoint used for evaluation.

Run it as a module from the repo root (like the other scripts/ files):

    # regenerate both doc GIFs
    python -m scripts.record_competition_gifs

    # one scenario of the multi-agent env at a chosen seed
    python -m scripts.record_competition_gifs --env ma --seeds 7 --out my_ma.gif

    # a 5-scenario reel of the single-agent env (cherry-pick the seeds)
    python -m scripts.record_competition_gifs --env sa --seeds 1 2 3 4 5 --out reel.gif
"""

import argparse
import math
import os

# Force pygame's headless "dummy" video backend so no window opens (works over
# SSH / in CI). setdefault => you can still override SDL_VIDEODRIVER yourself.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np                                  # noqa: E402
from PIL import Image                               # noqa: E402

from scripts.evaluate_competition import (           # noqa: E402
    actions_for_agents, load_policy, make_env, validate_policy_spaces, policy_reward_kwargs,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SA_GIF = os.path.join(REPO_ROOT, "docs", "media", "competition", "single_agent.gif")
DEFAULT_MA_GIF = os.path.join(REPO_ROOT, "docs", "media", "competition", "multi_agent.gif")


def policy_from_model(model_path, kind="ma", mdp="base"):
    """Return an ``act(obs) -> action`` callable, or None for a neutral policy.

    A checkpoint uses deterministic PPO predictions. Without one, each
    aircraft receives zero actions in the selected MDP.
    """
    if model_path is None:
        if mdp == "navigation-v5":
            raise ValueError("navigation-v5 lookahead requires an RL checkpoint; pass --model")
        return None
    return load_policy(kind, model_path, mdp=mdp)


def _step_limit(env, max_steps):
    """Cover the full simulated-time budget unless an explicit cap is requested."""
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        return max_steps
    raw_env = getattr(env, "unwrapped", env)
    return math.ceil(raw_env.episode_time_limit / (raw_env.action_frequency * raw_env.sim_dt))


def rollout_single_agent(seed, act=None, max_steps=None, mdp="base"):
    """Collect rgb_array frames for one CompetitionEnv-v0 episode."""
    env = make_env("sa", mdp=mdp, render_mode="rgb_array", **policy_reward_kwargs(act, mdp))
    frames = []
    try:
        validate_policy_spaces(act, env, "sa", mdp)
        step_limit = _step_limit(env, max_steps)
        obs, _ = env.reset(seed=seed)
        frames.append(env.render())
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < step_limit:
            action = np.zeros(2, dtype=np.float32) if act is None else act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            steps += 1
    finally:
        env.close()
    return frames


def rollout_multi_agent(seed, n_agents=10, act=None, max_steps=None, mdp="base"):
    """Collect rgb_array frames for one CompetitionZooEnv episode."""
    env = make_env("ma", n_agents=n_agents, mdp=mdp, render_mode="rgb_array",
                   **policy_reward_kwargs(act, mdp))
    frames = []
    try:
        validate_policy_spaces(act, env, "ma", mdp)
        step_limit = _step_limit(env, max_steps)
        observations, _ = env.reset(seed=seed)
        frames.append(env.render())
        steps = 0
        while env.agents and steps < step_limit:
            actions = actions_for_agents(act, observations, env.agents, env=env)
            observations, _, _, _, _ = env.step(actions)
            frames.append(env.render())
            steps += 1
    finally:
        env.close()
    return frames


def save_gif(frames, out_path, size=256, max_frames=120, duration=80):
    """Downscale, subsample and encode frames as a looping GIF."""
    if not frames:
        raise ValueError("no frames to save")
    if size < 1 or max_frames < 1 or duration < 1:
        raise ValueError("size, max_frames and duration must be positive")
    if len(frames) > max_frames:                    # keep the file small
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]       # include the terminal frame
    images = [Image.fromarray(f).resize((size, size), Image.BILINEAR) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   loop=0, duration=duration, optimize=True)
    print(f"wrote {out_path}  ({len(images)} frames)")


def record(env_kind, seeds, out_path, n_agents=10, model_path=None,
           mdp="base", max_steps=None, no_lookahead=None, **gif_kwargs):
    """Record one or more scenarios (concatenated) into a single GIF."""
    act = policy_from_model(model_path, kind=env_kind, mdp=mdp)
    if act is not None and no_lookahead is not None:
        act.disable_lookahead = no_lookahead
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one scenario seed is required")
    frames_per_scenario = max(2, math.ceil(gif_kwargs.get("max_frames", 120) / len(seeds)))
    frames = []
    for seed in seeds:
        if env_kind == "sa":
            scene = rollout_single_agent(seed, act=act, max_steps=max_steps, mdp=mdp)
        else:
            scene = rollout_multi_agent(seed, n_agents=n_agents, act=act,
                                        max_steps=max_steps, mdp=mdp)
        # Release unsampled frames between scenarios instead of retaining a
        # full five-episode 512x512 reel in memory until GIF encoding.
        if len(scene) > frames_per_scenario:
            indices = np.linspace(0, len(scene) - 1, frames_per_scenario, dtype=int)
            frames.extend(scene[i] for i in indices)
        else:
            frames.extend(scene)
        del scene
    save_gif(frames, out_path, **gif_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["sa", "ma", "both"], default="both",
                        help="single-agent, multi-agent, or both doc GIFs (default: both)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="scenario seed(s) to record (concatenated into one GIF)")
    parser.add_argument("--n-agents", type=int, default=10,
                        help="agents for the multi-agent env (competition config: 10)")
    parser.add_argument("--out", default=None, help="output GIF path (single --env only)")
    parser.add_argument("--model", default=None,
                        help="path to a trained model (see policy_from_model)")
    parser.add_argument("--mdp", choices=["base", "navigation", "navigation-v2", "navigation-v3", "navigation-v4", "navigation-v5"], default="base",
                        help="observation/reward/action design used to train the model")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="optional decision-step cap per scenario (default: full episode budget)")
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--no-lookahead", dest="no_lookahead", action="store_true", default=None,
                           help="record actor only; default follows checkpoint")
    inference.add_argument("--with-lookahead", dest="no_lookahead", action="store_false",
                           help="explicitly enable v5 lookahead")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.n_agents < 2:
        parser.error("--n-agents must be at least 2")

    if args.env == "both":
        if args.out is not None:
            parser.error("--out requires a single --env (sa or ma)")
        record("sa", args.seeds or [3], DEFAULT_SA_GIF, model_path=args.model,
               mdp=args.mdp, max_steps=args.max_steps, no_lookahead=args.no_lookahead)
        record("ma", args.seeds or [1], DEFAULT_MA_GIF,
               n_agents=args.n_agents, model_path=args.model,
               mdp=args.mdp, max_steps=args.max_steps, no_lookahead=args.no_lookahead)
    else:
        default_out = DEFAULT_SA_GIF if args.env == "sa" else DEFAULT_MA_GIF
        record(args.env, args.seeds or [1], args.out or default_out,
               n_agents=args.n_agents, model_path=args.model,
               mdp=args.mdp, max_steps=args.max_steps, no_lookahead=args.no_lookahead)


if __name__ == "__main__":
    main()
