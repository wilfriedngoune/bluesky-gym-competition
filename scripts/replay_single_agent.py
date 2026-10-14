"""Replay a saved development scenario without producing competition results."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_competition import load_policy


def replay(policy_dir, episode, render_mode="human"):
    from core.scenario import AgentSpec, Obstacle, Route, Scenario
    from bluesky_gym.envs.single_agent_env import SingleAgentEnv

    policy_dir = Path(policy_dir)
    with (policy_dir / "scenarios.jsonl").open(encoding="utf-8") as stream:
        entry = next((item for item in map(json.loads, stream) if item["episode_index"] == episode), None)
    if entry is None:
        raise ValueError(f"No saved scenario for episode {episode}")
    spec = entry["scenario"]
    scenario = Scenario(
        center=tuple(spec["center"]), sector=spec["sector"],
        obstacles=[Obstacle(**item) for item in spec["obstacles"]],
        agents=[AgentSpec(**item) for item in spec["agents"]],
        intruder_routes=[Route(**item) for item in spec["intruder_routes"]],
    )
    saved_model = policy_dir / "model.zip"
    act = load_policy("sa", saved_model if saved_model.exists() else None)
    env = SingleAgentEnv(scenario=scenario, render_mode=render_mode)
    try:
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(act(obs))
        print(json.dumps(info, indent=2))
        return info
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dir", help="Policy subdirectory containing scenarios.jsonl and optional model.zip")
    parser.add_argument("--episode", type=int, required=True, help="Zero-based episode index from failures.jsonl")
    parser.add_argument("--headless", action="store_true", help="Verify a replay without opening a window")
    args = parser.parse_args()
    replay(args.policy_dir, args.episode, None if args.headless else "human")


if __name__ == "__main__":
    main()
