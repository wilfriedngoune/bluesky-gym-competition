"""Experimental search-labelled actor imitation, followed separately by PPO.

The fixed teacher is RL-guided lookahead. Labels are collected only for live
aircraft on a mixture of student/teacher trajectories using a training seed.
Supervised actor fitting is separate from PPO; no search action is recorded
as an on-policy PPO sample. The saved student defaults to actor-only inference.
This does not establish safety, generalization or organizer approval.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bluesky_zoo.lookahead import resolve_inference_lookahead_config


@dataclass(frozen=True)
class DistillationConfig:
    seed: int = 20
    n_agents: int = 10
    iterations: int = 2
    decisions_per_iteration: int = 64
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-4
    teacher_mix: float = 0.5
    max_examples: int = 10_000

    def __post_init__(self):
        for name in ("iterations", "decisions_per_iteration", "epochs", "batch_size", "max_examples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(self.n_agents, bool) or not isinstance(self.n_agents, int) or not 2 <= self.n_agents <= 10:
            raise ValueError("Use 2..10 training aircraft")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.teacher_mix) or not 0 <= self.teacher_mix <= 1:
            raise ValueError("teacher_mix must be in [0, 1]")


class LiveExamples:
    """Bounded reservoir of copied live observations and bounded teacher labels."""

    def __init__(self, capacity=10_000, seed=20):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Example capacity must be a positive integer")
        self.capacity, self.seen = capacity, 0
        self.rng = np.random.default_rng(seed)
        self.rows, self.targets = [], []

    def add(self, observations, agents, targets):
        for agent in agents:
            target = np.asarray(targets[agent], dtype=np.float32)
            if target.shape != (2,) or not np.isfinite(target).all() or np.any(np.abs(target) > 1):
                raise ValueError("Teacher labels must be bounded finite heading/speed actions")
            row = {key: np.array(value, copy=True) for key, value in observations[agent].items()}
            if not all(np.isfinite(value).all() for value in row.values()):
                raise ValueError("Training observations must be finite")
            self.seen += 1
            index = len(self.rows) if len(self.rows) < self.capacity else int(self.rng.integers(self.seen))
            if index < self.capacity:
                if index == len(self.rows):
                    self.rows.append(row)
                    self.targets.append(target.copy())
                else:
                    self.rows[index], self.targets[index] = row, target.copy()

    def arrays(self):
        if not self.rows:
            raise ValueError("No live examples were collected")
        return ({key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]},
                np.stack(self.targets))


def actor_parameters(policy):
    """Update only actor MLP/head; reject feature sharing that changes the critic."""
    for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(policy, name, None)
        if extractor is not None and any(True for _ in extractor.parameters()):
            raise ValueError("Distillation currently requires parameter-free feature extractors to preserve the critic")
    if not hasattr(policy, "mlp_extractor") or not hasattr(policy.mlp_extractor, "policy_net"):
        raise ValueError("Distillation requires the standard separate PPO actor/critic architecture")
    parameters = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    critic = {id(p) for p in policy.mlp_extractor.value_net.parameters()} | {id(p) for p in policy.value_net.parameters()}
    if not parameters or any(id(p) in critic for p in parameters):
        raise ValueError("Actor and critic must have distinct trainable parameters")
    return parameters


def imitation_loss(mean, targets):
    """Heading has period two; the Gaussian mean must still stay in action bounds."""
    delta = math.pi * (mean[:, 0] - targets[:, 0])
    circular = torch.atan2(torch.sin(delta), torch.cos(delta)) / math.pi
    heading_bounds = torch.relu(torch.abs(mean[:, 0]) - 1.)
    return (circular.square() + (mean[:, 1] - targets[:, 1]).square() + heading_bounds.square()).mean()


def fit_actor(model, examples, *, epochs=5, batch_size=64, learning_rate=1e-4, seed=20):
    """Offline supervised phase; critic, Gaussian variance and PPO optimizer stay fixed."""
    observations, targets = examples.arrays()
    policy = model.policy
    policy.set_training_mode(False)
    parameters = actor_parameters(policy)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)

    def loss_for(indices):
        obs = {key: value[indices] for key, value in observations.items()}
        tensors, _ = policy.obs_to_tensor(obs)
        mean = policy.get_distribution(tensors).distribution.mean
        target = torch.as_tensor(targets[indices], dtype=mean.dtype, device=mean.device)
        return imitation_loss(mean, target)

    def full_loss():
        with torch.no_grad():
            return sum(float(loss_for(np.arange(start, min(start+batch_size, len(targets)))).cpu())
                       * min(batch_size, len(targets)-start)
                       for start in range(0, len(targets), batch_size)) / len(targets)

    initial = full_loss()
    best_loss, best_epoch = initial, 0
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(targets))
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(order[start:start+batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
        history.append(full_loss())
        if history[-1] < best_loss:
            best_loss, best_epoch = history[-1], epoch+1
            best_parameters = [parameter.detach().clone() for parameter in parameters]
    # Mini-batch updates can overshoot an already-correct teacher mean. Keep
    # the best fit on the aggregate training labels, including the initial fit.
    with torch.no_grad():
        for parameter, best in zip(parameters, best_parameters):
            parameter.copy_(best)
    return dict(initial_loss=initial, final_loss=best_loss, selected_epoch=best_epoch,
                epoch_losses=history, examples=len(targets))


def collect_examples(env, observations, teacher, student, examples, *, decisions, teacher_mix, rng):
    """Label live learner/teacher-mixture states, excluding departed/padded slots."""
    from scripts.evaluate_competition import actions_for_agents
    counts = dict(decisions=0, labelled_aircraft=0, teacher_actions=0, student_actions=0,
                  completed_aircraft=0, goal_arrivals=0, safety_events=0)
    student_act = SimpleNamespace(model=student, disable_lookahead=True)
    for _ in range(decisions):
        if not env.agents:
            observations, _ = env.reset()
        agents = list(env.agents)
        teacher_actions = actions_for_agents(teacher, observations, agents, env=env)
        student_actions = actions_for_agents(student_act, observations, agents, env=env)
        examples.add(observations, agents, teacher_actions)
        actions = {}
        for agent in agents:
            use_teacher = rng.random() < teacher_mix
            actions[agent] = teacher_actions[agent] if use_teacher else student_actions[agent]
            counts["teacher_actions" if use_teacher else "student_actions"] += 1
        observations, _, terms, truncs, infos = env.step(actions)
        for agent in agents:
            if terms[agent] or truncs[agent]:
                counts["completed_aircraft"] += 1
                counts["goal_arrivals"] += int(infos[agent].get("waypoint_reached", 0))
                counts["safety_events"] += sum(int(infos[agent].get(key, 0)) for key in
                                               ("intrusion_events", "restricted_area_events", "sector_exit_events"))
        counts["decisions"] += 1
        counts["labelled_aircraft"] += len(agents)
    return observations, counts


def output_paths(source, output):
    source = Path(source).resolve()
    if not source.is_file() and Path(str(source)+".zip").is_file():
        source = Path(str(source)+".zip")
    if not source.is_file():
        raise FileNotFoundError(source)
    base = Path(output).resolve()
    if base.suffix == ".zip":
        base = base.with_suffix("")
    paths = dict(model=Path(str(base)+".zip"), metadata=Path(str(base)+".distillation.json"),
                 dataset=Path(str(base)+".distillation.npz"))
    if paths["model"] == source:
        raise ValueError("Student output must be separate from the teacher checkpoint")
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Distillation output already exists: {path}")
    if Path(str(base)+".json").exists():
        raise FileExistsError(f"Experiment metadata already exists: {base}.json")
    return source, paths


def distill(model_path, output, *, config=None, lookahead_preset="saved",
            lookahead_simulations=None, lookahead_depth=None, dry_run=False):
    config = config or DistillationConfig()
    resolve_inference_lookahead_config(None, preset=lookahead_preset,
                                     simulations=lookahead_simulations, max_depth=lookahead_depth)
    source, paths = output_paths(model_path, output)
    plan = dict(teacher=str(source), outputs={key: str(value) for key, value in paths.items()},
                config=asdict(config), actor_only_student=True,
                note="Experimental imitation on training scenarios; PPO refinement is a separate phase.")
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    from stable_baselines3 import PPO
    from scripts import evaluate_competition as evaluation
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    teacher = evaluation.load_policy("ma", str(source), mdp="navigation-v5")
    teacher.disable_lookahead = False
    teacher.lookahead_config = resolve_inference_lookahead_config(
        teacher.lookahead_config, preset=lookahead_preset,
        simulations=lookahead_simulations, max_depth=lookahead_depth)
    student = PPO.load(source, device="cpu")
    actor_parameters(student.policy)
    if getattr(student, "competition_mdp", None) != "navigation-v5":
        raise ValueError("Distillation requires a navigation-v5 checkpoint")
    teacher.model.policy.set_training_mode(False)
    for parameter in teacher.model.policy.parameters():
        parameter.requires_grad_(False)
    if config.seed == 42:
        print("Development note: seed 42 overlaps the official evaluation stream; prefer other training seeds.")
    examples = LiveExamples(config.max_examples, config.seed)
    rng = np.random.default_rng(config.seed)
    env = evaluation.make_env("ma", n_agents=config.n_agents, mdp="navigation-v5",
                              **evaluation.policy_reward_kwargs(teacher, "navigation-v5"))
    history = []
    try:
        evaluation.validate_policy_spaces(teacher, env, "ma", "navigation-v5")
        observations, _ = env.reset(seed=config.seed)
        for iteration in range(config.iterations):
            observations, counts = collect_examples(
                env, observations, teacher, student, examples, decisions=config.decisions_per_iteration,
                teacher_mix=config.teacher_mix, rng=rng)
            fit = fit_actor(student, examples, epochs=config.epochs, batch_size=config.batch_size,
                            learning_rate=config.learning_rate, seed=config.seed+iteration)
            history.append(dict(iteration=iteration+1, collection=counts, fitting=fit))
            print(f"Distillation {iteration+1}/{config.iterations}: {len(examples.rows)} examples; "
                  f"loss {fit['initial_loss']:.6g} -> {fit['final_loss']:.6g}", flush=True)
    finally:
        env.close()
    metadata = dict(**plan, teacher_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    teacher_lookahead_config=asdict(teacher.lookahead_config), history=history,
                    examples_seen=examples.seen, retained_examples=len(examples.rows),
                    training_action_selection="Supervised actor imitation; no PPO updates in this script",
                    critic_unchanged=True, inference="actor-only", evaluated=False,
                    teacher_limit="Teacher labels can be unsafe or suboptimal; no safety guarantee.")
    student.competition_disable_lookahead = True
    student.competition_distillation = metadata
    # Recheck before any output write; preserve every existing experiment.
    _, paths = output_paths(source, output)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    student.save(paths["model"])
    observations, targets = examples.arrays()
    np.savez_compressed(paths["dataset"], actions=targets,
                        **{"obs__"+key: value for key, value in observations.items()})
    paths["metadata"].write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Saved experimental actor-only student: {paths['model']}")
    print("Evaluate it separately before promotion; PPO refinement uses train_zoo.py --init-model.")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ma_navigation_v5_fresh_v6/model.zip")
    parser.add_argument("--out", required=True)
    for name in ("seed", "n_agents", "iterations", "decisions_per_iteration", "epochs", "batch_size", "max_examples"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(DistillationConfig(), name))
    for name in ("learning_rate", "teacher_mix"):
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=getattr(DistillationConfig(), name))
    parser.add_argument("--lookahead-preset", choices=["saved", "fast"], default="saved")
    parser.add_argument("--lookahead-simulations", type=int, default=None)
    parser.add_argument("--lookahead-depth", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = DistillationConfig(**{name: getattr(args, name) for name in DistillationConfig.__dataclass_fields__})
        distill(args.model, args.out, config=config, lookahead_preset=args.lookahead_preset,
                lookahead_simulations=args.lookahead_simulations, lookahead_depth=args.lookahead_depth,
                dry_run=args.dry_run)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
