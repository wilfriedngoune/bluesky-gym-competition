"""Training-only scenario subsets; the official generator is never edited."""

from dataclasses import dataclass, replace

from core.scenario import ScenarioGenerator
from scripts.experiment_support import validate_environment, validate_world


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    obstacles: int
    intruders: int

    def __post_init__(self):
        if not 0 <= self.obstacles <= 5 or not 0 <= self.intruders <= 10:
            raise ValueError("Stages must fit the five obstacle and ten intruder observation slots")


STAGES = (
    CurriculumStage("navigation", 0, 0),
    CurriculumStage("obstacles_sparse", 2, 0),
    CurriculumStage("obstacles_full", 5, 0),
    CurriculumStage("traffic_sparse", 5, 3),
    CurriculumStage("traffic_medium", 5, 6),
    CurriculumStage("full", 5, 10),
)


def get_stage(name):
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"Unknown curriculum stage: {name}")


class ScenarioSubsetGenerator:
    """Generate an original scenario, then retain a prefix of its hazards.

    Starts, goals, sector shapes and hazard geometry are unchanged. No extra
    random draws are made, so a seed stream describes the same underlying worlds
    across stages. The final stage uses the original generator directly.
    """

    def __init__(self, original, stage):
        if type(original) is not ScenarioGenerator:
            raise ValueError("Curriculum must start from the original ScenarioGenerator")
        self.original = original
        self.stage = stage

    def generate(self, rng):
        scenario = self.original.generate(rng)
        return replace(scenario, obstacles=scenario.obstacles[:self.stage.obstacles],
                       intruder_routes=scenario.intruder_routes[:self.stage.intruders])


def set_training_stage(env, stage):
    """Change scenario difficulty at the next reset, without changing spaces."""
    validate_world(env)
    base = env.unwrapped
    if base._fixed_scenario is not None:
        raise ValueError("Curriculum training needs random scenarios")
    generator = base.scenario_generator
    original = generator.original if isinstance(generator, ScenarioSubsetGenerator) else generator
    if type(original) is not ScenarioGenerator:
        raise ValueError("Unexpected training scenario generator")
    base.scenario_generator = original if stage.name == "full" and (stage.obstacles, stage.intruders) == (5, 10) else ScenarioSubsetGenerator(original, stage)
    if type(base.scenario_generator) is ScenarioGenerator:
        validate_environment(env)


def make_training_env(stage):
    from bluesky_gym.envs.single_agent_env import SingleAgentEnv

    env = SingleAgentEnv(render_mode=None)
    try:
        validate_environment(env)
        set_training_stage(env, stage)
    except Exception:
        env.close()
        raise
    return env


def gate_passed(summary, goal_threshold, safe_threshold):
    return (summary["goal_completion_rate"] >= goal_threshold
            and summary["safe_goal_completion_rate"] >= safe_threshold)
