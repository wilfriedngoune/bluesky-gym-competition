"""Optional custom training examples; never installed by evaluation/replay."""

import math
import numpy as np

from core.scenario import AgentSpec, Obstacle, Scenario, agent_callsigns, _nm_to_latlon


def narrow_passage_scenario(n_agents=2, gap_nm=4, angle_deg=0, center=(52.0, 4.0)):
    if not 2 <= n_agents <= 10 or not 0 < gap_nm < 40:
        raise ValueError("Use 2..10 aircraft and a gap in (0, 40) NM")
    angle = math.radians(angle_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])

    def ll(north, east):
        return _nm_to_latlon(center, rotation @ np.array([north, east]))

    def box(north0, north1, east0, east1):
        return [ll(north0, east0), ll(north0, east1), ll(north1, east1), ll(north1, east0)]

    agents = []
    for index, name in enumerate(agent_callsigns(n_agents)):
        east = -15 if index % 2 == 0 else 15
        north = 45 + 10 * (index // 2)
        agents.append(AgentSpec(name, ll(-north, east), ll(north, east), angle_deg % 360, 150.0))
    return Scenario(
        center=center, sector=box(-110, 110, -100, 100), agents=agents,
        obstacles=[Obstacle(box(-20, 20, -25, -gap_nm / 2)),
                   Obstacle(box(-20, 20, gap_nm / 2, 25)),
                   Obstacle(box(-70, -60, 60, 70)), Obstacle(box(60, 70, -70, -60)),
                   Obstacle(box(40, 50, 60, 70))],
    )


class PassageTrainingGenerator:
    def __init__(self, original, n_agents, probability, center=(52.0, 4.0), *, profile="passage"):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("passage_training_probability must be in [0, 1]")
        if profile not in {"passage", "mixed"}:
            raise ValueError("passage_training_profile must be passage or mixed")
        if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)) or not 2 <= n_agents <= 10:
            raise ValueError("Use 2..10 aircraft for passage training")
        self.original, self.n_agents, self.probability, self.center = original, n_agents, probability, center
        self.profile = profile

    def generate(self, rng):
        if self.probability and rng.random() < self.probability:
            # Keep the legacy passage-only RNG stream exactly unchanged. The
            # additional type selection is used only by the explicit profile.
            if self.profile == "mixed" and rng.random() < 0.5:
                return detour_training_scenario(self.n_agents, rng=rng, center=self.center)
            return narrow_passage_scenario(self.n_agents, rng.uniform(3, 8), rng.uniform(0, 360), self.center)
        return self.original.generate(rng)


def detour_training_scenario(n_agents=2, rng=None, center=(52.0, 4.0)):
    """Seeded training example with a blocked direct route and two clear detours.

    All aircraft begin outside the obstacles with more than five NM separation.
    The large central obstacle blocks their initial goal bearing. Its dimensions,
    translation, scene rotation, formation and speeds vary with the supplied RNG.
    Four peripheral obstacles leave both sides of the central obstacle navigable.
    This custom distribution is for training only, never competition evaluation.
    Speeds remain BlueSky-native m/s, as required by ``AgentSpec``.
    """
    if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)) or not 2 <= n_agents <= 10:
        raise ValueError("Use 2..10 aircraft for detour training")
    if rng is None:
        raise ValueError("Supply the training RNG for reproducible detour examples")
    if len(center) != 2 or not all(math.isfinite(value) for value in center) or abs(center[0]) >= 89:
        raise ValueError("Use a finite latitude/longitude center away from the poles")
    angle_deg = rng.uniform(0, 360)
    angle = math.radians(angle_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])

    def ll(north, east):
        return _nm_to_latlon(center, rotation @ np.array([north, east]))

    def box(north0, north1, east0, east1):
        return [ll(north0, east0), ll(north0, east1), ll(north1, east1), ll(north1, east0)]

    half_north = rng.uniform(10, 18)
    half_east = rng.uniform(14, 22)
    east_offset = rng.uniform(-3, 3)
    obstacles = [Obstacle(box(-half_north, half_north,
                              east_offset - half_east, east_offset + half_east))]
    # Peripheral cells vary too, but remain away from the spawn/goal columns
    # and the simple detours at roughly 20..30 NM east/west of the origin.
    for north0, north1, east0, east1 in (
            (-30, -20, -65, -50), (25, 35, 45, 60),
            (-60, -45, 45, 60), (50, 65, -65, -50)):
        north_shift, east_shift = rng.uniform(-3, 3, size=2)
        obstacles.append(Obstacle(box(north0 + north_shift, north1 + north_shift,
                                       east0 + east_shift, east1 + east_shift)))

    start_north, goal_north = rng.uniform(35, 43, size=2)
    spacing = rng.uniform(8, 9)
    agents = []
    for index, name in enumerate(agent_callsigns(n_agents)):
        row = index // 2
        east = -8 if index % 2 == 0 else 8
        start = (-start_north - row * spacing + rng.uniform(-0.6, 0.6),
                 east + rng.uniform(-0.6, 0.6))
        goal = (goal_north + row * spacing + rng.uniform(-0.6, 0.6),
                east + rng.uniform(-0.6, 0.6))
        heading = (angle_deg + math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
                   + rng.uniform(-8, 8)) % 360
        agents.append(AgentSpec(name, ll(*start), ll(*goal), heading, rng.uniform(135, 155)))
    return Scenario(center=center, sector=box(-105, 105, -95, 95), agents=agents,
                    obstacles=obstacles)
