"""Custom training geometry remains seeded, feasible, and separate from evaluation."""

from dataclasses import asdict
import math
import unittest

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from bluesky_zoo.passage_scenarios import (
    PassageTrainingGenerator, detour_training_scenario, narrow_passage_scenario,
)


def nm(center, point):
    return np.array([(point[0] - center[0]) * 60,
                     (point[1] - center[1]) * 60 * math.cos(math.radians(center[0]))])


class TargetedScenarioTests(unittest.TestCase):
    def test_seeded_detours_have_safe_placements_and_clear_feasible_routes(self):
        for n_agents in (2, 4, 6, 10):
            for seed in range(25):
                with self.subTest(n_agents=n_agents, seed=seed):
                    scenario = detour_training_scenario(n_agents, rng=np.random.default_rng(seed))
                    self.assertEqual(len(scenario.agents), n_agents)
                    self.assertEqual(len(scenario.obstacles), 5)
                    sector = Polygon([nm(scenario.center, p) for p in scenario.sector])
                    obstacles = [Polygon([nm(scenario.center, p) for p in o.vertices])
                                 for o in scenario.obstacles]
                    self.assertTrue(all(o.is_valid and sector.contains(o) for o in obstacles))
                    starts = [nm(scenario.center, a.start) for a in scenario.agents]
                    goals = [nm(scenario.center, a.goal) for a in scenario.agents]
                    for index, start in enumerate(starts):
                        self.assertTrue(all(np.linalg.norm(start - other) > 5 for other in starts[:index]))
                    # Recover the rotation from the first rectangle's north edge.
                    corners = np.array(obstacles[0].exterior.coords)[:4]
                    north_axis = (corners[2] - corners[1]) / np.linalg.norm(corners[2] - corners[1])
                    east_axis = (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])
                    rotation = np.column_stack((north_axis, east_axis))
                    local_corners = corners @ rotation
                    low, high = local_corners.min(axis=0), local_corners.max(axis=0)
                    for aircraft, start, goal in zip(scenario.agents, starts, goals):
                        self.assertTrue(135 <= aircraft.speed <= 155)
                        self.assertTrue(0 <= aircraft.heading < 360)
                        self.assertTrue(sector.contains(Point(start)) and sector.contains(Point(goal)))
                        self.assertFalse(any(o.covers(Point(start)) or o.covers(Point(goal)) for o in obstacles))
                        self.assertTrue(obstacles[0].intersects(LineString((start, goal))))
                        local_start = start @ rotation
                        outside_east = low[1] - 6 if local_start[1] < (low[1] + high[1]) / 2 else high[1] + 6
                        waypoints = [start, rotation @ np.array([low[0] - 4, outside_east]),
                                     rotation @ np.array([high[0] + 4, outside_east]), goal]
                        route = LineString(waypoints)
                        self.assertTrue(sector.contains(route))
                        self.assertFalse(any(o.intersects(route) for o in obstacles))
                        # Conservative kinematic feasibility check with a 120 s
                        # allowance; this is not an A320 or multi-agent guarantee.
                        self.assertLess(route.length / (aircraft.speed / 1852) + 120, 3000)

    def test_detours_are_reproducible_and_vary_across_seeds(self):
        a = detour_training_scenario(10, rng=np.random.default_rng(17))
        b = detour_training_scenario(10, rng=np.random.default_rng(17))
        c = detour_training_scenario(10, rng=np.random.default_rng(18))
        self.assertEqual(asdict(a), asdict(b))
        self.assertNotEqual(asdict(a), asdict(c))

    def test_zero_probability_keeps_original_rng_stream_exact(self):
        class Original:
            def generate(self, rng):
                return rng.random(4)
        for profile in ("passage", "mixed"):
            actual_rng, expected_rng = np.random.default_rng(12), np.random.default_rng(12)
            result = PassageTrainingGenerator(Original(), 10, 0, profile=profile).generate(actual_rng)
            np.testing.assert_array_equal(result, Original().generate(expected_rng))
            self.assertEqual(actual_rng.bit_generator.state, expected_rng.bit_generator.state)

    def test_default_profile_preserves_legacy_custom_rng_sequence(self):
        rng, expected = np.random.default_rng(3), np.random.default_rng(3)
        expected.random()  # Legacy probability selection, even at probability one.
        wanted = narrow_passage_scenario(4, expected.uniform(3, 8), expected.uniform(0, 360))
        actual = PassageTrainingGenerator(None, 4, 1).generate(rng)
        self.assertEqual(asdict(actual), asdict(wanted))
        self.assertEqual(rng.bit_generator.state, expected.bit_generator.state)

    def test_mixed_profile_generates_both_custom_types_deterministically(self):
        rng, other = np.random.default_rng(44), np.random.default_rng(44)
        a, b = PassageTrainingGenerator(None, 6, 1, profile="mixed"), PassageTrainingGenerator(None, 6, 1, profile="mixed")
        scenarios = [a.generate(rng) for _ in range(20)]
        self.assertEqual([asdict(s) for s in scenarios], [asdict(b.generate(other)) for _ in range(20)])
        # Legacy passages have exactly 150 m/s; detours vary aircraft speeds.
        types = {all(agent.speed == 150 for agent in s.agents) for s in scenarios}
        self.assertEqual(types, {False, True})

    def test_invalid_options_fail_explicitly(self):
        for kwargs in (dict(probability=-.1), dict(probability=float("nan")),
                       dict(probability=1.1), dict(probability=.5, profile="unknown"),
                       dict(probability=.5, n_agents=11)):
            options = dict(n_agents=10, **kwargs) if "n_agents" not in kwargs else kwargs
            with self.assertRaises(ValueError):
                PassageTrainingGenerator(None, **options)
        for n_agents in (1, 11, 2.5, True):
            with self.assertRaises(ValueError):
                detour_training_scenario(n_agents, rng=np.random.default_rng(2))
        with self.assertRaises(ValueError):
            detour_training_scenario(2)


if __name__ == "__main__":
    unittest.main()
