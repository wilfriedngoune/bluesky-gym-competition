"""Focused checks for the shared MDP and aircraft-local navigation histories."""

import unittest
from unittest.mock import patch

import bluesky as bs
import numpy as np
from bluesky.tools.misc import txt2hdg

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.envs.single_agent_env import SingleAgentEnv
from bluesky_zoo.multi_agent_env import MultiAgentEnv
from core.scenario import AgentSpec, Obstacle, Scenario
from core.tools import kwikqdrdist


def fixed_scenario(first_at_goal=False):
    center = (52.0, 4.0)

    def latlon(north, east):
        return tuple(fn.nm_to_latlong(np.array(center), np.array([north, east])))

    return Scenario(
        center=center,
        sector=[latlon(-100, -100), latlon(-100, 100),
                latlon(100, 100), latlon(100, -100)],
        agents=[
            AgentSpec("A", latlon(-20, -20),
                      latlon(-20 if first_at_goal else 60, -20), 0, 150),
            AgentSpec("B", latlon(20, 20), latlon(-60, 20), 0, 150),
            AgentSpec("C", latlon(-20, 40), latlon(60, 40), 0, 150),
        ],
    )


class MultiAgentNavigationTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = MultiAgentEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_spaces_match_single_agent_and_intruder_slots_stay_fixed(self):
        # BlueSky is process-global: create/close these environments sequentially.
        single = SingleAgentEnv(n_obstacles=5)
        observation_space = single.observation_space
        action_space = single.action_space
        single.close()

        env = self.make_env(n_agents=8, n_obstacles=5)
        observations, _ = env.reset(seed=12)
        for agent, observation in observations.items():
            self.assertEqual(env.observation_space(agent), observation_space)
            self.assertEqual(env.action_space(agent), action_space)
            self.assertEqual(observation["intruders"].shape, (5, 7))
            self.assertTrue(env.observation_space(agent).contains(observation))

    def test_each_aircraft_uses_its_own_goal_and_action_history(self):
        env = self.make_env(scenario=fixed_scenario(), n_obstacles=0)
        observations, _ = env.reset(seed=4)
        np.testing.assert_allclose(observations["A"]["goal_direction"], [1, 0])
        np.testing.assert_allclose(observations["B"]["goal_direction"], [-1, 0], atol=1e-14)

        env._get_action("A", np.array([0.25, -0.5]))
        env._get_action("B", np.array([-0.75, 0.8]))
        np.testing.assert_array_equal(env._get_obs("A")["previous_action"], [0.25, -0.5])
        np.testing.assert_array_equal(env._get_obs("B")["previous_action"], [-0.75, 0.8])
        np.testing.assert_array_equal(env._get_obs("C")["previous_action"], [0, 0])
        self.assertAlmostEqual(env.navigation_states["A"].last_target_heading, 17.5)
        self.assertAlmostEqual(env.navigation_states["B"].last_target_heading, 127.5)
        self.assertAlmostEqual(env.navigation_states["A"].last_target_speed, 145)
        self.assertAlmostEqual(env.navigation_states["B"].last_target_speed, 210)

    def test_near_zero_heading_is_accepted_and_applied_by_bluesky(self):
        env = self.make_env(scenario=fixed_scenario(), n_obstacles=0)
        env.reset(seed=4)
        state = env.navigation_states["A"]
        for reference in (1e-5, -1e-5, 1e-10, 0.0, 179.0, -179.0):
            with self.subTest(reference=reference), \
                    patch.object(state, "_get_navigation_reference", return_value=(reference, 0)), \
                    patch.object(bs.stack, "stack", wraps=bs.stack.stack) as stack:
                env._get_action("A", np.zeros(2))
                heading_text = stack.call_args_list[0].args[0].split()[2]
                self.assertEqual(txt2hdg(heading_text), state.last_target_heading)
                # Exercise the actual command dispatcher and autopilot, too.
                bs.stack.process()
                self.assertEqual(bs.traf.ap.trk[bs.traf.id2idx("A")], state.last_target_heading)

    def test_invalid_navigation_values_fail_before_sending_commands(self):
        env = self.make_env(scenario=fixed_scenario(), n_obstacles=0)
        env.reset(seed=4)
        state = env.navigation_states["A"]
        with patch.object(bs.stack, "stack") as stack:
            for action in ([np.nan, 0], [0, np.inf], [0]):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, "Invalid navigation action for A"):
                    env._get_action("A", action)
            with patch.object(state, "_get_navigation_reference", return_value=(np.nan, 0)):
                with self.assertRaisesRegex(ValueError, "Invalid navigation target for A"):
                    env._get_action("A", np.zeros(2))
            stack.assert_not_called()

    def test_reward_deltas_use_the_corresponding_aircraft_baseline(self):
        env = self.make_env(scenario=fixed_scenario(), n_obstacles=0)
        env.reset(seed=4)
        env._reward_baseline = {a: dict(m) for a, m in env.metrics.items()}
        env.metrics["A"]["flight_time"] = 5
        env.metrics["B"]["flight_time"] = 1
        env.metrics["C"]["intrusion_events"] = 1
        self.assertAlmostEqual(env._get_reward("A"), -0.01)
        self.assertAlmostEqual(env._get_reward("B"), -0.002)
        self.assertAlmostEqual(env._get_reward("C"), -20)

    def test_restricted_area_hysteresis_is_local_to_the_aircraft(self):
        scenario = fixed_scenario()

        def latlon(north, east):
            return tuple(fn.nm_to_latlong(np.array(scenario.center), np.array([north, east])))

        scenario.obstacles = [Obstacle([
            latlon(-22, -22), latlon(-22, -18),
            latlon(-18, -18), latlon(-18, -22),
        ])]
        env = self.make_env(scenario=scenario, n_obstacles=1)
        observations, _ = env.reset(seed=4)
        self.assertEqual(observations["A"]["navigation_mode"][0], 1)
        self.assertEqual(observations["B"]["navigation_mode"][0], 0)
        self.assertEqual(env.navigation_states["A"].active_buffer_obstacle, 0)
        self.assertIsNone(env.navigation_states["B"].active_buffer_obstacle)

        idx = bs.traf.id2idx("A")
        bs.traf.lat[idx], bs.traf.lon[idx] = latlon(-14.5, -20)
        # At 3.5 NM, the 4 NM release threshold keeps A avoiding its obstacle.
        self.assertEqual(env._get_obs("A")["navigation_mode"][0], 1)
        self.assertEqual(env._get_obs("B")["navigation_mode"][0], 0)
        self.assertIsNone(env.navigation_states["B"].active_buffer_obstacle)

        bs.traf.lat[idx], bs.traf.lon[idx] = latlon(-13.5, -20)
        self.assertEqual(env._get_obs("A")["navigation_mode"][0], 0)
        self.assertIsNone(env.navigation_states["A"].active_buffer_obstacle)

    def test_reached_aircraft_is_removed_and_survivors_use_fresh_indices(self):
        env = self.make_env(scenario=fixed_scenario(first_at_goal=True), n_obstacles=0)
        env.reset(seed=4)
        observations, _, terminated, truncated, infos = env.step(
            {agent: np.zeros(2) for agent in env.agents}
        )
        self.assertTrue(terminated["A"])
        self.assertFalse(truncated["A"])
        self.assertEqual(infos["A"]["flight_time"], 1)
        self.assertEqual(infos["B"]["flight_time"], 5)
        self.assertEqual(env.agents, ["B", "C"])
        self.assertEqual(bs.traf.id2idx("B"), 0)
        self.assertEqual(env.navigation_states["C"].intruder_ids, ["B"])
        _, distance = kwikqdrdist(
            bs.traf.lat[bs.traf.id2idx("C")], bs.traf.lon[bs.traf.id2idx("C")],
            bs.traf.lat[bs.traf.id2idx("B")], bs.traf.lon[bs.traf.id2idx("B")],
        )
        self.assertAlmostEqual(observations["C"]["intruders"][0, 4], distance / 100)
        np.testing.assert_array_equal(observations["C"]["intruders"][1:], np.zeros((4, 7)))

    def test_reset_clears_histories_and_time_limit_truncates_all_agents(self):
        env = self.make_env(scenario=fixed_scenario(), n_obstacles=0, episode_time_limit=10)
        env.reset(seed=4)
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(
                {agent: np.array([0.1, -0.2]) for agent in env.agents}
            )
        self.assertTrue(all(truncated.values()))
        self.assertFalse(any(terminated.values()))
        self.assertEqual(env.agents, [])

        for state in env.navigation_states.values():
            state.previous_escape_clearance = 10
            state.was_inside_restricted = True
            state.active_buffer_obstacle = 99
        observations, infos = env.reset(seed=4)
        for agent, state in env.navigation_states.items():
            np.testing.assert_array_equal(observations[agent]["previous_action"], np.zeros(2))
            np.testing.assert_array_equal(state.previous_action, np.zeros(2))
            self.assertIsNone(state.previous_escape_clearance)
            self.assertFalse(state.was_inside_restricted)
            self.assertIsNone(state.active_buffer_obstacle)
            self.assertEqual(state.last_target_speed, 0)
            self.assertEqual(infos[agent]["flight_time"], 0)


if __name__ == "__main__":
    unittest.main()
