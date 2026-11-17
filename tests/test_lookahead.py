"""Search uses learned priors/values and forecasts independent horizontal states."""

from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError
import unittest

import numpy as np

from bluesky_zoo.lookahead import (
    AircraftState, LookaheadConfig, PolicyExpansion, Transition,
    gaussian_action_candidates, policy_value_search, predict_aircraft,
    resolve_lookahead_config,
)


class LookaheadSearchTests(unittest.TestCase):
    def test_search_accepts_initial_retreat_to_reach_a_better_future(self):
        calls = []

        def learned(state):
            calls.append(state)
            if state == "start":
                return PolicyExpansion([[-1, 0], [1, 0]], [0.2, 0.8], 0)
            # A learned positive continuation value allows an initial detour.
            return PolicyExpansion([[0, 0]], [1], 7)

        def transition(state, action):
            if state == "start":
                if action[0] < 0:
                    return Transition("around_obstacle", -2)
                return Transition("dead_end", 1, terminal=True)
            return Transition("goal", 12, terminal=True)

        result = policy_value_search("start", evaluate=learned, transition=transition,
                                     config=LookaheadConfig(simulations=30, max_depth=3, discount=1))
        self.assertEqual(result.action[0], -1)
        self.assertGreater(result.action_values[0], result.action_values[1])
        self.assertIn("around_obstacle", calls)
        self.assertNotIn("goal", calls)
        self.assertNotIn("dead_end", calls)
        self.assertGreaterEqual(result.deepest_depth, 2)
        self.assertEqual(result.visit_counts.sum(), 30)

    def test_changing_learned_leaf_value_changes_the_selected_action(self):
        def choose(preferred):
            return policy_value_search(
                "root", evaluate=lambda state: PolicyExpansion(
                    [[-1, 0], [1, 0]], [0.5, 0.5],
                    0 if state == "root" else (8 if state == preferred else -8)),
                transition=lambda state, action: Transition("left" if action[0] < 0 else "right", 0),
                config=LookaheadConfig(simulations=20, max_depth=1),
            )

        self.assertEqual(choose("left").action[0], -1)
        self.assertEqual(choose("right").action[0], 1)

    def test_policy_priors_drive_search_when_predicted_outcomes_tie(self):
        def choose(priors):
            return policy_value_search(
                None, evaluate=lambda state: PolicyExpansion([[-1, 0], [1, 0]], priors, 0),
                transition=lambda state, action: Transition(None, 0, terminal=True),
                config=LookaheadConfig(simulations=20),
            )

        self.assertEqual(choose([0.9, 0.1]).action[0], -1)
        self.assertEqual(choose([0.1, 0.9]).action[0], 1)

    def test_even_a_low_prior_root_direction_gets_one_trial(self):
        actions = []
        result = policy_value_search(
            None, evaluate=lambda state: PolicyExpansion([[-1, 0], [1, 0]], [1, 1e-8], 0),
            transition=lambda state, action: (actions.append(action.copy()) or Transition(None, 1, terminal=True)),
            config=LookaheadConfig(simulations=2),
        )
        self.assertEqual(len(actions), 2)
        np.testing.assert_array_equal(result.visit_counts, [1, 1])

    def test_simulations_depth_and_transitions_are_bounded_and_reproducible(self):
        root = {"history": (), "position": np.array([4.0, 9.0])}
        snapshot = deepcopy(root)
        evaluations = []
        transitions = []

        def evaluate(state):
            evaluations.append(len(state["history"]))
            return PolicyExpansion([[0, 0]], [1], 1)

        def transition(state, action):
            transitions.append(len(state["history"]))
            return Transition({"history": (*state["history"], action.copy()),
                               "position": state["position"].copy() + 1}, 0.1)

        cfg = LookaheadConfig(simulations=15, max_depth=3)
        first = policy_value_search(root, evaluate=evaluate, transition=transition, config=cfg)
        self.assertLessEqual(first.expanded_nodes, cfg.simulations + 1)
        self.assertEqual(first.deepest_depth, 3)
        self.assertLessEqual(max(evaluations), 3)
        self.assertEqual(len(transitions), 3)
        self.assertEqual(root["history"], snapshot["history"])
        np.testing.assert_array_equal(root["position"], snapshot["position"])
        second = policy_value_search(root, evaluate=evaluate, transition=transition, config=cfg)
        np.testing.assert_array_equal(first.action, second.action)
        np.testing.assert_array_equal(first.visit_counts, second.visit_counts)
        self.assertEqual(first.value, second.value)

    def test_macro_discount_and_terminal_reward_are_backed_up_once(self):
        result = policy_value_search(
            "root", evaluate=lambda state: PolicyExpansion([[0, 0]], [1], 100 if state == "leaf" else 0),
            transition=lambda state, action: Transition("leaf", 2, discount=0.5),
            config=LookaheadConfig(simulations=1, max_depth=1),
        )
        self.assertAlmostEqual(result.value, 52)
        terminal = policy_value_search(
            "root", evaluate=lambda state: PolicyExpansion([[0, 0]], [1], 100),
            transition=lambda state, action: Transition("goal", 20, terminal=True),
            config=LookaheadConfig(simulations=5),
        )
        self.assertEqual(terminal.value, 20)
        self.assertEqual(terminal.expanded_nodes, 1)

    def test_terminal_root_and_invalid_learned_predictions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "terminal root"):
            policy_value_search(None, evaluate=lambda state: PolicyExpansion([], [], 0, terminal=True),
                                transition=lambda state, action: None)
        for actions, priors, value in (([[2, 0]], [1], 0), ([[0, 0]], [0], 0),
                                      ([[0, 0]], [-1], 0), ([[0, 0]], [1], float("nan"))):
            with self.assertRaises(ValueError):
                PolicyExpansion(actions, priors, value)
        with self.assertRaises(ValueError):
            Transition(None, float("nan"))
        with self.assertRaises(ValueError):
            Transition(None, 0, discount=1.1)

    def test_expansions_own_their_arrays_and_prior_normalization_is_stable(self):
        actions = np.zeros((2, 2))
        priors = np.array([1e308, 1e308])
        expansion = PolicyExpansion(actions, priors, 0)
        actions[0, 0] = 1
        priors[0] = 0
        self.assertEqual(expansion.actions[0, 0], 0)
        np.testing.assert_allclose(expansion.priors, [0.5, 0.5])
        with self.assertRaises(ValueError):
            expansion.actions[0, 0] = 1


class AircraftPredictionTests(unittest.TestCase):
    def test_straight_flight_uses_knots_and_preserves_initial_state(self):
        initial = AircraftState(0, 0, 0, 180)
        samples = predict_aircraft(initial, 0, 180, 20)
        self.assertEqual(len(samples), 20)
        self.assertAlmostEqual(samples[-1].north_nm, 1)
        self.assertAlmostEqual(samples[-1].east_nm, 0)
        self.assertEqual(initial, AircraftState(0, 0, 0, 180))
        with self.assertRaises(FrozenInstanceError):
            initial.speed_kt = 10

    def test_turn_and_acceleration_are_limited_with_shortest_wraparound(self):
        initial = AircraftState(0, 0, 359, 150)
        cfg = LookaheadConfig(turn_rate_deg_s=2, speed_change_kt_s=1)
        samples = predict_aircraft(initial, 10, 200, 2.5, cfg)
        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[-1].heading_deg, 4)
        self.assertAlmostEqual(samples[-1].speed_kt, 152.5)
        self.assertGreater(samples[-1].north_nm, 0)
        # Direction is already near north while crossing the 360-degree seam.
        self.assertLess(abs(samples[-1].east_nm), 0.01)

    def test_prediction_and_search_configuration_reject_nonfinite_or_unbounded_values(self):
        for kwargs in ({"simulations": 0}, {"max_depth": 1.5}, {"candidate_count": 2},
                       {"discount": 0}, {"exploration": float("inf")}, {"rollout_seconds": 121},
                       {"turn_rate_deg_s": 0}, {"speed_change_kt_s": -1}):
            with self.assertRaises(ValueError):
                LookaheadConfig(**kwargs)
        cfg = LookaheadConfig(simulations=8)
        self.assertEqual(resolve_lookahead_config(asdict(cfg)), cfg)
        for seconds in (0, -1, float("nan"), 121):
            with self.assertRaises(ValueError):
                predict_aircraft(AircraftState(0, 0, 0, 150), 0, 150, seconds)

    def test_policy_candidates_are_reproducible_bounded_and_density_weighted(self):
        mean = np.array([0.3, -0.2])
        std = np.array([0.4, 0.4])
        actions, priors = gaussian_action_candidates(mean, std, count=9, seed=13)
        again, again_priors = gaussian_action_candidates(mean, std, count=9, seed=13)
        np.testing.assert_array_equal(actions, again)
        np.testing.assert_array_equal(priors, again_priors)
        np.testing.assert_array_equal(actions[0], mean)
        self.assertTrue(np.all(np.abs(actions) <= 1))
        self.assertAlmostEqual(priors.sum(), 1)
        self.assertEqual(int(np.argmax(priors)), 0)
        self.assertTrue(np.any(np.isclose(actions[:, 0], -0.2)))
        self.assertTrue(np.any(np.isclose(actions[:, 0], 0.8)))
        self.assertTrue(np.all(priors > 0))
        with self.assertRaises(ValueError):
            gaussian_action_candidates(mean, [0, 0.1])

    def test_confident_actor_still_proposes_both_tangents_and_one_distinct_uturn(self):
        actions, _ = gaussian_action_candidates([0, 0], [1e-6, 1e-6])
        np.testing.assert_array_equal(actions[:3], [[0, 0], [-0.5, 0], [0.5, 0]])
        self.assertIn(-1.0, actions[:, 0])
        # -1 and +1 represent the same physical heading: spend one slot on it.
        self.assertNotIn(1.0, actions[:, 0])
        wrapped, _ = gaussian_action_candidates([0.8, 0], [1e-6, 1e-6], count=3)
        np.testing.assert_allclose(wrapped[:, 0], [0.8, 0.3, -0.7])


if __name__ == "__main__":
    unittest.main()
