"""Safety selection must beat value only when forecast separation requires it."""
import math
import unittest
from bluesky_zoo.lookahead import LookaheadConfig, PolicyExpansion, Transition, policy_value_search
from bluesky_zoo.separation_filter import SeparationForecast as F, select_separation_candidate

class SeparationFilterTests(unittest.TestCase):
    def test_safe_candidate_beats_unsafe_high_value_and_visit_count(self):
        profiles = [F(4., 20., 30., 60.), F(8., 0., 0., 60.), F(9., 0., 0., 60.)]
        self.assertEqual(select_separation_candidate(profiles, [20, 1, 3], [100, -2, -10], [.8,.1,.1], 8.), 2)

    def test_no_safe_candidate_prioritizes_official_violation_before_buffer(self):
        profiles = [F(4., 1., 2., 60.), F(6., 0., 20., 60.), F(7., 0., 10., 60.)]
        self.assertEqual(select_separation_candidate(profiles, [20, 3, 1], [100, 50, -10], [.8,.1,.1], 8.), 2)

    def test_ties_and_no_neighbours_keep_learned_ranking(self):
        profiles = [F(math.inf, 0., 0., 60.)] * 3
        self.assertEqual(select_separation_candidate(profiles, [3, 4, 4], [100, 2, 5], [.8,.1,.1], 8.), 2)

    def test_config_defaults_and_invalid_values(self):
        self.assertFalse(LookaheadConfig().safety_filter)
        for option in ({"safety_filter":1}, {"safety_buffer_nm":-1}, {"safety_buffer_nm":float("nan")},
                       {"safety_horizon_seconds":0}, {"safety_horizon_seconds":121}):
            with self.assertRaises(ValueError):
                LookaheadConfig(**option)

    def test_search_observer_records_filtered_selection_without_extra_visits(self):
        import numpy as np
        observations=[]
        cfg=LookaheadConfig(simulations=4,max_depth=1,candidate_count=3)
        result=policy_value_search(0,evaluate=lambda _:PolicyExpansion(np.array([[0,0],[1,0]]),np.array([.9,.1]),0),
            transition=lambda _,action:Transition(1, float(action[0]),terminal=True),config=cfg,
            action_selector=lambda *args:0,observer=observations.append)
        np.testing.assert_array_equal(result.action,[0,0])
        self.assertEqual(sum(result.visit_counts),4)
        self.assertEqual(observations[0].selected_index,0)

class SeparationForecastTests(unittest.TestCase):
    def planner_and_scene(self):
        from types import SimpleNamespace
        from bluesky_zoo.lookahead import AircraftState
        from bluesky_zoo.lookahead_policy import LookaheadPolicy, ForecastScene, ForecastAircraft
        planner = LookaheadPolicy.__new__(LookaheadPolicy)
        planner.config = LookaheadConfig(safety_filter=True)
        planner.env = SimpleNamespace(episode_time_limit=3000., intrusion_distance=5., distance_margin=5.)
        planner._path_cache = {}
        planner._command = lambda aircraft, action:(aircraft.motion.heading_deg, aircraft.motion.speed_kt, None)
        def aircraft(agent, motion, goal):
            return ForecastAircraft(agent, motion, goal, (0.,0.), None, motion.heading_deg,
                                    motion.speed_kt, True, 1., None)
        scene = ForecastScene((aircraft("own", AircraftState(0.,0.,0.,0.), (100.,100.)),
                               aircraft("other", AircraftState(0.,10.,270.,360.), (0.,-100.))),0.)
        return planner, scene

    def test_known_closing_encounter_accumulates_deficit_with_one_second_samples(self):
        planner, scene = self.planner_and_scene()
        result = planner.separation_forecast(scene,0,[0,0],{"other":(270.,360.)})
        self.assertAlmostEqual(result.minimum_nm,4.)
        self.assertAlmostEqual(result.intrusion_deficit_nm_seconds,5.5)
        self.assertAlmostEqual(result.buffer_deficit_nm_seconds,82.)
        self.assertEqual(result.duration_seconds,60.)

    def test_fractional_time_limit_is_respected(self):
        from dataclasses import replace
        planner, scene = self.planner_and_scene()
        result = planner.separation_forecast(replace(scene,time=2999.5),0,[0,0],{"other":(270.,360.)})
        self.assertAlmostEqual(result.minimum_nm,9.95)
        self.assertEqual(result.duration_seconds,.5)

    def test_forecast_arrivals_are_removed_after_the_arrival_sample(self):
        from dataclasses import replace
        planner, scene = self.planner_and_scene()
        other = replace(scene.aircraft[1],goal=(0.,9.9))
        result = planner.separation_forecast(replace(scene,aircraft=(scene.aircraft[0],other)),0,[0,0],{"other":(270.,360.)})
        self.assertAlmostEqual(result.minimum_nm,9.9)
        self.assertEqual(result.duration_seconds,60.)
        own = replace(scene.aircraft[0],goal=(0.,0.))
        result = planner.separation_forecast(replace(scene,aircraft=(own,other)),0,[0,0],{"other":(270.,360.)})
        self.assertEqual(result.duration_seconds,1.)
