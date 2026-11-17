"""Forecast work is reused without changing geometry, timing or live state."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Point, Polygon

from bluesky_zoo import lookahead_policy as planning
from bluesky_zoo.lookahead import AircraftState, LookaheadConfig, predict_aircraft
from bluesky_zoo.lookahead_policy import LookaheadPolicy
import test_lookahead_pipeline as fixtures


class LookaheadPerformanceTests(unittest.TestCase):
    def geometry_planner(self):
        planner = object.__new__(LookaheadPolicy)
        planner.polygons = (
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]]),
            Polygon([(-5, -5), (5, -5), (0, 5)]),
        )
        planner._reset_forecast_cache()
        return planner

    def test_fast_occupancy_matches_covers_on_edges_vertices_holes_and_random_points(self):
        planner = self.geometry_planner()
        points = [(0, 0), (5, 0), (10, 10), (3, 3), (3, 5), (5, 5), (-5, -5), (1, 1), (12, 12)]
        points.extend(np.random.default_rng(73).uniform(-10, 15, size=(200, 2)))
        for position in points:
            expected = tuple(bool(polygon.covers(Point(position))) for polygon in planner.polygons)
            self.assertEqual(planner._inside_areas(position), expected)

    def test_area_cache_keeps_exact_nearest_geometry_and_returns_immutable_coordinates(self):
        planner = self.geometry_planner()
        position = np.array([12., 6.])
        for index, polygon in enumerate(planner.polygons):
            expected_point = polygon.exterior.interpolate(polygon.exterior.project(Point(position)))
            expected_xy = np.array(expected_point.coords[0])
            first = planner._area_state(position, index)
            second = planner._area_state(position, index)
            self.assertIs(first, second)
            self.assertEqual(first[0], float(np.linalg.norm(position - expected_xy)))
            self.assertEqual(first[1], polygon.covers(Point(position)))
            np.testing.assert_array_equal(first[2], expected_xy)
            self.assertFalse(first[2].flags.writeable)

    def test_forecasts_reuse_exact_immutable_samples_and_reset_between_decisions(self):
        planner = self.geometry_planner()
        planner.config = LookaheadConfig(simulations=1, max_depth=1)
        aircraft = SimpleNamespace(motion=AircraftState(1., 2., 0., 150.), tas_cas_ratio=1.)
        with patch.object(planning, "predict_aircraft", wraps=predict_aircraft) as predictor:
            first = planner._forecast_path(aircraft, 20., 170., 15.)
            second = planner._forecast_path(aircraft, 20., 170., 15.)
            self.assertIs(first, second)
            self.assertEqual(predictor.call_count, 1)
            changed = planner._forecast_path(aircraft, 30., 170., 15.)
            self.assertNotEqual(first, changed)
            self.assertEqual(predictor.call_count, 2)
            planner._reset_forecast_cache()
            again = planner._forecast_path(aircraft, 20., 170., 15.)
            self.assertEqual(first, again)
            self.assertIsNot(first, again)
            self.assertEqual(predictor.call_count, 3)
        self.assertEqual(aircraft.motion, AircraftState(1., 2., 0., 150.))

    def test_each_predicted_encounter_is_calculated_once_per_second(self):
        env, model = fixtures.LookaheadPipelineTests.make_model(self, n_agents=2)
        env.reset(seed=37)
        planner = LookaheadPolicy(model, env, LookaheadConfig(simulations=1, max_depth=1, rollout_seconds=5))
        scene = planner.snapshot()
        scene = replace(scene, aircraft=tuple(replace(a, goal=(1000., 1000.)) for a in scene.aircraft))
        commands = {a.agent: (a.motion.heading_deg, 150.) for a in scene.aircraft}
        with patch.object(planning, "_cpa", wraps=planning._cpa) as cpa:
            result = planner.transition(scene, 0, np.array([0, -0.4]), commands)
        self.assertEqual(result.state.time - scene.time, 5)
        # One initial separation query, then one shared result for each second's
        # intrusion, predictive warning and buffer calculations.
        self.assertEqual(cpa.call_count, 6)
        self.assertEqual(env.sim_time, scene.time)

    def test_live_snapshot_and_scenario_change_invalidate_all_work_caches(self):
        env, model = fixtures.LookaheadPipelineTests.make_model(self, n_agents=2)
        env.reset(seed=37)
        planner = LookaheadPolicy(model, env, LookaheadConfig(simulations=1, max_depth=1))
        scene = planner.snapshot()
        planner._area_state(np.array([0., 0.]), 0)
        planner._forecast_path(scene.aircraft[0], 20., 170., 15.)
        self.assertTrue(planner._area_cache and planner._inside_cache and planner._path_cache)
        planner.snapshot()
        self.assertEqual((planner._area_cache, planner._inside_cache, planner._path_cache), ({}, {}, {}))
        previous_polygons = planner.polygons
        env.reset(seed=38)
        planner.snapshot()
        self.assertIsNot(planner.polygons, previous_polygons)
        self.assertIs(planner._scenario, env.scenario)


if __name__ == "__main__":
    unittest.main()
