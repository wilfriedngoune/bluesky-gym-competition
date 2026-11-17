from dataclasses import asdict
import unittest

from bluesky_zoo.lookahead import LookaheadConfig, resolve_inference_lookahead_config


class InferenceLookaheadConfigTests(unittest.TestCase):
    def test_saved_default_preserves_all_settings(self):
        original = LookaheadConfig(simulations=25, max_depth=7, coordination_passes=2,
                                   safety_filter=True, safety_buffer_nm=4)
        self.assertEqual(resolve_inference_lookahead_config(original), original)
        self.assertEqual(resolve_inference_lookahead_config(asdict(original), preset="saved"), original)

    def test_fast_preserves_candidates_forecasts_discount_and_safety_settings(self):
        original = LookaheadConfig(coordination_passes=2, safety_filter=True)
        fast = resolve_inference_lookahead_config(original, preset="fast")
        self.assertEqual((fast.simulations, fast.max_depth, fast.coordination_passes), (12, 3, 1))
        for name in ("candidate_count", "rollout_seconds", "discount", "turn_rate_deg_s",
                     "speed_change_kt_s", "safety_filter", "safety_buffer_nm", "safety_horizon_seconds"):
            self.assertEqual(getattr(fast, name), getattr(original, name))

    def test_explicit_budget_overrides_preset_and_still_validates(self):
        fast = resolve_inference_lookahead_config(None, preset="fast", simulations=8, max_depth=2)
        self.assertEqual((fast.simulations, fast.max_depth), (8, 2))
        for kwargs in (dict(preset="unknown"), dict(simulations=0), dict(max_depth=0),
                       dict(simulations=1, coordination_passes=2)):
            with self.assertRaises(ValueError):
                resolve_inference_lookahead_config(None, **kwargs)
