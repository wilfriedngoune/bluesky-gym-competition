import unittest
from scripts.evaluate_safety_filter import offset_reset

class OffsetResetTests(unittest.TestCase):
    def test_replays_exact_seed_prefix_and_returns_assigned_scenario(self):
        for scenario in (1,9,20):
            seeds=[]
            def reset(*,seed=None):
                seeds.append(seed)
                return len(seeds)
            assigned=offset_reset(reset,scenario)
            self.assertEqual(assigned(seed=42),scenario)
            self.assertEqual(seeds,[42]+[None]*(scenario-1))
            self.assertEqual(assigned(seed=None),scenario+1)
            self.assertEqual(seeds,[42]+[None]*scenario)

    def test_other_seed_rejected_without_reset(self):
        assigned=offset_reset(lambda **kwargs:self.fail("Unexpected reset"),9)
        with self.assertRaises(ValueError):
            assigned(seed=43)
