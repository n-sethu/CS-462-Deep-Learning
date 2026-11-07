"""Boundary and ordering checks for cache-only adaptation interpretation."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('adaptation_analysis',Path(__file__).resolve().parents[1]/'results/adaptation/analyze.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_interval_touching_zero_is_inconclusive(self):
        self.assertEqual(analysis.direction(-1,0),'inconclusive')
        self.assertEqual(analysis.direction(0,1),'inconclusive')
        self.assertEqual(analysis.direction(-2,-1),'benefit')
        self.assertEqual(analysis.direction(1,2),'harm')
        self.assertEqual(analysis.direction(1,2,negative_better=False),'benefit')

    def test_ties_use_example_id_and_hand_computed_aurc(self):
        # Stable ID ordering gives error sequence 0,1,0,1 at equal risks.
        errors=np.array([1,0,1,0]);ids=np.array([3,0,1,2])
        values=analysis.ranked_metrics(errors,np.ones(4),ids)
        self.assertAlmostEqual(values['aurc'],(0+.5+1/3+.5)/4)
        self.assertEqual(values['accepted_errors80'],2)
        self.assertEqual(values['accepted90'],4)
        self.assertAlmostEqual(values['auroc'],.5)

    def test_invalid_score_alignment_rejected(self):
        with self.assertRaises(ValueError):analysis.ranked_metrics([0,1],[.1],[0,1])
        with self.assertRaises(ValueError):analysis.ranked_metrics([0,1],[np.nan,.5],[0,1])


if __name__=='__main__':unittest.main()
