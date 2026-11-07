"""Run from demo: python -m unittest test_demo -v."""
import json
import unittest
from pathlib import Path
import numpy as np
from calculations import ranked, choose_examples

class CachedCalculationTests(unittest.TestCase):
    def test_export_hashes(self):
        import hashlib
        root=Path(__file__).parent/'assets'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(manifest['default_ensemble'],1)
        self.assertEqual(manifest['ensemble_count'],10)
        for name,record in manifest['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),record['sha256'],name)

    def test_ties_use_source_id_and_ceil_coverage(self):
        ids=np.array([9,2,5]);errors=np.array([1,0,1]);scores=np.ones(3)
        result=ranked(errors,scores,ids,.5)
        self.assertEqual(result['kept'],2)
        np.testing.assert_array_equal(result['accepted'],[False,True,True])
        self.assertEqual(result['errors'],1)
        self.assertAlmostEqual(result['aurc'],(0+.5+2/3)/3)

    def test_gallery_categories_and_absence(self):
        errors=np.array([1,1,1,0]);ids=np.array([3,2,1,0]);a=np.array([1,0,1,1],bool);b=np.array([0,1,1,1],bool)
        self.assertEqual([i for _,i in choose_examples(errors,ids,a,b)],[0,1,2,3])
        self.assertIsNone(choose_examples(errors,ids,a,a)[0][1])

    def test_all_exported_operating_points(self):
        import csv
        root=Path(__file__).parent/'assets'
        with (root/'verification.csv').open() as f:checks=list(csv.DictReader(f))
        self.assertEqual(len(checks),800)
        for item in checks:
            with np.load(root/f'ensemble_{int(item["ensemble"]):02d}_{item["condition"]}.npz') as d:
                result=ranked(d['error'],d[f'{item["regime"]}_{item["detector"]}'],d['id'],float(item['coverage']))
            self.assertEqual(result['errors'],int(item['accepted_errors']))
            self.assertEqual(result['kept'],int(item['accepted']))

class StreamlitTests(unittest.TestCase):
    def test_presentation_path_and_coverage(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value,'140')
        self.assertEqual(at.metric[1].value,'133')
        at.selectbox[0].set_value('Gaussian noise').run()
        self.assertFalse(at.exception)
        at.radio[0].set_value('Noise-adapted').run()
        self.assertFalse(at.exception)
        at.selectbox[0].set_value('Motion blur').run()
        self.assertFalse(at.exception)
        at.slider[0].set_value(100).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value,at.metric[1].value)
        at.selectbox[1].set_value(10).run()
        self.assertFalse(at.exception)

if __name__=='__main__':unittest.main()
