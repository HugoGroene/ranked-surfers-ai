import json
from pathlib import Path
import tempfile
import unittest
from guard_monitor import GuardMonitor


class MonitorTests(unittest.TestCase):
    def test_late_uncertainty_and_menu_are_review_evidence_not_approved_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor=GuardMonitor(folder)
            monitor.observe(at=1,state='MENU',probabilities={'MENU':1})
            probabilities={'GAMEPLAY':.6,'OTHER':.4}
            row=monitor.observe(at=121,state='UNCERTAIN',probabilities=probabilities,
                run='run_0001',run_age=120,frame='run_0001/frames/0001.png')
            self.assertEqual(row['warning'],'non_gameplay_during_candidate')
            menu=monitor.observe(at=122,state='MENU',probabilities={'MENU':.99},run='run_0001',run_age=121)
            self.assertIn('review_death_or_false_positive',menu['warning'])
            self.assertEqual(probabilities,{'GAMEPLAY':.6,'OTHER':.4})
            monitor.close()
            result=json.loads((Path(folder)/'guard_summary.json').read_text())
            self.assertEqual(result['states']['MENU'],2)
            self.assertEqual(result['states_during_candidates']['MENU'],1)
            self.assertEqual(result['runs']['run_0001']['UNCERTAIN'],1)

    def test_exception_and_timing_logged_and_flushed_without_auto_label(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor=GuardMonitor(folder)
            monitor.observe(at=10,state='UNCERTAIN',probabilities={},error='test error',
                capture_ms=12,classifier_ms=3)
            rows=[json.loads(s) for s in (Path(folder)/'guard_observations.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['error'],'test error')
            self.assertEqual(rows[0]['warning'],'classifier_exception')
            self.assertEqual(rows[0]['capture_ms'],12)
            monitor.close()
            self.assertEqual(json.loads((Path(folder)/'guard_summary.json').read_text())['classifier_errors'],1)


if __name__=='__main__':unittest.main()
