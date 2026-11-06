import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from evaluate_completed import select_completed,execute

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def plan(self,status='COMPLETE',seconds=3601,audit='PASS_SETTLED_LEARNING',cleanup=None):
        train=self.root/'train';train.mkdir()
        report={'status':status,'training_seconds':seconds,'audit':{'status':audit},
                'finalization_errors':[] if cleanup is None else cleanup,'optimized_transitions':128}
        (train/'trial_report.json').write_text(json.dumps(report))
        with zipfile.ZipFile(train/'latest.zip','w') as z:
            z.writestr('policy.pth',b'offline fixture')
            z.writestr('data',json.dumps({'num_timesteps':128}))
        args=['--spacing','uniform','--seed','456','--minutes','60']
        return {'jobs':[{'mode':'train','name':'384_456','output':str(train),'command':args},
                        {'mode':'eval','name':'384_456','output':str(self.root/'eval'),'command':args}]}

    def test_complete_is_retained_not_retrained(self):
        t,e,om=select_completed(self.plan())
        self.assertEqual(len(e),1);self.assertTrue(t[0]['reused']);self.assertFalse(om)

    def test_partial_model_excluded(self):
        t,e,om=select_completed(self.plan(status='RUNNING',seconds=2500))
        self.assertFalse(t);self.assertFalse(e);self.assertEqual(len(om),1)

    def test_short_budget_excluded(self):
        self.assertFalse(select_completed(self.plan(seconds=1800))[0])

    def test_bad_audit_excluded(self):
        self.assertFalse(select_completed(self.plan(audit='FAIL'))[0])

    def test_cleanup_error_excluded(self):
        self.assertFalse(select_completed(self.plan(cleanup=['failed']))[0])

    def test_seed_filter_can_omit_384(self):
        self.assertFalse(select_completed(self.plan(),seed=123)[0])

    def test_live_training_refused_before_checks(self):
        with self.assertRaises(RuntimeError):execute(self.plan(),self.root)

    def test_spacing_filter(self):
        p=self.plan();p['jobs'][0]['command'][1]='golden'
        self.assertFalse(select_completed(p,'vision')[0])
        self.assertEqual(len(select_completed(p,'all')[0]),1)

if __name__=='__main__':unittest.main()
