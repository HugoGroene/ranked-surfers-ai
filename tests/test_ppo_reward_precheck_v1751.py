"""Auditor-only rollout-boundary precheck regression tests; no Unity actions."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from tools import patch_ppo_reward_precheck_v1751 as patch


def audit_candidate():
    source_path = ROOT / patch.TARGET
    source = source_path.read_bytes()
    # Test proposed production code, even when it has not been installed yet.
    if patch.sha(source) == patch.SHA_OLD:
        source = patch.prepare()[3]
    elif patch.NEW.encode('utf-8') not in source or patch.OLD.encode('utf-8') in source:
        raise AssertionError('Expected installed v1.7.5.1 auditor source')
    namespace = {'__name__': 'audit_candidate_v1751', '__file__': str(source_path)}
    exec(compile(source, str(source_path), 'exec'), namespace)
    return namespace['audit_training_log']


def mock_log(*, prechecks=1, freezes=1, menus=0, post_resume_menus=0):
    lines = ['[V612 TRAIN] synthetic offline check']
    for x in range(freezes):
        if x < prechecks:
            lines.append('[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.0 ml_gameplay=1.000')
        lines += [
            '[PPO PAUSE] Freeze verified after 4 captures; track_deltas=[0.0, 0.0, 0.0]',
            '[PPO PAUSE] Verified frozen gameplay before optimizer; paused-track delta=0.000',
            '[PPO GAP] '+json.dumps({'boundary':x+1, 'timesteps':128*(x+1)}),
            '[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '+json.dumps({
                'paused_wall_seconds':1.17,'corrected_episode_timer':True,
                'corrected_action_timer':True,'corrected_monitor_timer':True,
                'active_episode_clock_drift_s':0.0}),
        ]
        for frame in range(1,5):
            lines.append(f'[PPO RESYNC TRACE] frame={frame}/4 since_resume_ms=99.0 state=GAMEPLAY menu_score=0.0 ml_gameplay=1.0 ml_menu=0.0')
        lines.append('[PPO RESYNC] refreshed 4-frame stack; cached age now 8.3ms')
    for x in range(menus):
        lines += [
            '[PPO MENU HANDOFF] Boundary MENU confirmed by independent ML+template frames; no pause, reset or movement. Terminal reward deferred to next rollout step.',
            '[PPO GAP] '+json.dumps({'boundary':100+x, 'timesteps':128*(100+x)}),
            '[PPO MENU HANDOFF] MENU reverified after optimizer; menu_idle_seconds_excluded=1.123; next guarded PPO step must be TERMINAL and will record the actual death penalty; no movement key allowed.',
            '[PPO MENU HANDOFF] Terminal recorded by guarded env.step; death_reward=-1.0000; NO movement key sent.',
        ]
    for extra in range(max(0, prechecks - freezes)):
        lines.insert(-1, '[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY menu_score=0.0 ml_gameplay=1.000')
    lines.append('[V612 REPORT] '+json.dumps({'steps':256,'completed_requested_steps':True,'episodes':4,
                                             'mean_survival_seconds':4.1}))
    return '\n'.join(lines)


class AuditPrecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = staticmethod(audit_candidate())

    def run_audit(self, **kw):
        return self.audit(mock_log(**kw), min_resumes=1, min_steps=1)

    def test_normal_freeze_precheck_equal_pass(self):
        result=self.run_audit(prechecks=1,freezes=1,menus=0)
        self.assertEqual(result['status'],'PASS_STABILITY_CANDIDATE',result['failures'])

    def test_menu_race_explains_one_extra_precheck(self):
        result=self.run_audit(prechecks=2,freezes=1,menus=1)
        self.assertEqual(result['status'],'PASS_STABILITY_CANDIDATE',result['failures'])

    def test_menu_boundary_without_extra_precheck_also_passes(self):
        result=self.run_audit(prechecks=1,freezes=1,menus=1)
        self.assertEqual(result['status'],'PASS_STABILITY_CANDIDATE',result['failures'])

    def test_no_menu_extra_precheck_still_fails(self):
        result=self.run_audit(prechecks=2,freezes=1,menus=0)
        self.assertEqual(result['status'],'FAIL_SAFE_AUDIT')
        self.assertTrue(any('Pre-pause classifier checks' in f for f in result['failures']))

    def test_menu_cannot_excuse_two_extra_prechecks(self):
        result=self.run_audit(prechecks=3,freezes=1,menus=1)
        self.assertEqual(result['status'],'FAIL_SAFE_AUDIT')

    def test_missing_precheck_is_not_excused(self):
        result=self.run_audit(prechecks=1,freezes=2,menus=1)
        self.assertEqual(result['status'],'FAIL_SAFE_AUDIT')

    def test_installer_exact_source_dry_run_and_reject_second_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            dest=root/patch.TARGET
            dest.parent.mkdir(parents=True)
            src=ROOT/patch.TARGET
            # A post-install test still uses original backup for installer checks.
            source = src if patch.sha(src.read_bytes()) == patch.SHA_OLD else src.with_name(src.name+'.pre_ppo_reward_precheck_v1751.bak')
            self.assertTrue(source.exists())
            shutil.copy2(source,dest)
            target,bak,orig,updated=patch.prepare(root)
            self.assertEqual(patch.sha(orig),patch.SHA_OLD)
            self.assertNotEqual(orig,updated)
            self.assertFalse(bak.exists())
            backup=patch.apply(root)
            self.assertEqual(backup.read_bytes(),orig)
            self.assertEqual(target.read_bytes(),updated)
            with self.assertRaisesRegex(RuntimeError,'Unexpected auditor SHA256'):
                patch.prepare(root)


if __name__=='__main__': unittest.main()
