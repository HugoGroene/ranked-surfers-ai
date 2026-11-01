import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,Mock,patch
from dataclasses import replace
from road_config import ROOT
from rs_v612_gameplay_guard import screen_gate,sequence_decision
import tools.ppo_stream_menu_tail as tail
from menu_confirmation import candidate,install

def reading(score,state='MENU'):
    return screen_gate(state,{state:1.},score,confidence=.9,menu_threshold=.68)

def recorded_window():
    path=ROOT/'reports/temporal_batch_20261101_180312_752232/wide8_30_seed123/train/v612_guard_episode_0061.json'
    if path.exists():
        from rs_v612_gameplay_guard import GateReading
        return [GateReading(**r) for r in json.loads(path.read_text())['readings']]
    return [reading(v) for v in [.985,.541,.489,.667,.667,.600]]

class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_failure_is_only_candidate_until_fresh_fourth_support(self):
        readings=recorded_window()
        self.assertEqual(sequence_decision(readings),'ABORT')
        self.assertTrue(candidate(readings))
        with patch.object(tail,'pending_menu_tail',tail.pending_menu_tail):
            install()
            sample=AsyncMock(return_value=('new capture',reading(.60)))
            frame,result=await tail.confirm_menu_tail(readings,sample,AsyncMock())
        self.assertEqual((frame,result),('new capture','TERMINAL'))
        sample.assert_awaited_once()
        self.assertEqual(sequence_decision(readings[-6:]),'TERMINAL')

    async def test_six_weak_new_frames_still_abort(self):
        with patch.object(tail,'pending_menu_tail',tail.pending_menu_tail):
            install();sample=AsyncMock(return_value=('weak',reading(.1)))
            self.assertEqual((await tail.confirm_menu_tail(recorded_window(),sample,AsyncMock()))[1],'ABORT')
            self.assertEqual(sample.await_count,6)

    async def test_conflicting_new_evidence_never_allows_movement(self):
        for fresh in [reading(0,'GAMEPLAY'),reading(.8,'UNCERTAIN'),reading(float('nan')),
                      replace(reading(.6),visual_gameplay=True)]:
            with self.subTest(fresh=fresh),patch.object(tail,'pending_menu_tail',tail.pending_menu_tail):
                install();sample=AsyncMock(return_value=('conflict',fresh))
                self.assertEqual((await tail.confirm_menu_tail(recorded_window(),sample,AsyncMock()))[1],'ABORT')
                sample.assert_awaited_once()

    def test_candidate_requires_both_modalities_and_unbroken_menu(self):
        original=recorded_window()
        self.assertFalse(candidate(original[1:]))
        self.assertFalse(candidate([reading(.1)]*6))
        self.assertFalse(candidate([reading(.6)]*6))
        for i in range(6):
            broken=list(original);broken[i]=reading(0,'GAMEPLAY');self.assertFalse(candidate(broken))
            broken=list(original);broken[i]=replace(broken[i],visual_gameplay=True);self.assertFalse(candidate(broken))
        for i in (3,4,5):
            broken=list(original);broken[i]=reading(.1);self.assertFalse(candidate(broken))

class PlanTests(unittest.TestCase):
    def test_controls_call_real_predict_before_overriding_without_fake_probabilities(self):
        from control_eval import override_prediction
        forward=Mock(return_value=(2,{'max_probability':.5}))
        model,obs=object(),object()
        fn=override_prediction(forward,'jump')
        self.assertEqual(fn(model,obs,deterministic=True,sample_probabilities=True),(3,None))
        forward.assert_called_once_with(model,obs,deterministic=True,sample_probabilities=True)
        self.assertEqual(override_prediction(forward,'learned')(model,obs),(2,{'max_probability':.5}))

    def test_short_plan_contains_no_long_training_or_bc_or_extra_frames(self):
        from short_checks import plan_jobs
        jobs=plan_jobs(Path('reports/test_plan'),50)
        from road_config import PRESETS
        self.assertEqual(len(jobs),24+2*len(PRESETS))
        self.assertEqual(len({j['name'] for j in jobs}),len(jobs))
        trains=[j for j in jobs if j['kind']=='road_train'];self.assertEqual(len(trains),1+len(PRESETS))
        for job in trains:
            cmd=job['command']
            self.assertEqual(cmd[cmd.index('--frames')+1],'4')
            self.assertEqual(cmd[cmd.index('--minutes')+1],'1')
            self.assertNotIn('--bc-root',cmd)
        self.assertTrue(all('--benchmark-only' in j['command'] for j in jobs if j['kind']=='capture'))

    def test_incomplete_training_is_never_reused_but_clean_failure_can_continue(self):
        from short_checks import read_result
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trial_report.json'
            path.write_text(json.dumps(dict(status='STOPPED',finalization_errors=[])))
            result=read_result(dict(kind='road_train',report=str(path)),1)
            self.assertFalse(result['ok']);self.assertTrue(result['clean'])
            path.write_text(json.dumps(dict(status='STOPPED',finalization_errors=[],error='KeyboardInterrupt()')))
            self.assertFalse(read_result(dict(kind='road_train',report=str(path)),1)['clean'])
            path.write_text(json.dumps(dict(status='COMPLETE',audit={'status':'PASS_SETTLED_LEARNING'})))
            self.assertFalse(read_result(dict(kind='road_train',report=str(path)),0)['ok'])

    def test_missing_report_stops_suite(self):
        from short_checks import read_result
        with tempfile.TemporaryDirectory() as tmp:
            r=read_result(dict(kind='control',report=str(Path(tmp)/'missing.json')),1)
            self.assertFalse(r['ok']);self.assertFalse(r['clean'])

if __name__=='__main__':unittest.main()
