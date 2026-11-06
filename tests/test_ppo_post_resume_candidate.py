"""Replay the overnight weak-template MENU failure against current runtime."""
import asyncio
import types
import time
import unittest
from unittest.mock import AsyncMock, patch

from tools.ppo_resync_transient_v172 import ResyncWindow
from tests import test_ppo_current_runtime as runtime
from tests import test_ppo_pause_runtime_v179 as refresh
from tests import test_ppo_settled_learning as learning


def confirm(sequence, *, focused=True, candidate=None):
    env=runtime.actual_class_method('confirm_post_optimizer_menu')()
    env.page=types.SimpleNamespace(evaluate=AsyncMock(return_value=focused))
    env._guard_hard_stopped=False
    env.state_confidence=.9
    env.startup_motion_threshold=2.2
    env.args=types.SimpleNamespace(threshold=.68)
    env.readings=iter(sequence)
    env.captures=0
    env.keys=[]
    env._ppo_post_resume_menu_candidate=candidate
    async def capture():
        env.current=next(env.readings)
        env.captures+=1
        return None,env.current[1],None
    env._capture=capture
    env._read_screen_state=lambda:(env.current[0],{
        'GAMEPLAY':float(env.current[0]=='GAMEPLAY'),
        'MENU':float(env.current[0]=='MENU')})
    env._call=lambda coro,timeout:asyncio.run(coro)
    with patch('asyncio.sleep',new=AsyncMock()):
        result=env.confirm_post_optimizer_menu()
    return env,result


def overnight_refresh():
    with patch('asyncio.sleep',new=AsyncMock()):
        return refresh.run([refresh.frame(),
            refresh.frame('MENU',pg=0.,pm=.999,score=.484)])


class CandidateTests(unittest.TestCase):
    def test_callback_uses_reanchored_terminal_time_after_intervening_gameplay(self):
        terminal_durations=[]
        class ReanchoredProtocol(learning.ProtocolEnv):
            def refresh_after_ppo_optimizer(self):
                result=super().refresh_after_ppo_optimizer()
                if self.mode=='post_resume_death' and self.boundaries==1:
                    self.episode_start=time.monotonic()-5.
                    self._ppo_post_resume_menu_candidate=(time.monotonic()-2.,
                                                          time.perf_counter()-2.,time.time()-2.)
                return result

            def confirm_post_optimizer_menu(self):
                # Real confirmation resets the false early anchor after PLAY.
                self._ppo_post_resume_menu_candidate=(time.monotonic()-.01,
                                                      time.perf_counter()-.01,time.time()-.01)
                return True,{'checks':[{}]*11}

            def step(self,action):
                if self._ppo_boundary_menu_pending:
                    terminal_durations.append(self._ppo_boundary_menu_duration_at)
                return super().step(action)

        with patch.object(learning,'ProtocolEnv',ReanchoredProtocol):
            learning.SettledTests().run_case('post_resume_death')
        self.assertEqual(len(terminal_durations),1)
        self.assertGreater(terminal_durations[0],4.5)
        self.assertLess(terminal_durations[0],5.5)

    def test_late_menu_replays_failure_and_requires_three_more_corroborations(self):
        original=[('GAMEPLAY',0.)]*5+[('MENU',.99948),('MENU',.54292),('MENU',.56709)]
        env,(accepted,evidence)=confirm(original+[('MENU',.60)]*3,candidate=(1.,1.,1.))
        self.assertTrue(accepted)
        self.assertEqual(env.captures,11)
        self.assertEqual(evidence['max_checks'],12)
        self.assertTrue(all(r['corroborated'] for r in evidence['checks'][-4:]))
        self.assertGreater(env._ppo_post_resume_menu_candidate[0],1.)
        self.assertEqual(env.keys,[])

    def test_late_menu_weak_continuation_stops_at_twelve(self):
        original=[('GAMEPLAY',0.)]*5+[('MENU',.99948),('MENU',.54292),('MENU',.56709)]
        env,(accepted,evidence)=confirm(original+[('MENU',.10)]*4)
        self.assertFalse(accepted)
        self.assertEqual(env.captures,12)
        self.assertEqual(env.keys,[])

    def test_late_extension_conflict_stops_without_input(self):
        original=[('GAMEPLAY',0.)]*5+[('MENU',.99948),('MENU',.54292),('MENU',.56709)]
        for state in ('GAMEPLAY','UNCERTAIN','LOADING'):
            with self.subTest(state=state):
                env,(accepted,_)=confirm(original+[(state,0.)])
                self.assertFalse(accepted)
                self.assertEqual(env.captures,9)
                self.assertEqual(env.keys,[])

    def test_false_candidate_does_not_keep_early_menu_idle_anchor(self):
        env,(accepted,_)=confirm([('GAMEPLAY',0.)]*8,candidate=(1.,1.,1.))
        self.assertFalse(accepted)
        self.assertIsNone(env._ppo_post_resume_menu_candidate)

    def test_exact_overnight_frame_requests_confirmation_without_gameplay_vote(self):
        env,obs,reason,log=overnight_refresh()
        self.assertIsNone(obs)
        self.assertIn('state=MENU',reason)
        self.assertEqual(env.index,2)
        self.assertEqual(env.packed,1)
        self.assertEqual(len(env._ppo_post_resume_menu_candidate),3)
        self.assertIn('initial menu_score=0.484',log)
        self.assertEqual(env.keys,[])

    def test_weak_first_frame_then_four_corroborated_frames_confirms(self):
        env,(accepted,evidence)=confirm([('MENU',.484)]+[('MENU',.60)]*4)
        self.assertTrue(accepted)
        self.assertEqual(len(evidence['checks']),5)
        self.assertFalse(evidence['checks'][0]['corroborated'])
        self.assertEqual(env.keys,[])

    def test_persistent_ml_menu_without_template_still_stops_at_eight(self):
        env,(accepted,evidence)=confirm([('MENU',.484)]*8)
        self.assertFalse(accepted)
        self.assertEqual(env.captures,8)
        self.assertEqual(len(evidence['checks']),8)
        self.assertEqual(env.keys,[])

    def test_gameplay_disagreement_does_not_become_terminal(self):
        env,(accepted,_)=confirm([('GAMEPLAY',0.)]*8)
        self.assertFalse(accepted)
        self.assertEqual(env.keys,[])

    def test_focus_loss_stops_without_capture(self):
        env,(accepted,_)=confirm([],focused=False)
        self.assertFalse(accepted)
        self.assertEqual(env.captures,0)

    def test_invalid_evidence_or_low_confidence_cannot_request_confirmation(self):
        for pg,pm,score in [(0.,.89,.484),(0.,float('nan'),.484),
                (float('nan'),.999,.484),(0.,.999,float('nan')),
                (0.,1.1,.484),(0.,.999,-.1)]:
            with self.subTest(pg=pg,pm=pm,score=score):
                window=ResyncWindow()
                self.assertEqual(window.observe(state='MENU',probs={'GAMEPLAY':pg,'MENU':pm},
                    menu_score=score,confidence=.9,menu_threshold=.68),'ABORT')
                self.assertFalse(window.menu_candidate)

    def test_recovered_menu_is_a_passive_terminal_with_one_death_penalty(self):
        class WeakMenuProtocol(learning.ProtocolEnv):
            def refresh_after_ppo_optimizer(self):
                if self.mode=='post_resume_death' and self.boundaries==1:
                    env,obs,reason,_=overnight_refresh()
                    # Confirmation sleeps are mocked out. Supply a positive
                    # synthetic interval even on Windows' coarse monotonic tick.
                    self._ppo_post_resume_menu_candidate=tuple(
                        stamp-.05 for stamp in env._ppo_post_resume_menu_candidate)
                    return obs,reason,None
                return super().refresh_after_ppo_optimizer()

            def confirm_post_optimizer_menu(self):
                return confirm([('MENU',.484)]+[('MENU',.60)]*4)[1]

        with patch.object(learning,'ProtocolEnv',WeakMenuProtocol):
            rows,stats=learning.SettledTests().run_case('post_resume_death')
        terminal=next(r for r in rows if r.get('kind')=='passive_resync')
        self.assertTrue(terminal['done'])
        self.assertIsNone(terminal['sampled_action'])
        self.assertFalse(terminal['actor_mask'])
        self.assertIsNone(terminal['command'])
        self.assertEqual(sum(s['pieces']['death'] for s in terminal['reward_substeps']),-1.)
        self.assertEqual(stats['optimized_transitions'],8)


if __name__=='__main__':
    unittest.main()
