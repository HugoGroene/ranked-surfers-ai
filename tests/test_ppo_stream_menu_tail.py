from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
from rs_v612_gameplay_guard import GateReading,screen_gate,sequence_decision
from tools.ppo_stream_menu_tail import pending_menu_tail,confirm_menu_tail


def reading(state,score):
    return screen_gate(state,{state:1},score,confidence=.9,menu_threshold=.68)


def failed_window():
    return [reading('GAMEPLAY',.56)]+[reading('MENU',v) for v in [.97,.53,.63,.69,.58]]


def two_dip_failure():
    path=Path(__file__).parent/'fixtures/menu_transition_20261031_ep0112.json'
    return [GateReading(**row) for row in json.loads(path.read_text())['readings']]


class MenuTailTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_recent_corroborated_frames_request_fresh_fourth_vote_after_a_dip(self):
        path=Path(__file__).parent/'fixtures/menu_transition_20261031_six_ep0001.json'
        readings=[GateReading(**row) for row in json.loads(path.read_text())['readings']]
        self.assertEqual(sequence_decision(readings,max_frames=6),'ABORT')
        self.assertFalse(any(r.menu_evidence for r in readings))
        self.assertTrue(pending_menu_tail(readings))
        sample=AsyncMock(return_value=('fresh moderate MENU',reading('MENU',.60)))
        _,verdict=await confirm_menu_tail(readings,sample,AsyncMock())
        self.assertEqual(verdict,'TERMINAL')
        self.assertTrue(all(r.ml_template_corroborated for r in readings[-4:]))
        sample.assert_awaited_once()

    async def test_moderate_candidate_still_needs_new_support_and_expires_without_it(self):
        path=Path(__file__).parent/'fixtures/menu_transition_20261031_six_ep0001.json'
        original=[GateReading(**row) for row in json.loads(path.read_text())['readings']]
        sample=AsyncMock(return_value=('weak',reading('MENU',.10)))
        self.assertEqual((await confirm_menu_tail(list(original),sample,AsyncMock()))[1],'ABORT')
        self.assertEqual(sample.await_count,6)
        broken=list(original);broken[-2]=reading('MENU',.10)
        self.assertFalse(pending_menu_tail(broken))

    async def test_recorded_two_dip_candidate_requires_a_new_corroborated_capture(self):
        readings=two_dip_failure()
        self.assertEqual(sequence_decision(readings,max_frames=6),'ABORT')
        self.assertEqual(sum(r.ml_template_corroborated for r in readings[-5:]),3)
        self.assertEqual(sum(r.menu_evidence for r in readings[-5:]),2)
        self.assertTrue(pending_menu_tail(readings))
        sample=AsyncMock(return_value=('new independent frame',reading('MENU',.60)))
        wait=AsyncMock()
        frame,verdict=await confirm_menu_tail(readings,sample,wait)
        self.assertEqual((frame,verdict),('new independent frame','TERMINAL'))
        sample.assert_awaited_once()
        wait.assert_awaited_once_with(.045)
        self.assertEqual(sequence_decision(readings[-6:],max_frames=6),'TERMINAL')

    async def test_recorded_candidate_with_persistent_weak_templates_stops_at_budget(self):
        readings=two_dip_failure()
        sample=AsyncMock(return_value=('weak menu',reading('MENU',.10)))
        self.assertEqual((await confirm_menu_tail(readings,sample,AsyncMock()))[1],'ABORT')
        self.assertEqual(sample.await_count,6)

    async def test_recorded_candidate_aborts_on_any_new_gameplay_or_invalid_evidence(self):
        candidates=[reading('GAMEPLAY',0),reading('UNCERTAIN',.9),
                    reading('LOADING',.9),reading('PAUSED',.9),
                    reading('MENU',float('nan')),
                    replace(reading('MENU',.60),visual_gameplay=True)]
        for candidate in candidates:
            with self.subTest(reading=candidate):
                sample=AsyncMock(return_value=('conflicting frame',candidate))
                self.assertEqual((await confirm_menu_tail(two_dip_failure(),sample,AsyncMock()))[1],'ABORT')
                sample.assert_awaited_once()

    async def test_two_dip_route_needs_repeated_strong_and_recent_template_support(self):
        original=two_dip_failure()
        variants=[]
        weak_strong=list(original);weak_strong[1]=reading('MENU',.60);variants.append(weak_strong)
        weak_final=list(original);weak_final[-1]=reading('MENU',.10);variants.append(weak_final)
        weak_penultimate=list(original);weak_penultimate[-2]=reading('MENU',.10);variants.append(weak_penultimate)
        broken_menu=list(original);broken_menu[-3]=reading('GAMEPLAY',0);variants.append(broken_menu)
        visual=list(original);visual[-3]=replace(visual[-3],visual_gameplay=True);variants.append(visual)
        for readings in variants:
            with self.subTest(scores=[r.menu_score for r in readings]):
                self.assertFalse(pending_menu_tail(readings))
                sample=AsyncMock()
                self.assertEqual(await confirm_menu_tail(readings,sample,AsyncMock()),(None,'ABORT'))
                sample.assert_not_awaited()

    async def test_capture_failure_is_propagated_without_a_terminal_verdict(self):
        from rs_v612_gameplay_guard import GuardUncertainError
        sample=AsyncMock(side_effect=GuardUncertainError('stale capture'))
        with self.assertRaisesRegex(GuardUncertainError,'stale capture'):
            await confirm_menu_tail(two_dip_failure(),sample,AsyncMock())
        sample.assert_awaited_once()

    async def test_exact_failure_needs_one_more_corroborated_frame(self):
        readings=failed_window()
        self.assertEqual(sequence_decision(readings,max_frames=6),'ABORT')
        frame,verdict=await confirm_menu_tail(readings,AsyncMock(return_value=('frame',reading('MENU',.6))),AsyncMock())
        self.assertEqual((frame,verdict),('frame','TERMINAL'))
        self.assertEqual(len(readings),7)
        self.assertEqual(sequence_decision(readings[-6:],max_frames=6),'TERMINAL')

    async def test_positive_gameplay_never_authorizes_old_action(self):
        frame,verdict=await confirm_menu_tail(failed_window(),AsyncMock(return_value=('game',reading('GAMEPLAY',0))),AsyncMock())
        self.assertEqual(verdict,'ABORT')

    async def test_sustained_menu_needs_original_four_consecutive_template_votes(self):
        readings=[reading('MENU',v) for v in [.63,.94,.49,.67,.677,.58]]
        self.assertEqual(sequence_decision(readings,max_frames=6),'ABORT')
        sample=AsyncMock(return_value=('frame',reading('MENU',.6)))
        _,verdict=await confirm_menu_tail(readings,sample,AsyncMock())
        self.assertEqual(verdict,'TERMINAL')
        self.assertEqual(sequence_decision(readings[-6:],max_frames=6),'TERMINAL')

    async def test_ml_only_menu_is_not_enough(self):
        readings=[reading('MENU',.1) for _ in range(6)]
        self.assertFalse(pending_menu_tail(readings))
        sample=AsyncMock()
        self.assertEqual(await confirm_menu_tail(readings,sample,AsyncMock()),(None,'ABORT'))
        sample.assert_not_awaited()

    async def test_extra_capture_budget_is_bounded(self):
        sample=AsyncMock(return_value=('menu',reading('MENU',.1)))
        _,verdict=await confirm_menu_tail(failed_window(),sample,AsyncMock())
        self.assertEqual(verdict,'ABORT')
        self.assertEqual(sample.await_count,6)


if __name__=='__main__':
    unittest.main()
