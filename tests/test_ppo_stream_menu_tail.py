import unittest
from unittest.mock import AsyncMock
from rs_v612_gameplay_guard import screen_gate,sequence_decision
from tools.ppo_stream_menu_tail import pending_menu_tail,confirm_menu_tail


def reading(state,score):
    return screen_gate(state,{state:1},score,confidence=.9,menu_threshold=.68)


def failed_window():
    return [reading('GAMEPLAY',.56)]+[reading('MENU',v) for v in [.97,.53,.63,.69,.58]]


class MenuTailTests(unittest.IsolatedAsyncioTestCase):
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
