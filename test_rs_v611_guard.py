import unittest
from rs_v611_gameplay_guard import screen_gate, pair_decision

T = 0.68
C = 0.90

def gate(state, score=0.02, p=.99):
    probs = {'GAMEPLAY': p if state == 'GAMEPLAY' else 0.01,
             'MENU': p if state == 'MENU' else 0.01}
    return screen_gate(state, probs, score, confidence=C, menu_threshold=T)

class GuardTests(unittest.TestCase):
    def test_clear_gameplay_allows(self):
        self.assertEqual(gate('GAMEPLAY').verdict, 'PLAY')
    def test_menu_even_if_template_missing(self):
        self.assertEqual(gate('MENU', score=.01).verdict, 'MENU')
    def test_loading_never_allows_keypress(self):
        self.assertEqual(gate('LOADING').verdict, 'BLOCK')
    def test_uncertain_never_allows_keypress(self):
        self.assertEqual(gate('UNCERTAIN').verdict, 'BLOCK')
    def test_gameplay_predicted_but_template_high_blocks(self):
        self.assertEqual(gate('GAMEPLAY',score=.74).verdict, 'BLOCK')
    def test_two_menu_frames_force_terminal(self):
        self.assertEqual(pair_decision(gate('MENU'),gate('MENU')), 'TERMINAL')
    def test_transition_recover_requires_third_capture(self):
        self.assertEqual(pair_decision(gate('MENU'),gate('GAMEPLAY')), 'CONFIRM_PLAY')
    def test_second_menu_requires_third_capture(self):
        self.assertEqual(pair_decision(gate('LOADING'),gate('MENU')), 'CONFIRM_MENU')
    def test_two_uncertain_frames_hard_abort(self):
        self.assertEqual(pair_decision(gate('UNCERTAIN'),gate('LOADING')), 'ABORT')
    def test_invalid_probability_blocks(self):
        self.assertEqual(screen_gate('GAMEPLAY', {'GAMEPLAY':float('nan')}, .01,
                                     confidence=C, menu_threshold=T).verdict,'BLOCK')

if __name__=='__main__':
    unittest.main()
