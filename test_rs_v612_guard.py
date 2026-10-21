import unittest
from rs_v612_gameplay_guard import screen_gate, sequence_decision

T = .68
C = .90

def gate(state='UNCERTAIN', score=.02, confidence=.99):
    return screen_gate(state,
                       {'GAMEPLAY': confidence if state == 'GAMEPLAY' else .01,
                        'MENU': confidence if state == 'MENU' else .01},
                       score, confidence=C, menu_threshold=T)

class V612GuardTests(unittest.TestCase):
    def test_play_valid(self):
        self.assertEqual(gate('GAMEPLAY').verdict, 'PLAY')
    def test_menu_valid_even_low_template(self):
        self.assertEqual(gate('MENU').verdict, 'MENU')
    def test_template_high_blocks_play_even_if_ml_gameplay(self):
        self.assertEqual(gate('GAMEPLAY', .99).verdict, 'BLOCK')
    def test_uncertain_fails_closed(self):
        self.assertEqual(gate().verdict, 'BLOCK')
    def test_high_template_uncertain_counts_menu_evidence(self):
        self.assertTrue(gate('UNCERTAIN', 1.0).menu_evidence)
    def test_observed_crash_becomes_confirmed_menu(self):
        seq = [gate('UNCERTAIN', .57), gate('UNCERTAIN', 1.0)]
        self.assertEqual(sequence_decision(seq), 'CONTINUE')
        seq.append(gate('UNCERTAIN', 1.0))
        self.assertEqual(sequence_decision(seq), 'TERMINAL')
    def test_uncertain_followed_by_two_play_resumes(self):
        seq = [gate('UNCERTAIN', .57), gate('GAMEPLAY'), gate('GAMEPLAY')]
        self.assertEqual(sequence_decision(seq), 'PLAY')
    def test_single_strong_template_never_ends_episode(self):
        seq = [gate('UNCERTAIN', .57), gate('UNCERTAIN', 1.0), gate('UNCERTAIN', .57)]
        self.assertEqual(sequence_decision(seq), 'CONTINUE')
    def test_two_menu_ml_frames_confirm(self):
        self.assertEqual(sequence_decision([gate('MENU'), gate('MENU')]), 'TERMINAL')
    def test_loading_never_allows_action(self):
        self.assertEqual(sequence_decision([gate('LOADING'), gate('LOADING')]), 'CONTINUE')
    def test_six_uncertain_aborts(self):
        self.assertEqual(sequence_decision([gate()] * 6), 'ABORT')
    def test_menu_evidence_is_not_old(self):
        seq = [gate('MENU'), gate('GAMEPLAY'), gate('GAMEPLAY'), gate('UNCERTAIN', 1.0)]
        self.assertEqual(sequence_decision(seq), 'CONTINUE')
    def test_nan_blocks(self):
        self.assertEqual(screen_gate('GAMEPLAY', {'GAMEPLAY':float('nan')}, .02,
                                     confidence=C, menu_threshold=T).verdict, 'BLOCK')

if __name__ == '__main__':
    unittest.main()
