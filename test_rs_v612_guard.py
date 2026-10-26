import unittest
from rs_v612_gameplay_guard import screen_gate, sequence_decision

T = .68
C = .90

def gate(state='UNCERTAIN', score=.02, confidence=.99, *, moving=False):
    return screen_gate(state,
                       {'GAMEPLAY': confidence if state == 'GAMEPLAY' else .01,
                        'MENU': confidence if state == 'MENU' else .01},
                       score, confidence=C, menu_threshold=T,
                       track_motion=3.5 if moving else 0.,
                       track_texture=25. if moving else 0.,
                       hud_pixels=40 if moving else 0,
                       hud_changed_pixels=14 if moving else 0)

class V612GuardTests(unittest.TestCase):
    def test_play_valid(self):
        self.assertEqual(gate('GAMEPLAY').verdict, 'PLAY')
    def test_ml_menu_even_at_99_percent_is_not_menu_evidence(self):
        reading = gate('MENU')
        self.assertEqual(reading.verdict, 'BLOCK')
        self.assertTrue(reading.ml_menu)
        self.assertFalse(reading.menu_evidence)
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
    def test_two_ml_only_menu_frames_do_not_confirm_death(self):
        self.assertEqual(sequence_decision([gate('MENU'), gate('MENU')]), 'CONTINUE')
    def test_two_corroborated_menu_frames_confirm(self):
        self.assertEqual(sequence_decision([gate('MENU', .90), gate('MENU', .92)]), 'TERMINAL')
    def test_episode_12_real_menu_with_moderate_template_confirms_after_four(self):
        # Human-verified title menu: the template remained at .577191 while
        # the existing V4 classifier gave MENU >99.9% in six frames.
        probabilities = [.9997198, .9997024, .9996539,
                         .9996007, .9995453, .9994847]
        readings = [screen_gate('MENU', {'MENU': p}, .5771913909581744,
                                confidence=C, menu_threshold=T)
                    for p in probabilities]
        for length in (1, 2, 3):
            self.assertEqual(sequence_decision(readings[:length]), 'CONTINUE')
        self.assertEqual(sequence_decision(readings[:4]), 'TERMINAL')
        self.assertTrue(all(r.ml_template_corroborated for r in readings))
        self.assertFalse(any(r.menu_evidence for r in readings))
    def test_episode_0002_real_menu_with_fluctuating_template_scores(self):
        # Exact saved diagnostic from v612_guard_long_20261021_045943.
        # Previously aborted at six frames even though the screenshot was a menu.
        probabilities = [.9772104621, .9865583777, .9926880598,
                         .9947255254, .9968750477, .9976739287]
        menu_scores = [.6523618148, .9990614245, .6635738645,
                       .4681650196, .5785498361, .731]
        readings = [screen_gate('MENU', {'MENU': p}, score,
                                confidence=C, menu_threshold=T)
                    for p, score in zip(probabilities, menu_scores)]
        for length in range(1, 6):
            self.assertEqual(sequence_decision(readings[:length]), 'CONTINUE')
        self.assertEqual(sequence_decision(readings), 'TERMINAL')

    def test_spread_out_template_matches_need_continuous_ml_menu(self):
        # Two distant template peaks cannot bridge over actual GAMEPLAY.
        seq = [gate('MENU', .65), gate('MENU', .99), gate('MENU', .66),
               gate('GAMEPLAY', .02), gate('MENU', .58), gate('MENU', .73)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_spread_out_matches_require_multiple_moderate_template_frames(self):
        # Two isolated spikes in six ML MENU frames are not enough.
        seq = [gate('MENU', .04), gate('MENU', .99), gate('MENU', .05),
               gate('MENU', .03), gate('MENU', .04), gate('MENU', .74)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_spread_out_matches_reject_independent_visual_gameplay(self):
        # Even with sustained ML MENU, positive gameplay evidence prevents
        # the long-span menu accumulator from bridging over that frame.
        seq = [gate('MENU', .65), gate('MENU', .99), gate('MENU', .66),
               gate('MENU', .05, moving=True), gate('MENU', .58), gate('MENU', .73)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_single_strong_template_and_moderate_menu_does_not_confirm(self):
        # Do not allow a single high template false positive to end a run.
        seq = [gate('MENU', .65), gate('MENU', .99), gate('MENU', .66),
               gate('MENU', .47), gate('MENU', .58), gate('MENU', .67)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_1631_menu_with_moderate_final_template(self):
        # Exact six-capture evidence from the long-run stop. The screenshot
        # is the FIND RANKED MATCH menu, not active gameplay.
        menu_probs = [.9974838495, .9980006814, .9983571172,
                      .9986847043, .9989854693, .9992403984]
        scores = [.8445847368, .5076446684, .4936935573,
                  .6700425811, .7300076586, .6620018850]
        seq = [screen_gate('MENU', {'MENU': p, 'GAMEPLAY': 1-p}, score,
                           confidence=C, menu_threshold=T)
               for p, score in zip(menu_probs, scores)]
        for n in range(1, 6):
            self.assertEqual(sequence_decision(seq[:n]), 'CONTINUE')
        self.assertEqual([r.verdict for r in seq],
                         ['MENU', 'BLOCK', 'BLOCK', 'BLOCK', 'MENU', 'BLOCK'])
        self.assertEqual(sequence_decision(seq), 'TERMINAL')

    def test_episode_1631_rejects_unconfirmed_final_frame(self):
        # A late weak template reading must not be treated as a menu.
        scores = [.845, .58, .49, .67, .73, .50]
        self.assertEqual(sequence_decision([gate('MENU', x) for x in scores]),
                         'ABORT')

    def test_episode_1631_requires_strong_recent_template(self):
        # Peaks only at the start cannot confirm the final UI state.
        scores = [.84, .75, .58, .49, .65, .66]
        self.assertEqual(sequence_decision([gate('MENU', x) for x in scores]),
                         'ABORT')

    def test_episode_1631_requires_two_strong_template_hits(self):
        scores = [.845, .58, .49, .67, .66, .66]
        self.assertEqual(sequence_decision([gate('MENU', x) for x in scores]),
                         'ABORT')

    def test_episode_1631_rejects_positive_gameplay_evidence(self):
        seq = [gate('MENU', .845), gate('MENU', .58),
               gate('MENU', .02, moving=True), gate('MENU', .67),
               gate('MENU', .73), gate('MENU', .66)]
        self.assertTrue(seq[2].visual_gameplay)
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_1631_rejects_intervening_unknown_state(self):
        seq = [gate('MENU', .845), gate('MENU', .58),
               gate('UNCERTAIN', .59), gate('MENU', .67),
               gate('MENU', .73), gate('MENU', .66)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_0058_menu_animation_uncertain_then_menu(self):
        # Exact six-frame evidence from episode 58, human verified as the
        # FIND RANKED MATCH menu. Previously ended with ABORT.
        pmenus = [.4785635769, .6557241082, .8160337210,
                  .9248312712, .9532647729, .9813460708]
        scores = [.6523618148, .9990614245, .6635738645,
                  .4721082775, .5978451150, .7340090567]
        readings = []
        for p_menu, score in zip(pmenus, scores):
            state = 'MENU' if p_menu >= C else 'UNCERTAIN'
            readings.append(screen_gate(state,
                                        {'MENU': p_menu, 'GAMEPLAY': 1 - p_menu},
                                        score, confidence=C, menu_threshold=T))
        for length in range(1, 6):
            self.assertEqual(sequence_decision(readings[:length]), 'CONTINUE')
        self.assertEqual(sequence_decision(readings), 'TERMINAL')
        self.assertEqual([r.state for r in readings],
                         ['UNCERTAIN'] * 3 + ['MENU'] * 3)

    def test_episode_0058_requires_repeated_strong_template_evidence(self):
        seq = [gate('UNCERTAIN', .65), gate('UNCERTAIN', .99),
               gate('UNCERTAIN', .66), gate('MENU', .47),
               gate('MENU', .60), gate('MENU', .63)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_0058_rejects_actual_gameplay_in_sequence(self):
        seq = [gate('UNCERTAIN', .65), gate('UNCERTAIN', .99),
               gate('GAMEPLAY', .02), gate('MENU', .47),
               gate('MENU', .60), gate('MENU', .73)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_0058_rejects_visual_gameplay_evidence(self):
        seq = [gate('UNCERTAIN', .65), gate('UNCERTAIN', .99),
               gate('UNCERTAIN', .66), gate('MENU', .02, moving=True),
               gate('MENU', .60), gate('MENU', .73)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_0058_rejects_menu_votes_without_templates(self):
        seq = [gate('UNCERTAIN', .02), gate('UNCERTAIN', .03),
               gate('UNCERTAIN', .08), gate('MENU', .05),
               gate('MENU', .09), gate('MENU', .16)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_episode_0058_rejects_loading_interleaved(self):
        seq = [gate('UNCERTAIN', .65), gate('UNCERTAIN', .99),
               gate('LOADING', .66), gate('MENU', .47),
               gate('MENU', .60), gate('MENU', .73)]
        self.assertEqual(sequence_decision(seq), 'ABORT')

    def test_moderate_template_without_ml_menu_does_not_confirm(self):
        readings = [gate('UNCERTAIN', .58)] * 6
        self.assertEqual(sequence_decision(readings), 'ABORT')
    def test_ml_menu_with_weak_below_corrob_threshold_does_not_confirm(self):
        readings = [gate('MENU', .54)] * 6
        self.assertEqual(sequence_decision(readings), 'ABORT')
    def test_intermittent_ml_menu_with_moderate_template_does_not_confirm(self):
        readings = [gate('MENU', .58), gate('MENU', .58),
                    gate('UNCERTAIN', .58), gate('MENU', .58),
                    gate('MENU', .58), gate('UNCERTAIN', .58)]
        self.assertEqual(sequence_decision(readings), 'ABORT')
    def test_ml_menu_with_low_template_and_visual_progress_recovers(self):
        seq = [gate('UNCERTAIN'), gate('MENU', moving=True), gate('MENU', moving=True)]
        self.assertEqual(sequence_decision(seq), 'PLAY')
    def test_visual_motion_without_advancing_hud_never_allows_play(self):
        seq = [gate('MENU')]
        for _ in range(5):
            seq.append(screen_gate('MENU', {'MENU': .99}, .02, confidence=C,
                                   menu_threshold=T, track_motion=5.,
                                   track_texture=30., hud_pixels=70,
                                   hud_changed_pixels=0))
        self.assertEqual(sequence_decision(seq), 'ABORT')
    def test_hud_change_without_track_motion_never_allows_play(self):
        seq = [gate('MENU')]
        for _ in range(5):
            seq.append(screen_gate('MENU', {'MENU': .99}, .02, confidence=C,
                                   menu_threshold=T, track_motion=0.,
                                   track_texture=30., hud_pixels=70,
                                   hud_changed_pixels=20))
        self.assertEqual(sequence_decision(seq), 'ABORT')
    def test_episode_76_ml_menu_probabilities_do_not_confirm_death(self):
        # Gameplay at 41 m: the original guard falsely considered these
        # high MENU probabilities independent menu evidence.
        probabilities = [(.734, .266), (.495, .505), (.083, .917),
                         (.154, .846), (.105, .895), (.012, .988)]
        menu_scores = [.00, .04, .06, .11, .09, .16]
        readings = []
        for (p_game, p_menu), score in zip(probabilities, menu_scores):
            state = ('MENU' if p_menu >= C else
                     'GAMEPLAY' if p_game >= C else 'UNCERTAIN')
            readings.append(screen_gate(state, {'GAMEPLAY': p_game, 'MENU': p_menu},
                                        score, confidence=C, menu_threshold=T))
        self.assertEqual(sequence_decision(readings), 'ABORT')
        self.assertFalse(any(r.menu_evidence for r in readings))
    def test_episode_76_with_advancing_distance_recovers(self):
        readings = [gate('UNCERTAIN'), gate('UNCERTAIN', moving=True),
                    gate('MENU', moving=True)]
        self.assertEqual(sequence_decision(readings), 'PLAY')
    def test_menu_template_conflict_prevents_visual_recovery(self):
        seq = [gate('MENU', .9), gate('MENU', .05, moving=True),
               gate('MENU', .03, moving=True)]
        self.assertEqual(sequence_decision(seq), 'CONTINUE')
    def test_loading_never_allows_action(self):
        self.assertEqual(sequence_decision([gate('LOADING'), gate('LOADING')]), 'CONTINUE')
    def test_six_uncertain_aborts(self):
        self.assertEqual(sequence_decision([gate()] * 6), 'ABORT')
    def test_menu_evidence_is_not_old(self):
        seq = [gate('MENU', .9), gate('GAMEPLAY'), gate('GAMEPLAY'), gate('UNCERTAIN', 1.0)]
        self.assertEqual(sequence_decision(seq), 'CONTINUE')
    def test_nan_blocks(self):
        self.assertEqual(screen_gate('GAMEPLAY', {'GAMEPLAY':float('nan')}, .02,
                                     confidence=C, menu_threshold=T).verdict, 'BLOCK')

if __name__ == '__main__':
    unittest.main()
