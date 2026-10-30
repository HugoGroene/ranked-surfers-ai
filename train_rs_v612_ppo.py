#!/usr/bin/env python3
"""LOCAL Ranked Surfers V6.1.2: PPO with richer perception, rewards, input probe.

Unlike rs_v5_segmentation_live_benchmark.py, PPO's sampled movement is actually
executed and the nine-class YOLO features are included in the observation.
No demonstrations, scripted hazard decisions, reward from detector predictions,
or live/ranked website access. Original V4/V5 scripts and checkpoints untouched.

Modes: check (offline PPO wiring), probe (input audit), train, eval.
See README_RS_V61.md before training. V6 checkpoints incompatible by design.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from rs_v6_ppo_features import encode_hazards
from rs_v61_features import (AUX_DIM, V6_AUX_DIM, SCREEN_COLUMN_DIM, FEATURE_VERSION, ActionContext,
                             encode_aux, shape_description, ACTION_NAMES)
from rs_v61_rewards import ProgressRewards
from rs_v612_gameplay_guard import screen_gate, sequence_decision, GuardUncertainError

# The proven local Chromium/state-classifier controller is inherited rather
# than being rewritten; imported lazily to keep schema utilities lightweight.
import rs_v5_segmentation_live_benchmark as bench


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def build_env_class():
    from gymnasium import spaces
    import train_rs_v3_ppo as v3

    OldEnv = bench.build_live_env_class()

    from tools.ppo_background_display import BackgroundDisplayMixin

    class PPOV61Env(BackgroundDisplayMixin, OldEnv):
        """Preserve proven V4/V5 control, extend PPO observations without rules."""
        async def _open_browser(self):
            await super()._open_browser()
            if getattr(self,'_framing_profile','legacy') == 'canvas_v1' and not self._background_enabled():
                await self._calibrate_policy_canvas()
            if getattr(self, '_background_display_index', None) is not None:
                return
            from tools.ppo_browser_focus import ensure_startup_focus
            try:
                await ensure_startup_focus(self.page)
            except Exception as exc:
                raise GuardUncertainError(f'[PPO FOCUS] Startup stopped: {exc}') from exc

        def __init__(self, cfg, output, demos, detector, *, a):
            self._framing_profile = getattr(a,'framing_profile','legacy')
            self._background_display_index = getattr(a, 'background_display', None)
            self._background_canvas_policy = bool(getattr(a, 'canvas_policy_input', False))
            self._background_capture_mode = getattr(a, 'background_capture', 'mss')
            if a.benchmark_action != 'ppo':
                raise ValueError('V6 learning must execute the PPO-sampled action')
            self.reward_survival_rate = float(a.reward_survival_rate)
            self.reward_death_penalty = float(a.reward_death_penalty)
            self.reward_milestone_bonus = float(a.reward_milestone_bonus)
            self.reward_milestone_seconds = float(a.reward_milestone_seconds)
            self.survival_rewards = ProgressRewards(
                per_second=self.reward_survival_rate,
                milestone_bonus=self.reward_milestone_bonus,
                milestone_seconds=self.reward_milestone_seconds,
                death_penalty=self.reward_death_penalty)
            self.reward_parts = Counter()
            self.hazard_observation_count = 0
            self.stale_observation_count = 0
            self.hazard_nonzero_observations = 0
            self.last_aux = None
            self.action_context = ActionContext(assume_center=bool(a.assume_center_lane))
            self.column_nonzero_observations = 0
            self.guard_allowed = 0
            self.guard_rechecks = 0
            self.guard_confirmed_menu = 0
            self.guard_aborts = 0
            self.guard_ml_menu_disagreements = 0
            self.guard_visual_recoveries = 0
            self._guard_forced_terminal = False
            self._guard_hard_stopped = False
            self._ppo_boundary_menu_pending = False
            self._ppo_boundary_menu_duration_at = None
            self._ppo_boundary_menu_started_at = None
            self._ppo_menu_handoff_terminal_observed = False
            self._ppo_pause_visual_reference = None
            self._prepared_gameplay_reading = None
            self.prepared_guard_hits = 0
            self.prepared_guard_fallbacks = 0
            self.prepared_guard_late_fallbacks = 0
            self.reuse_observation_guard = bool(getattr(a, 'reuse_observation_guard', False))
            self.settled_learning = bool(getattr(a, 'settled_learning', False))
            self._learning_origin = None
            self._learning_verified_duration = 0.
            self._learning_excluded_pause = 0.
            self._learning_pause_started = None
            self._learning_passive_kind = None
            self._learning_current_observation = None
            self._learning_dispatched_key = None
            self._learning_capture_serial = 0
            if getattr(self, 'settled_learning', False):
                from tools.ppo_settled_learning import VerifiedProgressRewards
                self.survival_rewards = VerifiedProgressRewards(
                    self.reward_survival_rate,self.reward_milestone_bonus,
                    self.reward_milestone_seconds,self.reward_death_penalty)
            super().__init__(cfg, output, demos, detector, a=a)
            w, h = self.target_wh
            self.observation_space = spaces.Dict({
                'image': spaces.Box(0, 255, shape=(12, h, w), dtype=np.uint8),
                'aux': spaces.Box(0., 1., shape=(AUX_DIM,), dtype=np.float32),
            })

        def _pack(self, rgb_obs, *, start_detection=False):
            """Run one copy of V4's capture/async scheduling but encode V5 objects."""
            start = time.perf_counter()
            self._poll_result()
            frame, native = self._native_frames()
            if not self.native_history:
                for _ in range(bench.NFRAMES):
                    self.native_history.append(frame.copy())
            else:
                self.native_history.append(frame)
            packed = np.concatenate(tuple(self.native_history), axis=0)
            now = time.perf_counter()
            native_present = self._background_present_at()
            source_at = native_present if native_present is not None else (self.last_capture_end or now)
            if self.previous_observation_at is not None:
                self.frame_intervals.append(max(0., source_at-self.previous_observation_at))
            self.previous_observation_at = source_at

            # V5 segmentation runs on the same native RGB crop, in a single
            # worker, without waiting for inference on every PPO decision.
            if start_detection and self.detector_pool is not None:
                if self.job is None:
                    from rs_v5_shadow_segmentation import detect_v5
                    self.job = self.detector_pool.submit(
                        detect_v5, self.detector_model, native.copy(), source_at,
                        self.episode_tag, self.detector_requests, self.imgsz, self.conf)
                    self.detector_requests += 1
                else:
                    self.detector_skipped_busy += 1

            hazards, age_ms, fresh = encode_hazards(
                self.last_detection, now=now, episode_tag=self.episode_tag,
                max_stale_ms=self.max_stale_s*1000.)
            self.hazard_observation_count += 1
            if not fresh:
                self.stale_observation_count += 1
            if fresh and hazards[:-1].any():
                self.hazard_nonzero_observations += 1
            if fresh:
                self.fresh_steps += 1
                self.decision_ages_ms.append(age_ms)
            self.all_steps += 1
            aux = encode_aux(hazards, self.action_history,
                             now-self.episode_started, self.frame_intervals, age_ms,
                             snapshot=self.last_detection, episode_tag=self.episode_tag,
                             now=now, action_context=self.action_context,
                             max_stale_ms=self.max_stale_s*1000.)
            # Screen-column summaries are supplementary; they don't imply danger.
            if aux[V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM].any():
                self.column_nonzero_observations += 1
            self.last_aux = aux
            self._prepared_gameplay_reading = None
            if getattr(self, 'reuse_observation_guard', False) or getattr(self, 'settled_learning', False):
                captured = getattr(self, '_capture_menu_metadata', None)
                if captured is not None and captured[:2] == (id(self._v2_last_bgra), self.last_capture_end):
                    state, probabilities = self._read_screen_state()
                    menu_score = captured[2]
                    reading = screen_gate(state, probabilities, menu_score,
                        confidence=self.state_confidence, menu_threshold=self.args.threshold)
                    if reading.verdict == 'PLAY' and self.last_capture_end is not None:
                        if getattr(self, 'settled_learning', False) and self._learning_origin is not None:
                            at = float(source_at)
                            ongoing_pause = (max(0.,at-self._learning_pause_started)
                                if self._learning_pause_started is not None else 0.)
                            self._learning_verified_duration = max(self._learning_verified_duration,
                                at-self._learning_origin-self._learning_excluded_pause-ongoing_pause)
                        from tools.ppo_prepared_guard import PreparedGameplayReading
                        native_sample = getattr(getattr(self, 'grabber', None), 'last_sample', None)
                        self._prepared_gameplay_reading = PreparedGameplayReading(
                            id(self._v2_last_bgra), self.last_capture_end,
                            state, dict(probabilities), menu_score, presented_at=native_present,
                            source_acquired_at=native_sample.received_at if native_sample is not None else None)
            self.pack_ms.append((time.perf_counter()-start)*1000)
            result = {'image': packed, 'aux': aux}
            self._learning_current_observation = result
            self._learning_observation_capture = dict(
                serial=getattr(self,'_learning_capture_serial',None),captured_at=self.last_capture_end)
            if native_present is not None:
                self._learning_observation_capture['native_present_at'] = native_present
            return result, fresh, age_ms

        async def _capture(self):
            result = await super()._capture()
            self._learning_capture_serial = getattr(self,'_learning_capture_serial',0)+1
            self._capture_menu_metadata = (id(self._v2_last_bgra), self.last_capture_end, float(result[1]))
            return result

        def reset(self, *, seed=None, options=None):
            if self._guard_hard_stopped:
                raise GuardUncertainError('[V612 GUARD] Training latched off after uncertain UI; restart process after reviewing evidence.')
            if getattr(self, '_ppo_time_limit_pending', False):
                from tools.ppo_episode_reset import reload_after_time_limit
                self._call(reload_after_time_limit(self), timeout=120)
                self._ppo_time_limit_pending = False
            self._guard_forced_terminal = False
            self._ppo_boundary_menu_pending = False
            self._ppo_boundary_menu_duration_at = None
            self._ppo_boundary_menu_started_at = None
            self._ppo_menu_handoff_terminal_observed = False
            self._ppo_pause_visual_reference = None
            self.survival_rewards.reset()
            self.action_context.reset()
            self._learning_origin = None
            self._learning_verified_duration = 0.
            self._learning_excluded_pause = 0.
            self._learning_pause_started = None
            result = super().reset(seed=seed, options=options)
            self._learning_origin = time.perf_counter()
            return result

        def refresh_after_ppo_optimizer(self):
            """Rebuild four consecutive confident GAMEPLAY images, without keys.

            A near-GAMEPLAY UNCERTAIN frame just after overlay disappearance is
            neither gameplay proof nor MENU proof. Discard it and any partial
            image stack; allow a bounded number of independent fresh retries.
            No ML threshold changes, speculative restarts or reward changes.
            """
            import asyncio
            from tools.ppo_resync_transient_v172 import (
                ResyncWindow, MAX_RESYNC_CAPTURES)

            async def capture_fresh_stack():
                if self._guard_hard_stopped:
                    raise GuardUncertainError('Cannot refresh after guard hard stop')
                # Any earlier candidate belongs to a different rollout.
                self._ppo_post_resume_menu_candidate = None
                from tools.ppo_pause_visual_v179 import matches_prior_paused_panel
                reference = getattr(self, '_ppo_pause_visual_reference', None)
                if reference is None:
                    return None, 'Missing verified PAUSED visual reference; no PPO keys sent'
                overlay = self.page.locator(
                    '.game-pause-panel[aria-label="Game paused"]')
                paused_frame_skips = 0
                window = ResyncWindow()
                self.native_history.clear()  # never mix pre-/post-optimizer frames
                self.frame_intervals.clear()
                self.previous_observation_at = None
                observation = None
                accepted_traces = []
                previous_native_present = None
                for attempt in range(MAX_RESYNC_CAPTURES):
                    if attempt:
                        await asyncio.sleep(.033)
                    # Only the foreground local Practice may feed the PPO image.
                    focused = bool(await self.page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"'))
                    if not focused:
                        return None, 'browser focus/visibility lost while resynchronizing; no keys sent'
                    # React's DOM and the rendered screenshot can disagree for
                    # a frame after Resume. Neither may be PAUSED before any
                    # image is allowed into PPO's new observation stack.
                    pause_dom_before = bool(await overlay.is_visible())
                    legacy, menu_score, _ = await self._capture()
                    pause_dom_after = bool(await overlay.is_visible())
                    pause_visual, pause_difference = matches_prior_paused_panel(
                        self._v2_last_bgra, reference)
                    if pause_dom_before or pause_dom_after or pause_visual:
                        paused_frame_skips += 1
                        origin = ('DOM' if pause_dom_before or pause_dom_after
                                  else 'RENDERED_FRAME')
                        print('[PPO RESYNC PAUSED] runtime_state=PAUSED '
                              f'capture={attempt+1}/{MAX_RESYNC_CAPTURES} '
                              f'source={origin} visual_delta={pause_difference:.1f}; '
                              'NO movement or gameplay vote.', flush=True)
                        # Even an ML GAMEPLAY label MUST NOT override the panel.
                        # Drop any partial stack collected before this frame.
                        window = ResyncWindow()
                        self.native_history.clear()
                        self.frame_intervals.clear()
                        self.previous_observation_at = None
                        observation = None
                        accepted_traces.clear()
                        if paused_frame_skips >= 4:
                            return None, ('PAUSED overlay persisted after resume '
                                          f'for {paused_frame_skips} frames; no keys sent')
                        continue
                    state, probs = self._read_screen_state()
                    if state == 'GAMEPLAY' and getattr(self, '_background_capture_mode', 'mss') == 'dxgi':
                        distinct, previous_native_present = self._background_resync_present(previous_native_present)
                        if not distinct:
                            window = ResyncWindow()
                            self.native_history.clear()
                            self.frame_intervals.clear()
                            self.previous_observation_at = None
                            accepted_traces.clear()
                            observation = None
                            print('[PPO RESYNC NATIVE] Cached/old GPU frame excluded; no gameplay vote.', flush=True)
                            continue
                    # A terminal screen can appear AFTER Unity resumes, while
                    # we are passively capturing the four fresh PPO frames.
                    # Record its FIRST corroborated observation timestamp,
                    # but never authorize a menu from ML confidence alone.
                    if (state == 'MENU' and
                            probs.get('MENU', 0.) >= self.state_confidence and
                            menu_score >= float(self.args.threshold) * .80 and
                            self._ppo_post_resume_menu_candidate is None):
                        self._ppo_post_resume_menu_candidate = (
                            time.monotonic(), time.perf_counter(), time.time())
                    decision = window.observe(
                        state=state, probs=probs, menu_score=menu_score,
                        confidence=self.state_confidence,
                        menu_threshold=self.args.threshold)
                    resumed_at = getattr(self, '_ppo_resume_trace_start_perf', None)
                    since_resume_ms = ((time.perf_counter() - resumed_at) * 1000.0
                                       if resumed_at is not None else None)
                    if decision == 'RETRY':
                        print(f'[PPO RESYNC TRANSIENT] skipped={window.skipped} '
                              f'capture={attempt+1}/{MAX_RESYNC_CAPTURES} '
                              f'discarded_valid={window.discarded_before_last_retry} '
                              f'state={state} menu_score={menu_score:.3f} '
                              f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f} '
                              f'since_resume_ms={since_resume_ms}', flush=True)
                        # _pack has already added any earlier valid frames; they
                        # cannot be used across an uncertain intervening image.
                        self.native_history.clear()
                        self.frame_intervals.clear()
                        self.previous_observation_at = None
                        observation = None
                        accepted_traces.clear()  # rejected partial stacks must not appear in the audit
                        continue
                    if decision == 'ABORT':
                        # Exact unsafe frame, for manual inspection only.
                        if self._v2_last_bgra is not None:
                            try:
                                import cv2
                                image_path = (self.output_dir /
                                    f'ppo_resume_failure_ep{self.episode_number:04d}_frame{attempt+1}.jpg')
                                wrote = cv2.imwrite(str(image_path),
                                    cv2.cvtColor(self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
                                if wrote:
                                    print(f'[PPO RESYNC TRACE] failure_screenshot={image_path}', flush=True)
                            except Exception as trace_exc:
                                print(f'[PPO RESYNC TRACE] diagnostic screenshot unavailable: '
                                      f'{trace_exc}', flush=True)
                        return None, (f'{window.last_reason}; '
                                      f'capture={attempt+1}/{MAX_RESYNC_CAPTURES} '
                                      f'since_resume_ms={since_resume_ms}')
                    # Only confirmed GAMEPLAY frames are packed or counted in
                    # the existing four-frame audit. Never infer from motion.
                    if since_resume_ms is not None:
                        accepted_traces.append(
                            f'[PPO RESYNC TRACE] frame={window.accepted}/4 '
                            f'since_resume_ms={since_resume_ms:.1f} '
                            f'state={state} menu_score={menu_score:.3f} '
                            f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f} '
                            f'ml_menu={probs.get("MENU", 0.):.3f}')
                    observation, _fresh, _age = self._pack(
                        legacy, start_detection=(window.accepted == 1))
                    if decision == 'DONE':
                        # Only the four COMPLETE consecutive valid frames get
                        # audit TRACE events. Rejected partial stacks do not.
                        # The first PPO action can only use frames positively
                        # verified against BOTH Pause signals and the ML guard.
                        self._ppo_pause_visual_reference = None
                        for line in accepted_traces:
                            print(line, flush=True)
                        return observation, None
                return None, (f'No four consecutive GAMEPLAY frames within '
                              f'{MAX_RESYNC_CAPTURES} captures; '
                              f'transient_skips={window.skipped}; no keys sent')

            obs, reason = self._call(capture_fresh_stack(), timeout=15)
            return obs, reason, self.last_capture_end

        def confirm_post_optimizer_menu(self):
            """Passively verify a death after PPO updates using four recent MENU frames.

            Allow up to eight fresh captures to outlast a brief early animation dip.
            NEVER lower any guard threshold or restart based on ML alone.
            No keys are sent here; a separate inherited safe reset handles R.
            """
            import asyncio
            from rs_v612_gameplay_guard import screen_gate
            from tools.ppo_menu_guard_alignment_v14 import (
                accept_menu_after_transient, MAX_CONFIRM_CAPTURES)

            async def confirm():
                if self._guard_hard_stopped:
                    return False, 'gameplay guard already hard-stopped'
                readings = []
                for i in range(MAX_CONFIRM_CAPTURES):
                    if i:
                        await asyncio.sleep(.075)
                    # Focus can change while waiting. Do not authorize a reset
                    # using only the initial focus state.
                    try:
                        focused = bool(await self.page.evaluate(
                            'document.hasFocus() && document.visibilityState === "visible"'))
                    except Exception as exc:
                        return False, f'cannot verify browser focus: {exc}'
                    if not focused:
                        return False, 'browser not focused/visible'
                    if self._guard_hard_stopped:
                        return False, 'gameplay guard hard-stopped during confirmation'
                    _legacy, menu_score, _unused = await self._capture()
                    state, probs = self._read_screen_state()
                    reading = screen_gate(
                        state, probs, menu_score,
                        confidence=self.state_confidence,
                        menu_threshold=float(self.args.threshold),
                        motion_threshold=self.startup_motion_threshold)
                    readings.append(reading)
                    # Requires 4 most recent independent captures to corroborate
                    # MENU. A weak or gameplay capture breaks the streak.
                    if accept_menu_after_transient(readings, focused=focused):
                        confirmed = True
                        break
                else:
                    confirmed = False
                evidence = {
                    'checks': [
                        {'state': r.state,
                         'menu_probability': round(r.menu_probability, 4),
                         'menu_score': round(r.menu_score, 5),
                         'corroborated': bool(r.ml_template_corroborated)}
                        for r in readings],
                    'required': '4 consecutive confident ML MENU + >=80% template corroboration',
                    'max_checks': MAX_CONFIRM_CAPTURES,
                }
                return confirmed, evidence

            return self._call(confirm(), timeout=15)

        def pause_for_ppo_optimizer(self):
            """Freeze the *local* Practice game before PPO.train(), without keys.

            UI must report GAMEPLAY; verify the site pause overlay and that the
            captured track remains static. Never send Escape (fullscreen risk).
            The old cached PPO observation stays untouched until rollout-start
            refresh, after the game resumes.
            """
            import asyncio
            import time as _time
            import numpy as _np

            async def pause():
                if self._guard_hard_stopped or getattr(self, '_ppo_pause_pending', None):
                    raise GuardUncertainError('Guard stopped or pause already pending')
                pending_menu_suspicion = self.suspected_since is not None
                page = self.page
                if not bool(await page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"')):
                    raise GuardUncertainError('Browser not focused/visible before pause')
                overlay = page.locator('.game-pause-panel[aria-label="Game paused"]')
                pause_button = page.locator('button[aria-label="Pause game"]')
                if await overlay.is_visible():
                    raise GuardUncertainError('Game unexpectedly already paused')
                _legacy, menu_score, _ = await self._capture()
                state, probs = self._read_screen_state()
                # A1: an uncertain frame or a pending menu suspicion at an
                # optimizer boundary must NOT cause a blind Pause click. The
                # bounded recheck below requires two fresh PLAY frames + a
                # unique visible Pause button, or four corroborated MENU frames.
                needs_boundary_recheck = (
                    pending_menu_suspicion or state != 'GAMEPLAY' or
                    menu_score >= self.args.threshold * .85)
                if needs_boundary_recheck:
                    print('[PPO BOUNDARY RECHECK] Initial frame/suspicion '
                          f'state={state} menu_score={menu_score:.3f} '
                          f'pending_suspicion={pending_menu_suspicion}; '
                          'no keys sent.', flush=True)
                # The Pause control can vanish *before* click (v1.7.1), or
                # detach *while Playwright waits for a stable click target*
                # (v1.7.4). In both cases, use fresh classifier/template
                # captures before sending another click. A click timeout alone
                # is NOT evidence that the click failed: check overlay first.
                async def recheck_pause_control(reason, *, after_click=False):
                    nonlocal menu_score, probs
                    from tools.ppo_pause_control_recheck_v171 import (
                        PauseControlRecheck, MAX_RECHECK_CAPTURES)
                    window = PauseControlRecheck(
                        confidence=self.state_confidence,
                        threshold=self.args.threshold,
                        motion_threshold=self.startup_motion_threshold)
                    observation_monos = []
                    for recheck_index in range(MAX_RECHECK_CAPTURES):
                        if recheck_index:
                            await asyncio.sleep(.075)
                        focused = bool(await page.evaluate(
                            'document.hasFocus() && document.visibilityState === "visible"'))
                        if not focused:
                            raise GuardUncertainError(
                                f'Browser focus lost during {reason}; NO inputs sent')
                        if await overlay.is_visible():
                            if after_click:
                                print('[PPO PAUSE CLICK] Pause overlay appeared '
                                      f'during {reason}; verify frozen track, '
                                      'do not click again.', flush=True)
                                return 'OVERLAY'
                            raise GuardUncertainError(
                                'Unexpected Pause overlay during missing-control recheck')
                        count = await pause_button.count()
                        if count > 1:
                            raise GuardUncertainError(
                                'Ambiguous multiple site Pause game buttons; NO click')
                        visible = count == 1 and await pause_button.is_visible()
                        _legacy, score, _unused = await self._capture()
                        new_state, new_probs = self._read_screen_state()
                        captured_mono = _time.monotonic()
                        observation_monos.append(captured_mono)
                        decision = window.observe(
                            state=new_state, probabilities=new_probs,
                            menu_score=score, observed_mono=captured_mono,
                            focused=focused, button_count=count,
                            button_visible=visible)
                        if decision == 'MENU':
                            # Anchor to the first of the FINAL consecutive
                            # four confirmed menu observations, not a possible
                            # earlier unrelated MENU frame before a PLAY gap.
                            first_menu = observation_monos[-4]
                            if first_menu is None:
                                raise GuardUncertainError(
                                    'Corroborated MENU missing observation-time anchor')
                            self._ppo_boundary_menu_duration_at = max(
                                float(self.survival_rewards.last_duration),
                                first_menu - float(self.episode_start))
                            print('[PPO PAUSE CONTROL] ' + reason + '; MENU '
                                  f'corroborated after {len(window.readings)} frames; '
                                  'no movement, no restart.', flush=True)
                            # The unchanged rollout callback independently
                            # re-confirms MENU, then records the terminal
                            # penalty via the normal no-key input guard.
                            raise GuardUncertainError(
                                f'Not verified GAMEPLAY at rollout end: MENU, '
                                f'menu={score:.3f}')
                        if decision == 'READY':
                            # The pre-pause trace must describe the latest
                            # verified capture, not the original UNCERTAIN one.
                            menu_score, probs = score, new_probs
                            print('[PPO PAUSE CONTROL] ' + reason + '; Pause '
                                  'button visible with two GAMEPLAY frames '
                                  f'after {len(window.readings)} checks.', flush=True)
                            return 'READY'
                        if decision == 'ABORT':
                            break
                    import cv2
                    diagnostic = (self.output_dir /
                        f'ppo_pause_control_ep{self.episode_number:04d}.jpg')
                    if self._v2_last_bgra is not None:
                        cv2.imwrite(str(diagnostic), cv2.cvtColor(
                            self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
                    raise GuardUncertainError(
                        f'Pause control unresolved after {reason}; '
                        f'{window.diagnostic()}; diagnostic={diagnostic}; '
                        'NO inputs sent')

                if needs_boundary_recheck:
                    # Distinguish a transient camera pan from MENU without
                    # lowering either classifier or independent template rules.
                    # If a stale menu suspicion clears, the same two positive
                    # PLAY frames justify clearing it as in the per-action guard.
                    await recheck_pause_control('Rollout-boundary screen recheck')
                    if pending_menu_suspicion:
                        self.suspected_since = None
                        self.menu_scores.clear()
                        self.clear_streak = 0
                if await pause_button.count() != 1 or not await pause_button.is_visible():
                    await recheck_pause_control('Missing Pause button')
                print('[PPO PAUSE TRACE] pre_pause_classifier=GAMEPLAY '
                      f'menu_score={menu_score:.3f} '
                      f'ml_gameplay={probs.get("GAMEPLAY", 0.):.3f} '
                      f'boundary_rechecked={needs_boundary_recheck}', flush=True)

                if getattr(self, 'settled_learning', False):
                    # Preserve the last confident gameplay endpoint; paused
                    # overlay captures must never become policy observations.
                    self._learning_boundary_observation = self._pack(None, start_detection=False)[0]
                    # Conservative: exclude the UI transaction from reward
                    # time, including uncertain pause entry/verification.
                    self._learning_pause_started = time.perf_counter()

                from playwright.async_api import TimeoutError as PlaywrightTimeoutError
                pause_click_attempts = 1
                try:
                    await pause_button.click(timeout=1800)
                except PlaywrightTimeoutError as exc:
                    # DOM detachment or a timeout is not proof of a failed click.
                    print('[PPO PAUSE CLICK] Initial click timed out; '
                          'checking overlay and fresh screen evidence.', flush=True)
                    result = await recheck_pause_control(
                        'Pause click timeout', after_click=True)
                    if result == 'READY':
                        if await overlay.is_visible():
                            print('[PPO PAUSE CLICK] Overlay appeared before '
                                  'retry; no second click.', flush=True)
                        else:
                            pause_click_attempts += 1
                            try:
                                await pause_button.click(timeout=1200)
                            except PlaywrightTimeoutError as second_exc:
                                result = await recheck_pause_control(
                                    'Second Pause click timeout', after_click=True)
                                if result == 'READY':
                                    raise GuardUncertainError(
                                        'Pause control timed out twice despite '
                                        'verified gameplay; NO third click') from second_exc
                                # OVERLAY means click may have worked; still
                                # require independently verified frozen track.

                async def recover_disappeared_overlay(reason):
                    """At most one bounded recovery; no keys, no blind click.

                    MENU uses the existing deferred-terminal path; PLAY must
                    be established by two fresh images and unique Pause UI.
                    A successful second click is never assumed to freeze Unity.
                    """
                    nonlocal pause_click_attempts
                    print('[PPO PAUSE SETTLE] Overlay missing: ' + reason +
                          '; checking fresh UI/screen evidence; NO keys.', flush=True)
                    decision = await recheck_pause_control(
                        'Pause overlay vanished: ' + reason, after_click=True)
                    if decision == 'READY':
                        if pause_click_attempts >= 2:
                            raise GuardUncertainError(
                                'Pause overlay vanished after two Pause click attempts; '
                                'NO third click')
                        # It may have reappeared since the last PLAY frame.
                        if not await overlay.is_visible():
                            if (await pause_button.count() != 1 or
                                    not await pause_button.is_visible()):
                                raise GuardUncertainError(
                                    'Pause control changed before overlay recovery; NO click')
                            pause_click_attempts += 1
                            print('[PPO PAUSE SETTLE] Two fresh GAMEPLAY frames '
                                  'and unique Pause button; one verified click retry.',
                                  flush=True)
                            try:
                                await pause_button.click(timeout=1200)
                            except PlaywrightTimeoutError as exc:
                                decision = await recheck_pause_control(
                                    'Overlay recovery click timeout', after_click=True)
                                if decision != 'OVERLAY':
                                    raise GuardUncertainError(
                                        'Pause recovery click timed out; overlay not '
                                        'verified; NO third click') from exc
                    # Either the overlay returned by itself or a single
                    # verified re-click was made. Never count any prior frozen
                    # frames; caller resets the FreezeWindow.
                    try:
                        await overlay.wait_for(state='visible', timeout=1200)
                    except PlaywrightTimeoutError as exc:
                        raise GuardUncertainError(
                            'Pause overlay not visible after bounded recheck; '
                            'NO additional click') from exc

                try:
                    await overlay.wait_for(state='visible', timeout=2500)
                except PlaywrightTimeoutError:
                    # A completed click can race against a React transition.
                    await recover_disappeared_overlay('initial overlay wait')

                # Site overlay visibility is NOT proof Unity has stopped.
                # Require three frozen track pairs, even after any recovery.
                # If the overlay vanishes during settling, a bounded recheck
                # may confirm MENU or PLAY; ambiguous observations still stop.
                from tools.ppo_pause_freeze_window import FreezeWindow
                window = FreezeWindow(required_pairs=3, max_delta=.25)
                verified = None
                overlay_recoveries = 0
                attempt = 0
                while attempt < 24:
                    if not await overlay.is_visible():
                        if overlay_recoveries:
                            raise GuardUncertainError(
                                'Pause overlay vanished again during freeze settling; '
                                'NO more retries')
                        overlay_recoveries += 1
                        await recover_disappeared_overlay('freeze settling')
                        window = FreezeWindow(required_pairs=3, max_delta=.25)
                        verified = None
                        continue
                    await asyncio.sleep(.09)
                    attempt += 1
                    if not await overlay.is_visible():
                        if overlay_recoveries:
                            raise GuardUncertainError(
                                'Pause overlay vanished again during freeze settling; '
                                'NO more retries')
                        overlay_recoveries += 1
                        await recover_disappeared_overlay('settling sleep')
                        window = FreezeWindow(required_pairs=3, max_delta=.25)
                        verified = None
                        continue
                    _legacy, score, _unused = await self._capture()
                    # GAMEPLAY ML votes can be unreliable behind the pause
                    # overlay. Never relax the independent MENU exclusion or
                    # static-track threshold.
                    if score >= self.args.threshold * .85:
                        raise GuardUncertainError(
                            f'MENU template detected during pause: {score:.3f}')
                    if self._v2_last_bgra is None:
                        raise GuardUncertainError('Missing paused Unity capture')
                    track = self._motion_view(self._v2_last_bgra, self.crop)
                    verified = window.observe(
                        track, (_time.monotonic(), _time.perf_counter(), _time.time()))
                    if verified is not None:
                        # A frozen scene without a visible Pause overlay does
                        # not authorize starting the PPO optimizer.
                        if not await overlay.is_visible():
                            if overlay_recoveries:
                                raise GuardUncertainError(
                                    'Pause overlay vanished again after freeze '
                                    'verification; NO more retries')
                            overlay_recoveries += 1
                            await recover_disappeared_overlay('post-freeze check')
                            window = FreezeWindow(required_pairs=3, max_delta=.25)
                            verified = None
                            continue
                        break
                if verified is None:
                    raise GuardUncertainError(
                        'Unity did not demonstrate three consecutive frozen '
                        f'track intervals within {attempt} checks; '
                        f'changes={window.changes}')
                start_mono, start_perf, start_wall, mean_change = verified
                print(f'[PPO PAUSE] Freeze verified after {attempt} captures; '
                      f'track_deltas={window.changes}', flush=True)
                return start_mono, start_perf, start_wall, mean_change

            # Never silently proceed if the pause fails. Trainer catches the
            # exception, saves partial model, and closes the Practice browser.
            self._ppo_pause_visual_reference = None
            paused_mono, paused_perf, paused_wall, delta = self._call(pause(), timeout=15)
            # A DOM-hidden but visually stale Pause panel must not be mistaken
            # for GAMEPLAY after optimizer. Only use an anchor captured while
            # the panel AND three frozen-track pairs were verified.
            from tools.ppo_pause_visual_v179 import pause_panel_signature
            self._ppo_pause_visual_reference = pause_panel_signature(self._v2_last_bgra)
            self._ppo_pause_pending = (paused_mono, paused_perf, paused_wall,
                                       paused_mono - float(self.episode_start))
            return delta

        def resume_after_ppo_optimizer(self, monitor):
            """Resume, correct clocks, then let existing v1.4 resync the image."""
            import time as _time
            from tools.ppo_optimizer_pause_clock import compensate_pause_clocks

            token = getattr(self, '_ppo_pause_pending', None)
            if not (isinstance(token, tuple) and len(token) == 4):
                raise GuardUncertainError('No validated pause token at rollout start')

            # Validate the exact SB3 Monitor BEFORE unpausing.
            if getattr(monitor, 'env', None) is not self:
                raise GuardUncertainError('Expected exact Monitor(raw) wrapper')

            async def resume():
                page = self.page
                if not bool(await page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"')):
                    raise GuardUncertainError('Browser not focused/visible before resume')
                overlay = page.locator('.game-pause-panel[aria-label="Game paused"]')
                button = overlay.get_by_role('button', name='Resume', exact=True)
                if not await overlay.is_visible() or not await button.is_visible():
                    raise GuardUncertainError('Verified pause overlay/Resume button missing')
                from playwright.async_api import TimeoutError as PlaywrightTimeoutError
                try:
                    await button.click(timeout=1800)
                except PlaywrightTimeoutError:
                    # The first click might have *worked*. Always inspect the
                    # overlay before considering another click; hidden overlay
                    # still requires the strict four-frame GAMEPLAY resync.
                    print('[PPO RESUME CLICK] Timed out; checking overlay '
                          'before any retry.', flush=True)
                    if await overlay.is_visible():
                        if not bool(await page.evaluate(
                                'document.hasFocus() && document.visibilityState === "visible"')):
                            raise GuardUncertainError(
                                'Focus lost after timed-out Resume click; NO retry')
                        if await button.count() != 1 or not await button.is_visible():
                            raise GuardUncertainError(
                                'Resume control not unique and visible after timeout; NO retry')
                        # Before a second click, verify that Unity is STILL
                        # frozen behind the overlay. An overlay by itself may
                        # be stale; do not assume the game is paused from DOM.
                        import asyncio
                        from tools.ppo_pause_freeze_window import FreezeWindow
                        frozen = FreezeWindow(required_pairs=3, max_delta=.25)
                        for index in range(8):
                            await asyncio.sleep(.075)
                            if not await overlay.is_visible():
                                break
                            if not bool(await page.evaluate(
                                    'document.hasFocus() && document.visibilityState === "visible"')):
                                raise GuardUncertainError('Focus lost in Resume retry verification')
                            _legacy, score, _ = await self._capture()
                            if score >= self.args.threshold * .85 or self._v2_last_bgra is None:
                                raise GuardUncertainError(
                                    'Unsafe screen behind Resume overlay; NO retry')
                            track = self._motion_view(self._v2_last_bgra, self.crop)
                            found = frozen.observe(
                                track, (_time.monotonic(), _time.perf_counter(), _time.time()))
                            if found is not None:
                                break
                        if await overlay.is_visible():
                            if found is None:
                                raise GuardUncertainError(
                                    'Resume retry refused: frozen Unity not independently verified')
                            if await button.count() != 1 or not await button.is_visible():
                                raise GuardUncertainError(
                                    'Resume button changed after freeze verification; NO retry')
                            print('[PPO RESUME CLICK] Frozen track reverified; '
                                  'one guarded Resume retry.', flush=True)
                            try:
                                await button.click(timeout=1200)
                            except PlaywrightTimeoutError:
                                print('[PPO RESUME CLICK] Second click timed out; '
                                      'checking overlay; NO third click.', flush=True)
                                if await overlay.is_visible():
                                    raise GuardUncertainError(
                                        'Resume click timed out twice with overlay visible; NO third click')
                try:
                    await overlay.wait_for(state='hidden', timeout=2500)
                except PlaywrightTimeoutError:
                    if await overlay.is_visible():
                        raise GuardUncertainError('Resume overlay did not disappear; NO PPO action')
                if await overlay.is_visible():
                    raise GuardUncertainError('Resume overlay still visible; NO PPO action')
                if not bool(await page.evaluate(
                        'document.hasFocus() && document.visibilityState === "visible"')):
                    raise GuardUncertainError('Browser focus lost after Resume; NO PPO action')
                return _time.monotonic(), _time.perf_counter(), _time.time()

            finished_mono, finished_perf, finished_wall = self._call(resume(), timeout=12)
            # The timestamp is recorded immediately after the confirmed
            # overlay disappearance. This measures how long the game was
            # running before each of the existing four resync captures.
            self._ppo_resume_trace_start_perf = finished_perf
            dm = finished_mono - token[0]
            dp = finished_perf - token[1]
            # Adjust wall-time origins, but never fabricate a step/reward or
            # relax the screen-state classifier. If anything goes wrong,
            # fail closed BEFORE the next PPO action.
            report = compensate_pause_clocks(
                self, dm, dp, monitor=monitor,
                elapsed_wall=finished_wall - token[2])
            corrected_duration = _time.monotonic() - float(self.episode_start)
            duration_drift = corrected_duration - token[3]
            if not (-.10 <= duration_drift <= .20):
                raise GuardUncertainError(
                    f'Pause-aware episode timer drifted {duration_drift:.3f}s')
            report['active_episode_clock_drift_s'] = round(duration_drift, 3)
            self._ppo_pause_pending = None
            if getattr(self, 'settled_learning', False) and self._learning_pause_started is not None:
                self._learning_excluded_pause += max(0.,finished_perf-self._learning_pause_started)
                self._learning_pause_started = None
            return report

        @staticmethod
        def _meter_mask(bgra):
            """Guard-only distance HUD mask; supports visible Chrome or fullscreen."""
            from tools.ppo_guard_hud_v175 import meter_mask
            return meter_mask(bgra)

        def _guard_diagnostic(self, readings, *, reason, name=None):
            """Write screen + full gating evidence for any fail-closed stop."""
            import cv2
            target = self.output_dir / (name or f'v612_guard_episode_{self.episode_number:04d}.jpg')
            if self._v2_last_bgra is not None:
                cv2.imwrite(str(target), cv2.cvtColor(self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
            write_json(target.with_suffix('.json'), {
                'episode': int(self.episode_number),
                'reason': reason,
                'readings': [vars(x) for x in readings],
                'note': 'No movement command sent by this failed preflight.'
            })
            return target

        async def _step_browser(self, action):
            """Require gameplay on the most recent capture before sending keys.

            The inherited controller normally relies on menu text templates in
            its gameplay loop. We also use ML state every action. Ambiguity
            causes additional independent captures, never an automatic tap.
            """
            import asyncio
            if self._guard_hard_stopped:
                raise GuardUncertainError('[V612 GUARD] Inputs latched OFF after uncertain UI.')

            # These two views are derived from the exact raw full-screen BGRA
            # captures, not the PPO/YOLO game crop or a changed browser crop.
            # Never treat moving scenery alone as gameplay: animated death
            # screens can also move. The distance HUD must visibly advance.
            previous_track = None
            baseline_hud = None
            prepared = getattr(self, '_prepared_gameplay_reading', None)
            used_prepared = None
            self._prepared_gameplay_reading = None  # one decision only
            if (not getattr(self, 'reuse_observation_guard', False) or
                    self.suspected_since is not None or
                    getattr(self, '_ppo_boundary_menu_pending', False) or
                    prepared is None or not prepared.valid_for(
                        self._v2_last_bgra, self.last_capture_end, time.perf_counter())):
                prepared = None
                if getattr(self, 'reuse_observation_guard', False):
                    self.prepared_guard_fallbacks += 1

            async def sample():
                nonlocal previous_track, baseline_hud, prepared, used_prepared
                import cv2
                if prepared is not None:
                    frame = (self._v2_last_chw, prepared.menu_score, None)
                    state, probs = prepared.state, prepared.probabilities
                    used_prepared = prepared
                    prepared = None
                    self.prepared_guard_hits += 1
                else:
                    frame = await self._capture()
                    state, probs = self._read_screen_state()
                track_motion = track_texture = 0.
                hud_pixels = hud_changed_pixels = 0
                shot = self._v2_last_bgra
                if shot is not None:
                    track = self._motion_view(shot, self.crop)
                    hud = self._meter_mask(shot)
                    track_texture = float(track.std())
                    hud_pixels = int(cv2.countNonZero(hud))
                    if previous_track is not None:
                        track_motion = float(np.mean(cv2.absdiff(track, previous_track)))
                        # Compare to the first HUD sample of this recovery
                        # window, not just the last ~45ms sample: distance
                        # digits need not tick on every captured frame.
                        hud_changed_pixels = int(cv2.countNonZero(cv2.absdiff(hud, baseline_hud)))
                    if baseline_hud is None:
                        baseline_hud = hud
                    previous_track = track
                reading = screen_gate(state, probs, frame[1],
                                      confidence=self.state_confidence,
                                      menu_threshold=self.args.threshold,
                                      track_motion=track_motion,
                                      track_texture=track_texture,
                                      hud_pixels=hud_pixels,
                                      hud_changed_pixels=hud_changed_pixels,
                                      motion_threshold=self.startup_motion_threshold)
                return frame, reading

            # A verified rollout-boundary MENU survives the optimizer unchanged.
            # The NEXT normal env.step must deliver its real terminal reward;
            # never execute the policy-sampled movement on a MENU screen.
            boundary_menu_handoff = bool(getattr(self, '_ppo_boundary_menu_pending', False))
            frame, first = await sample()
            readings = [first]
            if first.ml_menu and not first.menu_evidence:
                self.guard_ml_menu_disagreements += 1
            needs_recovery = (first.verdict != 'PLAY' or
                              self.suspected_since is not None or boundary_menu_handoff)
            if needs_recovery:
                self.guard_rechecks += 1
                decision = 'CONTINUE'
                while decision == 'CONTINUE':
                    await asyncio.sleep(.045)
                    frame, reading = await sample()
                    readings.append(reading)
                    if reading.ml_menu and not reading.menu_evidence:
                        self.guard_ml_menu_disagreements += 1
                    decision = sequence_decision(readings, max_frames=6)
                if decision == 'TERMINAL':
                    self.guard_confirmed_menu += 1
                    self._guard_forced_terminal = True
                    if boundary_menu_handoff:
                        self._ppo_boundary_menu_pending = False
                        self._ppo_menu_handoff_terminal_observed = True
                    # Save the first few positive menu transitions as evidence.
                    if self.guard_confirmed_menu <= 6:
                        self._guard_diagnostic(readings, reason='menu confirmed without movement')
                    print(f'[V612 GUARD] MENU confirmed from {len(readings)} captures; '
                          f'NO key sent (ep={self.episode_number}; '
                          f'menu_scores={[round(r.menu_score, 2) for r in readings]}).',
                          flush=True)
                    return frame
                if decision == 'ABORT':
                    self.guard_aborts += 1
                    self._guard_hard_stopped = True
                    image = self._guard_diagnostic(readings, reason='ambiguous UI after six captures')
                    raise GuardUncertainError(
                        f'[V612 GUARD] SAFETY STOP after {len(readings)} '
                        f'unresolved captures; NO key sent. '
                        f'States={[r.state for r in readings]}, '
                        f'menu_scores={[round(r.menu_score, 2) for r in readings]}. '
                        f'Diagnostics: {image}')
                # A pending boundary death is NEVER allowed to resume gameplay
                # and execute the policy's sampled action. Require the ordinary
                # guard's TERMINAL verdict, otherwise fail closed before keys.
                if boundary_menu_handoff:
                    self.guard_aborts += 1
                    self._guard_hard_stopped = True
                    image = self._guard_diagnostic(
                        readings, reason='boundary MENU handoff became nonterminal')
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] MENU was not terminal on next step; '
                        f'NO key sent. Diagnostic: {image}')
                # Two affirmative ML gameplay frames, OR two consecutive
                # moving-track + advancing-distance-HUD pairs, permit action.
                if (readings[-1].visual_gameplay and readings[-2].visual_gameplay and
                        any(r.ml_menu and not r.menu_evidence for r in readings)):
                    self.guard_visual_recoveries += 1
                    if self.guard_visual_recoveries <= 6:
                        self._guard_diagnostic(
                            readings, reason='moving gameplay despite ML menu prediction',
                            name=(f'v612_disagreement_episode_{self.episode_number:04d}_'
                                  f'{self.guard_visual_recoveries:02d}.jpg'))
                # The inherited _step_browser suppresses keys when it thinks
                # menu detection is pending; clear that stale suspicion only
                # after *two* verified GAMEPLAY captures.
                if self.suspected_since is not None:
                    self.suspected_since = None
                    self.menu_scores.clear()
                    self.clear_streak = 0
                print(f'[V612 GUARD] Resumed after {len(readings)} captures: '
                      'two positive gameplay confirmations.', flush=True)

            if used_prepared is not None and not used_prepared.valid_for(
                    self._v2_last_bgra, self.last_capture_end, time.perf_counter()):
                # Invalidate before dispatch as well as before classification.
                # A scheduler stall must fall back to the original fresh guard.
                self.prepared_guard_late_fallbacks += 1
                return await self._step_browser(action)
            if (getattr(self, '_background_capture_mode', 'mss') == 'dxgi'
                    and not self._background_action_frame_current()):
                return await self._background_retry_action_frame(action)
            self.guard_allowed += 1
            if getattr(self, '_background_capture_mode', 'mss') == 'dxgi':
                # The post-action policy observation must have a new native
                # presentation. Static UI inspection remains explicitly separate.
                self.grabber.require_new = True
            self.action_context.note_action(int(action), time.perf_counter())
            if getattr(self, 'settled_learning', False):
                from train_rs_rl_early_death import ACTIONS
                self._learning_dispatched_key = ACTIONS[int(action)]
            # Preserve the inherited tap, pacing and post-key screenshot.
            return await super()._step_browser(int(action))

        def step(self, action):
            passive_kind = getattr(self, '_learning_passive_kind', None)
            if passive_kind is not None:
                self._learning_passive_kind = None
                return self._settled_passive_step(passive_kind)
            self._learning_dispatched_key = None
            self._guard_forced_terminal = False
            timings = getattr(self,'step_timings',[])
            timing_before = timings[-1] if timings else None
            obs, base_reward, died, truncated, info = super().step(int(action))
            if self._guard_forced_terminal:
                # Separate successful GAMEPLAY->MENU detector from a template
                # detector that could have missed the same menu transition.
                died, truncated = True, False
                info['v612_guard_confirmed_menu'] = True
            self._ppo_time_limit_pending = bool(truncated and not died)
            info['termination_reason'] = 'death' if died else ('time_limit' if truncated else None)
            info['v612_guard_allowed'] = self.guard_allowed
            info['v612_guard_rechecks'] = self.guard_rechecks
            # Only real elapsed gameplay (observed by the local game loop) is
            # rewarded. No coins, class confidence, lane assumptions, or
            # specific action get a reward. Avoid repeated-action exploits.
            if getattr(self, '_ppo_menu_handoff_terminal_observed', False) and not getattr(self, 'settled_learning', False):
                # Game was already dead when this rollout boundary was checked.
                # Do not credit optimizer / idle menu / guard confirmation time.
                anchor = self._ppo_boundary_menu_duration_at
                if anchor is None or not np.isfinite(anchor) or anchor < 0:
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] Missing trustworthy pre-optimizer death time')
                info['duration'] = round(
                    max(self.survival_rewards.last_duration, float(anchor)), 2)
            reward, pieces = self.survival_rewards.step(
                self._learning_verified_duration if getattr(self, 'settled_learning', False) else info.get('duration', 0.), died)
            if getattr(self, 'settled_learning', False):
                timing_after = self.step_timings[-1] if self.step_timings else None
                info.update(reward_duration_s=self.survival_rewards.last_duration,
                    learning_action_suppressed=bool(self._guard_forced_terminal),
                    learning_dispatched_key=self._learning_dispatched_key,
                    learning_dispatch_at=(timing_after.get('press_timestamp')
                        if timing_after is not timing_before and timing_after else None))
            self.reward_parts.update(pieces)
            info['reward_survival'] = round(pieces['survival'], 4)
            info['reward_milestones'] = round(pieces['milestones'], 4)
            info['reward_death'] = round(pieces['death'], 4)
            info['reward_v61'] = round(reward, 4)
            info['reward_old_ignored'] = round(float(base_reward), 4)
            if getattr(self, '_ppo_menu_handoff_terminal_observed', False):
                if not died or truncated or pieces['death'] >= 0:
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] Deferred death did not receive terminal penalty')
                self._ppo_menu_handoff_terminal_observed = False
                self._ppo_boundary_menu_duration_at = None
                info['ppo_menu_handoff_terminal'] = True
                print('[PPO MENU HANDOFF] Terminal recorded by guarded env.step; '
                      f'death_reward={pieces["death"]:.4f}; NO movement key sent.', flush=True)
            return obs, float(reward), died, truncated, info

        def _settled_passive_step(self, kind):
            """Emit elapsed-time/death evidence with no sampled or sent action."""
            terminal = bool(self._ppo_boundary_menu_pending)
            if terminal:
                # Existing independent guard must still confirm terminal MENU.
                self._guard_forced_terminal = False
                self._call(self._step_browser(0), timeout=25)
                if not self._guard_forced_terminal:
                    raise GuardUncertainError('Passive MENU settlement was not terminal')
                obs = self._pack(None,start_detection=False)[0]
            else:
                obs = (getattr(self,'_learning_boundary_observation',None)
                       if kind=='boundary' else self._learning_current_observation)
                if obs is None:
                    raise GuardUncertainError('Missing passive transition observation')
            reward,pieces = self.survival_rewards.step(self._learning_verified_duration,terminal)
            self.reward_parts.update(pieces)
            self.training_steps += 1
            self._ppo_menu_handoff_terminal_observed = False
            self._ppo_boundary_menu_duration_at = None
            info = dict(duration=round(self._learning_verified_duration,2),
                reward_duration_s=self._learning_verified_duration,
                reward_survival=pieces['survival'],reward_milestones=pieces['milestones'],
                reward_death=pieces['death'],reward_v61=reward,
                learning_action_suppressed=True,learning_dispatched_key=None,
                termination_reason='death' if terminal else None,
                ppo_menu_handoff_terminal=terminal,learning_passive_kind=kind)
            return obs,float(reward),terminal,False,info

        def hazard_report(self):
            report = super().hazard_report()
            report['v5_segmentation']['shadow_only'] = False
            report['v5_segmentation']['fed_into_ppo'] = True
            report['v5_segmentation']['feature_version'] = FEATURE_VERSION
            report['v5_segmentation']['screen_column_nonzero_observations'] = self.column_nonzero_observations
            report['v5_segmentation']['assumed_lane_unverified'] = self.action_context.assume_center
            report['v5_segmentation']['hazard_obs_total'] = self.hazard_observation_count
            report['v5_segmentation']['hazard_obs_stale_or_missing'] = self.stale_observation_count
            report['v5_segmentation']['hazard_obs_nonzero'] = self.hazard_nonzero_observations
            report['v612_live_input_guard'] = {
                'preflight_gameplay_allowed': self.guard_allowed,
                'preflight_rechecks_on_transient': self.guard_rechecks,
                'preflight_confirmed_menu': self.guard_confirmed_menu,
                'uncertain_state_abort': self.guard_aborts,
                'ml_menu_without_template': self.guard_ml_menu_disagreements,
                'visual_gameplay_recoveries': self.guard_visual_recoveries,
                'guard_hard_stopped': self._guard_hard_stopped,
                'preflight_checks_every_decision': True,
                'prepared_observation_guard_enabled': bool(getattr(self, 'reuse_observation_guard', False)),
                'prepared_observation_guard_hits': getattr(self, 'prepared_guard_hits', 0),
                'prepared_observation_guard_fallbacks': getattr(self, 'prepared_guard_fallbacks', 0),
                'prepared_observation_guard_late_fallbacks': getattr(self, 'prepared_guard_late_fallbacks', 0),
                'prepared_observation_max_age_ms': 20.,
                'warning': 'One screenshot cannot prevent a game ending between capture and keypress.'
            }
            return report

    return PPOV61Env


def make_model(env, a):
    import torch
    from stable_baselines3 import PPO
    from train_rs_v3_ppo import RGBHazardExtractor

    # The extractor reads the actual AUX size from the new observation_space,
    # and CNN takes the exact original four RGB screenshots (12 channels).
    return PPO(
        'MultiInputPolicy', env, device='cuda' if torch.cuda.is_available() else 'cpu',
        n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs,
        learning_rate=a.learning_rate, gamma=a.gamma, gae_lambda=.95,
        ent_coef=a.ent_coef, clip_range=.2, verbose=1, seed=a.seed,
        policy_kwargs={
            'features_extractor_class': RGBHazardExtractor,
            'features_extractor_kwargs': {'features_dim': 192},
            'net_arch': {'pi': [64], 'vf': [64]},
        },
    )


def offline_check(a):
    """Real SB3 PPO gradient update using fake observations, no game/detector."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    import torch

    class FakeEnv(gym.Env):
        metadata = {'render_modes': []}
        def __init__(self):
            self.observation_space = spaces.Dict({
                'image': spaces.Box(0,255,(12,a.resolution[1],a.resolution[0]),dtype=np.uint8),
                'aux': spaces.Box(0.,1.,(AUX_DIM,),dtype=np.float32),
            })
            self.action_space = spaces.Discrete(5)
            self.rng = np.random.default_rng(42)
            self.t = 0
        def _obs(self):
            return {'image': self.rng.integers(0,256,self.observation_space['image'].shape,
                                              dtype=np.uint8),
                    'aux': self.rng.random(AUX_DIM).astype(np.float32)}
        def reset(self, *, seed=None, options=None):
            self.t = 0
            return self._obs(), {}
        def step(self, action):
            self.t += 1
            return self._obs(), float(int(action)==(self.t%5))*.01, self.t>=8, False, {}

    sample = {
        'episode':5, 'captured_at':10., 'detections':[
            {'class':'coin','conf':.88,'box':[.4,.2,.45,.4],
             'polygon':[[.4,.2],[.45,.2],[.45,.4],[.4,.4]]},
            {'class':'train_roof','conf':.72,'box':[.3,.55,.8,.85],
             'polygon':[[.3,.55],[.8,.55],[.8,.85],[.3,.85]]},
        ]}
    hz, age, fresh = encode_hazards(sample, now=10.07, episode_tag=5)
    assert fresh and age > 0 and hz.shape[0] > 400 and hz[-1] == 1
    assert hz[7] == 1  # first coin upper-band slot is present
    bad, _, stale = encode_hazards(sample, now=10.50, episode_tag=5, max_stale_ms=300)
    assert not stale and not bad.any()
    incompatible, _, wrong = encode_hazards(sample, now=10.07, episode_tag=6)
    assert not wrong and not incompatible.any()
    from rs_v6_ppo_features import encode_aux as encode_v6_aux, AUX_DIM as V6_AUX_DIM
    aux = encode_v6_aux(hz, deque([0,3,4],maxlen=4), 1.7, deque([.03,.04],maxlen=3), age)
    assert aux.shape == (V6_AUX_DIM,) and float(aux.min())>=0 and float(aux.max())<=1
    ctx = ActionContext()
    extra = encode_aux(hz, deque([0,3,4],maxlen=4), 1.7,
                       deque([.03,.04],maxlen=3), age, snapshot=sample,
                       episode_tag=5, now=10.07, action_context=ctx)
    assert extra.shape == (AUX_DIM,) and np.isfinite(extra).all()
    assert not extra[V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM].all() and extra[V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM].any()
    assert extra[-9:-5].sum() == 0.  # no fake player lane without calibration
    print(f'[CHECK] Nine-class V6.1 observations OK: AUX_DIM={AUX_DIM}; RGB=12x{a.resolution[1]}x{a.resolution[0]}', flush=True)

    old=(a.n_steps,a.batch_size,a.n_epochs,a.learning_rate,a.gamma,a.ent_coef)
    a.n_steps, a.batch_size, a.n_epochs=16,8,1
    env=FakeEnv()
    try:
        model=make_model(env,a)
        before={k: v.detach().clone() for k,v in model.policy.features_extractor.state_dict().items()}
        model.learn(total_timesteps=32, progress_bar=False)
        changed=any(not torch.equal(before[k],v) for k,v in model.policy.features_extractor.state_dict().items())
        if not changed:
            raise RuntimeError('No visual/aux encoder weights changed during PPO update')
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d)/'policy')
            restored=PPO.load(Path(d)/'policy.zip', device=model.device)
            obs,_=env.reset()
            a1,_=model.predict(obs, deterministic=True)
            a2,_=restored.predict(obs, deterministic=True)
            assert int(a1)==int(a2)
        print('[CHECK] PASS: 9 classes, fresh/stale gating, 4 RGB frames, PPO gradients and save/load.',flush=True)
    finally:
        env.close()
        (a.n_steps,a.batch_size,a.n_epochs,a.learning_rate,a.gamma,a.ent_coef)=old




def read_policy_probs(model, obs):
    """Return actual current categorical policy probabilities (not action counts)."""
    if obs is None:
        return None
    try:
        import torch
        with torch.no_grad():
            tensor,_ = model.policy.obs_to_tensor(obs)
            categorical = model.policy.get_distribution(tensor).distribution
            probs = categorical.probs.detach().cpu().numpy()[0]
            return {'probabilities':[round(float(x),5) for x in probs],
                    'max_action':ACTION_NAMES[int(np.argmax(probs))],
                    'max_probability':round(float(np.max(probs)),5),
                    'entropy':round(float(-(probs*np.log(np.maximum(probs,1e-9))).sum()),5)}
    except (TypeError,ValueError,AttributeError) as exc:
        print(f'[POLICY DIAGNOSTIC] unavailable: {exc}',flush=True)
        return None


def _rgb_latest(obs):
    # V6.1 image is 4 x (3,H,W) RGB concatenated in channel dimension.
    import cv2
    arr=np.asarray(obs['image'][-3:],dtype=np.uint8).transpose(1,2,0)
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)


def input_probe(a):
    """Record browser key events and before/after image pairs with safe restart.

    After each pattern, wait for CONFIRMED death or episode timeout before
    reset, because the V4 screen-state restart requires a verified MENU.
    """
    import cv2
    from train_rs_v2_verify import RUN_ROOT
    dest=RUN_ROOT/a.run_name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Probe folder already exists: {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    picture_dir=dest/'action_pairs'
    picture_dir.mkdir()
    write_json(dest/'v61_probe_config.json',dict(vars(a),schema=shape_description()))
    monitored=raw=None
    trials=[]
    patterns=[('LEFT',[1,0,0,0,0,0]),('RIGHT',[2,0,0,0,0,0]),
              ('JUMP',[3,0,0,0,0,0]),('ROLL',[4,0,0,0,0,0]),
              ('LEFT_TWICE',[1,0,1,0,0,0,0,0])]
    finished_previous=True
    try:
        monitored,raw=create_live_env(a,dest)
        frame_hook_count=0
        for delay in a.probe_delays:
            raw.probe_press_delay_ms=int(delay)
            for repetition in range(a.probe_trials):
                for name,actions in patterns:
                    if not finished_previous:
                        raise RuntimeError('Previous test did not reach menu/death; cannot safely reset')
                    obs,_=monitored.reset()
                    # Game/iframe transitions can drop page event handlers.
                    # Re-install only after the guarded reset confirms GAMEPLAY.
                    try:
                        frame_hook_count=raw._call(raw._install_demo_hooks(),timeout=10)
                        raw._call(raw._drain_demo_events(),timeout=10)
                    except Exception as exc:
                        frame_hook_count=0
                        print('[INPUT PROBE] Event hooks unavailable:',repr(exc),flush=True)
                    initial=_rgb_latest(obs)
                    key_events=[]
                    ran=[]
                    done=False
                    for action in actions:
                        obs,_,dead,trunc,info=monitored.step(action)
                        ran.append(int(action))
                        if frame_hook_count:
                            key_events+=raw._call(raw._drain_demo_events(),timeout=10)
                        done=bool(dead or trunc)
                        ended_by_death=bool(dead)
                        if done: break
                    final=_rgb_latest(obs)
                    joined=np.concatenate([initial,final],axis=1)
                    cv2.putText(joined,f'{name} {delay}ms trial {repetition+1}',
                                (5,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                    file=f'{len(trials)+1:03d}_{delay}ms_{name}_trial{repetition+1}.jpg'
                    cv2.imwrite(str(picture_dir/file),joined,[cv2.IMWRITE_JPEG_QUALITY,90])
                    arrow={'LEFT':'ArrowLeft','RIGHT':'ArrowRight',
                           'JUMP':'ArrowUp','ROLL':'ArrowDown','LEFT_TWICE':'ArrowLeft'}[name]
                    presses=sum(ev.get('code')==arrow and ev.get('down') is True for ev in key_events)
                    releases=sum(ev.get('code')==arrow and ev.get('down') is False for ev in key_events)
                    rec={'pattern':name,'tap_duration_ms':delay,'trial':repetition+1,
                         'requested_actions':actions,'completed_actions':ran,
                         'terminal_during_test':done,'browser_keydowns':presses,
                         'browser_keyups':releases,
                         'event_hooks_installed':frame_hook_count>0,
                         'picture':str(Path('action_pairs')/file),
                         'movement_visually_verified':None,
                         'caveat':'Browser keydown is not proof Unity registered a movement.'}
                    trials.append(rec)
                    print('[INPUT PROBE] '+json.dumps({k:rec[k] for k in
                          ('pattern','tap_duration_ms','terminal_during_test',
                           'browser_keydowns','browser_keyups','picture')}),flush=True)
                    # Do NOT reset while still running! V4 only safely restarts
                    # from an independently verified MENU, not active gameplay.
                    drained=0
                    while not done and drained<800:
                        obs,_,dead,trunc,_=monitored.step(0)
                        done=bool(dead or trunc)
                        ended_by_death=bool(dead)
                        drained+=1
                    finished_previous=done and ended_by_death
                    if not finished_previous:
                        raise RuntimeError('Probe did not observe confirmed death; refusing to reset from active gameplay/truncation')
    finally:
        write_json(dest/'v61_input_probe.json',{'trials':trials,
                   'instruction':'Inspect before/after images and gameplay. Key event counts do not prove Unity movement.'})
        if monitored is not None:monitored.close()
        elif raw is not None:raw.close()
    print(f'[INPUT PROBE] Wrote {len(trials)} action pairs to {picture_dir}',flush=True)


def build_callback(a, output, raw=None):
    from stable_baselines3.common.callbacks import BaseCallback
    from tools.ppo_rollout_gap_audit import RolloutGapTracker
    from tools.ppo_rollout_resync import replace_cached_observation
    from tools.ppo_optimizer_menu_recovery import candidate_is_menu, validated_new_vec_episode
    from tools.ppo_rollout_menu_handoff_v17 import boundary_menu_candidate

    class V61Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.began = time.perf_counter()
            self.last_log = self.began
            self.last_save = self.began
            self.episode_durations = []
            self.episode_rewards = []
            self.recent_episodes = deque(maxlen=25)
            self.action_total = Counter()
            self.action_recent = Counter()
            self.reward_components = Counter()
            self.updates = 0
            self.policy_snapshots = []
            self.log_path = output/'progress.jsonl'
            self.log_handle = None
            self.reward_step_audit_handle = None
            self.rollout_gap_tracker = RolloutGapTracker()
            self.rollout_gap_log_path = output / 'ppo_rollout_gaps.jsonl'
            self._checkpoint_due_at_boundary = False

        def _on_training_start(self):
            self.log_handle = self.log_path.open('a',encoding='utf-8')
            if a.audit_reward_timing:
                self.reward_step_audit_handle = (output/'ppo_reward_step_audit.jsonl').open('w', encoding='utf-8')
                print('[PPO REWARD STEP AUDIT] Recording actual env reward components (observation only).', flush=True)

        def _on_rollout_end(self):
            # Before PPO.train() starts. No environment interaction.
            try:
                self.rollout_gap_tracker.finish_rollout(time.perf_counter())
            except Exception as exc:
                print(f"[PPO GAP] rollout-end diagnostic failed: {exc}", flush=True)
            if a.pause_ppo_optimizer:
                try:
                    if raw is None:
                        raise RuntimeError('No raw environment for PPO pause')
                    delta = raw.pause_for_ppo_optimizer()
                    print(f'[PPO PAUSE] Verified frozen gameplay before optimizer; '
                          f'paused-track delta={delta:.3f}', flush=True)
                except Exception as exc:
                    # MENU can appear after the 128th environment step but
                    # BEFORE the pause click. The previous step has already
                    # entered PPO's rollout buffer, so a direct reset here
                    # would LOSE the real death penalty. Keep the MENU idle
                    # during the optimizer and hand it to the NEXT normal
                    # env.step, whose existing guard records a real terminal.
                    if (a.resync_rollouts and boundary_menu_candidate(exc) and
                            getattr(raw, '_ppo_pause_pending', None) is None):
                        try:
                            confirmed, evidence = raw.confirm_post_optimizer_menu()
                        except Exception as verify_exc:
                            raise GuardUncertainError(
                                f'[PPO MENU HANDOFF] MENU confirmation failed: {verify_exc}') from verify_exc
                        if not confirmed:
                            raise GuardUncertainError(
                                f'[PPO PAUSE] Safe stop: rollout-boundary MENU not '
                                f'independently confirmed: {evidence}') from exc
                        if getattr(raw, '_ppo_boundary_menu_duration_at', None) is None:
                            raise GuardUncertainError(
                                '[PPO MENU HANDOFF] Missing pre-optimizer MENU duration anchor')
                        raw._ppo_boundary_menu_pending = True
                        raw._ppo_boundary_menu_started_at = (
                            time.monotonic(), time.perf_counter(), time.time())
                        print('[PPO MENU HANDOFF] Boundary MENU confirmed by '
                              'independent ML+template frames; no pause, reset or '
                              'movement. ' + ('Terminal settles before current rollout returns.'
                                  if getattr(a,'settled_learning',False) else
                                  'Terminal reward deferred to next rollout step.'),
                              flush=True)
                    else:
                        raise GuardUncertainError(
                            f'[PPO PAUSE] Safe stop at rollout-end: {exc}') from exc
            # Periodic saves are deferred until the verified frozen rollout
            # boundary (or independently confirmed terminal MENU). Never stall
            # active Unity in the middle of a PPO decision to write a ZIP.
            if self._checkpoint_due_at_boundary:
                if getattr(a,'settled_learning',False):
                    self.model._settled_checkpoint_due=True
                else:
                    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
                    save_atomic_checkpoint(self.model, output/'latest.zip')
                self.last_save = time.perf_counter()
                self._checkpoint_due_at_boundary = False
                if not getattr(a,'settled_learning',False):
                    print(f'[V612 SAVE] Atomic latest.zip at safe rollout boundary '
                          f'after {self.num_timesteps} PPO steps', flush=True)

        def _on_rollout_start(self):
            record = None
            # This runs after the preceding PPO gradient update. SB3 still
            # holds its previous _last_obs. Measure the gap, but DO NOT
            # alter observations, weights, rewards, or actions.
            try:
                record = self.rollout_gap_tracker.begin_rollout(
                    time.perf_counter(), timesteps=int(self.num_timesteps))
                if record is not None:
                    print("[PPO GAP] " + json.dumps(record), flush=True)
                    with self.rollout_gap_log_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(record) + "\n")
            except Exception as exc:
                print(f"[PPO GAP] rollout-start diagnostic failed: {exc}", flush=True)
            # MENU needs no freeze while the optimizer works: the game is
            # already terminal and idle. Re-confirm before handing the next
            # step to the input guard. No R, no direct VecEnv.reset, and no
            # stale-image movement are permitted by the handoff gate.
            if (a.pause_ppo_optimizer and record is not None and raw is not None
                    and getattr(raw, '_ppo_boundary_menu_pending', False)):
                if getattr(raw, '_ppo_pause_pending', None) is not None:
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] Unexpected pause token during terminal handoff')
                confirmed, evidence = raw.confirm_post_optimizer_menu()
                if not confirmed:
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] MENU not independently confirmed '
                        f'after optimizer: {evidence}; NO inputs sent.')
                token = getattr(raw, '_ppo_boundary_menu_started_at', None)
                if token is None or len(token) != 3:
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] Missing idle-menu timer anchor')
                vec = self.model.env
                if (self.model.n_envs != 1 or not hasattr(vec, 'envs') or
                        len(vec.envs) != 1 or getattr(vec.envs[0], 'env', None) is not raw):
                    raise GuardUncertainError(
                        '[PPO MENU HANDOFF] Expected exact single-env Monitor')
                from tools.ppo_optimizer_pause_clock import compensate_pause_clocks
                now_mono, now_perf, now_wall = (
                    time.monotonic(), time.perf_counter(), time.time())
                try:
                    idle = compensate_pause_clocks(
                        raw, now_mono-token[0], now_perf-token[1],
                        monitor=vec.envs[0], elapsed_wall=now_wall-token[2])
                except Exception as exc:
                    raise GuardUncertainError(
                        f'[PPO MENU HANDOFF] Idle-menu clock correction failed: {exc}') from exc
                raw._ppo_boundary_menu_started_at = None
                print('[PPO MENU HANDOFF] MENU reverified after optimizer; '
                      f'menu_idle_seconds_excluded={idle["paused_wall_seconds"]:.3f}; '
                      'next guarded PPO step must be TERMINAL and will record '
                      'the actual death penalty; no movement key allowed.', flush=True)
                return
            # Resume BEFORE the existing fresh-stack capture and PPO prediction.
            if a.pause_ppo_optimizer and record is not None:
                try:
                    if raw is None:
                        raise RuntimeError('No raw environment for PPO resume')
                    vec = self.model.env
                    if self.model.n_envs != 1 or not hasattr(vec, 'envs') or len(vec.envs) != 1:
                        raise RuntimeError('Pause clock correction requires one DummyVecEnv')
                    monitor = vec.envs[0]
                    if getattr(monitor, 'env', None) is not raw:
                        raise RuntimeError('Pause requires Monitor(raw) without other wrappers')
                    clock_report = raw.resume_after_ppo_optimizer(monitor)
                    print('[PPO PAUSE] Resumed after optimizer with pause-aware clocks: '
                          + json.dumps(clock_report), flush=True)
                except Exception as exc:
                    raise GuardUncertainError(f'[PPO PAUSE] Safe stop at rollout-start: {exc}') from exc
            # Run after the optimizer, before SB3 predicts its first action.
            # Only active with the explicit experimental flag.
            if a.resync_rollouts and record is not None:
                try:
                    if raw is None:
                        raise RuntimeError("No raw environment for resync")
                    observation, skip_reason, capture_at = raw.refresh_after_ppo_optimizer()
                    if observation is None:
                        # v1.7.3: MENU may appear DURING resync after the game
                        # resumes. Confirm with four fresh ML+template captures,
                        # then let the existing guarded env.step record the
                        # actual terminal and death penalty with NO movement.
                        # The cached PPO observation is stale, but the guard
                        # MUST intercept the sampled action before any key.
                        candidate = getattr(raw, '_ppo_post_resume_menu_candidate', None)
                        if candidate is not None:
                            if (not a.pause_ppo_optimizer or
                                    getattr(raw, '_ppo_pause_pending', None) is not None or
                                    getattr(raw, '_ppo_boundary_menu_pending', False)):
                                raise GuardUncertainError(
                                    '[PPO POST RESUME MENU] Invalid pause/menu state; no keys')
                            confirmed, evidence = raw.confirm_post_optimizer_menu()
                            if not confirmed:
                                raise GuardUncertainError(
                                    '[PPO POST RESUME MENU] MENU not independently '
                                    f'confirmed: {evidence}; no inputs sent')
                            vec = self.model.env
                            if (self.model.n_envs != 1 or not hasattr(vec, 'envs') or
                                    len(vec.envs) != 1 or
                                    getattr(vec.envs[0], 'env', None) is not raw):
                                raise GuardUncertainError(
                                    '[PPO POST RESUME MENU] Expected single Monitor(raw)')
                            if len(candidate) != 3:
                                raise GuardUncertainError(
                                    '[PPO POST RESUME MENU] Missing first-menu time anchor')
                            from tools.ppo_optimizer_pause_clock import compensate_pause_clocks
                            now_mono, now_perf, now_wall = (
                                time.monotonic(), time.perf_counter(), time.time())
                            duration = max(
                                float(raw.survival_rewards.last_duration),
                                float(candidate[0]) - float(raw.episode_start))
                            if not np.isfinite(duration) or duration < 0:
                                raise GuardUncertainError(
                                    '[PPO POST RESUME MENU] Invalid terminal duration')
                            clock = compensate_pause_clocks(
                                raw, now_mono-candidate[0], now_perf-candidate[1],
                                monitor=vec.envs[0], elapsed_wall=now_wall-candidate[2])
                            raw._ppo_boundary_menu_duration_at = duration
                            raw._ppo_boundary_menu_pending = True
                            raw._ppo_post_resume_menu_candidate = None
                            print('[PPO POST RESUME MENU] Confirmed MENU after '
                                  f'{len(evidence["checks"])} fresh ML+template captures; '
                                  f'menu_idle_seconds_excluded={clock["paused_wall_seconds"]:.3f}; '
                                  'deferred guarded terminal with NO movement.', flush=True)
                            return
                        # If the game died *during the optimizer*, the preceding
                        # rollout was already trained on. Safely reset a NEW episode
                        # without inventing an action or a synthetic terminal reward.
                        # An abandoned in-progress episode is logged explicitly.
                        if not (a.recover_rollout_menu and candidate_is_menu(skip_reason)):
                            raise GuardUncertainError(
                                f"Cannot verify gameplay after PPO update: {skip_reason}. "
                                "Stopping instead of acting on the stale cached image.")
                        verified, evidence = raw.confirm_post_optimizer_menu()
                        if not verified:
                            raise GuardUncertainError(
                                f"Post-update MENU not independently confirmed: {evidence}. "
                                "Stopping without restarting active gameplay.")
                        # Use the *monitored VecEnv*, not raw.reset(), so monitor
                        # and SB3 wrappers remain synchronized. Only confirmed MENU
                        # can reach the inherited safe R-restart routine.
                        new_vec_obs = self.model.env.reset()
                        reset_obs, episode_starts = validated_new_vec_episode(
                            self.model._last_obs, new_vec_obs, self.model.n_envs)
                        fresh_at = getattr(raw, 'last_capture_end', None)
                        reset_age_ms = ((time.perf_counter() - fresh_at) * 1000
                                        if fresh_at is not None else float('inf'))
                        if not (0 <= reset_age_ms < 100.0):
                            raise GuardUncertainError(
                                f"Reset gameplay capture is stale ({reset_age_ms:.1f}ms).")
                        self.model._last_obs = reset_obs
                        self.model._last_episode_starts = episode_starts
                        print('[PPO RESYNC] Verified MENU during optimizer; '
                              f'restarted local Practice with fresh observation '
                              f'({reset_age_ms:.1f}ms old), new_episode=True; '
                              'ABANDONED partial episode without death reward.', flush=True)
                        return
                    age_ms = (time.perf_counter() - capture_at) * 1000
                    if not (0 <= age_ms < 100.0):
                        raise GuardUncertainError(
                            f"Refresh itself is stale ({age_ms:.1f}ms). "
                            "Stopping instead of acting on an old observation.")
                    replace_cached_observation(self.model, observation)
                    print(f"[PPO RESYNC] refreshed 4-frame stack; cached age now {age_ms:.1f}ms", flush=True)
                except Exception as exc:
                    raise GuardUncertainError(f"[PPO RESYNC] Safe stop: {exc}") from exc

        def _on_step(self):
            now = time.perf_counter()
            try:
                # raw is the same unwrapped env supplied to Monitor,
                # preserving original source-frame timing even if SB3
                # adds VecEnv observation wrappers.
                capture_at = getattr(raw, 'last_capture_end', None) if raw is not None else None
                done = bool(np.any(self.locals.get('dones', [])))
                self.rollout_gap_tracker.record_step(now, capture_at=capture_at, done=done)
            except Exception as exc:
                if not getattr(self, '_rollout_gap_step_error_shown', False):
                    self._rollout_gap_step_error_shown = True
                    print(f"[PPO GAP] step diagnostic failed: {exc}", flush=True)
            for act in np.asarray(self.locals.get('actions', []), dtype=np.int32).ravel():
                self.action_total[int(act)] += 1
                self.action_recent[int(act)] += 1
            if a.audit_reward_timing:
                # Read env->Monitor->DummyVecEnv INFO, after the actual
                # policy action. Absolutely no actions/rewards/dones changed.
                audited_infos = self.locals.get('infos', [])
                audited_dones = np.asarray(self.locals.get('dones', []), dtype=bool).ravel()
                audited_rewards = np.asarray(self.locals.get('rewards', []), dtype=float).ravel()
                if len(audited_infos) != 1 or len(audited_dones) != 1 or len(audited_rewards) != 1:
                    raise RuntimeError('[PPO REWARD STEP AUDIT] Expected single env INFO/DONE/reward')
                audit_info = audited_infos[0]
                required = ('duration', 'reward_survival', 'reward_milestones',
                            'reward_death', 'reward_v61')
                if any(k not in audit_info for k in required):
                    raise RuntimeError('[PPO REWARD STEP AUDIT] Missing raw reward fields')
                audit_row = {
                    'step': int(self.num_timesteps),
                    'episode_duration_s': float(audit_info['duration']),
                    'done': bool(audited_dones[0]),
                    'truncated': bool(audit_info.get('TimeLimit.truncated', False)),
                    'environment_reward': float(audited_rewards[0]),
                    'reward_survival': float(audit_info['reward_survival']),
                    'reward_milestones': float(audit_info['reward_milestones']),
                    'reward_death': float(audit_info['reward_death']),
                    'reward_v61': float(audit_info['reward_v61']),
                    'ppo_menu_handoff_terminal': bool(
                        audit_info.get('ppo_menu_handoff_terminal', False)),
                }
                self.reward_step_audit_handle.write(json.dumps(audit_row)+'\n')
                self.reward_step_audit_handle.flush()
            for info in self.locals.get('infos', []):
                for key, dest in (('reward_survival','survival'),('reward_milestones','milestones'),
                                  ('reward_death','death')):
                    self.reward_components[dest] += float(info.get(key,0.))
                if 'episode' in info:
                    secs=float(info.get('reward_duration_s',info.get('duration',info['episode']['t']))
                        if getattr(a,'settled_learning',False) else info.get('duration',info['episode']['t']))
                    self.episode_durations.append(secs)
                    self.episode_rewards.append(float(info['episode']['r']))
                    self.recent_episodes.append(secs)
            if now - self.last_log >= a.log_every_seconds:
                values=list(self.recent_episodes)
                policy_snapshot = read_policy_probs(
                    self.model, self.locals.get('new_obs'))
                if policy_snapshot is not None:
                    self.policy_snapshots.append(policy_snapshot)
                record={
                    'policy_snapshot': policy_snapshot,
                    'seconds':round(now-self.began,1),'steps':int(self.num_timesteps),
                    'episodes':len(self.episode_durations),
                    'mean_recent_25_seconds':round(float(np.mean(values)),2) if values else None,
                    'median_recent_25_seconds':round(float(np.median(values)),2) if values else None,
                    'best_duration_seconds':round(max(self.episode_durations),2) if self.episode_durations else None,
                    'recent_actions':dict(self.action_recent),
                    'total_actions':dict(self.action_total),
                    'reward_components':{k:round(v,3) for k,v in self.reward_components.items()},
                }
                print('[V612 PROGRESS] '+json.dumps(record),flush=True)
                self.log_handle.write(json.dumps(record)+'\n');self.log_handle.flush()
                self.action_recent.clear()
                self.last_log=now
            if now - self.last_save >= a.save_every_minutes*60:
                if a.pause_ppo_optimizer:
                    self._checkpoint_due_at_boundary = True
                else:
                    from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
                    save_atomic_checkpoint(self.model, output/'latest.zip')
                    print(f'[V612 SAVE] Atomic latest.zip after {self.num_timesteps} PPO steps',flush=True)
                    self.last_save=now
            return (True if getattr(a,'settled_learning',False)
                    else (now-self.began) < a.max_minutes*60)

        def _on_training_end(self):
            if self.reward_step_audit_handle is not None:
                self.reward_step_audit_handle.close()
            if self.log_handle is not None:
                self.log_handle.close()

    return V61Callback()


def create_live_env(a, output, *, browser_session=None):
    from stable_baselines3.common.monitor import Monitor
    import train_rs_v2_verify as v2
    from train_rs_v3_ppo import v2_args
    from ultralytics import YOLO
    from rs_v5_shadow_segmentation import check_model_names, detect_v5

    detector = YOLO(str(a.detector))
    check_model_names(detector)
    if a.torch_threads > 0:
        import torch
        torch.set_num_threads(a.torch_threads)
    # Warm up before measuring, avoiding GPU first-run stalls.
    _ = detect_v5(detector,np.zeros((576,1024,3),dtype=np.uint8),
                  time.perf_counter(),0,0,a.imgsz,a.conf)
    arg = argparse.Namespace(url='http://localhost:8000', fps=a.fps,
                             max_minutes=a.max_minutes,fullscreen=True)
    cfg=v2_args(arg)
    cfg.episode_max_seconds=a.episode_max_seconds
    if browser_session is not None:
        cfg.persistent_browser_session = browser_session
    Env=build_env_class()
    demos=v2.DemoStore(v2.DEMO_ROOT,max_samples=100,load_existing=False)
    raw=Env(cfg,output,demos,detector,a=a)
    if a.mode == 'train':
        # Full reward audits stream to disk; keep only recent debug timing
        # dictionaries in RAM during multi-hour training.
        raw.step_timings = deque(maxlen=10000)
    if browser_session is not None:
        write_json(output/'persistent_browser.json', browser_session.summary())
    try:
        monitored=Monitor(raw,filename=str(output/'episodes.monitor.csv'))
    except BaseException:
        raw.close()
        raise
    return monitored, raw


def build_report(a, raw, *, steps, started, callback=None, mode='train'):
    durations = callback.episode_durations if callback is not None else []
    report = {
        'mode':mode,'feature_schema':shape_description(),
        'ppo_actions_sent_only_when_gameplay_verified': True, 
        'input_note':'Actions sent through Playwright, Unity response not automatically verified.',
        'no_demonstration_training':True,
        'reward_terms':{'survival_per_real_second':a.reward_survival_rate,
                        'milestone_bonus':a.reward_milestone_bonus,
                        'milestone_seconds':a.reward_milestone_seconds,
                        'confirmed_death_penalty':a.reward_death_penalty,
                        'reward_for_yolo_predictions':0,'reward_for_specific_actions':0},
        'steps':int(steps),'wall_seconds':round(time.perf_counter()-started,2),
        'avg_steps_per_sec_including_resets':round(int(steps)/max(.001,time.perf_counter()-started),2),
        'episodes':len(durations), 'mean_survival_seconds':round(float(np.mean(durations)),2) if durations else None,
        'max_survival_seconds':round(max(durations),2) if durations else None,
        'action_counts':dict(callback.action_total) if callback is not None else dict(raw.action_counts),
        'gameplay_only_intervals_ms':bench.as_report(raw.gameplay_only_interval_ms),
        'startup_guard_failures':raw.startup_failures,
        'detector':raw.hazard_report(),
        'observation_aux_dim':AUX_DIM,'observation_image_shape':list(raw.observation_space['image'].shape),
        'episode_time_limit_seconds':float(a.episode_max_seconds),
        'debug_step_timing_history_limit':getattr(raw.step_timings, 'maxlen', None),
        'reward_parts':dict(raw.reward_parts),
        'first_25_mean_seconds':round(float(np.mean(durations[:25])),3) if len(durations)>=25 else None,
        'last_25_mean_seconds':round(float(np.mean(durations[-25:])),3) if len(durations)>=25 else None,
        'policy_snapshots':(callback.policy_snapshots if callback is not None else []),
    }
    return report


def train(a, *, browser_session=None):
    import torch
    from stable_baselines3 import PPO
    from train_rs_v2_verify import RUN_ROOT

    dest=RUN_ROOT/a.run_name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Run directory already exists: {dest}. Use unique --run-name.')
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'v61_config.json', dict(vars(a), schema=shape_description()))
    started=time.perf_counter()
    monitored=raw=model=callback=None
    completed=False
    try:
        monitored,raw=(create_live_env(a,dest) if browser_session is None else
                       create_live_env(a,dest,browser_session=browser_session))
        if a.resume:
            from tools.ppo_checkpoint_settings import load_training_checkpoint
            model, effective_settings = load_training_checkpoint(
                a, monitored, device='cuda' if torch.cuda.is_available() else 'cpu')
            write_json(dest/'v61_effective_ppo_settings.json', effective_settings)
            if (model.n_steps != a.n_steps or model.batch_size != a.batch_size):
                print('[RESUME] Using checkpoint PPO rollout/batch hyperparameters; '
                      'CLI --n-steps/--batch-size do not override a loaded checkpoint.',flush=True)
            print(f'[RESUME] PPO weights loaded from {a.resume}',flush=True)
        else:
            model=make_model(monitored,a)
        model.framing_profile = getattr(a,'framing_profile','legacy')
        if getattr(a,'settled_learning',False):
            from tools.ppo_settled_learning import SettledPPO,reference_seconds_for
            model.__class__ = SettledPPO
            reference=reference_seconds_for(model,a.discount_reference_seconds)
            model.configure_settled(raw,dest,reference,a.max_minutes)
        callback=build_callback(a,dest,raw=raw)
        print(f'[V612 TRAIN] PPO executing its OWN actions; no fixed jump; no demos. '
              f'RGB={a.resolution}, AUX={AUX_DIM}, YOLO={a.detector}, '
              f'steps={a.steps}, max_minutes={a.max_minutes}, device={model.device}',flush=True)
        model.learn(total_timesteps=a.steps,callback=callback,progress_bar=False,
                    reset_num_timesteps=not bool(a.resume))
        completed=True
    except KeyboardInterrupt:
        print('[V612] Interrupted by user. Saving current weights.',flush=True)
    except GuardUncertainError as exc:
        print(str(exc), flush=True)
        print('[V612] UI safety stop. Partial checkpoint/report will be saved; DO NOT resume until evidence is reviewed.',flush=True)
    finally:
        if callback is not None and callback.log_handle is not None and not callback.log_handle.closed:
            callback.log_handle.flush()
            callback.log_handle.close()
        if model is not None:
            from tools.ppo_atomic_checkpoint_v176 import save_atomic_checkpoint
            save_atomic_checkpoint(model, dest/'latest.zip')
            print(f'[V612] Saved model atomically: {dest/"latest.zip"}',flush=True)
        if raw is not None:
            report=build_report(a,raw,steps=raw.training_steps,
                                started=started,callback=callback)
            if getattr(a,'settled_learning',False):
                stats=model._settled_stats
                stats['discarded_partial_transitions']=max(
                    stats['discarded_partial_transitions'],
                    stats['collected_transitions']-stats['completed_rollout_transitions'])
                stats['completed_interactions_not_ledgered']=max(0,
                    model.num_timesteps-model._settled_initial_timesteps-stats['collected_transitions'])
                report['raw_environment_operations'] = raw.training_steps
                report['steps'] = model._settled_stats['collected_transitions']
                report['learning_transitions'] = model._settled_stats
                report['reward_clock'] = 'verified_gameplay_only_excluding_pause_UI'
                report['discount_reference_seconds'] = model.settled_reference_seconds
                report['survival_reward_cap'] = 'uncapped_verified_time; no confirmation-time credit'
                report['legacy_callback_action_counts'] = report['action_counts']
                selected_counts,dispatched_counts=Counter(),Counter()
                with (dest/'learning_transitions.jsonl').open() as ledger:
                    for line in ledger:
                        row=json.loads(line)
                        if row['event']=='collected':
                            if row['sampled_action'] is not None:
                                selected_counts[str(row['sampled_action'])]+=1
                            if row['command'] is not None:
                                dispatched_counts[row['command']]+=1
                report['action_counts']=dict(selected_counts)
                report['dispatched_key_counts']=dict(dispatched_counts)
                report['avg_steps_per_sec_including_resets'] = round(
                    report['steps']/max(.001,report['wall_seconds']),2)
            report['completed_requested_steps']=completed and (
                model._settled_stats['completed_rollout_transitions']>=a.steps
                if getattr(a,'settled_learning',False) else raw.training_steps>=a.steps)
            report['interrupted_or_time_limited']=not report['completed_requested_steps']
            write_json(dest/'v61_report.json',report)
            print('[V612 REPORT] '+json.dumps({k:report[k] for k in ('steps','episodes',
                  'mean_survival_seconds','max_survival_seconds','avg_steps_per_sec_including_resets',
                  'gameplay_only_intervals_ms','startup_guard_failures')} | {
                      'live_guard': report['detector']['v612_live_input_guard'],
                      'completed_requested_steps': report['completed_requested_steps']
                  },default=str),flush=True)
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()
    print(f'[V612] Training outputs: {dest}',flush=True)


def evaluate(a, *, browser_session=None):
    import torch
    from stable_baselines3 import PPO
    from train_rs_v2_verify import RUN_ROOT
    dest=RUN_ROOT/a.run_name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Evaluation folder exists: {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'v61_eval_config.json', dict(vars(a),schema=shape_description()))
    started=time.perf_counter()
    monitored=raw=None
    durations=[];scores=[];actions=Counter(); steps=0
    termination_reasons=Counter()
    prob_sums=np.zeros(5,dtype=np.float64)
    prob_samples=0
    prob_maxima=[]
    try:
        monitored,raw=(create_live_env(a,dest) if browser_session is None else
                       create_live_env(a,dest,browser_session=browser_session))
        eval_policy=('stochastic' if a.stochastic_eval else a.eval_policy)
        model=(PPO.load(str(a.resume),env=monitored,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
               if eval_policy in ('deterministic','stochastic') else None)
        rng=np.random.default_rng(a.seed)
        obs,info=monitored.reset()
        while len(durations)<a.eval_episodes and time.perf_counter()-started<a.max_minutes*60:
            if eval_policy == 'random':
                action=int(rng.integers(0,5))
            elif eval_policy == 'none':
                action=0
            else:
                from tools.ppo_eval_prediction import predict_with_optional_probabilities
                action, probs = predict_with_optional_probabilities(model, obs,
                    deterministic=(eval_policy=='deterministic'), sample_probabilities=(steps % 8 == 0))
                action=int(action)
                if probs is not None:
                    prob_sums += np.asarray(probs['probabilities'],dtype=float)
                    prob_samples+=1
                    prob_maxima.append(probs['max_probability'])
            obs,r,done,trunc,info=monitored.step(action)
            actions[action]+=1;steps+=1
            if done or trunc:
                termination_reasons['death' if done else 'time_limit'] += 1
                durations.append(float(info.get('duration',0.)))
                scores.append(float(info.get('episode',{}).get('r',0.)))
                print(f'[V61 EVAL] episode {len(durations)}/{a.eval_episodes}: '
                      f'{durations[-1]:.2f}s; return={scores[-1]:.2f}; '
                      f'actions={dict(actions)}',flush=True)
                if len(durations)<a.eval_episodes:
                    obs,info=monitored.reset()
    except KeyboardInterrupt:
        print('[V61 EVAL] Interrupted; saving partial statistics.',flush=True)
    except GuardUncertainError as exc:
        print(str(exc), flush=True)
        print('[V612 EVAL] UI safety stop. Partial evaluation will be saved.',flush=True)
    finally:
        if raw is not None:
            report=build_report(a,raw,steps=steps,started=started,mode='eval')
            report.update({'eval_episodes':len(durations),'durations_seconds':durations,
                           'eval_termination_counts':dict(termination_reasons),
                           'mean_survival_seconds':round(float(np.mean(durations)),2) if durations else None,
                           'median_survival_seconds':round(float(np.median(durations)),2) if durations else None,
                           'best_survival_seconds':max(durations) if durations else None,
                           'returns':scores,'action_counts':dict(actions),
                           'deterministic':eval_policy=='deterministic', 'eval_policy':eval_policy,
                            'mean_action_probabilities':dict(zip(ACTION_NAMES,
                                [round(float(x),4) for x in prob_sums/max(1,prob_samples)])),
                            'mean_policy_max_probability':round(float(np.mean(prob_maxima)),4) if prob_maxima else None,
                            'policy_probability_samples':prob_samples})
            write_json(dest/'v61_eval_report.json',report)
            print(f'[V61 EVAL REPORT] {dest/"v61_eval_report.json"}',flush=True)
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',required=True,choices=['check','probe','train','eval'])
    p.add_argument('--run-name',default='v61_ppo_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--resolution',type=bench.shape,default=(512,288))
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--framing-profile',choices=['legacy','canvas_v1'],default='legacy',
                   help='canvas_v1: whole calibrated game canvas for PPO/detector; separate guard inputs')
    p.add_argument('--background-display',type=int,default=None,
                   help='EXPERIMENTAL non-primary MSS display index; page focus emulation, no OS activation')
    p.add_argument('--background-capture',choices=['mss','dxgi'],default='mss',
                   help='EXPERIMENTAL background capture backend; DXGI requires isolated optional dependencies')
    p.add_argument('--canvas-policy-input',action='store_true',
                   help='With --background-display, crop policy/detector inside pixel-calibrated canvas (changes framing)')
    p.add_argument('--rate-mode',choices=['uncapped','capped'],default='uncapped')
    p.add_argument('--reuse-observation-guard', action='store_true',
                   help='EXPERIMENTAL: reuse the confident gameplay check on the exact policy image for at most 20 ms; otherwise use normal preflight capture')
    learning_mode=p.add_mutually_exclusive_group()
    learning_mode.add_argument('--settled-learning',action='store_true',
                   help='Corrected collector: settled boundary terminals, masked passive transitions, verified-time rewards and duration-aware discounts')
    learning_mode.add_argument('--legacy-learning',action='store_true',
                   help='Historical comparisons only: retain old transition/reward/discount semantics')
    p.add_argument('--discount-reference-seconds',type=float,default=None,
                   help='Explicit time unit for corrected gamma/lambda; otherwise inherit checkpoint unit, or 0.05s for legacy sources')
    p.add_argument('--resync-rollouts', action='store_true',
                   help='EXPERIMENTAL: passively rebuild 4-frame observation after PPO updates')
    p.add_argument('--pause-ppo-optimizer', action='store_true',
                   help='EXPERIMENTAL: freeze local Practice during PPO gradient updates, correct wall clocks; requires --resync-rollouts')
    p.add_argument('--recover-rollout-menu', action='store_true',
                   help='EXPERIMENTAL: after verified death during PPO update, reset local Practice with episode flag; skipped death reward is logged')
    p.add_argument('--crop',nargs=4,type=float,default=bench.DEFAULT_CROP)
    p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--max-minutes',type=float,default=25.)
    p.add_argument('--n-steps',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--n-epochs',type=int,default=5)
    p.add_argument('--learning-rate',type=float,default=3.0e-4)
    p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--ent-coef',type=float,default=.002)
    p.add_argument('--target-kl',type=float,default=None)
    p.add_argument('--override-ppo-settings',nargs='+',default=[],
                   choices=['n_steps','batch_size','n_epochs','learning_rate','gamma','ent_coef','target_kl','seed'],
                   help='Explicitly override only these saved settings when resuming training; all others retain checkpoint values')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--detector',type=Path,default=Path('ai_data/v5_hazards/runs/segment_round04/weights/best.pt'))
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--conf',type=float,default=.20)
    p.add_argument('--detect-every',type=int,default=3)
    p.add_argument('--max-stale-ms',type=int,default=300)
    p.add_argument('--press-delay-ms',type=int,default=10)
    p.add_argument('--capture-mode',choices=['fast','legacy'],default='fast')
    p.add_argument('--resize-method',choices=['linear','area'],default='linear')
    p.add_argument('--torch-threads',type=int,default=1)
    p.add_argument('--startup-grace',type=float,default=.10)
    p.add_argument('--restart-grace',type=float,default=.10)
    p.add_argument('--state-model',type=Path,default=Path('ai_data/v4_screen_states/screen_state_best.pt'))
    p.add_argument('--state-confidence',type=float,default=.90)
    p.add_argument('--menu-ready-timeout',type=float,default=30.)
    p.add_argument('--startup-timeout',type=float,default=8.)
    p.add_argument('--startup-motion-threshold',type=float,default=2.2)
    p.add_argument('--save-overlays',type=int,default=6)
    p.add_argument('--preview-delay',type=float,default=.2)
    p.add_argument('--episode-max-seconds',type=float,default=300.,
                   help='External episode time limit; reported separately from death. Use 15 explicitly for historical benchmarks.')
    p.add_argument('--reward-survival-rate',type=float,default=.8)
    p.add_argument('--reward-death-penalty',type=float,default=1.0)
    p.add_argument('--reward-milestone-bonus',type=float,default=.15)
    p.add_argument('--reward-milestone-seconds',type=float,default=.5)
    p.add_argument('--log-every-seconds',type=float,default=30.)
    p.add_argument('--save-every-minutes',type=float,default=4.)
    p.add_argument('--audit-reward-timing', action='store_true',
                   help='OPT-IN: write read-only per-step reward/timing evidence for offline audit')
    p.add_argument('--resume',type=Path,default=None,
                   help='Only V6.1 checkpoints work; V6 has a different aux dimension.')
    p.add_argument('--eval-episodes',type=int,default=20)
    p.add_argument('--stochastic-eval',action='store_true',help='Compatibility alias for --eval-policy stochastic')
    p.add_argument('--eval-policy',choices=['deterministic','stochastic','random','none'],
                   default='deterministic', help='random and none allow matched no-checkpoint baselines')
    p.add_argument('--assume-center-lane',action='store_true',
                   help='EXPERIMENTAL command-only lane odometry. NOT verified by vision; off by default.')
    p.add_argument('--probe-trials',type=int,default=2,help='Repetitions per action and tap duration')
    p.add_argument('--probe-delays-ms',type=str,default='5,10',
                   help='Comma-separated Playwright tap durations for manual action audit')
    a=p.parse_args(argv)
    if a.framing_profile == 'canvas_v1' and (a.resolution != (512,288) or a.resize_method != 'linear'
                                             or tuple(a.crop) != tuple(bench.DEFAULT_CROP)):
        p.error('canvas_v1 fixes 512x288 linear whole-canvas preprocessing; custom crop/resize settings are incompatible')
    if a.background_display is not None and (a.background_display < 1 or a.capture_mode != 'fast'):
        p.error('--background-display requires index >=1 and --capture-mode fast')
    if a.canvas_policy_input and a.background_display is None:
        p.error('--canvas-policy-input requires --background-display')
    if a.background_capture != 'mss' and a.background_display is None:
        p.error('--background-capture dxgi requires --background-display')
    if a.mode=='train' and a.pause_ppo_optimizer and a.resync_rollouts and not a.legacy_learning:
        a.settled_learning=True
    if a.settled_learning and (a.mode!='train' or not a.pause_ppo_optimizer or not a.resync_rollouts):
        p.error('--settled-learning requires training with optimizer pause and rollout resync')
    if a.settled_learning and a.recover_rollout_menu:
        p.error('Corrected learning forbids abandoning episodes with --recover-rollout-menu')
    if a.discount_reference_seconds is not None and not .001<=a.discount_reference_seconds<=1.:
        p.error('Invalid discount reference interval')
    if a.override_ppo_settings and (a.mode != 'train' or a.resume is None):
        p.error('--override-ppo-settings requires resumed training')
    if a.target_kl is not None and not 0 < a.target_kl < 1:
        p.error('--target-kl must be positive and below 1')
    if a.reuse_observation_guard and a.mode == 'train' and not (a.pause_ppo_optimizer and a.resync_rollouts):
        p.error('--reuse-observation-guard training requires --pause-ppo-optimizer and --resync-rollouts')
    if a.reuse_observation_guard and a.capture_mode != 'fast':
        p.error('--reuse-observation-guard requires the fast capture path')
    if a.audit_reward_timing and not (a.mode == 'train' and a.pause_ppo_optimizer and a.resync_rollouts):
        p.error('--audit-reward-timing requires --mode train --pause-ppo-optimizer --resync-rollouts')
    if a.pause_ppo_optimizer and (a.mode != 'train' or not a.resync_rollouts):
        p.error('--pause-ppo-optimizer requires --mode train --resync-rollouts')
    a.crop=bench.validate_crop(a.crop)
    a.benchmark_action='ppo'  # NEVER override PPO actions in a learning run.
    try:
        a.probe_delays = [int(x.strip()) for x in a.probe_delays_ms.split(',')]
    except ValueError:
        p.error('--probe-delays-ms must be comma-separated integers e.g. 5,10,20')
    if not a.probe_delays or any(x<1 or x>100 for x in a.probe_delays) or not 1<=a.probe_trials<=10:
        p.error('Probe timing must be 1..100ms and trials 1..10')
    if a.mode=='check':
        if a.resolution==(512,288):
            print('[CHECK] Tip: use --resolution 160x96 for a faster offline test.',flush=True)
    else:
        if not a.detector.is_file() or not a.state_model.is_file():
            p.error(f'Missing V5 model or V4 screen state classifier: {a.detector}, {a.state_model}')
        if not 0.5 <= a.state_confidence < 1.0:
            p.error('State confidence must be in [0.5,1)')
        if not (0 < a.max_minutes <= 480 and 1 <= a.fps <= 30 and
                0 <= a.torch_threads <= 32 and a.steps >= a.n_steps and
                1 <= a.detect_every <= 50 and 1 <= a.press_delay_ms <= 100 and
                10 <= a.max_stale_ms <= 5000 and 1 <= a.menu_ready_timeout <= 90 and
                0 <= a.startup_grace <= 3 and 0 <= a.restart_grace <= 3 and
                .5 <= a.startup_timeout <= 20 and 0 < a.episode_max_seconds <= 600):
            p.error('Invalid live timing settings')
        if a.mode=='train' and a.steps%a.n_steps:
            p.error('--steps must be divisible by --n-steps')
        if a.mode=='eval' and (a.stochastic_eval or a.eval_policy in ('deterministic','stochastic')) and (a.resume is None or not a.resume.is_file()):
            p.error('--mode eval with learned policy requires an existing --resume model.zip')
        if a.mode=='train' and a.resume and not a.resume.is_file():
            p.error(f'--resume checkpoint does not exist: {a.resume}')
        if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):
            p.error('--run-name must be a simple directory name')
    if not (a.n_steps>=16 and a.batch_size>=1 and a.n_steps%a.batch_size==0 and
            a.n_epochs>=1 and 1e-6<=a.learning_rate<=1e-2 and 0<a.gamma<1 and
            0<=a.ent_coef<=1 and a.reward_survival_rate>=0 and
            a.reward_death_penalty>=0 and a.reward_milestone_bonus>=0 and
            a.reward_milestone_seconds>0):
        p.error('Invalid PPO or reward settings')
    if not (a.imgsz>=160 and a.imgsz%32==0 and 0<a.conf<1):
        p.error('YOLO --imgsz must be a multiple of 32, and conf within (0,1)')
    return a


if __name__=='__main__':
    args=parse_args()
    if args.mode=='check':
        offline_check(args)
    elif args.mode=='probe':
        input_probe(args)
    elif args.mode=='train':
        train(args)
    else:
        evaluate(args)
