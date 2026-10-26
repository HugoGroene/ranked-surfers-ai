#!/usr/bin/env python3
"""LOCAL Ranked Surfers V6.1.1: PPO with richer perception, rewards, input probe.

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
from rs_v611_gameplay_guard import screen_gate, pair_decision

# The proven local Chromium/state-classifier controller is inherited rather
# than being rewritten; imported lazily to keep schema utilities lightweight.
import rs_v5_segmentation_live_benchmark as bench


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def build_env_class():
    from gymnasium import spaces
    import train_rs_v3_ppo as v3

    OldEnv = bench.build_live_env_class()

    class PPOV61Env(OldEnv):
        """Preserve proven V4/V5 control, extend PPO observations without rules."""
        def __init__(self, cfg, output, demos, detector, *, a):
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
            self._guard_forced_terminal = False
            self._guard_hard_stopped = False
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
            source_at = self.last_capture_end or now
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
            self.pack_ms.append((time.perf_counter()-start)*1000)
            return {'image': packed, 'aux': aux}, fresh, age_ms

        def reset(self, *, seed=None, options=None):
            if self._guard_hard_stopped:
                raise RuntimeError('[V611 GUARD] Training latched off after uncertain UI; restart process after reviewing evidence.')
            self._guard_forced_terminal = False
            self.survival_rewards.reset()
            self.action_context.reset()
            return super().reset(seed=seed, options=options)

        def _guard_diagnostic(self, first, second, *, reason):
            import cv2
            data = {
                'episode': int(self.episode_number),
                'reason': reason,
                'first': vars(first),
                'second': vars(second),
                'note': 'No movement command was sent on this failed preflight.',
            }
            target = self.output_dir / f'v611_guard_episode_{self.episode_number:04d}.jpg'
            if self._v2_last_bgra is not None:
                cv2.imwrite(str(target), cv2.cvtColor(self._v2_last_bgra, cv2.COLOR_BGRA2BGR))
            write_json(target.with_suffix('.json'), data)
            return target

        async def _step_browser(self, action):
            """Check a NEW full-screen screenshot immediately before every key.

            The inherited controller only checks the full-screen classifier at
            restart; it normally uses template-only death detection while
            playing. This gate does not let PPO press during MENU/LOADING/
            OTHER/UNCERTAIN, even if the template detector misses the UI.
            """
            import asyncio
            if self._guard_hard_stopped:
                raise RuntimeError('[V611 GUARD] Inputs disabled after uncertain screen state.')
            first_frame = await self._capture()
            state, probs = self._read_screen_state()
            first = screen_gate(state, probs, first_frame[1],
                                confidence=self.state_confidence,
                                menu_threshold=self.args.threshold)
            if first.verdict != 'PLAY':
                await asyncio.sleep(.045)
                second_frame = await self._capture()
                state, probs = self._read_screen_state()
                second = screen_gate(state, probs, second_frame[1],
                                     confidence=self.state_confidence,
                                     menu_threshold=self.args.threshold)
                decision = pair_decision(first, second)
                if decision == 'TERMINAL':
                    self.guard_confirmed_menu += 1
                    self._guard_forced_terminal = True
                    print(f'[V611 GUARD] Menu confirmed BEFORE action; movement suppressed '
                          f'(ep={self.episode_number}, template={second.menu_score:.2f}).', flush=True)
                    return second_frame
                # If one capture recovers, require ANOTHER independent capture
                # with the same state. Never give PPO a reward for an action we
                # silently replaced with no-op.
                if decision in ('CONFIRM_PLAY', 'CONFIRM_MENU'):
                    self.guard_rechecks += 1
                    await asyncio.sleep(.045)
                    third_frame = await self._capture()
                    state, probs = self._read_screen_state()
                    third = screen_gate(state, probs, third_frame[1],
                                        confidence=self.state_confidence,
                                        menu_threshold=self.args.threshold)
                    if decision == 'CONFIRM_MENU' and third.verdict == 'MENU':
                        self.guard_confirmed_menu += 1
                        self._guard_forced_terminal = True
                        print(f'[V611 GUARD] Menu confirmed after extra check; '
                              f'movement suppressed (ep={self.episode_number}).',flush=True)
                        return third_frame
                    if decision == 'CONFIRM_PLAY' and third.verdict == 'PLAY':
                        print(f'[V611 GUARD] Transient {first.state} revalidated '
                              'with two consecutive GAMEPLAY frames; action permitted.',flush=True)
                    else:
                        self.guard_aborts += 1
                        self._guard_hard_stopped = True
                        image = self._guard_diagnostic(first, third, reason='third frame failed reconfirmation')
                        raise RuntimeError(f'[V611 GUARD] ABORT: unstable UI state '
                                           f'(states={first.state},{second.state},{third.state}). '
                                           f'NO key sent. Screenshot: {image}')
                else:
                    self.guard_aborts += 1
                    self._guard_hard_stopped = True
                    image = self._guard_diagnostic(first, second, reason='two non-GAMEPLAY readings')
                    raise RuntimeError(f'[V611 GUARD] ABORT: two consecutive screen captures '
                                       f'not confirmed GAMEPLAY (states={first.state},{second.state}; '
                                       f'menu_scores={first.menu_score:.2f},{second.menu_score:.2f}). '
                                       f'NO key sent. Screenshot: {image}')
            self.guard_allowed += 1
            if self.suspected_since is None:
                self.action_context.note_action(int(action), time.perf_counter())
            # Preserve the existing configurable tap, pacing and post-key capture.
            # The inherited TimedEnv method returns (RGB, template_score, raw).
            return await super()._step_browser(int(action))

        def step(self, action):
            self._guard_forced_terminal = False
            obs, base_reward, died, truncated, info = super().step(int(action))
            if self._guard_forced_terminal:
                # Separate successful GAMEPLAY->MENU detector from a template
                # detector that could have missed the same menu transition.
                died, truncated = True, False
                info['v611_guard_confirmed_menu'] = True
            info['v611_guard_allowed'] = self.guard_allowed
            info['v611_guard_rechecks'] = self.guard_rechecks
            # Only real elapsed gameplay (observed by the local game loop) is
            # rewarded. No coins, class confidence, lane assumptions, or
            # specific action get a reward. Avoid repeated-action exploits.
            reward, pieces = self.survival_rewards.step(
                info.get('duration', 0.), died)
            self.reward_parts.update(pieces)
            info['reward_survival'] = round(pieces['survival'], 4)
            info['reward_milestones'] = round(pieces['milestones'], 4)
            info['reward_death'] = round(pieces['death'], 4)
            info['reward_v61'] = round(reward, 4)
            info['reward_old_ignored'] = round(float(base_reward), 4)
            return obs, float(reward), died, truncated, info

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
            report['v611_live_input_guard'] = {
                'preflight_gameplay_allowed': self.guard_allowed,
                'preflight_rechecks_on_transient': self.guard_rechecks,
                'preflight_confirmed_menu': self.guard_confirmed_menu,
                'uncertain_state_abort': self.guard_aborts,
                'guard_hard_stopped': self._guard_hard_stopped,
                'preflight_checks_every_decision': True,
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


def build_callback(a, output):
    from stable_baselines3.common.callbacks import BaseCallback

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

        def _on_training_start(self):
            self.log_handle = self.log_path.open('a',encoding='utf-8')

        def _on_step(self):
            now = time.perf_counter()
            for act in np.asarray(self.locals.get('actions', []), dtype=np.int32).ravel():
                self.action_total[int(act)] += 1
                self.action_recent[int(act)] += 1
            for info in self.locals.get('infos', []):
                for key, dest in (('reward_survival','survival'),('reward_milestones','milestones'),
                                  ('reward_death','death')):
                    self.reward_components[dest] += float(info.get(key,0.))
                if 'episode' in info:
                    secs=float(info.get('duration',info['episode']['t']))
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
                print('[V611 PROGRESS] '+json.dumps(record),flush=True)
                self.log_handle.write(json.dumps(record)+'\n');self.log_handle.flush()
                self.action_recent.clear()
                self.last_log=now
            if now - self.last_save >= a.save_every_minutes*60:
                self.model.save(output/'latest')
                print(f'[V611 SAVE] latest.zip after {self.num_timesteps} PPO steps',flush=True)
                self.last_save=now
            return (now-self.began) < a.max_minutes*60

        def _on_training_end(self):
            if self.log_handle is not None:
                self.log_handle.close()

    return V61Callback()


def create_live_env(a, output):
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
    Env=build_env_class()
    demos=v2.DemoStore(v2.DEMO_ROOT,max_samples=100,load_existing=False)
    raw=Env(cfg,output,demos,detector,a=a)
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
        'actual_ppo_actions_executed': mode=='train' or (mode=='eval' and a.eval_policy in ('deterministic','stochastic')), 
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
        'reward_parts':dict(raw.reward_parts),
        'first_25_mean_seconds':round(float(np.mean(durations[:25])),3) if len(durations)>=25 else None,
        'last_25_mean_seconds':round(float(np.mean(durations[-25:])),3) if len(durations)>=25 else None,
        'policy_snapshots':(callback.policy_snapshots if callback is not None else []),
    }
    return report


def train(a):
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
        monitored,raw=create_live_env(a,dest)
        if a.resume:
            model=PPO.load(str(a.resume),env=monitored,
                           device='cuda' if torch.cuda.is_available() else 'cpu')
            if (model.n_steps != a.n_steps or model.batch_size != a.batch_size):
                print('[RESUME] Using checkpoint PPO rollout/batch hyperparameters; '
                      'CLI --n-steps/--batch-size do not override a loaded checkpoint.',flush=True)
            print(f'[RESUME] PPO weights loaded from {a.resume}',flush=True)
        else:
            model=make_model(monitored,a)
        callback=build_callback(a,dest)
        print(f'[V611 TRAIN] PPO executing its OWN actions; no fixed jump; no demos. '
              f'RGB={a.resolution}, AUX={AUX_DIM}, YOLO={a.detector}, '
              f'steps={a.steps}, max_minutes={a.max_minutes}, device={model.device}',flush=True)
        model.learn(total_timesteps=a.steps,callback=callback,progress_bar=False,
                    reset_num_timesteps=not bool(a.resume))
        completed=True
    except KeyboardInterrupt:
        print('[V611] Interrupted by user. Saving current weights.',flush=True)
    finally:
        if callback is not None and callback.log_handle is not None and not callback.log_handle.closed:
            callback.log_handle.flush()
            callback.log_handle.close()
        if model is not None:
            model.save(dest/'latest')
            print(f'[V611] Saved model: {dest/"latest.zip"}',flush=True)
        if raw is not None:
            report=build_report(a,raw,steps=raw.training_steps,
                                started=started,callback=callback)
            report['completed_requested_steps']=completed and raw.training_steps>=a.steps
            report['interrupted_or_time_limited']=not report['completed_requested_steps']
            write_json(dest/'v61_report.json',report)
            print('[V611 REPORT] '+json.dumps({k:report[k] for k in ('steps','episodes',
                  'mean_survival_seconds','max_survival_seconds','avg_steps_per_sec_including_resets',
                  'gameplay_only_intervals_ms','startup_guard_failures')},default=str),flush=True)
        if monitored is not None:
            monitored.close()
        elif raw is not None:
            raw.close()
    print(f'[V611] Training outputs: {dest}',flush=True)


def evaluate(a):
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
    prob_sums=np.zeros(5,dtype=np.float64)
    prob_samples=0
    prob_maxima=[]
    try:
        monitored,raw=create_live_env(a,dest)
        eval_policy=('stochastic' if a.stochastic_eval else a.eval_policy)
        model=(PPO.load(str(a.resume),env=monitored,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
               if eval_policy in ('deterministic','stochastic') else None)
        rng=np.random.default_rng(a.seed)
        obs,info=monitored.reset()
        while len(durations)<a.eval_episodes and time.perf_counter()-started<a.max_minutes*60:
            if model is not None and steps % 8 == 0:
                probs=read_policy_probs(model,obs)
                if probs is not None:
                    prob_sums += np.asarray(probs['probabilities'],dtype=float)
                    prob_samples+=1
                    prob_maxima.append(probs['max_probability'])
            if eval_policy == 'random':
                action=int(rng.integers(0,5))
            elif eval_policy == 'none':
                action=0
            else:
                action,_=model.predict(obs,deterministic=(eval_policy=='deterministic'))
                action=int(action)
            obs,r,done,trunc,info=monitored.step(action)
            actions[action]+=1;steps+=1
            if done or trunc:
                durations.append(float(info.get('duration',0.)))
                scores.append(float(info.get('episode',{}).get('r',0.)))
                print(f'[V61 EVAL] episode {len(durations)}/{a.eval_episodes}: '
                      f'{durations[-1]:.2f}s; return={scores[-1]:.2f}; '
                      f'actions={dict(actions)}',flush=True)
                if len(durations)<a.eval_episodes:
                    obs,info=monitored.reset()
    except KeyboardInterrupt:
        print('[V61 EVAL] Interrupted; saving partial statistics.',flush=True)
    finally:
        if raw is not None:
            report=build_report(a,raw,steps=steps,started=started,mode='eval')
            report.update({'eval_episodes':len(durations),'durations_seconds':durations,
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


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',required=True,choices=['check','probe','train','eval'])
    p.add_argument('--run-name',default='v61_ppo_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--resolution',type=bench.shape,default=(512,288))
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--rate-mode',choices=['uncapped','capped'],default='uncapped')
    p.add_argument('--crop',nargs=4,type=float,default=bench.DEFAULT_CROP)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--max-minutes',type=float,default=25.)
    p.add_argument('--n-steps',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--n-epochs',type=int,default=5)
    p.add_argument('--learning-rate',type=float,default=3.0e-4)
    p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--ent-coef',type=float,default=.002)
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
    p.add_argument('--episode-max-seconds',type=float,default=15.)
    p.add_argument('--reward-survival-rate',type=float,default=.8)
    p.add_argument('--reward-death-penalty',type=float,default=1.0)
    p.add_argument('--reward-milestone-bonus',type=float,default=.15)
    p.add_argument('--reward-milestone-seconds',type=float,default=.5)
    p.add_argument('--log-every-seconds',type=float,default=30.)
    p.add_argument('--save-every-minutes',type=float,default=4.)
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
    a=p.parse_args()
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
