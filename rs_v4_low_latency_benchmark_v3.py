#!/usr/bin/env python3
"""Ranked Surfers V4 optimized controller / menu/active-gameplay-verified startup and performance benchmark.

Nothing here saves/replaces a trained PPO model or modifies existing scripts.
It keeps the V3 reward/death/restart mechanics as-is; this is a speed/memory
and observation-validation experiment, NOT a new winning agent.

Modes:
  preview: Show precisely which pixels game-only cropping keeps, using already
           collected ORIGINAL native gameplay screenshots.
  check:   Offline real SB3 MultiInputPolicy learning update with fake images;
           catches resolution/GPU-memory/integration errors without browser.
  live:    Train a THROWAWAY new PPO model for a few rollouts in live Practice,
           with native game crop -> 4 RGB frames, and asynchronous YOLO on
           the same cropped original-resolution frames. No checkpoint saved.

Run from project root with active Windows venv. See README_V4_FULL_BENCH.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
import numpy as np

DEFAULT_CROP = (0.0, 0.055, 1.0, 0.965)  # normalized L,T,R,B; preview/adjust!
NFRAMES = 4


def shape(value):
    try:
        w, h = (int(x) for x in value.lower().split('x'))
        if w < 128 or h < 72 or w > 1920 or h > 1080 or w % 16 or h % 8:
            raise ValueError()
        return w, h
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('Use WxH (multiples 16x8), e.g. 512x288')


def validate_crop(crop):
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError('Crop must be normalized left top right bottom, 0..1')
    if (right-left) < .50 or (bottom-top) < .50:
        raise ValueError('Crop excludes over half the screen. Check values.')
    return tuple(float(x) for x in crop)


def crop_rect(frame_shape, crop):
    height, width = frame_shape[:2]
    l, t, r, b = crop
    x1, y1 = round(l*width), round(t*height)
    x2, y2 = round(r*width), round(b*height)
    return x1, y1, x2, y2


def crop_game(frame, crop):
    x1, y1, x2, y2 = crop_rect(frame.shape, crop)
    return frame[y1:y2, x1:x2]


def fit_rgb_to_target(native_rgb, size, interpolation=cv2.INTER_AREA):
    """Letterbox instead of stretching the perspective/obstacle geometry."""
    target_w, target_h = size
    h, w = native_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    small_w = max(1, min(target_w, round(w * scale)))
    small_h = max(1, min(target_h, round(h * scale)))
    scaled = cv2.resize(native_rgb, (small_w, small_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w-small_w)//2
    y = (target_h-small_h)//2
    canvas[y:y+small_h, x:x+small_w] = scaled
    return canvas


def image_stats(frame):
    return {'shape': list(frame.shape), 'mean': round(float(frame.mean()), 2),
            'std': round(float(frame.std()), 2)}


def as_report(values):
    if not values:
        return {'count': 0}
    a = np.asarray(values, dtype=float)
    return {'count': int(len(a)), 'median_ms': round(float(np.median(a)), 2),
            'p95_ms': round(float(np.percentile(a,95)), 2),
            'mean_ms': round(float(np.mean(a)), 2)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def _draw_crop_preview(bgr, crop, title):
    vis = bgr.copy()
    x1, y1, x2, y2 = crop_rect(bgr.shape, crop)
    cv2.rectangle(vis, (x1,y1), (x2-1,y2-1), (0,220,255), max(3,bgr.shape[1]//500))
    cv2.putText(vis, title, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,220,255), 2, cv2.LINE_AA)
    return vis


def preview(a):
    src_dir = a.frames_dir
    paths = sorted(src_dir.glob('*.jpg')) + sorted(src_dir.glob('*.png'))
    if not paths:
        raise SystemExit(f'No frames in {src_dir}. Use your V4 capture V2 native_frames folder.')
    a.output.mkdir(parents=True, exist_ok=True)
    paths = paths[:a.preview_frames]
    report = {'crop': a.crop, 'frames': [], 'note': 'Yellow line shows retained game crop. Verify visually.'}
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        x1,y1,x2,y2 = crop_rect(bgr.shape, a.crop)
        cut = crop_game(bgr, a.crop)
        cv2.imwrite(str(a.output / f'{i:02d}_native_with_crop.jpg'),
                    _draw_crop_preview(bgr, a.crop, 'YELLOW OUTLINE = GAME-ONLY ROI'),
                    [cv2.IMWRITE_JPEG_QUALITY, 91])
        cv2.imwrite(str(a.output / f'{i:02d}_game_crop.jpg'), cut,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        for size in a.resolutions:
            w,h = size
            small = fit_rgb_to_target(cut, (w,h))
            # Upscale only for human visual inspection; actual model sees w x h.
            cv2.imwrite(str(a.output / f'{i:02d}_ppo_{w}x{h}.jpg'),
                        cv2.resize(small, (768,432), interpolation=cv2.INTER_NEAREST))
        report['frames'].append({'source':str(p), 'native_wh':[bgr.shape[1],bgr.shape[0]],
                                 'crop_xyxy':[x1,y1,x2,y2],
                                 'crop_wh':[x2-x1,y2-y1], 'pixels':image_stats(cut)})
    write_json(a.output / 'crop_preview_report.json', report)
    print(f'[PREVIEW] Cropped {len(report["frames"])} real frames -> {a.output}', flush=True)
    print('[PREVIEW] Verify all gameplay content remains visible and browser/taskbar are gone.',flush=True)
    if report['frames']:
        print('[PREVIEW] Crop rectangle:',report['frames'][0]['crop_xyxy'],flush=True)


def make_v3_ppo(env, a):
    from train_rs_v3_ppo import RGBHazardExtractor
    from stable_baselines3 import PPO
    import torch
    policy_args = dict(features_extractor_class=RGBHazardExtractor,
                       features_extractor_kwargs={'features_dim':192},
                       net_arch={'pi':[64], 'vf':[64]})
    return PPO('MultiInputPolicy', env, device='cuda' if torch.cuda.is_available() else 'cpu',
               n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs,
               learning_rate=3e-4, gamma=.99, gae_lambda=.95, ent_coef=.02,
               clip_range=.2, verbose=0, seed=123, policy_kwargs=policy_args)


def offline_check(a):
    """Full PPO update, at requested RGB resolution, no game or YOLO required."""
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    from train_rs_v3_ppo import AUX_DIM

    class FakeFrames(gym.Env):
        metadata = {'render_modes':[]}
        def __init__(self):
            self.observation_space = spaces.Dict({
                'image':spaces.Box(0,255,shape=(12,a.resolution[1],a.resolution[0]),dtype=np.uint8),
                'aux':spaces.Box(0.,1.,shape=(AUX_DIM,),dtype=np.float32),
            })
            self.action_space=spaces.Discrete(5)
            self.rng=np.random.default_rng(123)
            self.t=0
        def _obs(self):
            # Structured, identical-shaped visual data; tests software not skill.
            return {'image':self.rng.integers(0,256,self.observation_space['image'].shape,dtype=np.uint8),
                    'aux':np.zeros(AUX_DIM,np.float32)}
        async def _capture(self):
            """Preserve menu detection and base-env shape, avoid unused INTER_AREA RGB resize.

            LocalPracticeEnv still requires a 3x144x256 frame for its legacy
            four-frame deque. V4 PPO NEVER consumes this frame: _pack reads
            native BGRA and builds its own 12xHxW observation from game crop.
            No changes to reward, death detector or key behavior here.
            """
            if self.capture_mode == 'legacy':
                return await super()._capture()
            t0=time.perf_counter()
            await self._sync_capture_region()
            t1=time.perf_counter()
            self.capture_alignment_ms.append((t1-t0)*1000)
            t0=t1
            shot=np.asarray(self.grabber.grab(self.region))
            t1=time.perf_counter()
            self.capture_grab_ms.append((t1-t0)*1000)
            self._metric('mss_grab',t1-t0)
            self._last_bgra=shot
            self._v2_last_bgra=shot
            t0=t1
            score=float(self._fast_menu_score(shot))
            t1=time.perf_counter()
            self.capture_menu_ms.append((t1-t0)*1000)
            self._metric('menu_detect',t1-t0)
            t0=t1
            # INTER_LINEAR is used only for the throwaway compatibility image;
            # the REAL 512x288 PPO image retains configurable AREA/LINEAR.
            shrunk=cv2.resize(shot,(256,144),interpolation=cv2.INTER_LINEAR)
            rgb=cv2.cvtColor(shrunk,cv2.COLOR_BGRA2RGB)
            chw=np.ascontiguousarray(rgb.transpose(2,0,1))
            self._v2_last_chw=chw
            t1=time.perf_counter()
            self.capture_compat_ms.append((t1-t0)*1000)
            self._metric('image_resize',t1-t0)
            self.last_capture_end=t1
            return chw,score,None

        @staticmethod
        def _motion_view(bgra,crop):
            """Low-cost motion audit in the central track, not browser/HUD."""
            game=crop_game(bgra,crop)
            h,w=game.shape[:2]
            track=game[int(h*.32):int(h*.86),int(w*.27):int(w*.73)]
            tiny=cv2.resize(track,(128,96),interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(tiny,cv2.COLOR_BGRA2GRAY)

        async def _start_episode(self):
            """Never release movement actions on the transitional loading image.

            The inherited R restart considers two non-menu captures 'gameplay',
            which includes loading screens. We wait for real central scene motion
            AFTER an explicit grace period, with a hard fail-closed timeout.
            If an actual menu is detected, R is retried before returning.
            """
            begin=time.perf_counter()
            frame,raw=await super()._start_episode()
            deadline=time.perf_counter()+self.startup_timeout_s
            grace_s=(self.startup_grace_s if self.episode_number==1
                     else self.restart_grace_s)
            release_after=time.perf_counter()+grace_s
            prev_view=None
            motion_votes=0
            menu_votes=0
            retries_this_episode=0
            last_score=None
            last_motion=0.0
            print(f'[START GUARD] R transition observed; suppressing arrows for >= '
                  f'{grace_s:.2f}s until scene motion is confirmed.',flush=True)
            while time.perf_counter()<deadline:
                await asyncio.sleep(.10)
                frame,score,_=await self._capture()
                last_score=float(score)
                bgra=self._v2_last_bgra
                if bgra is None:
                    continue
                view=self._motion_view(bgra,self.crop)
                last_motion=(float(np.mean(cv2.absdiff(view,prev_view)))
                             if prev_view is not None else 0.0)
                prev_view=view
                self.startup_motion_samples.append(last_motion)
                # The menu is a strong visual stop signal, even if R was
                # incorrectly acknowledged during loading.
                if score>=self.args.threshold*.8:
                    menu_votes+=1
                    motion_votes=0
                else:
                    menu_votes=0
                    if (time.perf_counter()>=release_after and
                        score<self.args.threshold*.55 and
                        last_motion>=self.startup_motion_threshold and
                        float(view.std())>=12.0):
                        motion_votes+=1
                    else:
                        motion_votes=0
                if menu_votes>=2:
                    # Returned to menu before a confirmed running frame.
                    retries_this_episode+=1
                    self.startup_retries+=1
                    print(f'[START GUARD] Menu observed after R (score={score:.2f}); retrying R '
                          f'({retries_this_episode}/4).',flush=True)
                    if retries_this_episode>4:
                        break
                    frame=await self._press_r_restart()
                    deadline=time.perf_counter()+self.startup_timeout_s
                    release_after=time.perf_counter()+grace_s
                    motion_votes=0
                    menu_votes=0
                    prev_view=None
                    continue
                if motion_votes>=2:
                    waited_ms=(time.perf_counter()-begin)*1000
                    self.startup_guard_ms.append(waited_ms)
                    print(f'[START GUARD] Gameplay motion confirmed; '
                          f'hold={waited_ms:.0f}ms, motion={last_motion:.2f}, menu={score:.2f}.',flush=True)
                    # Save episode-1 preview AFTER gameplay is confirmed.
                    return frame,(self._encode_last_frame() if self.episode_number==1 else None)
            self.startup_failures+=1
            path=self.output_dir/f'startup_guard_failed_ep{self.episode_number:04d}.jpg'
            if self._v2_last_bgra is not None:
                cv2.imwrite(str(path),cv2.cvtColor(self._v2_last_bgra,cv2.COLOR_BGRA2BGR))
            raise RuntimeError(f'[START GUARD] Could not confirm actual moving gameplay '
                               f'within {self.startup_timeout_s:.1f}s (menu={last_score}, '
                               f'motion={last_motion:.2f}). Saved {path}. '
                               'No movement keys were sent.')

        def reset(self,*,seed=None,options=None):
            self.t=0
            return self._obs(),{}
        def step(self, action):
            self.t+=1
            return self._obs(), float(self.t%5==int(action)), self.t>=25, False, {}

    print(f'[CHECK] Full V3 PPO policy at {a.resolution[0]}x{a.resolution[1]}, '
          f'rollout {a.n_steps}, batch {a.batch_size}, epochs {a.n_epochs}',flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    env=FakeFrames()
    model=make_v3_ppo(env,a)
    model.learn(total_timesteps=a.n_steps, progress_bar=False)
    elapsed=time.perf_counter()-t0
    mem_gpu=round(torch.cuda.max_memory_reserved()/1024**2,1) if torch.cuda.is_available() else None
    report={'resolution':a.resolution, 'elapsed_s':round(elapsed,2),
            'rollout_steps':a.n_steps,'gpu_peak_reserved_mib':mem_gpu,
            'image_rollout_estimate_mib':round(a.n_steps*12*a.resolution[0]*a.resolution[1]*4/1024**2,1)}
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'offline_check_report.json',report)
    print('[CHECK] PASS '+json.dumps(report),flush=True)


def build_live_env_class():
    """Deferred imports: preview works without Playwright/UI/GPU."""
    import asyncio
    import gymnasium as gym
    from gymnasium import spaces
    import train_rs_v3_ppo as v3
    from rs_v2_live_hazards_v2 import hazard_features, detect

    class NativeCroppedV4Env(v3.RGBHazardPracticeEnv):
        def __init__(self, cfg, output, demos, detector, *, a):
            self.target_wh=a.resolution
            self.capture_mode=a.capture_mode
            self.resize_method=a.resize_method
            self.startup_grace_s=a.startup_grace
            self.restart_grace_s=a.restart_grace
            self.startup_timeout_s=a.startup_timeout
            self.menu_ready_timeout_s=a.menu_ready_timeout
            self.startup_motion_threshold=a.startup_motion_threshold
            self.capture_grab_ms=[]
            self.capture_menu_ms=[]
            self.capture_compat_ms=[]
            self.capture_alignment_ms=[]
            self.capture_end_to_pack_ms=[]
            self.pack_ms=[]
            self.startup_guard_ms=[]
            self.startup_retries=0
            self.startup_failures=0
            self.startup_motion_samples=[]
            self.first_decision_at=None
            self.last_decision_at=None
            self.crop=a.crop
            self.rate_mode=a.rate_mode
            self.target_period=1.0/a.fps
            self._prev_step_start=None
            self.native_history=deque(maxlen=NFRAMES)
            self.native_image_wh=None
            self.native_sample=None
            self.game_crop_ms=[]
            self.step_elapsed_ms=[]
            self.actual_press_interval_ms=[]
            self.step_started_times=[]
            self.reset_times_ms=[]
            self.last_source_native=None
            self._raw_source_save=False
            self.saved_overlay_count=0
            self._saved_episode_tags=set()
            self._preview_delay_s=a.preview_delay
            self._preview_dir=output/'matched_detection_previews'
            self._preview_dir.mkdir(exist_ok=True)
            self._preview_limit=a.save_overlays
            self.training_steps=0
            self.experienced_deaths=0
            self.experienced_timeouts=0
            self.action_counts=Counter()
            super().__init__(cfg, output, demos, detector,
                             detect_every=a.detect_every, imgsz=a.imgsz,
                             conf=a.conf, max_stale_ms=a.max_stale_ms,
                             press_delay_ms=a.press_delay_ms)
            w,h=self.target_wh
            self.observation_space=spaces.Dict({
                'image':spaces.Box(low=0,high=255,shape=(12,h,w),dtype=np.uint8),
                'aux':spaces.Box(low=0.,high=1.,shape=(v3.AUX_DIM,),dtype=np.float32),
            })

        async def _capture(self):
            """Preserve menu detection and base-env shape, avoid unused INTER_AREA RGB resize.

            LocalPracticeEnv still requires a 3x144x256 frame for its legacy
            four-frame deque. V4 PPO NEVER consumes this frame: _pack reads
            native BGRA and builds its own 12xHxW observation from game crop.
            No changes to reward, death detector or key behavior here.
            """
            if self.capture_mode == 'legacy':
                return await super()._capture()
            t0=time.perf_counter()
            await self._sync_capture_region()
            t1=time.perf_counter()
            self.capture_alignment_ms.append((t1-t0)*1000)
            t0=t1
            shot=np.asarray(self.grabber.grab(self.region))
            t1=time.perf_counter()
            self.capture_grab_ms.append((t1-t0)*1000)
            self._metric('mss_grab',t1-t0)
            self._last_bgra=shot
            self._v2_last_bgra=shot
            t0=t1
            score=float(self._fast_menu_score(shot))
            t1=time.perf_counter()
            self.capture_menu_ms.append((t1-t0)*1000)
            self._metric('menu_detect',t1-t0)
            t0=t1
            # INTER_LINEAR is used only for the throwaway compatibility image;
            # the REAL 512x288 PPO image retains configurable AREA/LINEAR.
            shrunk=cv2.resize(shot,(256,144),interpolation=cv2.INTER_LINEAR)
            rgb=cv2.cvtColor(shrunk,cv2.COLOR_BGRA2RGB)
            chw=np.ascontiguousarray(rgb.transpose(2,0,1))
            self._v2_last_chw=chw
            t1=time.perf_counter()
            self.capture_compat_ms.append((t1-t0)*1000)
            self._metric('image_resize',t1-t0)
            self.last_capture_end=t1
            return chw,score,None

        @staticmethod
        def _motion_view(bgra,crop):
            """Low-cost motion audit in the central track, not browser/HUD."""
            game=crop_game(bgra,crop)
            h,w=game.shape[:2]
            track=game[int(h*.32):int(h*.86),int(w*.27):int(w*.73)]
            tiny=cv2.resize(track,(128,96),interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(tiny,cv2.COLOR_BGRA2GRAY)

        @staticmethod
        def _meter_mask(bgra):
            """Locate bright-yellow gameplay distance digits in the fullscreen HUD.

            This intentionally reads the RAW screen, not the gameplay crop:
            our game-only crop removes some of the top HUD. Works for the
            local Ranked Surfers HUD placement; fail closed if it changes.
            """
            h, w = bgra.shape[:2]
            # Normalized from the game's native 1920x1080 fullscreen capture:
            # yellow distance number and trailing 'm' around x=0.11..0.14,
            # y=0.02..0.036. Not dependent on screenshot downscaling.
            roi = bgra[int(h*.011):int(h*.048), int(w*.101):int(w*.151), :3]
            if roi.size == 0:
                return np.zeros((24,64),dtype=np.uint8)
            hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([16,40,135]), np.array([46,255,255]))
            # fixed-size shape so cosmetic resizing has limited effect
            return cv2.resize(mask,(64,24),interpolation=cv2.INTER_NEAREST)

        async def _await_ready_menu(self):
            """DO NOT send the first R while Unity is still loading.

            On each restart, wait for two visible 'FIND RANKED MATCH' scores.
            The calibrated Windows template supplies this signal. A timer on
            its own is not sufficient; animated loading can outlast it.
            """
            deadline=time.perf_counter()+self.menu_ready_timeout_s
            seen=0
            score=0.0
            while time.perf_counter()<deadline:
                _,score,_=await self._capture()
                seen=(seen+1 if score>=self.args.threshold*.87 else 0)
                if seen>=2:
                    print(f'[START STATE] MENU_READY (score={score:.2f}); now R is allowed.',flush=True)
                    if self.episode_number<=2 and self._v2_last_bgra is not None:
                        self.output_dir.mkdir(parents=True,exist_ok=True)
                        cv2.imwrite(str(self.output_dir/f'state_ep{self.episode_number:04d}_menu_ready.jpg'),
                                    cv2.cvtColor(self._v2_last_bgra,cv2.COLOR_BGRA2BGR))
                    return
                await asyncio.sleep(.11)
            self.startup_failures+=1
            self.output_dir.mkdir(parents=True,exist_ok=True)
            path=self.output_dir/f'menu_ready_failed_ep{self.episode_number:04d}.jpg'
            if self._v2_last_bgra is not None:
                cv2.imwrite(str(path),cv2.cvtColor(self._v2_last_bgra,cv2.COLOR_BGRA2BGR))
            raise RuntimeError(f'[START STATE] Menu was not confirmed within '
                               f'{self.menu_ready_timeout_s:.1f}s (score={score:.2f}). '
                               f'No R or arrow keys sent. Saved {path}')

        @staticmethod
        def _pause_icon_fraction(bgra):
            """Positive gameplay signal: bright blue PAUSE icon in top-right corner.

            Avoid using the distance digits as a mandatory condition: typography,
            HUD alignment and color differ across machines/frames. We check for
            a blue icon *and* a moving gameplay scene while the death menu is absent.
            """
            h, w = bgra.shape[:2]
            roi = bgra[int(h*.006):int(h*.075), int(w*.959):int(w*.999), :3]
            if roi.size == 0:
                return 0.0
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            return float(np.count_nonzero(cv2.inRange(hsv,
                        np.array([90,95,65]),np.array([135,255,255])))) / (roi.shape[0]*roi.shape[1])

        async def _start_episode(self):
            """Menu -> R -> visually verified live gameplay; never use HUD digit change.

            Unlike v2, requires a blue top-right pause icon and central scene
            movement in successive captures. This is a screen-based heuristic;
            diagnostic frames are saved for easy calibration if Windows differs.
            """
            import asyncio
            began = time.perf_counter()
            await self._await_ready_menu()
            frame, raw = await super()._start_episode()
            print('[START STATE V3] R_SENT; checking live gameplay, NO arrows yet.',flush=True)
            grace = self.startup_grace_s if self.episode_number == 1 else self.restart_grace_s
            deadline = time.perf_counter() + self.startup_timeout_s
            grace_end = time.perf_counter() + grace
            prev_view = None
            votes = 0
            menu_votes = 0
            retries = 0
            peak_blue = 0.0
            peak_motion = 0.0
            latest_score = 0.0
            max_meter = 0
            last_trace = -1.0
            trace_dir = self.output_dir / f'startup_trace_ep{self.episode_number:04d}'
            if self.episode_number <= 2:
                trace_dir.mkdir(parents=True,exist_ok=True)
            while time.perf_counter() < deadline:
                await asyncio.sleep(.065)
                frame, score, _ = await self._capture()
                bgra = self._v2_last_bgra
                if bgra is None:
                    continue
                latest_score = float(score)
                blue = self._pause_icon_fraction(bgra)
                peak_blue = max(peak_blue, blue)
                view = self._motion_view(bgra, self.crop)
                motion = (float(np.mean(cv2.absdiff(view,prev_view)))
                          if prev_view is not None else 0.0)
                peak_motion = max(motion,peak_motion)
                prev_view = view
                meter_count = int(cv2.countNonZero(self._meter_mask(bgra)))
                max_meter = max(meter_count,max_meter)
                elapsed = time.perf_counter()-began
                # Save the *sequence*, not just the last menu frame. This is
                # the missing evidence if the heuristic fails on Windows.
                if self.episode_number <= 2 and (elapsed-last_trace)>.30:
                    last_trace=elapsed
                    dest=trace_dir/f'{int(elapsed*1000):05d}ms_menu{score:.2f}_blue{blue:.2f}_motion{motion:.1f}.jpg'
                    cv2.imwrite(str(dest),cv2.cvtColor(bgra,cv2.COLOR_BGRA2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY,80])
                    print(f'[START STATE V3] t={elapsed:.2f}s '
                          f'menu={score:.2f} pause_blue={blue:.3f} '
                          f'motion={motion:.2f} dist_pixels={meter_count} '
                          f'votes={votes}',flush=True)
                menu_on = score >= self.args.threshold*.80
                if menu_on:
                    menu_votes += 1
                    votes = 0
                else:
                    menu_votes=0
                    # Positive gameplay check: pause icon is visible AND the
                    # scene moves between snapshots. Distance pixels are
                    # logged but NOT a mandatory gate.
                    active = (blue >= .055 and motion >= self.startup_motion_threshold
                              and float(view.std()) >= 12.0
                              and time.perf_counter() >= grace_end)
                    votes = votes + 1 if active else 0
                if votes >= 2:
                    waited_ms = (time.perf_counter()-began)*1000
                    self.startup_guard_ms.append(waited_ms)
                    print(f'[START STATE V3] GAMEPLAY_CONFIRMED in {waited_ms:.0f}ms '
                          f'(blue={blue:.3f} motion={motion:.2f} menu={score:.2f}). '
                          f'Arrow input now enabled.',flush=True)
                    if self.episode_number<=2:
                        cv2.imwrite(str(self.output_dir/f'state_ep{self.episode_number:04d}_gameplay.jpg'),
                                    cv2.cvtColor(bgra,cv2.COLOR_BGRA2BGR))
                    return frame,(self._encode_last_frame() if self.episode_number==1 else None)
                if menu_votes>=2:
                    retries += 1
                    self.startup_retries += 1
                    if retries > 2:
                        break
                    print('[START STATE V3] Menu returned before gameplay confirmation; '
                          f'retrying R ({retries}/2).',flush=True)
                    await self._press_r_restart()
                    grace_end=time.perf_counter()+grace
                    deadline=time.perf_counter()+self.startup_timeout_s
                    prev_view=None
                    votes=menu_votes=0
            self.startup_failures += 1
            path=self.output_dir/f'startup_guard_failed_ep{self.episode_number:04d}.jpg'
            if self._v2_last_bgra is not None:
                cv2.imwrite(str(path),cv2.cvtColor(self._v2_last_bgra,cv2.COLOR_BGRA2BGR))
            raise RuntimeError(f'[START STATE V3] Gameplay not confirmed after '
                               f'{self.startup_timeout_s:.1f}s; '
                               f'last_menu={latest_score:.2f}, '
                               f'peak_pause_blue={peak_blue:.3f}, '
                               f'peak_motion={peak_motion:.2f}, '
                               f'max_dist_pixels={max_meter}. '
                               f'NO arrow keys sent. Inspect {trace_dir} and {path}')

        def reset(self,*,seed=None,options=None):
            began=time.perf_counter()
            self.native_history.clear()
            self._prev_step_start=None
            obs,info=super().reset(seed=seed,options=options)
            self.reset_times_ms.append((time.perf_counter()-began)*1000)
            return obs,info

        def _native_frames(self):
            shot=self._v2_last_bgra
            if shot is None or shot.ndim!=3 or shot.shape[2]!=4:
                raise RuntimeError('No BGRA source from V2 capture; aborting benchmark')
            began=time.perf_counter()
            box=crop_game(shot,self.crop)
            if not box.size:
                raise RuntimeError('Empty game crop')
            self.native_image_wh=(int(box.shape[1]),int(box.shape[0]))
            # Only crop and resize once per observation. YOLO gets native cropped RGB.
            native_rgb=cv2.cvtColor(box,cv2.COLOR_BGRA2RGB)
            interp=cv2.INTER_AREA if self.resize_method=='area' else cv2.INTER_LINEAR
            small=fit_rgb_to_target(native_rgb,self.target_wh,interpolation=interp)
            chw=np.ascontiguousarray(small.transpose(2,0,1))
            self.game_crop_ms.append((time.perf_counter()-began)*1000)
            self.last_source_native=native_rgb
            return chw,native_rgb

        def _pack(self, rgb_obs, *, start_detection=False):
            pack_begin=time.perf_counter()
            # Preserve the original PPO auxiliary feature semantics; only the
            # RGB source/resolution and YOLO SOURCE images are different.
            self._poll_result()
            frame,native=self._native_frames()
            if not self.native_history:
                for _ in range(NFRAMES):
                    self.native_history.append(frame.copy())
            else:
                self.native_history.append(frame)
            packed=np.concatenate(tuple(self.native_history),axis=0)
            now=time.perf_counter()
            source_at=self.last_capture_end or now
            if self.previous_observation_at is not None:
                self.frame_intervals.append(max(0.,source_at-self.previous_observation_at))
            self.previous_observation_at=source_at
            if start_detection and self.detector_pool is not None:
                if self.job is None:
                    self.job=self.detector_pool.submit(
                        detect,self.detector_model,native.copy(),source_at,
                        self.episode_tag,self.detector_requests,self.imgsz,self.conf)
                    self.detector_requests+=1
                else:
                    self.detector_skipped_busy+=1
            hazard,age_ms,fresh=hazard_features(self.last_detection,now=now,max_age_s=self.max_stale_s)
            if fresh:
                self.fresh_steps+=1
                self.decision_ages_ms.append(age_ms)
            self.all_steps+=1
            aux=v3.form_aux(hazard,self.action_history,now-self.episode_started,
                            self.frame_intervals,age_ms)
            self.pack_ms.append((time.perf_counter()-pack_begin)*1000)
            return {'image':packed,'aux':aux},fresh,age_ms

        def _write_sample(self, info, terminated):
            if (self._preview_limit<=0 or self.saved_overlay_count>=self._preview_limit or
                self.episode_tag in self._saved_episode_tags or terminated or
                time.perf_counter()-self.episode_started < self._preview_delay_s or
                float(info.get('menu_similarity',0.))>self.args.threshold*.5):
                return
            snap=self.last_detection
            if (snap is None or snap['episode']!=self.episode_tag or
                not snap['detections'] or time.perf_counter()-snap['captured_at']>.3):
                return
            # Matched exact source, not boxes on a different video frame.
            path=self._preview_dir/f'{self.saved_overlay_count:02d}_detected.jpg'
            # Overlay helper triples image size, so produce native crop source
            # ourselves at a usable size for user inspection.
            native_rgb=snap['source_frame']
            vis=cv2.cvtColor(native_rgb,cv2.COLOR_RGB2BGR)
            h,w=vis.shape[:2]
            cv2.imwrite(str(self._preview_dir/f'{self.saved_overlay_count:02d}_source.jpg'),vis,
                        [cv2.IMWRITE_JPEG_QUALITY,94])
            for d in snap['detections']:
                x1,y1,x2,y2=d['box']
                c={'coin':(0,210,255),'train':(220,130,20),'barrier':(20,40,220)}.get(d['class'],(255,255,255))
                p1=(int(x1*w),int(y1*h)); p2=(int(x2*w),int(y2*h))
                cv2.rectangle(vis,p1,p2,c,2)
                cv2.putText(vis,f"{d['class']} {d['conf']:.2f}",
                            (p1[0],max(15,p1[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,c,2)
            cv2.imwrite(str(path),vis,[cv2.IMWRITE_JPEG_QUALITY,94])
            self._saved_episode_tags.add(self.episode_tag)
            self.saved_overlay_count+=1
            print(f'[OVERLAY] Saved exact-source gameplay pair #{self.saved_overlay_count} '
                  f'from episode {self.episode_tag}',flush=True)

        def step(self,action):
            # Limit decisions ONLY if ahead of schedule; don't introduce the
            # legacy per-step sleep. Actual FPS is measured separately.
            if self.rate_mode=='capped' and self._prev_step_start is not None:
                remaining=self.target_period-(time.perf_counter()-self._prev_step_start)
                if remaining>0:
                    time.sleep(remaining)
            started=time.perf_counter()
            self._prev_step_start=started
            obs,reward,dead,trunc,info=super().step(action)
            self.step_elapsed_ms.append((time.perf_counter()-started)*1000)
            if self.step_started_times:
                self.actual_press_interval_ms.append((started-self.step_started_times[-1])*1000)
            self.step_started_times.append(started)
            if self.first_decision_at is None:
                self.first_decision_at=started
            self.last_decision_at=started
            self.training_steps+=1
            self.action_counts[int(action)]+=1
            self.experienced_deaths+=bool(dead)
            self.experienced_timeouts+=bool(trunc)
            self._write_sample(info,dead or trunc)
            return obs,reward,dead,trunc,info

    return NativeCroppedV4Env


def live(a):
    import torch
    from stable_baselines3.common.monitor import Monitor
    import train_rs_v2_verify as v2
    from train_rs_v3_ppo import v2_args
    from rs_v2_live_hazards_v2 import detect
    from ultralytics import YOLO

    # Nothing from V3 is overwritten. Create a fresh, disposable PPO policy.
    run_dir=v2.RUN_ROOT/a.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f'Output exists: {run_dir}. Choose a new --run-name.')
    run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/'benchmark_settings.json',vars(a))
    if a.torch_threads>0:
        torch.set_num_threads(a.torch_threads)
        print(f'[SPEED] torch CPU threads={torch.get_num_threads()}',flush=True)
    detector=None if a.no_yolo else YOLO(str(a.detector))
    print(f'[LIVE] Torch {torch.__version__}; CUDA={torch.cuda.is_available()}',flush=True)
    # Cold YOLO initialization outside timings; same crop-ish aspect ratio.
    if detector is not None:
        _=detect(detector,np.zeros((576,1024,3),dtype=np.uint8),time.perf_counter(),0,0,a.imgsz,a.conf)
    import argparse as _argparse
    fake=_argparse.Namespace(url='http://localhost:8000',fps=a.fps,
                             max_minutes=a.max_minutes,fullscreen=True)
    cfg=v2_args(fake)
    cfg.episode_max_seconds=15  # Watchdog for missed death, identical across sizes.
    Env=build_live_env_class()
    demos=v2.DemoStore(v2.DEMO_ROOT,max_samples=100,load_existing=False)
    raw_env=None
    monitored=None
    model=None
    measured_train_ms=[]
    policy_forward_dispatch_ms=[]
    started=time.perf_counter()
    peak_process_rss_mib=None
    try:
        import psutil
        process=psutil.Process()
    except ImportError:
        process=None
    successful=False
    try:
        raw_env=Env(cfg,run_dir,demos,detector,a=a)
        monitored=Monitor(raw_env,filename=str(run_dir/'episodes.monitor.csv'))
        model=make_v3_ppo(monitored,a)
        # Match V3 multimodal model, train the policy/critic as well as encoder.
        original_forward=model.policy.forward
        def timed_forward(*args,**kwargs):
            t=time.perf_counter()
            ret=original_forward(*args,**kwargs)
            policy_forward_dispatch_ms.append((time.perf_counter()-t)*1000)
            return ret
        model.policy.forward=timed_forward
        original_train=model.train
        def timed_train():
            began=time.perf_counter()
            result=original_train()
            took=(time.perf_counter()-began)*1000
            measured_train_ms.append(took)
            print(f'[PPO UPDATE] {took:.1f}ms (complete model + actor/critic)',flush=True)
            return result
        model.train=timed_train
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f'[LIVE] PPO {a.resolution[0]}x{a.resolution[1]}, '
              f'crop={a.crop}, YOLO source=native GAME CROP, imgsz={a.imgsz}, '
              f'detect_every={a.detect_every}, target_fps={a.fps}, '
              f'rate_mode={a.rate_mode}, total_steps={a.steps}, '
              f'capture_mode={a.capture_mode}, resize={a.resize_method}, '
              f'startup_grace={a.startup_grace}, restart_grace={a.restart_grace}',flush=True)
        class TimeGuard:
            def __init__(self):
                from stable_baselines3.common.callbacks import BaseCallback
                class Callback(BaseCallback):
                    def __init__(self,limit):
                        super().__init__()
                        self.limit=limit
                        self.began=time.perf_counter()
                    def _on_step(self):
                        nonlocal peak_process_rss_mib
                        if process is not None and self.n_calls % 16 == 0:
                            mib=process.memory_info().rss/1024**2
                            peak_process_rss_mib=max(peak_process_rss_mib or 0,mib)
                        return time.perf_counter()-self.began<self.limit
                self.callback=Callback(a.max_minutes*60)
        cb=TimeGuard().callback
        model.learn(total_timesteps=a.steps,callback=cb,progress_bar=False)
        successful=True
    except KeyboardInterrupt:
        print('[LIVE] Interrupted; closing game and writing partial report.',flush=True)
    finally:
        wall=max(.001,time.perf_counter()-started)
        if process is not None:
            peak_process_rss_mib=max(peak_process_rss_mib or 0,process.memory_info().rss/1024**2)
        if raw_env is not None:
            det_report=raw_env.hazard_report()
            report={
                'note':'Real local Practice full V3 PPO updates, no checkpoint saved. '
                       'Start menu and distance-HUD progression must confirm gameplay before arrows; preprocessor is optional. '
                       'Not an evaluation of game-playing skill.',
                'capture_mode':a.capture_mode, 'resize_method':a.resize_method,
                'torch_threads':torch.get_num_threads(),
                'menu_ready_timeout_s':a.menu_ready_timeout,
                'startup_grace_s':a.startup_grace,
                'restart_grace_s':a.restart_grace,
                'startup_guard_wait_ms':as_report(raw_env.startup_guard_ms),
                'startup_guard_retries':raw_env.startup_retries,
                'startup_guard_failures':raw_env.startup_failures,
                'startup_motion_mean_abs_gray': (
                    round(float(np.mean(raw_env.startup_motion_samples)),2)
                    if raw_env.startup_motion_samples else None),
                'resolution_wh':a.resolution, 'crop_normalized':a.crop,
                'native_game_crop_wh':raw_env.native_image_wh,
                'fps_target':a.fps,'rate_mode':a.rate_mode,
                'complete':successful, 'wall_seconds':round(wall,2),
                'game_steps':raw_env.training_steps,
                'steps_per_second_including_resets_and_ppo_updates':round(raw_env.training_steps/wall,2),
                'steps_per_second_after_first_decision_including_updates': (
                    round((raw_env.training_steps-1) / max(.001,(raw_env.last_decision_at-raw_env.first_decision_at)),2)
                    if raw_env.training_steps>1 else None),
                'death_episodes':raw_env.experienced_deaths,
                'watchdog_timeouts':raw_env.experienced_timeouts,
                'action_counts':{['NONE','LEFT','RIGHT','JUMP','ROLL'][int(k)]:int(v)
                                 for k,v in raw_env.action_counts.items()},
                'capture_and_browser_step_ms':as_report(raw_env.step_elapsed_ms),
                'keypress_roundtrip_ms':as_report([s['press_roundtrip_ms']
                                                    for s in raw_env.step_timings if s.get('press')]),
                'action_browser_capture_ms':as_report([s['capture_ms']
                                                       for s in raw_env.step_timings]),
                'game_crop_resize_ms':as_report(raw_env.game_crop_ms),
                'capture_grab_ms':as_report(raw_env.capture_grab_ms),
                'capture_menu_ms':as_report(raw_env.capture_menu_ms),
                'capture_legacy_compat_image_ms':as_report(raw_env.capture_compat_ms),
                'capture_alignment_ms':as_report(raw_env.capture_alignment_ms),
                'observation_pack_ms':as_report(raw_env.pack_ms),
                'ppo_policy_forward_dispatch_ms':as_report(policy_forward_dispatch_ms),
                'decision_start_intervals_ms':as_report(raw_env.actual_press_interval_ms),
                'episode_resets_ms':as_report(raw_env.reset_times_ms),
                'ppo_complete_update_ms':as_report(measured_train_ms),
                'async_yolo':det_report,
                'yolo_enabled':not a.no_yolo,
                'process_peak_sampled_rss_mib':round(peak_process_rss_mib,1) if peak_process_rss_mib is not None else None,
                'gpu_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,1)
                                          if torch.cuda.is_available() else None,
                'gpu_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/1024**2,1)
                                          if torch.cuda.is_available() else None,
                'gpu_device_total_mib':round(torch.cuda.get_device_properties(0).total_memory/1024**2,1)
                                       if torch.cuda.is_available() else None,
                'image_rollout_estimate_mib':round(a.n_steps*12*a.resolution[0]*a.resolution[1]*4/1024**2,1),
                'matched_overlays':raw_env.saved_overlay_count,
            }
            write_json(run_dir/'v4_low_latency_report.json',report)
            print('[V4 RESULT] '+json.dumps(report,indent=2),flush=True)
            print(f'[LIVE] Report: {run_dir/"v4_low_latency_report.json"}',flush=True)
        if monitored is not None:
            monitored.close()
        elif raw_env is not None:
            raw_env.close()
    if not successful:
        print('[LIVE] Partial or failed benchmark; check report before comparing results.',flush=True)


def options():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='mode',required=True)
    for name in ('preview','check','live'):
        q=sub.add_parser(name)
        q.add_argument('--output',type=Path,default=Path('ai_data/v4_crop_preview') if name=='preview' else None)
        q.add_argument('--resolution',type=shape,default=(512,288))
        q.add_argument('--crop',type=float,nargs=4,metavar=('LEFT','TOP','RIGHT','BOTTOM'),
                       default=DEFAULT_CROP,help='Normalized fraction of original screen, default 0 .055 1 .965')
        if name=='preview':
            q.add_argument('--frames-dir',type=Path,
                           default=Path('ai_data/v4_resolution_gameplay_v2/native_frames'))
            q.add_argument('--resolutions',type=shape,nargs='+',default=[(384,216),(512,288)])
            q.add_argument('--preview-frames',type=int,default=4)
        else:
            q.add_argument('--n-steps',type=int,default=128)
            q.add_argument('--batch-size',type=int,default=32)
            q.add_argument('--n-epochs',type=int,default=2)
        if name=='live':
            q.add_argument('--run-name',default='v4_full_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
            q.add_argument('--fps',type=int,default=30)
            q.add_argument('--rate-mode',choices=['capped','uncapped'],default='capped')
            q.add_argument('--max-minutes',type=float,default=4)
            q.add_argument('--steps',type=int,default=256)
            q.add_argument('--detector',type=Path,default=Path('checkpoints/yolo_hazards_best.pt'))
            q.add_argument('--no-yolo',action='store_true', help='Run same PPO training without YOLO for an ablation')
            q.add_argument('--imgsz',type=int,default=640)
            q.add_argument('--conf',type=float,default=.20)
            q.add_argument('--detect-every',type=int,default=3)
            q.add_argument('--max-stale-ms',type=int,default=600)
            q.add_argument('--press-delay-ms',type=int,default=20)
            q.add_argument('--capture-mode',choices=['legacy','fast'],default='fast',
                           help='fast skips redundant expensive 256x144 INTER_AREA compatibility resize')
            q.add_argument('--resize-method',choices=['area','linear'],default='area',
                           help='PPO image downsampling; AREA higher quality, LINEAR faster; compare previews')
            q.add_argument('--torch-threads',type=int,default=1,
                           help='0=PyTorch default; 1 often lowers CPU dispatch latency on CUDA')
            q.add_argument('--startup-grace',type=float,default=.6,
                           help='Minimum seconds after preliminary R confirmation before movement')
            q.add_argument('--restart-grace',type=float,default=.15,
                           help='Minimum seconds after R on subsequent episodes; motion still required')
            q.add_argument('--menu-ready-timeout',type=float,default=25.,
                           help='Wait for 2 Windows-calibrated menu detections BEFORE first R')
            q.add_argument('--startup-timeout',type=float,default=8.,
                           help='Max seconds waiting for confirmed moving game per R attempt')
            q.add_argument('--startup-motion-threshold',type=float,default=2.2,
                           help='Minimum mean grayscale change in central track across samples')
            q.add_argument('--save-overlays',type=int,default=8)
            q.add_argument('--preview-delay',type=float,default=.9)
    a=p.parse_args()
    try:
        a.crop=validate_crop(a.crop)
    except ValueError as exc:
        p.error(str(exc))
    if a.mode in ('check','live'):
        if a.n_steps<16 or a.batch_size<1 or a.n_steps%a.batch_size!=0 or a.n_epochs<1:
            p.error('--n-steps >=16 and multiple of --batch-size; --n-epochs>=1')
        if a.output is None:
            a.output=Path('ai_data/v4_offline_'+str(a.resolution[0]))
    if a.mode=='live':
        if not (0 < a.max_minutes <= 120 and 0 < a.max_stale_ms <= 5000 and 1 <= a.detect_every <= 1000 and 1 <= a.press_delay_ms <= 100):
            p.error('Invalid max-minutes/max-stale-ms/detect-every/press-delay-ms')
        if not (0 <= a.torch_threads <= 32 and 0 <= a.startup_grace <= 3
                and 0 <= a.restart_grace <= 3 and .5 <= a.startup_timeout <= 20 and
                1 <= a.menu_ready_timeout <= 90 and
                0.1 <= a.startup_motion_threshold <= 30):
            p.error('Invalid torch-threads/startup-grace/restart-grace/startup-timeout/startup-motion-threshold')
        if a.fps<2 or a.fps>30 or a.steps<a.n_steps or a.steps%a.n_steps:
            p.error('--fps 2..30; --steps >= --n-steps and divisible by --n-steps')
        if not a.no_yolo and not a.detector.is_file():
            p.error(f'Missing detector: {a.detector}')
        if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):
            p.error('--run-name must be a simple folder name')
        if a.imgsz<160 or a.imgsz%32 or not 0<a.conf<1:
            p.error('YOLO --imgsz must be divisible by 32 and confidence in (0,1)')
    if a.mode=='preview':
        preview(a)
    elif a.mode=='check':
        offline_check(a)
    else:
        live(a)


if __name__=='__main__':
    options()
