"""Opt-in process-local adapter; never imported by production training."""
import asyncio
from collections import deque
import contextlib
import copy
import math
import threading
import time
import numpy as np


class ContextHistory:
    """Missing history is zero-filled with explicit validity and age fields."""
    def __init__(self, frames, span):
        if frames not in (4,6,7,8,9,10,11,12) or not math.isfinite(span) or not 0 < span <= 2:
            raise ValueError('Invalid temporal configuration')
        self.frames, self.span = frames, span
        self.lock = threading.Lock()
        self.items = deque(maxlen=256)
        self.enabled = False
        self.generation = 0
        self.after = float('-inf')

    def clear(self):
        with self.lock:
            self.items.clear()
            self.enabled = False
            self.generation += 1
            self.after = time.perf_counter()

    def token(self):
        with self.lock:
            return self.generation if self.enabled else None

    def add(self, frame, stamp, generation):
        with self.lock:
            if generation is None or not self.enabled or generation != self.generation or stamp < self.after:
                return
            if not math.isfinite(stamp):
                raise ValueError('Invalid temporal capture timestamp')
            if self.items and stamp <= self.items[-1][0]:
                raise ValueError('Nonmonotonic temporal capture')
            self.items.append((stamp,frame.copy()))
            while self.items and self.items[0][0] < stamp-self.span-.06:
                self.items.popleft()

    def observation(self, latest, stamp, *, enable=True):
        if not math.isfinite(stamp):
            raise ValueError('Invalid observation timestamp')
        with self.lock:
            past = [(t,f) for t,f in self.items if t < stamp]
            self.enabled = enable
        past.append((stamp,latest))
        times = np.array([t for t,_ in past])
        indices = np.searchsorted(times,stamp-np.linspace(self.span,0,self.frames),side='right')-1
        indices[-1] = len(past)-1
        images, ages, valid, used = [], [], [], set()
        for target,i in zip(stamp-np.linspace(self.span,0,self.frames),indices):
            good = i >= 0 and i not in used and target-times[i] <= .060000001
            images.append(past[i][1] if good else np.zeros_like(latest))
            ages.append(min(1.,max(0.,(stamp-times[i])/(self.span+.06))) if good else 1.)
            valid.append(float(good))
            if good: used.add(i)
        return np.concatenate(images),np.array(ages+valid,np.float32)


class CapturePump:
    """One capture in flight, one latest frame; no unbounded frame FIFO."""
    def __init__(self, source, callback, hz, token=lambda: None):
        if not math.isfinite(hz) or not 1 <= hz <= 120:
            raise ValueError('Capture Hz must be 1..120')
        self.source = source
        self.token = token
        self.callback = callback
        self.period = 1/hz
        self.condition = asyncio.Condition()
        self.wanted = asyncio.Event()
        self.latest = None
        self.error = None
        self.last_timing = {}
        self.delivery_serial = -1
        self.count = 0
        self.delivery_stale_discards = 0
        self.delivery_stale_max_ms = 0.
        self.source_stale_exhaustions = 0
        self.source_recoveries = 0
        self.source_last_error = None
        self.stopping = False
        self.started = time.perf_counter()
        self.task = asyncio.create_task(self.run())

    async def run(self):
        try:
            while not self.stopping:
                # During startup, reset, or optimizer pause, capture only on
                # guard demand. Avoid competing with PPO updates for readback.
                if self.token() is None:
                    await self.wanted.wait()
                if self.stopping:break
                self.wanted.clear()
                began = time.perf_counter()
                generation = self.token()
                image = await self.capture_source()
                if self.stopping:break
                timing = copy.deepcopy(self.source.last_timing)
                # This is a conservative request-time bound, NOT a Unity tick.
                stamp = timing['request_started_at']
                if not math.isfinite(stamp) or not began <= stamp <= time.perf_counter():
                    raise ValueError('Capture source timestamp outside request')
                self.callback(image,stamp,generation)
                async with self.condition:
                    self.latest = (stamp,image,timing)
                    self.count += 1
                    self.condition.notify_all()
                await asyncio.sleep(max(0,self.period-(time.perf_counter()-began)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self.condition:
                self.error = exc
                self.condition.notify_all()

    async def capture_source(self):
        """Recover only fully consumed stale replies, never protocol/clock errors.

        Electron already tries three future paints per call. Allow at most three
        such calls here (nine paints total); never publish their rejected pixels.
        Other errors and cancellation propagate immediately.
        """
        from tools.ppo_electron_capture import StaleElectronFrame
        for attempt in range(3):
            try:
                image = await self.source.capture()
                if attempt:self.source_recoveries += 1
                return image
            except StaleElectronFrame as exc:
                self.source_stale_exhaustions += 1
                self.source_last_error = str(exc)
                if attempt == 2:raise
                # Yield under contention before demanding another future paint.
                await asyncio.sleep(.01)

    async def capture(self):
        original_request = time.perf_counter()
        for attempt in range(3):
            requested = time.perf_counter()
            self.wanted.set()
            async with self.condition:
                await asyncio.wait_for(self.condition.wait_for(
                    lambda: self.error is not None or (self.latest is not None and self.latest[0]>=requested)),5)
                if self.error is not None: raise self.error
                stamp,image,timing = self.latest
                age=time.perf_counter()-stamp
                if not math.isfinite(age) or age<0:raise ValueError('Invalid temporal delivery clock')
                if timing['electron_serial'] <= self.delivery_serial:
                    raise ValueError('Temporal pump attempted duplicate serial delivery')
                if age > .05:
                    # No frame escapes to a guard/policy. Next attempt demands a
                    # capture STARTED after the retry, never the cached candidate.
                    self.delivery_stale_discards += 1
                    self.delivery_stale_max_ms=max(self.delivery_stale_max_ms,age*1000)
                    continue
                self.delivery_serial = timing['electron_serial']
                self.last_timing = copy.deepcopy(timing)
                self.last_timing.update(temporal_consumer_requested_at=original_request,
                    temporal_delivery_retries=attempt,temporal_delivery_age_ms=age*1000)
                return image
        from rs_v612_gameplay_guard import GuardUncertainError
        raise GuardUncertainError('Temporal pump delivery exceeds 50ms after 3 fresh requests; no frame delivered')

    async def close(self):
        # Python 3.10 wait_for can race cancellation with a completed receive.
        # A persistent stop flag also terminates that successfully returned path.
        self.stopping = True
        self.wanted.set()
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError): await self.task


def install(trainer, config):
    from tools.ppo_headless_env import install as headless
    headless(trainer,'electron',electron_transport='shared_memory')
    base_build = trainer.build_env_class
    import tools.ppo_observation_contract as contracts
    base_contract = contracts.requested_contract

    def contract(a):
        value = base_contract(a)
        value['image_shape'][0] = 3*config['frames']
        value['aux_dim'] += 2*config['frames']
        value['temporal'] = dict(config, version=2, padding='zero_with_age_and_validity',
                                 ordering='oldest_to_newest', selection_tolerance_ms=60)
        return value
    contracts.requested_contract = contract
    original_shape = trainer.shape_description
    def shape():
        value = original_shape()
        value.update(image_channels=3*config['frames'], image_frames=config['frames'],raw_rgb_history_frames=config['frames'],
                     aux_dim=557+2*config['frames'], temporal=dict(config))
        return value
    trainer.shape_description = shape

    def build():
        base = base_build()
        class TemporalEnv(base):
            def __init__(self,*args,**kwargs):
                self.temporal = ContextHistory(config['frames'],config['span_ms']/1000)
                self.pump = None
                self._temporal_resync = False
                self._temporal_pausing = False
                self.temporal_packs = 0
                self.temporal_valid_slots = 0
                super().__init__(*args,**kwargs)
                from gymnasium import spaces
                old = self.observation_space
                self.observation_space = spaces.Dict({
                    'image':spaces.Box(0,255,(config['frames']*3,*old['image'].shape[1:]),np.uint8),
                    'aux':spaces.Box(0.,1.,(old['aux'].shape[0]+2*config['frames'],),np.float32)})

            async def _open_browser(self):
                await super()._open_browser()
                from tools.ppo_framing import canvas_observation
                import cv2
                def record(image,stamp,generation):
                    if generation is not None:
                        frame,_ = canvas_observation(image,self.target_wh,
                            cv2.INTER_AREA if self.resize_method=='area' else cv2.INTER_LINEAR)
                        self.temporal.add(frame,stamp,generation)
                source = self._headless_source
                self.pump = CapturePump(source,record,config['capture_hz'],self.temporal.token)
                # Keep delivery timing separate from the producer's next capture.
                self._headless_source = self.pump

            def _pack(self,*args,**kwargs):
                result,fresh,age = super()._pack(*args,**kwargs)
                # Anchor to the guarded policy observation, never a later pump frame.
                stamp = self._headless_capture_timing['request_started_at']
                image,temporal = self.temporal.observation(result['image'][-3:],stamp,
                    enable=not (self._temporal_resync or self._temporal_pausing))
                result = dict(image=image,aux=np.concatenate((result['aux'],temporal)))
                self.temporal_packs += 1
                self.temporal_valid_slots += int(temporal[-config['frames']:].sum())
                self._learning_current_observation = result
                return result,fresh,age

            def reset(self,*args,**kwargs):
                self.temporal.clear()
                return super().reset(*args,**kwargs)

            def pause_for_ppo_optimizer(self):
                self.temporal.clear()
                self._temporal_pausing = True
                try:
                    return super().pause_for_ppo_optimizer()
                finally:
                    self.temporal.clear()
                    self._temporal_pausing = False

            def refresh_after_ppo_optimizer(self):
                self.temporal.clear()
                self._temporal_resync = True
                try:
                    return super().refresh_after_ppo_optimizer()
                finally:
                    self._temporal_resync = False

            async def _close_browser(self):
                try:
                    if self.pump is not None:
                        import json
                        await self.pump.close()
                        elapsed = time.perf_counter()-self.pump.started
                        (self.output_dir/'temporal_capture.json').write_text(json.dumps(dict(
                            captures=self.pump.count,wall_seconds=elapsed,
                            achieved_capture_hz=self.pump.count/max(elapsed,1e-9),
                            requested_capture_hz=config['capture_hz'],includes_pauses=True,
                            delivery_stale_discards=self.pump.delivery_stale_discards,
                            source_stale_exhaustions=self.pump.source_stale_exhaustions,
                            source_recoveries=self.pump.source_recoveries,
                            source_last_error=self.pump.source_last_error,
                            delivery_stale_max_ms=self.pump.delivery_stale_max_ms,
                            packs=self.temporal_packs,
                            valid_slot_fraction=self.temporal_valid_slots/max(1,self.temporal_packs*config['frames'])),indent=2))
                finally:
                    await super()._close_browser()
        return TemporalEnv
    trainer.build_env_class = build
