"""Opt-in extension of the fixed4 pipeline; production files remain untouched."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'development/staged/history_training_compare_20261103'
sys.path[:0] = [str(ROOT), str(BASE)]
import parallel_adapter as base
from history import CommandHistory, EXTRA_DIM, SCHEMA
from pause_history import PauseHistory, PAUSE_SCHEMA


def history_env(base_class):
    class CommandHistoryEnv(base_class):
        def __init__(self, *args, **kwargs):
            self.command_timeline = CommandHistory()
            self._command_last_timing = None
            self._pause_history = PauseHistory()
            self._history_resync_frames = None
            super().__init__(*args, **kwargs)
            from gymnasium import spaces
            old = self.observation_space
            self.observation_space = spaces.Dict(dict(image=old['image'],
                aux=spaces.Box(0.,1.,(old['aux'].shape[0]+EXTRA_DIM,),np.float32)))

        async def _step_browser(self, action):
            prior = self.step_timings[-1] if self.step_timings else None
            result = await super()._step_browser(action)
            row = self.step_timings[-1] if self.step_timings else None
            # A guard stop or passive capture has no new dispatched-step timing.
            # Recursive guard retries must not insert a command twice.
            if row is not None and row is not prior and row is not self._command_last_timing:
                if row['press']:
                    self.command_timeline.add(row['action'], row['press_timestamp'])
                elif row['action'] == 0:
                    stamp = self.action_context.last_action_times[0]
                    if stamp is not None:
                        self.command_timeline.add(0, stamp)
                self._command_last_timing = row
            return result

        def _pack(self, *args, **kwargs):
            result, fresh, age = super()._pack(*args, **kwargs)
            if self._history_resync_frames is not None:
                self._history_resync_frames.append((self._headless_capture_timing['request_started_at'],
                                                   result['image'][-3:].copy()))
            extra = self.command_timeline.vector(self._headless_capture_timing['request_started_at'])
            result = dict(result, aux=np.concatenate((result['aux'],extra)))
            self._learning_current_observation = result
            return result, fresh, age

        def reset(self, *args, **kwargs):
            self.command_timeline.clear()
            self._command_last_timing = None
            self._pause_history.discard()
            return super().reset(*args, **kwargs)

        def _history_episode(self):
            return (getattr(self,'episode_number',None),getattr(self,'episode_tag',None))

        def pause_for_ppo_optimizer(self):
            self._pause_history.snapshot(self.temporal,self.command_timeline,self._history_episode())
            self.command_timeline.clear()
            try:
                result=super().pause_for_ppo_optimizer()
                self._pause_history.paused(getattr(self,'_ppo_pause_pending',None),self._history_episode())
                return result
            except BaseException:
                self._pause_history.discard()
                raise

        def resume_after_ppo_optimizer(self, monitor):
            token=getattr(self,'_ppo_pause_pending',None)
            try:
                result=super().resume_after_ppo_optimizer(monitor)
                self._pause_history.resumed(token,getattr(self,'_ppo_resume_trace_start_perf',None),
                                             self._history_episode())
                return result
            except BaseException:
                self._pause_history.discard()
                raise

        def _read_screen_state(self):
            state,probs=super()._read_screen_state()
            if getattr(self,'_temporal_resync',False) and state!='GAMEPLAY':
                self._pause_history.discard()
            return state,probs

        def refresh_after_ppo_optimizer(self):
            self.command_timeline.clear()
            from collections import deque
            self._history_resync_frames=deque(maxlen=4)
            try:
                result=super().refresh_after_ppo_optimizer()
                obs,reason,captured_at=result
                if obs is None or reason is not None or getattr(self,'_guard_hard_stopped',False):
                    self._pause_history.discard()
                    return result
                anchor=self._headless_capture_timing['request_started_at']
                if self._pause_history.restore(self.temporal,self.command_timeline,
                        list(self._history_resync_frames),anchor,self._history_episode()):
                    image,timing=self.temporal.observation(obs['image'][-3:],anchor)
                    # Keep original 557 aux fields; replace only image ages/validity
                    # and appended command history. No guard timestamp is shifted.
                    obs=dict(obs,image=image,aux=np.concatenate((obs['aux'][:557],timing,
                                                               self.command_timeline.vector(anchor))))
                    self._learning_current_observation=obs
                return obs,reason,captured_at
            except BaseException:
                self._pause_history.discard()
                raise
            finally:
                self._history_resync_frames=None

        async def _close_browser(self):
            try:
                return await super()._close_browser()
            finally:
                (self.output_dir/'command_history.json').write_text(
                    json.dumps(dict(self.command_timeline.report(),pause_history=self._pause_history.report()),indent=2))
    return CommandHistoryEnv


def install():
    if os.environ.get('RS_COMPARISON_LAYOUT','fixed4') != 'fixed4':
        raise ValueError('Command-history experiment requires fixed4')
    os.environ['RS_COMPARISON_LAYOUT'] = 'fixed4'
    trainer = base.install()
    if getattr(trainer,'_command_history_ready',False):
        return trainer
    import tools.ppo_observation_contract as contracts
    original_contract = contracts.requested_contract
    def requested(args):
        value = original_contract(args)
        if value['aux_dim'] != 565:
            raise ValueError('Expected the unchanged 565-value fixed4 prefix')
        value['aux_dim'] += EXTRA_DIM
        value['command_history'] = dict(SCHEMA)
        value['history_pause'] = dict(PAUSE_SCHEMA)
        return value
    contracts.requested_contract = requested
    original_build = trainer.build_env_class
    trainer.build_env_class = lambda: history_env(original_build())
    original_shape = trainer.shape_description
    def shape():
        value = original_shape()
        value.update(aux_dim=565+EXTRA_DIM, command_history=dict(SCHEMA))
        return value
    trainer.shape_description = shape
    trainer._command_history_ready = True
    return trainer


def sources():
    result = base.sources()
    for file in HERE.glob('*.py'):
        result[file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    if os.environ.get('RS_REUSE_PUMP_CROP')=='1':
        file=ROOT/'development/experiments/multimap_classifier/crop_reuse.py'
        result[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    if os.environ.get('RS_DEFER_STARTUP_PREVIEWS')=='1':
        file=ROOT/'development/experiments/multimap_classifier/deferred_previews.py'
        result[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def configuration():
    return dict(base.configuration(), command_history=dict(SCHEMA))


def contract():
    install()
    return base.contract()
