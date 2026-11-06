"""Retain history only across an independently verified same-episode freeze."""
from collections import deque
import math

PAUSE_SCHEMA = dict(version=1, clock='capture_request_time_minus_verified_frozen_duration',
    restore='same_episode_after_successful_guarded_resync',
    discard='episode_reset_pause_failure_resume_failure_uncertain_or_menu_resync',
    freshness='latest_capture_guard_timestamps_are_never_shifted')


class PauseHistory:
    def __init__(self):
        self.saved = None
        self.snapshots = self.restores = self.discards = 0
        self.shifted_seconds = 0.
        self.restored_frames = self.restored_commands = 0

    def discard(self):
        if self.saved is not None:self.discards += 1
        self.saved = None

    def snapshot(self, temporal, commands, episode):
        self.discard()
        with temporal.lock:
            self.saved = dict(items=list(temporal.items), recent=list(temporal.recent),
                first=temporal.first_observation, commands=list(commands.events), episode=episode)
            # Invalidate any capture already in flight; it cannot enter history.
            temporal.enabled = False
            temporal.generation += 1
        self.snapshots += 1

    def paused(self, token, episode):
        if self.saved is None:return
        if not isinstance(token,tuple) or len(token)!=4 or episode!=self.saved['episode']:
            self.discard();return
        stamp=float(token[1])
        if not math.isfinite(stamp):self.discard();return
        self.saved['paused_at']=stamp

    def resumed(self, token, finished, episode):
        if self.saved is None:return
        start=self.saved.get('paused_at')
        if (start is None or not isinstance(token,tuple) or len(token)!=4 or
            token[1]!=start or episode!=self.saved['episode'] or finished is None):
            self.discard();return
        delta=float(finished)-start
        if not math.isfinite(delta) or not 0<delta<=30:
            self.discard();return
        self.saved.update(delta=delta,resumed_at=float(finished))

    def restore(self, temporal, commands, fresh, anchor, episode):
        saved=self.saved
        if (saved is None or 'delta' not in saved or saved['episode']!=episode or
            not fresh or not math.isfinite(anchor) or anchor<saved['resumed_at']):
            self.discard();return False
        delta=saved['delta'];cutoff=anchor-temporal.span-temporal.tolerance
        old=[(t+delta,f) for t,f in saved['items']
             if t<=saved['paused_at'] and cutoff<=t+delta<anchor]
        fresh=[(t,f) for t,f in fresh if saved['resumed_at']<=t<=anchor]
        if not fresh or fresh[-1][0]!=anchor:
            self.discard();return False
        merged=sorted(dict(old+fresh).items())
        restored_commands=[(t+delta,a) for t,a in saved['commands']
            if t<=saved['paused_at'] and 0<=anchor-(t+delta)<=1.]
        with temporal.lock:
            temporal.items=deque(merged,maxlen=256)
            temporal.recent=deque(fresh[-temporal.recent_count:],maxlen=temporal.recent_count)
            temporal.first_observation=(saved['first']+delta if saved['first'] is not None else fresh[0][0])
            temporal.after=saved['resumed_at']
            temporal.generation+=1
            # The returned observation enables the pump only after restoration.
            temporal.enabled=False
        commands.events=deque(restored_commands)
        self.restores+=1;self.shifted_seconds+=delta
        self.restored_frames+=len(old);self.restored_commands+=len(restored_commands)
        self.saved=None
        return True

    def report(self):
        return dict(schema=PAUSE_SCHEMA,snapshots=self.snapshots,restores=self.restores,
            discarded_snapshots=self.discards,excluded_frozen_seconds=self.shifted_seconds,
            restored_image_entries=self.restored_frames,restored_command_entries=self.restored_commands)
