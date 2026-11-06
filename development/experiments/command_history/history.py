"""Worker-local command timeline, anchored to the policy image request clock."""
from collections import deque
import math
import numpy as np

SLOTS = 32
WIDTH = 7  # five-way one-hot action, age in seconds / 1 second, validity
EXTRA_DIM = SLOTS * WIDTH
SCHEMA = dict(version=2, slots=SLOTS, seconds=1., actions=['NONE','LEFT','RIGHT','JUMP','ROLL'],
    slot_fields=['one_hot_5','age_seconds_divided_by_1','valid'],
    ordering='oldest_to_newest_right_aligned', padding='zeros_invalid',
    anchor='policy_image_request_started_at',
    events='completed_keyboard_press_dispatch_start_or_guard_accepted_noop',
    resets='episode_or_uncertain_continuity', pause='retain_shift_verified_frozen_duration',
    overflow='keep_newest_32_count_dropped')


class CommandHistory:
    def __init__(self):
        self.events = deque()
        self.total_events = self.overflows = self.resets = 0
        self.pack_count = self.valid_slots = 0

    def clear(self):
        self.events.clear()
        self.resets += 1

    def add(self, action, at):
        if isinstance(action, bool) or int(action) != action or not 0 <= int(action) < 5:
            raise ValueError('Invalid command action')
        if not math.isfinite(at) or (self.events and at < self.events[-1][0]):
            raise ValueError('Nonmonotonic command clock')
        while self.events and self.events[0][0] < at - 1.:
            self.events.popleft()
        if len(self.events) >= SLOTS:
            self.events.popleft()
            self.overflows += 1
        self.events.append((float(at), int(action)))
        self.total_events += 1

    def vector(self, anchor):
        if not math.isfinite(anchor):
            raise ValueError('Invalid observation clock')
        selected = [(t,a) for t,a in self.events if 0 <= anchor-t <= 1.]
        result = np.zeros((SLOTS, WIDTH), np.float32)
        for i, (t, action) in enumerate(selected, SLOTS-len(selected)):
            result[i,action] = 1.
            result[i,5] = anchor-t
            result[i,6] = 1.
        self.pack_count += 1
        self.valid_slots += len(selected)
        return result.ravel()

    def report(self):
        return dict(schema=SCHEMA, total_events=self.total_events, overflow_events=self.overflows,
            resets=self.resets, observations=self.pack_count,
            mean_valid_entries=self.valid_slots/max(1,self.pack_count))
