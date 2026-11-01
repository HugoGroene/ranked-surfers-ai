"""Independent nonblocking rate gate; missed deadlines never cause bursts."""
import math


class RateGate:
    def __init__(self, hz):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('Rate must be finite and positive')
        self.period = 1 / hz
        self.next_at = None
        self.last_at = None

    def due(self, now):
        if not math.isfinite(now) or (self.last_at is not None and now < self.last_at):
            raise ValueError('Invalid or backwards clock')
        self.last_at = now
        if self.next_at is not None and now < self.next_at:
            return False
        self.next_at = now + self.period
        return True
