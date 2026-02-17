from __future__ import annotations
import time


class ExponentialBackoff:
    def __init__(self, initial: float = 1.0, maximum: float = 30.0, factor: float = 2.0):
        self.initial = float(initial)
        self.maximum = float(maximum)
        self.factor = float(factor)
        self._current = self.initial

    def wait(self):
        time.sleep(self._current)
        self._current = min(self.maximum, self._current * self.factor)

    def reset(self):
        self._current = self.initial
