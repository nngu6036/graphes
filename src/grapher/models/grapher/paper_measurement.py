"""Opt-in measurements for paper experiments; inactive in ordinary runs.

No random values are drawn and no model/configuration is changed. The profile
runner activates one Recorder and synchronizes CUDA at stage boundaries.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_active = ContextVar("graphes_paper_recorder", default=None)


class Recorder:
    def __init__(self, synchronize=None):
        self.synchronize = synchronize or (lambda: None)
        self.stages = defaultdict(lambda: {"calls": 0, "failures": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0, "maximum_call_seconds": 0.0, "call_seconds": []})
        self.stack = []
        self.attempted_samples = 0
        self.completed_endpoints = 0
        self.attempt_batches = []

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    @contextmanager
    def stage(self, name):
        self.synchronize()
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.synchronize()
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.stages[name]
            row["calls"] += 1
            row["failures"] += int(failed)
            row["inclusive_seconds"] += elapsed
            row["exclusive_seconds"] += max(0.0, elapsed - frame[1])
            row["maximum_call_seconds"] = max(row["maximum_call_seconds"], elapsed)
            row["call_seconds"].append(elapsed)


@contextmanager
def stage(name):
    recorder = _active.get()
    if recorder is None:
        yield
    else:
        with recorder.stage(name):
            yield


def attempted(count):
    recorder = _active.get()
    if recorder is not None:
        recorder.attempted_samples += int(count)
        recorder.attempt_batches.append(int(count))


def completed(count=1):
    recorder = _active.get()
    if recorder is not None:
        recorder.completed_endpoints += int(count)
