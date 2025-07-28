# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 snu-comparch contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
# -----------------------------------------------------------------------------------

"""Global timer for profiling."""
from collections import namedtuple
import time
from typing import Callable, Any


class _Timer:
    """An internal timer."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.start_time = None

        # start-stop timestamp pairs
        self.start_times = []
        self.stop_times = []
        self.costs = []

    def start(self, sync_func: Callable = None):
        """Start the timer."""
        assert not self.started, f"timer {self.name} has already been started."
        if sync_func:
            sync_func()

        self.start_time = time.perf_counter()
        self.start_times.append(self.start_time)
        self.started = True

    def stop(self, sync_func: Callable = None):
        """Stop the timer."""
        assert self.started, f"timer {self.name} is not started."
        if sync_func:
            sync_func()

        stop_time = time.perf_counter()
        self.costs.append(stop_time - self.start_time)
        self.stop_times.append(stop_time)
        self.started = False

    def reset(self):
        """Reset timer."""
        self.started = False
        self.start_time = None
        self.start_times = []
        self.stop_times = []
        self.costs = []

    def elapsed(self, mode: str = "average"):
        """Calculate the elapsed time."""
        if not self.costs:
            return 0.0
        if mode == "average":
            return sum(self.costs) / len(self.costs)
        elif mode == "sum":
            return sum(self.costs)
        else:
            raise RuntimeError("Supported mode is: average | sum")


class Timers:
    """A group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name: str):
        return name in self.timers


timers = Timers()

Event = namedtuple("Event", ("tstamp", "name", "info"))


class Tracer:
    """An activity tracer."""

    def __init__(self):
        self.events = []

    def log(self, name: str, info: Any, sync_func: Callable = None):
        if sync_func:
            sync_func()

        self.events.append(Event(time.perf_counter(), name, info))


tracer = Tracer()

