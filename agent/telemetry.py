"""
FactTrace Agent Telemetry
=========================
Tracks stage-by-stage latencies, cache metrics, and API call efficiency
for production observability.
"""

import time
from typing import Any


class AgentTelemetry:
    """High-resolution latency and API call telemetry tracker."""

    def __init__(self):
        self.timers: dict[str, float] = {}
        self._start_times: dict[str, float] = {}
        self.cache_hits: int = 0
        self.serp_calls: int = 0

    def start(self, agent_name: str) -> None:
        self._start_times[agent_name] = time.perf_counter()

    def stop(self, agent_name: str) -> float:
        start_time = self._start_times.get(agent_name, time.perf_counter())
        duration = time.perf_counter() - start_time
        self.timers[agent_name] = duration
        return duration

    def to_dict(self) -> dict[str, Any]:
        total = sum(self.timers.values())
        return {
            "total_latency": round(total, 2),
            "planner_latency": round(self.timers.get("planner", 0.0), 2),
            "retrieval_latency": round(self.timers.get("retrieval", 0.0), 2),
            "critic_latency": round(self.timers.get("critic", 0.0), 2),
            "synthesis_latency": round(self.timers.get("synthesis", 0.0), 2),
            "cache_hits": self.cache_hits,
            "serp_calls": self.serp_calls,
        }
