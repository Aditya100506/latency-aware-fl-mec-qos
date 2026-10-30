"""Discrete-event network simulator for federated-learning update traffic.

This is a controlled abstraction, not a standards-compliant 5G/6G simulator.
All times are simulated seconds; all payload sizes are MB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import heapq
import random
import statistics
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class Job:
    job_id: int
    client_id: int
    created_at: float
    size_mb: float
    qos_class: str
    path_delay_s: float
    edge_id: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    dropped: bool = False


@dataclass
class Metrics:
    scenario: str
    clients: int
    round_id: int
    jobs_created: int
    jobs_completed: int
    jobs_dropped: int
    mean_latency_s: float
    p95_latency_s: float
    mean_queue_delay_s: float
    throughput_jobs_s: float
    drop_ratio: float
    simulated_duration_s: float


@dataclass
class Server:
    server_id: int
    bandwidth_mbps: float
    queue_limit: int
    propagation_s: float
    queues: Dict[str, Deque[Job]] = field(default_factory=lambda: {
        "critical": deque(), "standard": deque(), "bulk": deque()
    })
    deficits: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.0, "standard": 0.0, "bulk": 0.0
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "critical": 4.0, "standard": 2.0, "bulk": 1.0
    })
    busy_until: float = 0.0
    busy: bool = False

    def queued_count(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def enqueue(self, job: Job) -> bool:
        if self.queued_count() >= self.queue_limit:
            job.dropped = True
            return False
        self.queues[job.qos_class].append(job)
        return True

    def pop_drr(self) -> Optional[Job]:
        # Weighted deficit round robin over the three QoS classes.
        classes = ("critical", "standard", "bulk")
        for _ in range(len(classes) * 3):
            for cls in classes:
                if not self.queues[cls]:
                    continue
                self.deficits[cls] += self.weights[cls] * 0.25
                head = self.queues[cls][0]
                cost = max(0.05, head.size_mb / max(self.bandwidth_mbps, 0.1))
                if self.deficits[cls] >= cost:
                    self.deficits[cls] -= cost
                    return self.queues[cls].popleft()
        # Ensure progress for small simulations even when a job exceeds one quantum.
        for cls in classes:
            if self.queues[cls]:
                self.deficits[cls] = 0.0
                return self.queues[cls].popleft()
        return None


class NetworkSimulation:
    def __init__(self, scenario: str, clients: int, rounds: int, seed: int,
                 queue_limit: int = 100, edge_count: int = 4):
        self.scenario = scenario
        self.clients = clients
        self.rounds = rounds
        self.rng = random.Random(seed + clients * 17 + len(scenario))
        self.queue_limit = queue_limit
        self.edge_count = edge_count
        self.servers: List[Server] = []
        self.jobs: List[Job] = []
        self.metrics: List[Metrics] = []
        self.clock = 0.0
        self.job_id = 0
        self._build_servers()

    def _build_servers(self):
        if self.scenario == "cloud_baseline":
            # One shared cloud ingress/service bottleneck with higher propagation delay.
            self.servers = [Server(0, bandwidth_mbps=35.0,
                                   queue_limit=self.queue_limit, propagation_s=0.045)]
        else:
            self.servers = [
                Server(i, bandwidth_mbps=55.0 + 5.0 * (i % 2),
                       queue_limit=self.queue_limit, propagation_s=0.008 + 0.002 * (i % 3))
                for i in range(self.edge_count)
            ]

    def _choose_server(self, client_id: int) -> Server:
        if self.scenario != "mec_qos_latency_aware":
            if self.scenario == "cloud_baseline":
                return self.servers[0]
            # Deterministic client-to-edge assignment for plain MEC + QoS.
            return self.servers[client_id % len(self.servers)]
        # Latency-aware assignment: estimated path delay plus current queued work.
        base_path = [0.006 + 0.0015 * ((client_id + i * 3) % 9)
                     for i in range(len(self.servers))]
        scores = []
        for i, server in enumerate(self.servers):
            queued_mb = sum(j.size_mb for q in server.queues.values() for j in q)
            estimated_queue = queued_mb / max(server.bandwidth_mbps, 0.1)
            scores.append((base_path[i] + estimated_queue, server.server_id))
        return self.servers[min(scores)[1]]

    def _make_job(self, client_id: int, now: float) -> Job:
        self.job_id += 1
        size = self.rng.uniform(0.25, 1.5)
        roll = self.rng.random()
        qos = "critical" if roll < 0.20 else ("standard" if roll < 0.75 else "bulk")
        server = self._choose_server(client_id)
        path = server.propagation_s
        if self.scenario == "cloud_baseline":
            path += self.rng.uniform(0.035, 0.070)
        else:
            path += self.rng.uniform(0.004, 0.018)
        return Job(self.job_id, client_id, now, size, qos, path, server.server_id)

    def _serve_server(self, server: Server, now: float, events: list):
        if server.busy or now < server.busy_until:
            return
        job = server.pop_drr()
        if job is None:
            return
        server.busy = True
        job.started_at = now
        service_s = (job.size_mb * 8.0) / max(server.bandwidth_mbps, 0.1)
        finish = now + service_s + job.path_delay_s
        server.busy_until = finish
        heapq.heappush(events, (finish, "finish", server.server_id, job))

    def run(self) -> List[Metrics]:
        for round_id in range(1, self.rounds + 1):
            round_start = (round_id - 1) * 0.25
            round_jobs: List[Job] = []
            for client_id in range(self.clients):
                # Small, bounded jitter emulates staggered client upload arrivals.
                arrival = round_start + self.rng.uniform(0.0, 0.12)
                job = self._make_job(client_id, arrival)
                round_jobs.append(job)
            events = []
            for job in round_jobs:
                server = self.servers[job.edge_id]
                if server.enqueue(job):
                    heapq.heappush(events, (job.created_at, "arrival", server.server_id, job))
                else:
                    self.jobs.append(job)
            completed: List[Job] = []
            round_dropped = sum(1 for j in round_jobs if j.dropped)
            now = round_start
            # Discrete-event processing until all jobs for this round finish.
            while events:
                now, event_type, server_id, job = heapq.heappop(events)
                server = self.servers[server_id]
                if event_type == "arrival":
                    self._serve_server(server, now, events)
                elif event_type == "finish":
                    server.busy = False
                    job.finished_at = now
                    completed.append(job)
                    self._serve_server(server, now, events)
                # If queues have work but no future event, start them.
                for srv in self.servers:
                    if not srv.busy and srv.queued_count() > 0:
                        self._serve_server(srv, now, events)
            # Jobs may be queued at round boundary; drain every server to avoid silent omissions.
            drain_events = []
            for srv in self.servers:
                srv.busy = False
                self._serve_server(srv, now, drain_events)
            while drain_events:
                now, event_type, server_id, job = heapq.heappop(drain_events)
                srv = self.servers[server_id]
                if event_type == "finish":
                    job.finished_at = now
                    completed.append(job)
                    srv.busy = False
                    self._serve_server(srv, now, drain_events)
            self.jobs.extend(round_jobs)
            latencies = [j.finished_at - j.created_at for j in completed if j.finished_at is not None]
            queue_delays = [max(0.0, j.started_at - j.created_at) for j in completed if j.started_at is not None]
            duration = max([j.finished_at or round_start for j in completed] + [round_start]) - round_start
            sorted_lat = sorted(latencies)
            p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))] if sorted_lat else 0.0
            done = len(completed)
            dropped = round_dropped
            created = len(round_jobs)
            self.metrics.append(Metrics(
                scenario=self.scenario, clients=self.clients, round_id=round_id,
                jobs_created=created, jobs_completed=done, jobs_dropped=dropped,
                mean_latency_s=statistics.mean(latencies) if latencies else 0.0,
                p95_latency_s=p95,
                mean_queue_delay_s=statistics.mean(queue_delays) if queue_delays else 0.0,
                throughput_jobs_s=(done / duration) if duration > 0 else 0.0,
                drop_ratio=(dropped / created) if created else 0.0,
                simulated_duration_s=duration
            ))
            self.clock = now
        return self.metrics
