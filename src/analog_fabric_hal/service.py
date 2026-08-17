"""Serialized hardware-service boundary for the analog fabric MVP.

This module intentionally hides topology and driver ownership from clients.
It is an in-process stand-in for a future separate RPC service: its public
methods are the contract to preserve when HTTP, gRPC, or a message queue is
added later.

Design note: this module is deliberately conservative. A driver call is an
atomic, hardware-facing action; the service may cancel only before the next
action. A normal driver return is dispatch evidence, never proof that a device
changed. Explicit measurement is the only route to verified state.

Future work: replace the in-process queue with a server-owned durable job
queue, persist the desired/observed state, attach authenticated actors to every
request, and add recovery rules for device communication failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable
from uuid import uuid4

from .analog_fabric import Compiler, Edge, FabricError, FabricProgram, Simulator, Waveform
from .audit import AuditEvent, JsonlAuditLog
from .drivers import DriverRegistry
from .device_registry import DeviceRegistry
from .security import AuthorizationPolicy


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    MISMATCH = "mismatch"


class LifecycleState(str, Enum):
    OFF = "off"
    ON = "on"
    FAULTED = "faulted"


@dataclass(frozen=True)
class WeightUpdate:
    source: int
    destination: int
    weight: float


@dataclass(frozen=True)
class ItemResult:
    """One result per requested weight; batches are never collapsed to pass/fail."""
    index: int
    update: WeightUpdate
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class MeasurementRequest:
    """A logical experiment for validating one programmed edge."""
    source: int
    destination: int
    stimulus_v: float
    tolerance_v: float


@dataclass(frozen=True)
class MeasurementResult:
    request: MeasurementRequest
    expected_v: float | None
    observed_v: float | None
    state: VerificationState | None
    reason: str | None = None


@dataclass(frozen=True)
class InputWaveformRequest:
    port: int
    waveform: Waveform


@dataclass(frozen=True)
class WaveformJobResult:
    port: int
    dispatched: bool
    waveform: object | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LifecycleResult:
    state: LifecycleState
    completed: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass
class JobResult:
    job_id: str
    state: JobState = JobState.QUEUED
    items: list[ItemResult] = field(default_factory=list)
    measurement: MeasurementResult | None = None
    waveform: WaveformJobResult | None = None
    lifecycle: LifecycleResult | None = None


@dataclass
class _Job:
    result: JobResult
    updates: tuple[WeightUpdate, ...]
    measurement_request: MeasurementRequest | None = None
    input_request: InputWaveformRequest | None = None
    capture_port: int | None = None
    lifecycle_action: str | None = None
    actor: str = "system"
    cancelled: Event = field(default_factory=Event)


class HardwareService:
    """Owns a fabric configuration and applies all physical changes serially.

    The service is the trust boundary between logical clients and shared
    hardware. Its private registries hold topology, wiring, and instruments;
    callers only provide logical ports, edges, and desired waveforms.
    """

    def __init__(self, program: FabricProgram | None = None, before_operation: Callable[[], None] | None = None, driver_registry: DriverRegistry | None = None, audit_log: JsonlAuditLog | None = None, device_registry: DeviceRegistry | None = None, authorization_policy: AuthorizationPolicy | None = None):
        self._program = program or FabricProgram()
        self._before_operation = before_operation
        self._driver_registry = driver_registry
        self._audit_log = audit_log
        self._device_registry = device_registry
        self._authorization_policy = authorization_policy
        if device_registry is not None and device_registry.shape != self._program.shape:
            raise FabricError("device registry shape must match the fabric program")
        self._lifecycle_state = LifecycleState.OFF
        self._jobs: dict[str, _Job] = {}
        self._queue: Queue[_Job | None] = Queue()
        self._lock = Lock()
        self._safe_stop = Event()
        self._verification: dict[tuple[int, int], VerificationState] = {}
        self._worker = Thread(target=self._work, name="analog-fabric-worker", daemon=True)
        self._worker.start()

    def submit_weights(self, updates: list[WeightUpdate], actor: str = "system") -> str:
        """Queue logical edge-weight requests without exposing physical wiring."""
        self._authorize(actor, "program")
        job_id = str(uuid4())
        job = _Job(JobResult(job_id), tuple(updates), actor=actor)
        with self._lock:
            if self._safe_stop.is_set():
                job.result.state = JobState.STOPPED
                self._jobs[job_id] = job
                return job_id
            self._jobs[job_id] = job
        self._queue.put(job)
        self._audit("weight-batch-submitted", "queued", f"items={len(updates)}", job_id, actor)
        return job_id

    def status(self, job_id: str) -> JobResult:
        with self._lock:
            job = self._jobs[job_id]
            return JobResult(job.result.job_id, job.result.state, list(job.result.items), job.result.measurement, job.result.waveform, job.result.lifecycle)

    def edge_verification(self, source: int, destination: int) -> VerificationState | None:
        """Return evidence status, never an assumed physical readback."""
        with self._lock:
            return self._verification.get((source, destination))

    def submit_edge_measurement(self, request: MeasurementRequest, actor: str = "system") -> str:
        """Queue an explicit observation job after programming has completed."""
        self._authorize(actor, "measure")
        job_id = str(uuid4())
        job = _Job(JobResult(job_id), (), request, actor=actor)
        with self._lock:
            if self._safe_stop.is_set():
                job.result.state = JobState.STOPPED
                self._jobs[job_id] = job
                return job_id
            self._jobs[job_id] = job
        self._queue.put(job)
        return job_id

    def submit_input_waveform(self, request: InputWaveformRequest, actor: str = "system") -> str:
        """Queue a logical source request; physical source selection stays private."""
        self._authorize(actor, "source")
        return self._submit_waveform_job(_Job(JobResult(str(uuid4())), (), input_request=request, actor=actor))

    def submit_output_capture(self, port: int, actor: str = "system") -> str:
        """Queue a logical capture request; physical capture selection stays private."""
        self._authorize(actor, "capture")
        return self._submit_waveform_job(_Job(JobResult(str(uuid4())), (), capture_port=port, actor=actor))

    def submit_power_on(self, actor: str = "system") -> str:
        self._authorize(actor, "power")
        return self._submit_lifecycle_job("power_on", actor)

    def submit_safe_stop(self, actor: str = "system") -> str:
        self._authorize(actor, "power")
        return self._submit_lifecycle_job("safe_stop", actor)

    def _submit_lifecycle_job(self, action: str, actor: str) -> str:
        job = _Job(JobResult(str(uuid4())), (), lifecycle_action=action, actor=actor)
        with self._lock:
            self._jobs[job.result.job_id] = job
        self._queue.put(job)
        self._audit(action, "queued", job_id=job.result.job_id, actor=actor)
        return job.result.job_id

    def _submit_waveform_job(self, job: _Job) -> str:
        with self._lock:
            if self._safe_stop.is_set():
                job.result.state = JobState.STOPPED
                self._jobs[job.result.job_id] = job
                return job.result.job_id
            self._jobs[job.result.job_id] = job
        self._queue.put(job)
        self._audit("waveform-job-submitted", "queued", job_id=job.result.job_id, actor=job.actor)
        return job.result.job_id

    def cancel(self, job_id: str) -> None:
        """Cancellation is observed only before the next atomic update."""
        with self._lock:
            self._jobs[job_id].cancelled.set()

    def emergency_stop(self) -> None:
        """Prevent new programming work and make the fabric's simulated state safe."""
        self._safe_stop.set()
        result = self._driver_registry.safe_stop() if self._driver_registry is not None else None
        with self._lock:
            # In a physical driver this is where outputs would be disabled and
            # supplies commanded to their pre-approved safe settings.
            self._program = FabricProgram(shape=self._program.shape, outputs=self._program.outputs)
            self._lifecycle_state = LifecycleState.FAULTED if result and result.failures else LifecycleState.OFF
        outcome = "faulted" if result and result.failures else "safe"
        self._audit("emergency-stop", outcome, "; ".join(result.failures) if result else "simulator state cleared")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=1)

    def _work(self) -> None:
        # This is the only worker allowed to execute hardware-facing jobs.  Do
        # not parallelize it until a capability model proves operations are
        # independent across devices and shared resources.
        while (job := self._queue.get()) is not None:
            with self._lock:
                if self._safe_stop.is_set():
                    job.result.state = JobState.STOPPED
                    continue
                job.result.state = JobState.RUNNING
            self._audit("job-started", "running", job_id=job.result.job_id, actor=job.actor)
            if job.lifecycle_action is not None:
                self._run_lifecycle(job)
                continue
            if job.measurement_request is not None:
                self._measure(job)
                continue
            if job.input_request is not None:
                self._source_waveform(job)
                continue
            if job.capture_port is not None:
                self._capture_waveform(job)
                continue
            for index, update in enumerate(job.updates):
                # Cancellation is intentionally checked between updates, never
                # while a fixed driver call may be modifying physical hardware.
                if job.cancelled.is_set():
                    with self._lock:
                        job.result.state = JobState.CANCELLED
                    break
                if self._safe_stop.is_set():
                    with self._lock:
                        job.result.state = JobState.STOPPED
                    break
                try:
                    if self._before_operation:
                        self._before_operation()
                    self._apply(update)
                    item = ItemResult(index, update, True)
                except (FabricError, ValueError) as error:
                    item = ItemResult(index, update, False, str(error))
                with self._lock:
                    job.result.items.append(item)
                self._audit("weight-update", "accepted" if item.accepted else "rejected", item.reason or "", job.result.job_id, job.actor)
            else:
                with self._lock:
                    job.result.state = JobState.COMPLETE

    def _run_lifecycle(self, job: _Job) -> None:
        if self._driver_registry is None:
            result = LifecycleResult(LifecycleState.FAULTED, (), ("no private driver registry is configured",))
        elif job.lifecycle_action == "power_on":
            safety = self._driver_registry.power_on()
            result = LifecycleResult(LifecycleState.FAULTED if safety.failures else LifecycleState.ON, safety.completed, safety.failures)
        else:
            safety = self._driver_registry.safe_stop()
            result = LifecycleResult(LifecycleState.FAULTED if safety.failures else LifecycleState.OFF, safety.completed, safety.failures)
        with self._lock:
            self._lifecycle_state = result.state
            job.result.lifecycle = result
            job.result.state = JobState.COMPLETE
        self._audit(job.lifecycle_action or "lifecycle", result.state.value, "; ".join(result.failures), job.result.job_id, job.actor)

    def _measure(self, job: _Job) -> None:
        request = job.measurement_request
        assert request is not None
        key = (request.source, request.destination)
        with self._lock:
            edge = next((e for e in self._program.edges if (e.source, e.destination) == key), None)
            program = self._program
        if edge is None:
            result = MeasurementResult(request, None, None, None, "edge has not been programmed")
        elif request.tolerance_v < 0:
            result = MeasurementResult(request, None, None, None, "tolerance_v cannot be negative")
        else:
            inputs = [Waveform((0.0,), 1.0) for _ in range(program.shape.inputs)]
            inputs[request.source] = Waveform((request.stimulus_v,), 1.0)
            observed = Simulator().run(Compiler().compile(program), inputs)[request.destination].samples_v[0]
            expected = min(program.outputs[request.destination].max_v, max(program.outputs[request.destination].min_v, program.outputs[request.destination].bias_v + edge.weight * request.stimulus_v))
            state = VerificationState.VERIFIED if abs(observed - expected) <= request.tolerance_v else VerificationState.MISMATCH
            result = MeasurementResult(request, expected, observed, state)
            with self._lock:
                self._verification[key] = state
        with self._lock:
            job.result.measurement = result
            job.result.state = JobState.COMPLETE
        self._audit("measurement", result.state.value if result.state else "rejected", result.reason or "", job.result.job_id, job.actor)

    def _source_waveform(self, job: _Job) -> None:
        request = job.input_request
        assert request is not None
        if request.port < 0 or request.port >= self._program.shape.inputs:
            result = WaveformJobResult(request.port, False, reason="input port is outside the declared network shape")
        elif self._device_registry is not None:
            try:
                self._device_registry.validate_input_waveform(request.port, request.waveform)
                result = None
            except FabricError as error:
                result = WaveformJobResult(request.port, False, reason=str(error))
            if result is not None:
                with self._lock:
                    job.result.waveform = result
                    job.result.state = JobState.COMPLETE
                self._audit("input-waveform", "rejected", result.reason or "", job.result.job_id, job.actor)
                return
            if self._driver_registry is None:
                result = WaveformJobResult(request.port, False, reason="no private driver registry is configured")
            else:
                try:
                    self._driver_registry.send_waveform(request.waveform)
                    result = WaveformJobResult(request.port, True)
                except (ValueError, FabricError) as error:
                    result = WaveformJobResult(request.port, False, reason=str(error))
        elif self._driver_registry is None:
            result = WaveformJobResult(request.port, False, reason="no private driver registry is configured")
        else:
            try:
                self._driver_registry.send_waveform(request.waveform)
                result = WaveformJobResult(request.port, True)
            except (ValueError, FabricError) as error:
                result = WaveformJobResult(request.port, False, reason=str(error))
        with self._lock:
            job.result.waveform = result
            job.result.state = JobState.COMPLETE
        self._audit("input-waveform", "dispatched" if result.dispatched else "rejected", result.reason or "", job.result.job_id, job.actor)

    def _capture_waveform(self, job: _Job) -> None:
        port = job.capture_port
        assert port is not None
        if port < 0 or port >= self._program.shape.outputs:
            result = WaveformJobResult(port, False, reason="output port is outside the declared network shape")
        elif self._device_registry is not None:
            try:
                self._device_registry.validate_output_capture(port)
                result = None
            except FabricError as error:
                result = WaveformJobResult(port, False, reason=str(error))
            if result is not None:
                with self._lock:
                    job.result.waveform = result
                    job.result.state = JobState.COMPLETE
                self._audit("output-capture", "rejected", result.reason or "", job.result.job_id, job.actor)
                return
            if self._driver_registry is None:
                result = WaveformJobResult(port, False, reason="no private driver registry is configured")
            else:
                try:
                    waveform = self._driver_registry.capture_waveform()
                    result = WaveformJobResult(port, True, waveform=waveform)
                except (ValueError, FabricError) as error:
                    result = WaveformJobResult(port, False, reason=str(error))
        elif self._driver_registry is None:
            result = WaveformJobResult(port, False, reason="no private driver registry is configured")
        else:
            try:
                waveform = self._driver_registry.capture_waveform()
                result = WaveformJobResult(port, True, waveform=waveform)
            except (ValueError, FabricError) as error:
                result = WaveformJobResult(port, False, reason=str(error))
        with self._lock:
            job.result.waveform = result
            job.result.state = JobState.COMPLETE
        self._audit("output-capture", "dispatched" if result.dispatched else "rejected", result.reason or "", job.result.job_id, job.actor)

    def _apply(self, update: WeightUpdate) -> None:
        candidate = Edge(update.source, update.destination, update.weight)
        # Validate the full logical update before a real driver sees it. This
        # ordering is a safety rule: invalid coordinates must never become an
        # attempted write to an instrument merely because a batch is mixed.
        with self._lock:
            new_edges = [edge for edge in self._program.edges if (edge.source, edge.destination) != (candidate.source, candidate.destination)]
            new_edges.append(candidate)
            candidate_program = FabricProgram(shape=self._program.shape, edges=new_edges, outputs=self._program.outputs)
            Compiler().compile(candidate_program)
        if self._driver_registry is not None:
            # A return proves dispatch only; verification remains a distinct job.
            self._driver_registry.program_weight(candidate.weight, candidate.source, candidate.destination)
        with self._lock:
            self._program = candidate_program
            self._verification[(candidate.source, candidate.destination)] = VerificationState.UNVERIFIED

    def _authorize(self, actor: str, action: str) -> None:
        if self._authorization_policy is None:
            return
        try:
            self._authorization_policy.require(actor, action)
        except FabricError as error:
            self._audit("authorization", "denied", f"{action}: {error}", actor=actor)
            raise

    def _audit(self, action: str, outcome: str, detail: str = "", job_id: str | None = None, actor: str = "system") -> None:
        if self._audit_log is not None:
            self._audit_log.record(AuditEvent(action=action, outcome=outcome, detail=detail, job_id=job_id, actor=actor))
