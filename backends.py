"""Common execution contract for simulated and driver-backed fabric backends.

The contract is intentionally expressed in logical operations: program weights,
source a waveform, capture outputs, measure edges, and manage power. A future
AI-framework bridge should target this boundary rather than importing a vendor
driver or the simulator directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from analog_fabric import Compiler, Edge, FabricError, FabricProgram, Simulator, Waveform
from experiment_api import ExperimentRequest
from service import HardwareService, JobState


@dataclass(frozen=True)
class BackendItem:
    index: int
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class BackendMeasurement:
    source: int
    destination: int
    expected_v: float
    observed_v: float
    verified: bool


@dataclass(frozen=True)
class BackendRun:
    backend: str
    items: tuple[BackendItem, ...]
    captures: dict[int, Waveform]
    measurements: tuple[BackendMeasurement, ...]


class FabricBackend(Protocol):
    """Stable contract shared by simulator and physical-driver implementations."""
    name: str
    def execute(self, request: ExperimentRequest) -> BackendRun: ...


class SimulatorBackend:
    """Reference backend used for prediction, parity tests, and regression tests."""
    name = "simulator"

    def execute(self, request: ExperimentRequest) -> BackendRun:
        program = FabricProgram(shape=request.shape)
        items: list[BackendItem] = []
        for index, update in enumerate(request.weights):
            try:
                candidate = Edge(update.source, update.destination, update.weight)
                program = FabricProgram(shape=program.shape, edges=[*program.edges, candidate], outputs=program.outputs)
                Compiler().compile(program)
                items.append(BackendItem(index, True))
            except FabricError as error:
                items.append(BackendItem(index, False, str(error)))
        simulator = Simulator()
        inputs = [Waveform((0.0,), request.inputs[0].waveform.sample_rate_hz if request.inputs else 1.0) for _ in range(request.shape.inputs)]
        for source in request.inputs:
            inputs[source.port] = source.waveform
        outputs = simulator.run(Compiler().compile(program), inputs)
        captures = {port: outputs[port] for port in request.captures}
        measurements: list[BackendMeasurement] = []
        for request_measurement in request.measurements:
            stimulus = [Waveform((0.0,), 1.0) for _ in range(request.shape.inputs)]
            stimulus[request_measurement.source] = Waveform((request_measurement.stimulus_v,), 1.0)
            observed = simulator.run(Compiler().compile(program), stimulus)[request_measurement.destination].samples_v[0]
            # In the simulator the reference prediction and observation use the
            # same numerical model; physical backends will diverge as noise and
            # calibration are added, which is precisely what parity tests expose.
            expected = observed
            measurements.append(BackendMeasurement(request_measurement.source, request_measurement.destination, expected, observed, abs(expected - observed) <= request_measurement.tolerance_v))
        return BackendRun(self.name, tuple(items), captures, tuple(measurements))


class DriverBackend:
    """Backend adapter that executes the same request through HardwareService.

    The current demo uses simulated drivers, so this is a driver-backed mock.
    Replacing the adapters with vendor integrations leaves this API unchanged.
    """
    name = "driver-backed"

    def __init__(self, service: HardwareService, actors: dict[str, str] | None = None) -> None:
        self._service = service
        self._actors = actors or {}

    def _wait(self, job_id: str):
        while True:
            result = self._service.status(job_id)
            if result.state not in {JobState.QUEUED, JobState.RUNNING}:
                return result

    def execute(self, request: ExperimentRequest) -> BackendRun:
        actor = lambda action: self._actors.get(action, "system")
        if request.power_on:
            self._wait(self._service.submit_power_on(actor=actor("power")))
        for source in request.inputs:
            self._wait(self._service.submit_input_waveform(source, actor=actor("source")))
        programmed = self._wait(self._service.submit_weights(list(request.weights), actor=actor("program")))
        captures: dict[int, Waveform] = {}
        for port in request.captures:
            capture = self._wait(self._service.submit_output_capture(port, actor=actor("capture"))).waveform
            if capture and capture.dispatched and isinstance(capture.waveform, Waveform):
                captures[port] = capture.waveform
        measurements: list[BackendMeasurement] = []
        for requested in request.measurements:
            result = self._wait(self._service.submit_edge_measurement(requested, actor=actor("measure"))).measurement
            if result and result.expected_v is not None and result.observed_v is not None:
                measurements.append(BackendMeasurement(requested.source, requested.destination, result.expected_v, result.observed_v, result.state is not None and result.state.value == "verified"))
        if request.safe_stop:
            self._wait(self._service.submit_safe_stop(actor=actor("power")))
        return BackendRun(self.name, tuple(BackendItem(item.index, item.accepted, item.reason) for item in programmed.items), captures, tuple(measurements))


class NonIdealDriverBackend:
    """Deterministic physical-nonideality model wrapped around a driver backend.

    This is for parity and calibration demonstrations, not a claim about a
    particular instrument. Real backends should replace this profile with
    measured calibration data, noise models, drift, and device-specific limits.
    """
    name = "nonideal-driver-backed"

    def __init__(self, backend: FabricBackend, gain: float = 1.0, offset_v: float = 0.0) -> None:
        self._backend = backend
        self._gain = gain
        self._offset_v = offset_v

    def _transform(self, value: float) -> float:
        return value * self._gain + self._offset_v

    def execute(self, request: ExperimentRequest) -> BackendRun:
        base = self._backend.execute(request)
        captures = {
            port: Waveform(tuple(self._transform(value) for value in waveform.samples_v), waveform.sample_rate_hz)
            for port, waveform in base.captures.items()
        }
        measurements = tuple(
            BackendMeasurement(item.source, item.destination, item.expected_v, self._transform(item.observed_v), abs(item.expected_v - self._transform(item.observed_v)) <= next(
                request_measurement.tolerance_v for request_measurement in request.measurements
                if request_measurement.source == item.source and request_measurement.destination == item.destination
            ))
            for item in base.measurements
        )
        return BackendRun(self.name, base.items, captures, measurements)


def compare_runs(reference: BackendRun, candidate: BackendRun, tolerance_v: float = 1e-9) -> tuple[bool, tuple[str, ...]]:
    """Compare high-value parity signals without exposing backend internals."""
    differences: list[str] = []
    if [item.accepted for item in reference.items] != [item.accepted for item in candidate.items]:
        differences.append("weight-item outcomes differ")
    for expected, observed in zip(reference.measurements, candidate.measurements):
        if abs(expected.observed_v - observed.observed_v) > tolerance_v:
            differences.append(f"measurement {expected.source}->{expected.destination} differs")
    if len(reference.measurements) != len(candidate.measurements):
        differences.append("measurement count differs")
    for port, expected_waveform in reference.captures.items():
        observed_waveform = candidate.captures.get(port)
        if observed_waveform is None or expected_waveform.samples_v != observed_waveform.samples_v:
            differences.append(f"capture on output {port} differs")
    return not differences, tuple(differences)
