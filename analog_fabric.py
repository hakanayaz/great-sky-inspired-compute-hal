"""A small, hardware-shaped compiler and simulator for analog signal fabrics.

Public values use SI units: volts, amperes, seconds, and hertz.  The simulator
uses discrete samples internally, which lets real instrument drivers replace
the simulated drivers later without changing compiled programs.

The simulator is not a claim that a physical analog device is sample-based.
It is a testable approximation used to validate the compiler contract before
hardware-specific calibration, noise, fractional delay, and live streaming are
available. Those are intentional future extensions, not hidden omissions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi, tanh
from typing import Callable, Mapping, Protocol


class FabricError(ValueError):
    """Raised when a graph cannot safely be compiled or simulated."""


@dataclass(frozen=True)
class Waveform:
    samples_v: tuple[float, ...]
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise FabricError("sample_rate_hz must be positive")
        if not self.samples_v:
            raise FabricError("waveform must contain at least one sample")


@dataclass(frozen=True)
class NetworkShape:
    """Logical dimensions; physical wiring remains private to the HAL.
    """
    inputs: int = 4
    outputs: int = 4

    def __post_init__(self) -> None:
        if self.inputs <= 0 or self.outputs <= 0:
            raise FabricError("network dimensions must both be positive")


@dataclass(frozen=True)
class Edge:
    """One physical connection: a programmable weight and fixed transfer behavior."""
    source: int
    destination: int
    weight: float
    bandwidth_hz: float | None = None
    delay_s: float = 0.0
    transfer: str = "linear"

    def __post_init__(self) -> None:
        if self.source < 0 or self.destination < 0:
            raise FabricError("edge ports cannot be negative")
        if not -1.0 <= self.weight <= 1.0:
            raise FabricError("weight must be in the normalized range -1..1")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0:
            raise FabricError("bandwidth_hz must be positive")
        if self.delay_s < 0:
            raise FabricError("delay_s cannot be negative")
        if not self.transfer.strip():
            raise FabricError("edge transfer must have a non-empty fixed identifier")


@dataclass(frozen=True)
class OutputConfig:
    bias_v: float = 0.0
    min_v: float = -1.0
    max_v: float = 1.0
    bias_group: str = "default"
    bias_current_a: float | None = None

    def __post_init__(self) -> None:
        if self.min_v >= self.max_v:
            raise FabricError("output min_v must be lower than max_v")
        if not self.min_v <= self.bias_v <= self.max_v:
            raise FabricError("bias_v must be within the output range")
        if self.bias_current_a is not None and self.bias_current_a < 0:
            raise FabricError("bias_current_a cannot be negative")


@dataclass
class FabricProgram:
    shape: NetworkShape = field(default_factory=NetworkShape)
    edges: list[Edge] = field(default_factory=list)
    outputs: list[OutputConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.outputs:
            self.outputs = [OutputConfig() for _ in range(self.shape.outputs)]
        if len(self.outputs) != self.shape.outputs:
            raise FabricError("output configuration count must match network shape")
        for edge in self.edges:
            if edge.source >= self.shape.inputs or edge.destination >= self.shape.outputs:
                raise FabricError("edge port is outside the declared network shape")


@dataclass(frozen=True)
class CompiledFabric:
    program: FabricProgram
    warnings: tuple[str, ...]


class Compiler:
    """Validates topology and prepares a program for a particular device shape."""

    def compile(self, program: FabricProgram) -> CompiledFabric:
        warnings: list[str] = []
        destinations = {edge.destination for edge in program.edges}
        for port in range(program.shape.outputs):
            if port not in destinations:
                warnings.append(f"output {port} has no incoming edge")
        for edge in program.edges:
            if edge.weight < 0:
                warnings.append(
                    f"edge {edge.source}->{edge.destination} uses a signed weight; "
                    "physical hardware may map it to differential positive/negative paths"
                )
        return CompiledFabric(program, tuple(warnings))


def render_ir(program: FabricProgram, symbol: str = "network") -> str:
    """Render a compact, MLIR-inspired view of a logical fabric program.

    This is an explanatory IR for the demo, not an implementation of MLIR.
    A future compiler could lower the same model into a formal dialect or a
    backend-specific instruction stream.
    """
    lines = [f"fabric.network @{symbol} {{", f"  fabric.shape inputs={program.shape.inputs} outputs={program.shape.outputs}"]
    for edge in sorted(program.edges, key=lambda item: (item.source, item.destination)):
        bandwidth = "unlimited" if edge.bandwidth_hz is None else f"{edge.bandwidth_hz}Hz"
        lines.append(
            f"  fabric.edge %in{edge.source} -> %out{edge.destination} "
            f"{{weight={edge.weight}, transfer={edge.transfer}, bandwidth={bandwidth}, delay={edge.delay_s}s}}"
        )
    for port, output in enumerate(program.outputs):
        lines.append(
            f"  fabric.output %out{port} {{bias={output.bias_v}V, range=[{output.min_v}V, {output.max_v}V]}}"
        )
    return "\n".join(lines + ["}"])


def _low_pass(samples: list[float], sample_rate_hz: float, bandwidth_hz: float | None) -> list[float]:
    if bandwidth_hz is None:
        return samples
    alpha = 1.0 - exp(-2.0 * pi * bandwidth_hz / sample_rate_hz)
    result: list[float] = []
    previous = 0.0
    for value in samples:
        previous += alpha * (value - previous)
        result.append(previous)
    return result


TransferFunction = Callable[[float], float]


def _linear(value: float) -> float:
    return value


def _apply_transfer(
    samples: list[float], transfer: str, functions: Mapping[str, TransferFunction]
) -> list[float]:
    """Apply an edge's fixed response before its programmable gain.

    The edge stores a stable function identifier, not executable user code.
    A simulation or physical backend decides which identifiers it supports.
    """
    try:
        function = functions[transfer]
    except KeyError as error:
        raise FabricError(f"simulator does not implement fixed edge transfer {transfer!r}") from error
    return [function(value) for value in samples]


def _delay(samples: list[float], sample_rate_hz: float, delay_s: float) -> list[float]:
    steps = round(delay_s * sample_rate_hz)
    return [0.0] * steps + samples[: len(samples) - steps] if steps else samples


class Simulator:
    """Sampled approximation of a continuous-time analog fabric."""

    def __init__(self, transfer_functions: Mapping[str, TransferFunction] | None = None) -> None:
        # These are only convenient toy implementations. A device backend can
        # advertise different fixed functions without changing Edge or the HAL.
        self._transfer_functions: dict[str, TransferFunction] = {
            "linear": _linear,
            "tanh": tanh,
        }
        if transfer_functions:
            self._transfer_functions.update(transfer_functions)

    def run(self, compiled: CompiledFabric, inputs: list[Waveform]) -> list[Waveform]:
        if len(inputs) != compiled.program.shape.inputs:
            raise FabricError("input waveform count must match network shape")
        rate = inputs[0].sample_rate_hz
        length = len(inputs[0].samples_v)
        if any(w.sample_rate_hz != rate or len(w.samples_v) != length for w in inputs):
            raise FabricError("all inputs must share a sample rate and length")
        sums = [[0.0] * length for _ in range(compiled.program.shape.outputs)]
        for edge in compiled.program.edges:
            signal = _apply_transfer(list(inputs[edge.source].samples_v), edge.transfer, self._transfer_functions)
            signal = _low_pass(signal, rate, edge.bandwidth_hz)
            signal = _delay(signal, rate, edge.delay_s)
            for index, value in enumerate(signal):
                sums[edge.destination][index] += edge.weight * value
        outputs: list[Waveform] = []
        for port, config in enumerate(compiled.program.outputs):
            clipped = tuple(min(config.max_v, max(config.min_v, value + config.bias_v)) for value in sums[port])
            outputs.append(Waveform(clipped, rate))
        return outputs


class WaveformSource(Protocol):
    def produce(self) -> Waveform: ...


class WaveformSink(Protocol):
    def capture(self, waveform: Waveform) -> None: ...


@dataclass
class SimulatedAWG:
    waveform: Waveform

    def produce(self) -> Waveform:
        return self.waveform


@dataclass
class SimulatedScope:
    captures: list[Waveform] = field(default_factory=list)

    def capture(self, waveform: Waveform) -> None:
        self.captures.append(waveform)


class SimulatedRFSoC(SimulatedAWG):
    """Placeholder with the same source contract as a future RFSoC driver."""
