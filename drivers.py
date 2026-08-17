"""Private adapters for fixed, instrument-specific driver signatures.

The underlying drivers may return no value.  A normal return therefore means
only that a command was dispatched, not that physical state is verified.

The Protocol classes below intentionally mirror external signatures rather
than improving them. Adapters are where the HAL translates those diverse,
fixed shapes into logical capabilities. Future adapters can add new hardware
without requiring application algorithms to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Capability(str, Enum):
    EDGE_WEIGHT = "edge-weight"
    OUTPUT_BIAS = "output-bias"
    WAVEFORM_SOURCE = "waveform-source"
    WAVEFORM_CAPTURE = "waveform-capture"


@dataclass(frozen=True)
class DispatchReceipt:
    capability: Capability
    adapter_name: str


@dataclass(frozen=True)
class SafetyResult:
    completed: tuple[str, ...]
    failures: tuple[str, ...]


class VoltageSource(Protocol):
    """Fixed external driver signature; the HAL does not change it."""
    def output(self, on_or_off: bool) -> None: ...
    def set_volt(self, value: float) -> None: ...
    def get_volt(self) -> float: ...
    def set_memory(self, value: float) -> None: ...


class Memory_controller(Protocol):
    """Fixed external driver signature; no return value is required."""
    def set_memory(self, value: float, row: int, col: int) -> None: ...


class AWG(Protocol):
    def set_volt(self, value: float) -> None: ...
    def set_waveform(self, wave: object) -> None: ...
    def output_on(self) -> None: ...
    def output_off(self) -> None: ...


class Scope(Protocol):
    def capture_waveform(self) -> None: ...
    def get_waveform(self) -> object: ...


class RFSoC(Protocol):
    def set_waveform(self, wave: object) -> None: ...
    def capture_waveform(self) -> None: ...
    def get_waveform(self) -> object: ...
    def output_on(self) -> None: ...
    def output_off(self) -> None: ...


class DriverAdapter(Protocol):
    name: str
    capabilities: frozenset[Capability]


@dataclass
class MemoryControllerAdapter:
    name: str
    driver: Memory_controller
    capabilities: frozenset[Capability] = frozenset({Capability.EDGE_WEIGHT})

    def program_weight(self, value: float, source: int, destination: int) -> DispatchReceipt:
        self.driver.set_memory(value, source, destination)
        return DispatchReceipt(Capability.EDGE_WEIGHT, self.name)


@dataclass
class VoltageSourceAdapter:
    name: str
    driver: VoltageSource
    capabilities: frozenset[Capability] = frozenset({Capability.OUTPUT_BIAS})

    def set_output_bias(self, volts: float) -> DispatchReceipt:
        self.driver.set_volt(volts)
        return DispatchReceipt(Capability.OUTPUT_BIAS, self.name)

    def power_on(self) -> None:
        self.driver.output(True)

    def safe_stop(self) -> None:
        self.driver.output(False)


@dataclass
class VoltageSourceMemoryAdapter:
    """Use a voltage source's fixed set_memory method for one bound edge."""
    name: str
    driver: VoltageSource
    edge: tuple[int, int]
    capabilities: frozenset[Capability] = frozenset({Capability.EDGE_WEIGHT})

    def program_weight(self, value: float, source: int, destination: int) -> DispatchReceipt:
        if (source, destination) != self.edge:
            raise ValueError("voltage source is not wired to the requested logical edge")
        self.driver.set_memory(value)
        return DispatchReceipt(Capability.EDGE_WEIGHT, self.name)


@dataclass
class RFSoCAdapter:
    name: str
    driver: RFSoC
    capabilities: frozenset[Capability] = frozenset({Capability.WAVEFORM_SOURCE, Capability.WAVEFORM_CAPTURE})

    def send_waveform(self, wave: object) -> DispatchReceipt:
        self.driver.set_waveform(wave)
        self.driver.output_on()
        return DispatchReceipt(Capability.WAVEFORM_SOURCE, self.name)

    def capture_waveform(self) -> object:
        self.driver.capture_waveform()
        return self.driver.get_waveform()

    def power_on(self) -> None:
        self.driver.output_on()

    def safe_stop(self) -> None:
        self.driver.output_off()


@dataclass
class AWGScopeAdapter:
    """One logical source/capture provider backed by two physical drivers."""
    name: str
    awg: AWG
    scope: Scope
    capabilities: frozenset[Capability] = frozenset({Capability.WAVEFORM_SOURCE, Capability.WAVEFORM_CAPTURE})

    def send_waveform(self, wave: object) -> DispatchReceipt:
        self.awg.set_waveform(wave)
        self.awg.output_on()
        return DispatchReceipt(Capability.WAVEFORM_SOURCE, self.name)

    def capture_waveform(self) -> object:
        self.scope.capture_waveform()
        return self.scope.get_waveform()

    def power_on(self) -> None:
        self.awg.output_on()

    def safe_stop(self) -> None:
        self.awg.output_off()


class DriverRegistry:
    """Private capability-to-adapter map, owned by one hardware service.

    This is not a public service locator: exposing it would leak physical
    wiring and allow callers to bypass queueing, validation, and audit rules.
    """

    def __init__(self) -> None:
        self._adapters: dict[Capability, DriverAdapter] = {}

    def register(self, adapter: DriverAdapter) -> None:
        for capability in adapter.capabilities:
            if capability in self._adapters:
                raise ValueError(f"capability already has an adapter: {capability.value}")
            self._adapters[capability] = adapter

    def poll_driver(self, attribute: str | Capability) -> DriverAdapter:
        """Return the private adapter serving a logical hardware attribute."""
        try:
            capability = attribute if isinstance(attribute, Capability) else Capability(attribute)
        except ValueError as error:
            raise ValueError(f"unknown driver attribute: {attribute}") from error
        adapter = self._adapters.get(capability)
        if adapter is None:
            raise ValueError(f"no private driver adapter provides {capability.value}")
        return adapter

    def program_weight(self, value: float, source: int, destination: int) -> DispatchReceipt:
        adapter = self.poll_driver(Capability.EDGE_WEIGHT)
        method = getattr(adapter, "program_weight", None)
        if method is None:
            raise ValueError("selected adapter cannot program an edge weight")
        return method(value, source, destination)

    def send_waveform(self, wave: object) -> DispatchReceipt:
        adapter = self.poll_driver(Capability.WAVEFORM_SOURCE)
        method = getattr(adapter, "send_waveform", None)
        if method is None:
            raise ValueError("selected adapter cannot source a waveform")
        return method(wave)

    def capture_waveform(self) -> object:
        adapter = self.poll_driver(Capability.WAVEFORM_CAPTURE)
        method = getattr(adapter, "capture_waveform", None)
        if method is None:
            raise ValueError("selected adapter cannot capture a waveform")
        return method()

    def power_on(self) -> SafetyResult:
        return self._run_safety_action("power_on")

    def safe_stop(self) -> SafetyResult:
        return self._run_safety_action("safe_stop")

    def _run_safety_action(self, method_name: str) -> SafetyResult:
        # A safe-stop attempts every independently controllable output even if
        # one adapter fails. The caller receives all failures and must treat
        # the resulting physical state as faulted rather than silently safe.
        completed: list[str] = []
        failures: list[str] = []
        seen: set[int] = set()
        for adapter in self._adapters.values():
            if id(adapter) in seen:
                continue
            seen.add(id(adapter))
            method = getattr(adapter, method_name, None)
            if method is None:
                continue
            try:
                method()
                completed.append(adapter.name)
            except Exception as error:
                failures.append(f"{adapter.name}: {error}")
        return SafetyResult(tuple(completed), tuple(failures))
