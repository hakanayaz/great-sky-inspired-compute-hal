"""Private physical attachment and safety configuration for one HAL device.

This configuration is intentionally separate from a user's experiment request.
It represents facts established during hardware commissioning: approved port
attachments, electrical limits, shared bias ownership, and future sequencing
metadata. Keeping it private prevents an experiment from changing its own
safety envelope.
"""
from __future__ import annotations

from dataclasses import dataclass

from analog_fabric import FabricError, NetworkShape, Waveform
from drivers import Capability


@dataclass(frozen=True)
class PortAttachment:
    """One approved logical-port connection and its device-level limits."""
    logical_port: int
    capability: Capability
    min_volts: float
    max_volts: float
    max_bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if self.min_volts >= self.max_volts:
            raise FabricError("port voltage limits must form a non-empty range")
        if self.max_bandwidth_hz is not None and self.max_bandwidth_hz <= 0:
            raise FabricError("port bandwidth limit must be positive")


@dataclass(frozen=True)
class BiasGroup:
    """Outputs that share one bias resource and must be managed together."""
    name: str
    output_ports: tuple[int, ...]
    min_current_a: float
    max_current_a: float
    adapter_name: str

    def __post_init__(self) -> None:
        if self.min_current_a < 0 or self.min_current_a > self.max_current_a:
            raise FabricError("bias-current limits are invalid")


@dataclass(frozen=True)
class SafetyPlan:
    """Preferred adapter order; enforcement is the next production increment."""
    power_on_adapters: tuple[str, ...] = ()
    safe_stop_adapters: tuple[str, ...] = ()


class DeviceRegistry:
    """Non-public description of one physical device and its approved wiring."""

    def __init__(self, shape: NetworkShape, inputs: list[PortAttachment], outputs: list[PortAttachment], bias_groups: list[BiasGroup] | None = None, safety_plan: SafetyPlan = SafetyPlan()):
        self.shape = shape
        self.inputs = self._index(inputs, shape.inputs, Capability.WAVEFORM_SOURCE, "input")
        self.outputs = self._index(outputs, shape.outputs, Capability.WAVEFORM_CAPTURE, "output")
        self.bias_groups = tuple(bias_groups or [])
        self.safety_plan = safety_plan
        for group in self.bias_groups:
            if any(port not in self.outputs for port in group.output_ports):
                raise FabricError(f"bias group {group.name} refers to an unconfigured output")

    @staticmethod
    def _index(attachments: list[PortAttachment], bound: int, capability: Capability, label: str) -> dict[int, PortAttachment]:
        result: dict[int, PortAttachment] = {}
        for attachment in attachments:
            if attachment.logical_port < 0 or attachment.logical_port >= bound:
                raise FabricError(f"{label} attachment is outside the declared network shape")
            if attachment.capability != capability:
                raise FabricError(f"{label} attachment has the wrong capability")
            if attachment.logical_port in result:
                raise FabricError(f"duplicate {label} attachment")
            result[attachment.logical_port] = attachment
        return result

    def validate_input_waveform(self, port: int, waveform: Waveform) -> None:
        # Validate before resolving a driver so an unsafe waveform is rejected
        # at the HAL boundary, not discovered by a physical instrument.
        attachment = self.inputs.get(port)
        if attachment is None:
            raise FabricError("logical input has no approved physical attachment")
        if any(value < attachment.min_volts or value > attachment.max_volts for value in waveform.samples_v):
            raise FabricError("waveform exceeds the attached input voltage limits")
        if attachment.max_bandwidth_hz is not None and waveform.sample_rate_hz / 2 > attachment.max_bandwidth_hz:
            raise FabricError("waveform sample rate exceeds the attached input bandwidth")

    def validate_output_capture(self, port: int) -> None:
        if port not in self.outputs:
            raise FabricError("logical output has no approved physical attachment")
