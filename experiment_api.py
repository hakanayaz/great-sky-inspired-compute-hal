"""Declarative, high-level experiment request for the analog-fabric HAL.

The JSON request is intentionally a statement of desired experiment behavior,
not a backdoor to physical configuration. It names logical resources only;
the device and driver registries remain the authority on safe attachments,
limits, and instrument selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from analog_fabric import Edge, FabricError, NetworkShape, Waveform
from service import InputWaveformRequest, MeasurementRequest, WeightUpdate


@dataclass(frozen=True)
class ExperimentRequest:
    name: str
    shape: NetworkShape
    weights: tuple[WeightUpdate, ...]
    inputs: tuple[InputWaveformRequest, ...]
    captures: tuple[int, ...]
    measurements: tuple[MeasurementRequest, ...]
    power_on: bool
    safe_stop: bool

    def valid_edges(self) -> list[Edge]:
        """Edges that can appear in the compiler preview before dispatch."""
        edges: list[Edge] = []
        for update in self.weights:
            try:
                candidate = Edge(update.source, update.destination, update.weight)
                if candidate.source < self.shape.inputs and candidate.destination < self.shape.outputs:
                    edges.append(candidate)
            except FabricError:
                pass
        return edges


def load_experiment(path: str | Path) -> ExperimentRequest:
    """Parse a user-facing use-case description into the HAL's typed request.

    Future work: replace this small JSON loader with a versioned API schema,
    richer diagnostics, authenticated request metadata, and a network-facing
    HTTP or gRPC endpoint.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        shape = NetworkShape(int(data["network"]["inputs"]), int(data["network"]["outputs"]))
        weights = tuple(WeightUpdate(int(item["source"]), int(item["destination"]), float(item["weight"])) for item in data.get("weights", []))
        inputs = tuple(
            InputWaveformRequest(int(item["port"]), Waveform(tuple(float(value) for value in item["samples_v"]), float(item["sample_rate_hz"])))
            for item in data.get("inputs", [])
        )
        measurements = tuple(
            MeasurementRequest(int(item["source"]), int(item["destination"]), float(item["stimulus_v"]), float(item["tolerance_v"]))
            for item in data.get("measurements", [])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FabricError(f"invalid experiment request: {error}") from error
    return ExperimentRequest(
        name=str(data.get("name", "experiment")), shape=shape, weights=weights, inputs=inputs,
        captures=tuple(int(port) for port in data.get("captures", [])), measurements=measurements,
        power_on=bool(data.get("power_on", False)), safe_stop=bool(data.get("safe_stop", False)),
    )
