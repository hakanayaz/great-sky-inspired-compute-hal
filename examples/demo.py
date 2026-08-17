"""Closed-loop 4x4 HAL demonstration with safety and audit behavior."""
from pathlib import Path
from time import sleep

from analog_fabric_hal.analog_fabric import Compiler, FabricProgram, Waveform, render_ir
from analog_fabric_hal.audit import JsonlAuditLog
from analog_fabric_hal.backends import DriverBackend, SimulatorBackend, compare_runs
from analog_fabric_hal.device_registry import DeviceRegistry, PortAttachment
from analog_fabric_hal.drivers import Capability, DriverRegistry, MemoryControllerAdapter, RFSoCAdapter, VoltageSourceAdapter
from analog_fabric_hal.experiment_api import load_experiment
from analog_fabric_hal.security import AuthorizationError, AuthorizationPolicy, Role
from analog_fabric_hal.service import HardwareService, JobState


class PrototypeFabricMemoryController:
    """Mock edge-memory programmer for the prototype compute fabric."""
    def __init__(self) -> None:
        self.writes: list[tuple[float, int, int]] = []

    def set_memory(self, value: float, row: int, col: int) -> None:
        self.writes.append((value, row, col))


class RFSoCTestSystem:
    """Mock RFSoC test endpoint that sources and captures analog waveforms."""
    def __init__(self) -> None:
        self.source_waveform: Waveform | None = None
        self.output_enabled = False
        self.captured_waveform = Waveform((0.4,), 1_000.0)

    def set_waveform(self, wave: Waveform) -> None:
        self.source_waveform = wave

    def capture_waveform(self) -> None:
        pass

    def get_waveform(self) -> Waveform:
        return self.captured_waveform

    def output_on(self) -> None:
        self.output_enabled = True

    def output_off(self) -> None:
        self.output_enabled = False


class PrototypeBiasSupply:
    """Mock shared bias supply for the prototype compute fabric."""
    def __init__(self) -> None:
        self.enabled = False

    def output(self, on_or_off: bool) -> None:
        self.enabled = on_or_off

    def set_volt(self, value: float) -> None:
        pass

    def get_volt(self) -> float:
        return 0.0

    def set_memory(self, value: float) -> None:
        pass


def wait_for(service: HardwareService, job_id: str):
    while True:
        result = service.status(job_id)
        if result.state not in {JobState.QUEUED, JobState.RUNNING}:
            return result
        sleep(0.001)


def main() -> None:
    request_path = Path(__file__).with_name("demo_experiment.json")
    request = load_experiment(request_path)
    shape = request.shape
    fabric = FabricProgram(shape=shape)
    requested_edges = request.valid_edges()
    requested_program = FabricProgram(shape=shape, edges=requested_edges)

    print("\n=== 0. Declarative experiment request ===")
    print("Use case:", request.name)
    print("Request file:", request_path.name)

    print("\n=== 1. Logical program (MLIR-inspired IR) ===")
    print(render_ir(requested_program, request.name))

    print("\n=== 2. Compiler validation ===")
    compiled = Compiler().compile(requested_program)
    print("shape:", f"{shape.inputs} inputs x {shape.outputs} outputs")
    print("warnings:", compiled.warnings or "none")

    controller, rfsoc, bias = PrototypeFabricMemoryController(), RFSoCTestSystem(), PrototypeBiasSupply()
    drivers = DriverRegistry()
    drivers.register(MemoryControllerAdapter("prototype-fabric-memory", controller))
    drivers.register(RFSoCAdapter("rfsoc-test-system", rfsoc))
    drivers.register(VoltageSourceAdapter("prototype-bias-supply", bias))
    device = DeviceRegistry(
        shape,
        inputs=[PortAttachment(port, Capability.WAVEFORM_SOURCE, -1.0, 1.0, 1_000.0) for port in (0, 1)],
        outputs=[PortAttachment(port, Capability.WAVEFORM_CAPTURE, -1.0, 1.0) for port in (2, 3)],
    )
    audit_path = Path(__file__).with_name("demo_audit.jsonl")
    policy = AuthorizationPolicy({"observer": Role.INPUT_OUTPUT, "operator": Role.OPERATOR})
    service = HardwareService(program=fabric, driver_registry=drivers, device_registry=device, audit_log=JsonlAuditLog(audit_path), authorization_policy=policy)
    try:
        print("\n=== 3. HAL lowering ===")
        print("edge-weight -> prototype-fabric-memory; waveform source/capture -> rfsoc-test-system; output bias -> prototype-bias-supply")

        if request.power_on:
            print("\n=== 4. Queued power-on ===")
            power = wait_for(service, service.submit_power_on(actor="operator"))
            print("Power state:", power.lifecycle.state.value, "adapters enabled:", power.lifecycle.completed)

        print("\n=== 5. Logical waveform source job ===")
        for input_request in request.inputs:
            source = wait_for(service, service.submit_input_waveform(input_request, actor="observer"))
            print(f"Input port {input_request.port}:", "dispatched" if source.waveform.dispatched else source.waveform.reason)

        print("\n=== 6. Per-item programming batch ===")
        try:
            service.submit_weights([request.weights[0]], actor="observer")
        except AuthorizationError as error:
            print("Observer programming attempt: denied (", error, ")", sep="")
        programmed = wait_for(service, service.submit_weights(list(request.weights), actor="operator"))
        for item in programmed.items:
            print(f"  item {item.index}: edge {item.update.source}->{item.update.destination}, weight={item.update.weight}: ", end="")
            print("accepted" if item.accepted else f"rejected ({item.reason})")
        print("Private driver writes:", controller.writes)

        print("\n=== 7. Logical output capture ===")
        for port in request.captures:
            capture = wait_for(service, service.submit_output_capture(port, actor="observer"))
            print(f"Output port {port} capture:", capture.waveform.waveform.samples_v if capture.waveform.dispatched else capture.waveform.reason)

        print("\n=== 8. Explicit edge measurements ===")
        for measurement_request in request.measurements:
            measurement = wait_for(service, service.submit_edge_measurement(measurement_request, actor="operator")).measurement
            print(f"  edge {measurement_request.source}->{measurement_request.destination}:", measurement.state.value, "expected", measurement.expected_v, "observed", measurement.observed_v)

        if request.safe_stop:
            print("\n=== 9. Queued safe-stop and audit ===")
            safe_stop = wait_for(service, service.submit_safe_stop(actor="operator"))
            print("Power state:", safe_stop.lifecycle.state.value, "adapters disabled:", safe_stop.lifecycle.completed)
        audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
        print("Durable audit events written:", len(audit_lines))
        print("Latest audit event:", audit_lines[-1])

        print("\n=== 10. Backend parity check ===")
        simulated = SimulatorBackend().execute(request)
        driver_backed = DriverBackend(service, {"source": "observer", "capture": "observer", "program": "operator", "measure": "operator", "power": "operator"}).execute(request)
        matches, differences = compare_runs(simulated, driver_backed)
        print("Simulator and driver-backed runs match:", matches)
        print("Parity differences:", differences or "none")
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
