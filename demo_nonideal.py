"""Demo 2: use backend parity to expose deterministic physical nonidealities."""
from pathlib import Path

from audit import JsonlAuditLog
from backends import DriverBackend, NonIdealDriverBackend, SimulatorBackend, compare_runs
from demo import PrototypeBiasSupply, PrototypeFabricMemoryController, RFSoCTestSystem
from device_registry import DeviceRegistry, PortAttachment
from drivers import Capability, DriverRegistry, MemoryControllerAdapter, RFSoCAdapter, VoltageSourceAdapter
from experiment_api import load_experiment
from security import AuthorizationPolicy, Role
from service import HardwareService


def main() -> None:
    request = load_experiment(Path(__file__).with_name("demo_experiment.json"))
    controller, rfsoc, bias = PrototypeFabricMemoryController(), RFSoCTestSystem(), PrototypeBiasSupply()
    drivers = DriverRegistry()
    drivers.register(MemoryControllerAdapter("prototype-fabric-memory", controller))
    drivers.register(RFSoCAdapter("rfsoc-test-system", rfsoc))
    drivers.register(VoltageSourceAdapter("prototype-bias-supply", bias))
    device = DeviceRegistry(
        request.shape,
        inputs=[PortAttachment(port, Capability.WAVEFORM_SOURCE, -1.0, 1.0, 1_000.0) for port in (0, 1)],
        outputs=[PortAttachment(port, Capability.WAVEFORM_CAPTURE, -1.0, 1.0) for port in (2, 3)],
    )
    policy = AuthorizationPolicy({"observer": Role.INPUT_OUTPUT, "operator": Role.OPERATOR})
    audit = JsonlAuditLog(Path(__file__).with_name("demo_nonideal_audit.jsonl"))
    service = HardwareService(driver_registry=drivers, device_registry=device, audit_log=audit, authorization_policy=policy)
    try:
        print("=== Demo 2: calibration/parity scenario ===")
        print("Physical profile: gain = 0.92, offset = 0.01 V")
        reference = SimulatorBackend().execute(request)
        driver = DriverBackend(service, {"source": "observer", "capture": "observer", "program": "operator", "measure": "operator", "power": "operator"})
        observed = NonIdealDriverBackend(driver, gain=0.92, offset_v=0.01).execute(request)
        matches, differences = compare_runs(reference, observed, tolerance_v=1e-9)
        for measurement in observed.measurements:
            print(f"edge {measurement.source}->{measurement.destination}: expected {measurement.expected_v:.4f} V, observed {measurement.observed_v:.4f} V, verified={measurement.verified}")
        print("Parity match:", matches)
        print("Differences:", differences or "none")
        print("Interpretation: this is an expected calibration/regression signal, not a queue or authorization failure.")
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
