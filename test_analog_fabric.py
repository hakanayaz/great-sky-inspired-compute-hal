import unittest
from time import sleep
from tempfile import TemporaryDirectory

from analog_fabric import Compiler, Edge, FabricError, FabricProgram, NetworkShape, OutputConfig, Simulator, Waveform
from audit import JsonlAuditLog
from backends import SimulatorBackend, compare_runs
from experiment_api import ExperimentRequest
from device_registry import DeviceRegistry, PortAttachment
from service import HardwareService, InputWaveformRequest, JobState, LifecycleState, MeasurementRequest, VerificationState, WeightUpdate
from drivers import AWGScopeAdapter, DriverRegistry, MemoryControllerAdapter, RFSoCAdapter, VoltageSourceAdapter
from drivers import Capability
from security import AuthorizationError, AuthorizationPolicy, Role


class FabricTests(unittest.TestCase):
    def test_weighted_routing_and_bias(self):
        input0 = Waveform((0.0, 1.0, 1.0), 1_000.0)
        quiet = Waveform((0.0, 0.0, 0.0), 1_000.0)
        program = FabricProgram(edges=[Edge(0, 2, 0.5)], outputs=[OutputConfig() for _ in range(2)] + [OutputConfig(bias_v=0.1)] + [OutputConfig()])
        output = Simulator().run(Compiler().compile(program), [input0, quiet, quiet, quiet])[2]
        self.assertEqual(output.samples_v, (0.1, 0.6, 0.6))

    def test_output_saturates(self):
        signal = Waveform((1.0,), 1_000.0)
        program = FabricProgram(edges=[Edge(0, 0, 1.0)], outputs=[OutputConfig(bias_v=0.5, max_v=1.0)] + [OutputConfig()] * 3)
        self.assertEqual(Simulator().run(Compiler().compile(program), [signal] * 4)[0].samples_v, (1.0,))

    def test_edge_owns_a_fixed_tanh_transfer(self):
        signal = Waveform((1.0,), 1_000.0)
        program = FabricProgram(
            shape=NetworkShape(1, 1),
            edges=[Edge(0, 0, 0.5, transfer="tanh")],
        )
        output = Simulator().run(Compiler().compile(program), [signal])[0]
        self.assertAlmostEqual(output.samples_v[0], 0.5 * 0.7615941559557649)

    def test_simulator_can_register_a_device_specific_fixed_transfer(self):
        signal = Waveform((0.5,), 1_000.0)
        program = FabricProgram(
            shape=NetworkShape(1, 1),
            edges=[Edge(0, 0, 0.5, transfer="square")],
        )
        output = Simulator({"square": lambda value: value * value}).run(Compiler().compile(program), [signal])[0]
        self.assertEqual(output.samples_v, (0.125,))

    def test_rejects_inconsistent_inputs(self):
        program = FabricProgram()
        with self.assertRaises(FabricError):
            Simulator().run(Compiler().compile(program), [Waveform((0.0,), 1.0)] * 3)

    def test_batch_reports_each_item(self):
        service = HardwareService()
        try:
            job = service.submit_weights([WeightUpdate(0, 0, 0.5), WeightUpdate(1, 4, 0.2), WeightUpdate(2, 2, -0.5)])
            for _ in range(100):
                result = service.status(job)
                if result.state == JobState.COMPLETE:
                    break
                sleep(0.001)
            self.assertEqual(result.state, JobState.COMPLETE)
            self.assertEqual([item.accepted for item in result.items], [True, False, True])
            self.assertIn("outside the declared network shape", result.items[1].reason)
        finally:
            service.shutdown()

    def test_global_stop_rejects_new_work(self):
        service = HardwareService()
        try:
            service.emergency_stop()
            job = service.submit_weights([WeightUpdate(0, 0, 0.5)])
            self.assertEqual(service.status(job).state, JobState.STOPPED)
        finally:
            service.shutdown()

    def test_measurement_verifies_a_programmed_edge(self):
        service = HardwareService()
        try:
            write = service.submit_weights([WeightUpdate(0, 1, 0.5)])
            while service.status(write).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertEqual(service.edge_verification(0, 1), VerificationState.UNVERIFIED)
            measure = service.submit_edge_measurement(MeasurementRequest(0, 1, stimulus_v=0.8, tolerance_v=1e-9))
            while service.status(measure).state != JobState.COMPLETE:
                sleep(0.001)
            result = service.status(measure).measurement
            self.assertEqual(result.state, VerificationState.VERIFIED)
            self.assertAlmostEqual(result.observed_v, 0.4)
        finally:
            service.shutdown()

    def test_arbitrary_network_shape(self):
        shape = NetworkShape(inputs=2, outputs=3)
        program = FabricProgram(shape=shape, edges=[Edge(1, 2, 0.5)])
        quiet = Waveform((0.0, 0.0), 1_000.0)
        signal = Waveform((1.0, 0.5), 1_000.0)
        outputs = Simulator().run(Compiler().compile(program), [quiet, signal])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[2].samples_v, (0.5, 0.25))

    def test_rejects_edge_outside_shape(self):
        with self.assertRaises(FabricError):
            FabricProgram(shape=NetworkShape(2, 2), edges=[Edge(2, 0, 0.1)])

    def test_memory_controller_is_private_to_the_service(self):
        class FakeMemoryController:
            def __init__(self):
                self.calls = []

            def set_memory(self, value, row, col):
                self.calls.append((value, row, col))

        driver = FakeMemoryController()
        registry = DriverRegistry()
        adapter = MemoryControllerAdapter("lab-memory", driver)
        registry.register(adapter)
        self.assertIs(registry.poll_driver("edge-weight"), adapter)
        service = HardwareService(driver_registry=registry)
        try:
            job = service.submit_weights([WeightUpdate(1, 2, 0.25)])
            while service.status(job).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertEqual(driver.calls, [(0.25, 1, 2)])
        finally:
            service.shutdown()

    def test_invalid_weight_update_never_reaches_driver(self):
        class FakeMemoryController:
            def __init__(self): self.calls = []
            def set_memory(self, value, row, col): self.calls.append((value, row, col))

        driver = FakeMemoryController()
        registry = DriverRegistry()
        registry.register(MemoryControllerAdapter("lab-memory", driver))
        service = HardwareService(program=FabricProgram(shape=NetworkShape(2, 2)), driver_registry=registry)
        try:
            job = service.submit_weights([WeightUpdate(2, 0, 0.1)])
            while service.status(job).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertFalse(service.status(job).items[0].accepted)
            self.assertEqual(driver.calls, [])
        finally:
            service.shutdown()

    def test_rfsoc_and_awg_scope_supply_the_same_logical_waveform_capabilities(self):
        class FakeRFSoC:
            def __init__(self): self.wave = None; self.enabled = False
            def set_waveform(self, wave): self.wave = wave
            def capture_waveform(self): pass
            def get_waveform(self): return self.wave
            def output_on(self): self.enabled = True
            def output_off(self): self.enabled = False

        rfsoc = FakeRFSoC()
        registry = DriverRegistry()
        registry.register(RFSoCAdapter("rfsoc", rfsoc))
        registry.send_waveform("pulse")
        self.assertEqual(registry.capture_waveform(), "pulse")
        self.assertTrue(rfsoc.enabled)

    def test_service_queues_logical_waveform_source_and_capture(self):
        class FakeRFSoC:
            def __init__(self): self.wave = None
            def set_waveform(self, wave): self.wave = wave
            def capture_waveform(self): pass
            def get_waveform(self): return self.wave
            def output_on(self): pass
            def output_off(self): pass

        registry = DriverRegistry()
        registry.register(RFSoCAdapter("rfsoc", FakeRFSoC()))
        service = HardwareService(program=FabricProgram(shape=NetworkShape(2, 3)), driver_registry=registry)
        wave = Waveform((0.1, 0.2), 1_000.0)
        try:
            source = service.submit_input_waveform(InputWaveformRequest(1, wave))
            while service.status(source).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertTrue(service.status(source).waveform.dispatched)
            capture = service.submit_output_capture(2)
            while service.status(capture).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertEqual(service.status(capture).waveform.waveform, wave)
        finally:
            service.shutdown()

    def test_power_sequence_and_audit_log(self):
        class FakeVoltageSource:
            def __init__(self): self.history = []
            def output(self, enabled): self.history.append(enabled)
            def set_volt(self, value): pass
            def get_volt(self): return 0.0
            def set_memory(self, value): pass

        source = FakeVoltageSource()
        registry = DriverRegistry()
        registry.register(VoltageSourceAdapter("bias-source", source))
        with TemporaryDirectory() as directory:
            audit = JsonlAuditLog(f"{directory}/audit.jsonl")
            service = HardwareService(driver_registry=registry, audit_log=audit)
            try:
                power_on = service.submit_power_on()
                while service.status(power_on).state != JobState.COMPLETE:
                    sleep(0.001)
                self.assertEqual(service.status(power_on).lifecycle.state, LifecycleState.ON)
                safe_stop = service.submit_safe_stop()
                while service.status(safe_stop).state != JobState.COMPLETE:
                    sleep(0.001)
                self.assertEqual(service.status(safe_stop).lifecycle.state, LifecycleState.OFF)
                self.assertEqual(source.history, [True, False])
                self.assertIn('"action": "power_on"', audit.path.read_text())
                self.assertIn('"action": "safe_stop"', audit.path.read_text())
            finally:
                service.shutdown()

    def test_device_registry_rejects_unattached_or_overrange_waveforms(self):
        class FakeRFSoC:
            def set_waveform(self, wave): pass
            def capture_waveform(self): pass
            def get_waveform(self): return Waveform((0.0,), 1_000.0)
            def output_on(self): pass
            def output_off(self): pass

        shape = NetworkShape(2, 3)
        device = DeviceRegistry(
            shape,
            inputs=[PortAttachment(0, Capability.WAVEFORM_SOURCE, -0.5, 0.5, 1_000.0)],
            outputs=[PortAttachment(2, Capability.WAVEFORM_CAPTURE, -1.0, 1.0)],
        )
        drivers = DriverRegistry()
        drivers.register(RFSoCAdapter("rfsoc", FakeRFSoC()))
        service = HardwareService(program=FabricProgram(shape=shape), driver_registry=drivers, device_registry=device)
        try:
            unattached = service.submit_input_waveform(InputWaveformRequest(1, Waveform((0.1,), 1_000.0)))
            while service.status(unattached).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertFalse(service.status(unattached).waveform.dispatched)
            overrange = service.submit_input_waveform(InputWaveformRequest(0, Waveform((0.8,), 1_000.0)))
            while service.status(overrange).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertIn("voltage limits", service.status(overrange).waveform.reason)
        finally:
            service.shutdown()

    def test_role_policy_allows_io_but_blocks_weight_programming(self):
        policy = AuthorizationPolicy({"viewer": Role.INPUT_OUTPUT, "operator": Role.OPERATOR})
        service = HardwareService(authorization_policy=policy)
        try:
            with self.assertRaises(AuthorizationError):
                service.submit_weights([WeightUpdate(0, 0, 0.1)], actor="viewer")
            with self.assertRaises(AuthorizationError):
                service.submit_power_on(actor="viewer")
            # The role check admits a logical input request; it may later be
            # rejected by missing hardware configuration, which is separate.
            job = service.submit_input_waveform(InputWaveformRequest(0, Waveform((0.1,), 1_000.0)), actor="viewer")
            while service.status(job).state != JobState.COMPLETE:
                sleep(0.001)
            self.assertFalse(service.status(job).waveform.dispatched)
        finally:
            service.shutdown()

    def test_simulator_backend_is_self_consistent(self):
        request = ExperimentRequest(
            "parity", NetworkShape(1, 1), (WeightUpdate(0, 0, 0.5),),
            (InputWaveformRequest(0, Waveform((0.8,), 1_000.0)),), (0,),
            (MeasurementRequest(0, 0, 0.8, 1e-9),), False, False,
        )
        run = SimulatorBackend().execute(request)
        matches, differences = compare_runs(run, run)
        self.assertTrue(matches)
        self.assertEqual(differences, ())


if __name__ == "__main__":
    unittest.main()
