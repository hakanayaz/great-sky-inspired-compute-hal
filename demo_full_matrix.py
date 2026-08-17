"""Demo 3: a fully connected 4x4 analog matrix-vector operation."""
from pathlib import Path

from analog_fabric import FabricProgram, render_ir
from backends import SimulatorBackend
from experiment_api import load_experiment


def main() -> None:
    request = load_experiment(Path(__file__).with_name("demo_full_matrix.json"))
    program = FabricProgram(shape=request.shape, edges=request.valid_edges())
    print("=== Demo 3: fully connected 4x4 fabric ===")
    print("Every input is connected to every output: 4 x 4 = 16 edges.\n")
    print("Weight matrix W (rows = outputs, columns = inputs):")
    for destination in range(request.shape.outputs):
        row = [next(weight.weight for weight in request.weights if weight.source == source and weight.destination == destination) for source in range(request.shape.inputs)]
        print("  [" + ", ".join(f"{value:5.2f}" for value in row) + "]")
    vector = [source.waveform.samples_v[0] for source in request.inputs]
    print("\nInput vector x (V):", vector)
    print("\nLogical IR:")
    print(render_ir(program, request.name))
    run = SimulatorBackend().execute(request)
    print("\nResult y = W × x (V):")
    for port in range(request.shape.outputs):
        print(f"  output {port}: {run.captures[port].samples_v[0]:.3f} V")


if __name__ == "__main__":
    main()
