"""Demo 4: a visible 8-node, 64-edge recurrent analog-fabric toy.

The normal FabricProgram represents a directional input-to-output layer. This
demo makes recurrence explicit by treating the four logical inputs and four
logical outputs as one combined eight-node state. A row is the receiving node;
a column is the sending node. Each step computes ``tanh(W @ state)``.

It is deliberately small and deterministic: a visual explanation of feedback
and inhibition, not a biological model or an instrument-control implementation.
"""
from __future__ import annotations

import json
from math import tanh
from pathlib import Path


NODE_NAMES = ("in1", "in2", "in3", "in4", "out1", "out2", "out3", "out4")


def next_state(weights: list[list[float]], state: list[float]) -> list[float]:
    """Apply all 64 programmable edges once, then a stable toy node response."""
    return [tanh(sum(weight * value for weight, value in zip(row, state))) for row in weights]


def format_matrix(weights: list[list[float]]) -> str:
    """Make the combined adjacency matrix readable in a terminal transcript."""
    header = "receiver \\ sender | " + " ".join(f"{name:>6}" for name in NODE_NAMES)
    rule = "-" * len(header)
    rows = [header, rule]
    for name, row in zip(NODE_NAMES, weights):
        rows.append(f"{name:>17} | " + " ".join(f"{value:>6.2f}" for value in row))
    return "\n".join(rows)


def main() -> None:
    data = json.loads(Path(__file__).with_name("demo_recurrent.json").read_text())
    weights = data["weights"]
    state = data["initial_state_v"]
    if len(weights) != len(NODE_NAMES) or any(len(row) != len(NODE_NAMES) for row in weights):
        raise ValueError("Demo 4 requires an 8 by 8 matrix: 64 programmable edges")
    if len(state) != len(NODE_NAMES):
        raise ValueError("initial_state_v must contain one value for each of the 8 nodes")

    print("=== Demo 4: fully connected input/output feedback fabric ===")
    print("8 logical nodes = in1..in4 + out1..out4; 8 x 8 = 64 programmed edges.")
    print("Rows receive signals; columns send them. All four matrix blocks are active:")
    print("input<-input, input<-output feedback, output<-input feedforward, output<-output.\n")
    print(format_matrix(weights))
    print("\nstep | " + " | ".join(f"{name:>6}" for name in NODE_NAMES))
    print("-----+" + "-".join("--------" for _ in NODE_NAMES))
    for step in range(data["steps"] + 1):
        print(f"{step:>4} | " + " | ".join(f"{value:>6.3f}" for value in state))
        state = next_state(weights, state)
    print("\nResult: every node is affected by every other node at each feedback step.")
    print("Negative values model inhibitory paths; positive values model excitatory paths.")
    print("This is a discrete-time analog-fabric teaching model, not a brain or PDE simulation.")


if __name__ == "__main__":
    main()
