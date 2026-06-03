from dataclasses import dataclass

@dataclass
class EvaluationProtocol:
    num_generated_graphs: int = 1024
    num_reference_graphs: int = 1024
    num_runs: int = 3
    seed: int = 42
