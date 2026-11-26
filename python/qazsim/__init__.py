"""QazSim — quantum circuit simulator of the KazNU Quantum AI Laboratory.

Architecture follows Qiskit: the user API is Python, the numerical core is
written in Rust and exposed through PyO3 as ``qazsim._accelerate``.
"""

from . import _accelerate, qasm2
from .circuit import (Instruction, Parameter, ParameterExpression,
                      ParameterVector, QuantumCircuit)
from .devices import GeminiCalibration, GeminiMiniPro, gemini_fleet
from .noise import (NoiseModel, QuantumError, ReadoutError, amplitude_damping_error,
                    depolarizing_error, pauli_error, phase_damping_error,
                    thermal_relaxation_error)
from .quantum_info import DensityMatrix, Operator, Statevector, state_fidelity
from .simulator import Job, QazSim, QazSimError, Result
from .transpiler import TranspilerError, transpile

__version__ = "0.1.4"

__all__ = [
    "QuantumCircuit", "Instruction", "Parameter", "ParameterVector", "ParameterExpression",
    "QazSim", "Job", "Result", "QazSimError",
    "Statevector", "DensityMatrix", "Operator", "state_fidelity",
    "NoiseModel", "QuantumError", "ReadoutError", "depolarizing_error", "pauli_error",
    "amplitude_damping_error", "phase_damping_error", "thermal_relaxation_error",
    "transpile", "TranspilerError",
    "GeminiMiniPro", "GeminiCalibration", "gemini_fleet",
    "qasm2", "__version__",
]
