"""QazSim — quantum circuit simulator of the Quantum AI Laboratory.

The user API is Python; the numerical core is written in Rust and exposed
through PyO3 as ``qazsim._accelerate``.

Two ways to run circuits:

* compiled programs (0.2): ``Circuit`` -> ``compile`` -> ``Program`` with gate
  fusion, batched parameters and adjoint gradients;
* the circuit/backend API (0.1): ``QuantumCircuit`` -> ``QazSim().run(...)``,
  with measurements, noise models and the Gemini Mini Pro emulator.
"""

from . import _accelerate, qasm2
from .circuit import (Instruction, Parameter, ParameterExpression,
                      ParameterVector, QuantumCircuit)
from .devices import GeminiCalibration, GeminiMiniPro, gemini_fleet
from .program import Circuit, Param, Program, compile, params, pauli_terms
from .noise import (NoiseModel, QuantumError, ReadoutError, amplitude_damping_error,
                    depolarizing_error, pauli_error, phase_damping_error,
                    thermal_relaxation_error)
from .quantum_info import DensityMatrix, Operator, Statevector, state_fidelity
from .simulator import Job, QazSim, QazSimError, Result
from .transpiler import TranspilerError, transpile

__version__ = "0.2.0"

__all__ = [
    "Circuit", "Param", "params", "compile", "Program", "pauli_terms",
    "QuantumCircuit", "Instruction", "Parameter", "ParameterVector", "ParameterExpression",
    "QazSim", "Job", "Result", "QazSimError",
    "Statevector", "DensityMatrix", "Operator", "state_fidelity",
    "NoiseModel", "QuantumError", "ReadoutError", "depolarizing_error", "pauli_error",
    "amplitude_damping_error", "phase_damping_error", "thermal_relaxation_error",
    "transpile", "TranspilerError",
    "GeminiMiniPro", "GeminiCalibration", "gemini_fleet",
    "qasm2", "__version__",
]
