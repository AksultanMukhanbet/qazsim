"""Emulator of the SpinQ Gemini Mini Pro desktop NMR quantum computer.

Device facts used here:
* 2 qubits, both coupled;
* at most 80 one-qubit or 40 two-qubit operations per circuit;
* NMR works with a molecular ensemble, so the device returns probabilities
  and a reconstructed density matrix rather than single-shot samples.

Default noise values come from the SpinQ Gemini paper (arXiv:2101.10017):
T1 ≈ 4 s, T2 ≈ 0.3 s, 1-qubit gate ≈ 20 µs, CNOT ≈ 800 µs, gate fidelities
≈ 0.99 / 0.98. Replace them with the calibration of your own devices.
The native gate set and the exact meaning of "80/40" should be checked
against SpinQ's documentation for the Mini Pro.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from ..simulator import QazSim, QazSimError
from ..transpiler import transpile

ONE_QUBIT_GATES = ["id", "x", "y", "z", "h", "rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz"]


@dataclass
class GeminiCalibration:
    t1: float = 4.0            # s
    t2: float = 0.3            # s
    time_1q: float = 20e-6     # s
    time_2q: float = 800e-6    # s
    depol_1q: float = 0.01     # depolarizing parameter after a 1-qubit gate
    depol_2q: float = 0.02     # depolarizing parameter after a 2-qubit gate
    readout_error: float = 0.0  # NMR readout is an ensemble average


class GeminiLimitError(QazSimError):
    pass


def gemini_noise_model(cal: GeminiCalibration) -> NoiseModel:
    nm = NoiseModel(basis_gates=ONE_QUBIT_GATES + TWO_QUBIT_GATES)
    relax_1q = thermal_relaxation_error(cal.t1, cal.t2, cal.time_1q)
    relax_2q = thermal_relaxation_error(cal.t1, cal.t2, cal.time_2q)
    e1 = relax_1q.compose(depolarizing_error(cal.depol_1q, 1))
    e2 = relax_2q.expand(relax_2q).compose(depolarizing_error(cal.depol_2q, 2))
    nm.add_all_qubit_quantum_error(e1, [g for g in ONE_QUBIT_GATES if g != "id"])
    nm.add_all_qubit_quantum_error(e2, TWO_QUBIT_GATES)
    return nm


class GeminiMiniPro(QazSim):
    """QazSim backend that behaves like one SpinQ Gemini Mini Pro.

    Circuits are transpiled to the native gates automatically and rejected if
    they exceed 2 qubits or the 80/40 operation limit.

    >>> dev = GeminiMiniPro()                    # default calibration
    >>> dev = GeminiMiniPro(GeminiCalibration(t2=0.25), name="gemini_3")
    """

    num_qubits = 2
    basis_gates = ONE_QUBIT_GATES + TWO_QUBIT_GATES
    max_1q_ops = 80
    max_2q_ops = 40

    def __init__(self, calibration: GeminiCalibration | None = None,
                 name: str = "gemini_mini_pro", noisy: bool = True,
                 seed_simulator: int | None = None):
        self.calibration = calibration or GeminiCalibration()
        super().__init__(method="density_matrix",
                         noise_model=gemini_noise_model(self.calibration) if noisy else None,
                         seed_simulator=seed_simulator, name=name)

    def check_limits(self, circuit) -> dict:
        """Raise GeminiLimitError if the circuit does not fit the device."""
        if circuit.num_qubits > self.num_qubits:
            raise GeminiLimitError(f"Gemini Mini Pro has {self.num_qubits} qubits, "
                                   f"circuit has {circuit.num_qubits}")
        ops = circuit.count_ops()
        bad = [g for g in ops if g not in self.basis_gates and g not in
               ("measure", "barrier", "save_density_matrix", "save_probabilities")]
        if bad:
            raise GeminiLimitError(f"not native to Gemini Mini Pro: {bad} (transpile first)")
        n2 = sum(ops.get(g, 0) for g in TWO_QUBIT_GATES)
        n1 = sum(ops.get(g, 0) for g in ONE_QUBIT_GATES if g != "id")
        # Assumption: one 2-qubit operation uses the budget of two 1-qubit ones.
        if n2 > self.max_2q_ops or n1 + 2 * n2 > self.max_1q_ops:
            raise GeminiLimitError(f"too many operations: {n1} one-qubit, {n2} two-qubit "
                                   f"(limit {self.max_1q_ops}/{self.max_2q_ops})")
        return {"one_qubit": n1, "two_qubit": n2}

    def _validate(self, circuit):
        self.check_limits(circuit)

    def run(self, circuits, shots: int = 1024, parameter_binds=None, **kwargs):
        from ..circuit import QuantumCircuit
        circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
        if parameter_binds is not None:
            circuits = [c.assign_parameters(b) for c, b in zip(circuits, parameter_binds)]
        return super().run(transpile(circuits, backend=self), shots=shots, **kwargs)


def gemini_fleet(n: int = 7, calibrations=None) -> list:
    """Backends for the laboratory's devices: gemini_1 ... gemini_n."""
    cals = calibrations or [GeminiCalibration() for _ in range(n)]
    return [GeminiMiniPro(cal, name=f"gemini_{i + 1}") for i, cal in enumerate(cals)]
