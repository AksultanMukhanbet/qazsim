"""QazSim tests. Run: pytest -q

Reference results are computed independently with dense NumPy matrices
(kron products). If Qiskit is installed in the test environment, random
circuits are also compared against qiskit.quantum_info.Statevector; QazSim
itself never imports Qiskit.
"""

import math

import numpy as np
import pytest

from qazsim import (DensityMatrix, GeminiCalibration, GeminiMiniPro, NoiseModel,
                    Operator, Parameter, ParameterVector, QazSim, QuantumCircuit,
                    ReadoutError, Statevector, amplitude_damping_error,
                    depolarizing_error, gemini_fleet, qasm2, state_fidelity,
                    thermal_relaxation_error, transpile)
from qazsim.devices.gemini import GeminiLimitError
from qazsim.gates import STANDARD_GATES, gate_matrix

RNG = np.random.default_rng(1234)


# ------------------------------------------------------------ reference model
def dense_op(matrix, qubits, n):
    """Full 2^n x 2^n operator of ``matrix`` on ``qubits`` (little-endian)."""
    k = len(qubits)
    dim = 2 ** n
    out = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        sub = sum(((col >> q) & 1) << b for b, q in enumerate(qubits))
        for r in range(2 ** k):
            row = col
            for b, q in enumerate(qubits):
                row = (row & ~(1 << q)) | (((r >> b) & 1) << q)
            out[row, col] += matrix[r, sub]
    return out


def reference_state(qc):
    psi = np.zeros(2 ** qc.num_qubits, dtype=complex)
    psi[0] = 1
    for ins in qc.data:
        if ins.is_gate:
            psi = dense_op(ins.to_matrix(), ins.qubits, qc.num_qubits) @ psi
    return psi * np.exp(1j * qc.global_phase)


def random_circuit(n, depth, rng=RNG):
    qc = QuantumCircuit(n)
    names = [g for g, (k, _, _) in STANDARD_GATES.items() if k <= n]
    for _ in range(depth):
        g = names[rng.integers(len(names))]
        k, npar, _ = STANDARD_GATES[g]
        qs = rng.choice(n, size=k, replace=False).tolist()
        qc.append(g, qs, tuple(rng.uniform(-math.pi, math.pi, npar)))
    return qc


# -------------------------------------------------------------------- gates
@pytest.mark.parametrize("name", list(STANDARD_GATES))
def test_gates_are_unitary(name):
    k, npar, _ = STANDARD_GATES[name]
    m = gate_matrix(name, tuple(RNG.uniform(-3, 3, npar)))
    assert np.allclose(m @ m.conj().T, np.eye(2 ** k))


# -------------------------------------------------------------- statevector
def test_bell_and_ghz():
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(1, 2)
    sv = Statevector.from_instruction(qc)
    expected = np.zeros(8, complex)
    expected[0] = expected[7] = 1 / math.sqrt(2)
    assert np.allclose(sv.data, expected)


def test_little_endian_like_qiskit():
    qc = QuantumCircuit(3)
    qc.x(0)  # |001> -> index 1
    assert np.isclose(abs(Statevector.from_instruction(qc).data[1]), 1)
    assert Statevector.from_instruction(qc).probabilities_dict() == {"001": 1.0}


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_random_circuits_match_reference(n):
    for _ in range(5):
        qc = random_circuit(n, 40)
        assert np.allclose(Statevector.from_instruction(qc).data, reference_state(qc), atol=1e-10)


def test_large_state_parallel_kernel():
    """16 qubits: above the Rust parallel threshold."""
    n = 16
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.rz(0.3, 15).ccx(0, 7, 15)
    sv = Statevector.from_instruction(qc)
    assert np.isclose(np.linalg.norm(sv.data), 1)
    back = sv.evolve(qc.inverse())
    assert np.isclose(abs(back.data[0]), 1)


def test_marginal_probabilities():
    qc = QuantumCircuit(3)
    qc.x(2).h(0)
    sv = Statevector.from_instruction(qc)
    assert np.allclose(sv.probabilities([2]), [0, 1])
    assert np.allclose(sv.probabilities([0]), [0.5, 0.5])
    # qargs order: first listed qubit is least significant bit
    assert sv.probabilities_dict([2, 1], decimals=10) == {"01": 1.0}


def test_expectation_values():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    sv = Statevector.from_instruction(qc)
    assert np.isclose(sv.expectation_value("ZZ"), 1)
    assert np.isclose(sv.expectation_value("XX"), 1)
    assert np.isclose(sv.expectation_value("ZI"), 0)
    assert np.isclose(sv.expectation_value([("ZZ", 0.5), ("XX", 0.25)]), 0.75)


# ----------------------------------------------------------------- backend
def test_counts_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0).cx(0, 1).measure([0, 1], [0, 1])
    counts = QazSim(seed_simulator=7).run(qc, shots=4000).result().get_counts()
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] - 2000) < 200


def test_clbit_mapping_and_memory():
    qc = QuantumCircuit(2, 3)
    qc.x(0).measure(0, 2).measure(1, 0)
    res = QazSim().run(qc, shots=10, memory=True).result()
    assert res.get_counts() == {"100": 10}
    assert res.get_memory() == ["100"] * 10


def test_measure_all_and_save():
    qc = QuantumCircuit(2)
    qc.h(0).save_statevector().measure_all()
    res = QazSim().run(qc, shots=100).result()
    assert set(res.get_counts()) <= {"00", "01"}
    assert np.allclose(res.get_statevector().probabilities(), [0.5, 0.5, 0, 0])


def test_mid_circuit_measurement_and_reset():
    qc = QuantumCircuit(1, 2)
    qc.x(0).measure(0, 0).reset(0).measure(0, 1)
    counts = QazSim().run(qc, shots=50).result().get_counts()
    assert counts == {"01": 50}


def test_multiple_circuits_and_parameter_binds():
    th = Parameter("θ")
    qc = QuantumCircuit(1, 1)
    qc.ry(th, 0).measure(0, 0)
    res = QazSim().run([qc, qc], shots=200, parameter_binds=[{th: 0}, {th: math.pi}]).result()
    assert res.get_counts(0) == {"0": 200}
    assert res.get_counts(1) == {"1": 200}


def test_parameters_expressions_and_vector():
    v = ParameterVector("x", 2)
    qc = QuantumCircuit(1)
    qc.rx(2 * v[0] - v[1] / 2, 0)
    assert qc.num_parameters == 2
    bound = qc.assign_parameters([1.0, 2.0])
    assert np.isclose(bound.data[0].params[0], 1.0)


def test_unbound_parameter_raises():
    qc = QuantumCircuit(1)
    qc.rx(Parameter("a"), 0)
    with pytest.raises(Exception):
        QazSim().run(qc)


# ---------------------------------------------------------- density matrix
def test_density_matrix_equals_pure_state_without_noise():
    qc = random_circuit(3, 30)
    dm = DensityMatrix.from_instruction(qc)
    sv = Statevector.from_instruction(qc)
    assert np.allclose(dm.data, np.outer(sv.data, sv.data.conj()), atol=1e-10)


def test_full_depolarizing_gives_maximally_mixed():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(1.0, 1), ["h"])
    qc = QuantumCircuit(1)
    qc.h(0).save_density_matrix()
    dm = QazSim(noise_model=nm).run(qc).result().get_density_matrix()
    assert np.allclose(dm.data, np.eye(2) / 2)


def test_amplitude_damping_decays_to_zero():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(amplitude_damping_error(1.0), ["x"])
    qc = QuantumCircuit(1)
    qc.x(0).save_density_matrix()
    dm = QazSim(noise_model=nm).run(qc).result().get_density_matrix()
    assert np.allclose(dm.data, [[1, 0], [0, 0]])


def test_thermal_relaxation_coherence():
    t1, t2, t = 4.0, 0.3, 0.1
    err = thermal_relaxation_error(t1, t2, t)
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], complex)
    out = sum(k @ rho @ k.conj().T for k in err.kraus)
    assert np.isclose(out[0, 1].real, 0.5 * math.exp(-t / t2))
    assert np.isclose(np.trace(out).real, 1)


def test_readout_error():
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.1, 0.9]]))
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    counts = QazSim(noise_model=nm, seed_simulator=3).run(qc, shots=10000).result().get_counts()
    assert abs(counts.get("1", 0) - 1000) < 200


# --------------------------------------------------------------- transpiler
@pytest.mark.parametrize("basis", [["rz", "ry", "cx"], ["rz", "rx", "cz"], ["u", "cx"],
                                   GeminiMiniPro.basis_gates])
def test_transpile_preserves_operator(basis):
    for _ in range(5):
        qc = random_circuit(3, 25)
        tq = transpile(qc, basis_gates=basis)
        assert set(tq.count_ops()) <= set(basis)
        assert Operator(tq).equiv(Operator(qc))
        # exact including global phase
        assert np.allclose(Statevector.from_instruction(tq).data, reference_state(qc), atol=1e-8)


def test_transpile_optimizes():
    qc = QuantumCircuit(2)
    qc.h(0).h(0).cx(0, 1).cx(0, 1).rz(0.1, 1).rz(-0.1, 1)
    tq = transpile(qc, basis_gates=["rz", "ry", "cx"])
    assert tq.size() == 0


# ----------------------------------------------------------- Gemini Mini Pro
def test_gemini_bell_fidelity_and_ensemble_output():
    dev = GeminiMiniPro(seed_simulator=1)
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).save_density_matrix().save_probabilities()
    res = dev.run(qc).result()
    ideal = Statevector.from_instruction(QuantumCircuit(2).h(0).cx(0, 1))
    f = state_fidelity(res.get_density_matrix(), ideal)
    assert 0.9 < f < 1.0
    assert set(res.get_probabilities()) == {"00", "01", "10", "11"}


def test_gemini_noiseless_is_exact():
    dev = GeminiMiniPro(noisy=False)
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).swap(0, 1).save_density_matrix()
    dm = dev.run(qc).result().get_density_matrix()
    ideal = Statevector.from_instruction(QuantumCircuit(2).h(0).cx(0, 1).swap(0, 1))
    assert np.isclose(state_fidelity(dm, ideal), 1)


def test_gemini_rejects_three_qubits():
    with pytest.raises(Exception):
        GeminiMiniPro().run(QuantumCircuit(3).h(0))


def test_gemini_rejects_too_many_operations():
    qc = QuantumCircuit(2)
    for _ in range(41):
        qc.cz(0, 1).h(0)
    with pytest.raises(GeminiLimitError):
        GeminiMiniPro().check_limits(transpile(qc, basis_gates=GeminiMiniPro.basis_gates,
                                               optimization_level=0))


def test_gemini_fleet():
    fleet = gemini_fleet(7, [GeminiCalibration(t2=0.2 + 0.02 * i) for i in range(7)])
    assert [d.name for d in fleet] == [f"gemini_{i}" for i in range(1, 8)]


# ---------------------------------------------------------------------- qasm
def test_qasm_roundtrip():
    qc = random_circuit(3, 20)
    qc2 = qasm2.loads(qasm2.dumps(qc))
    assert np.allclose(Statevector.from_instruction(qc2).data,
                       Statevector.from_instruction(qc).data, atol=1e-10)


def test_qasm_with_pi_expressions():
    src = """OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2];
    u2(0, pi) q[0]; cx q[0], q[1]; rz(-pi/4) q[1]; measure q[0] -> c[0];"""
    qc = qasm2.loads(src)
    assert qc.count_ops() == {"u": 1, "cx": 1, "rz": 1, "measure": 1}


# ------------------------------------------------------- optional: vs Qiskit
def test_against_qiskit_if_installed():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector as QStatevector
    for n in (2, 4, 6):
        for _ in range(5):
            qc = random_circuit(n, 40)
            q = qiskit.QuantumCircuit.from_qasm_str(qasm2.dumps(qc))
            assert np.allclose(Statevector.from_instruction(qc).data,
                               QStatevector(q).data, atol=1e-9)


def test_draw():
    qc = QuantumCircuit(2, 2)
    qc.h(0).cx(0, 1).measure([0, 1], [0, 1])
    text = qc.draw()
    assert "q_0" in text and "H" in text and "M1" in text
