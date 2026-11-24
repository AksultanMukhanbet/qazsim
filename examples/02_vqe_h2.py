"""VQE for H2 (2-qubit Hamiltonian, bond length 0.735 Å) on the Gemini Mini Pro emulator.

Hamiltonian coefficients: standard 2-qubit reduction of H2 in STO-3G
(parity mapping); exact ground energy ≈ -1.857 Ha (electronic part).
"""
import numpy as np

from qazsim import GeminiMiniPro, Parameter, QazSim, QuantumCircuit, Statevector

H = {"II": -1.052373245772859, "IZ": 0.39793742484318045, "ZI": -0.39793742484318045,
     "ZZ": -0.01128010425623538, "XX": 0.18093119978423156}

theta = Parameter("θ")
ansatz = QuantumCircuit(2)
ansatz.x(0)
ansatz.ry(theta, 1)
ansatz.cx(1, 0)


def energy_ideal(t):
    return Statevector.from_instruction(ansatz.assign_parameters([t])).expectation_value(H)


def energy_device(t, dev=GeminiMiniPro()):
    qc = ansatz.assign_parameters([t])
    qc.save_density_matrix()
    return dev.run(qc).result().get_density_matrix().expectation_value(H)


grid = np.linspace(-np.pi, np.pi, 121)
e_ideal = [energy_ideal(t) for t in grid]
e_dev = [energy_device(t) for t in grid]
exact = min(np.linalg.eigvalsh(sum(c * np.kron(*[{"I": np.eye(2), "X": [[0, 1], [1, 0]],
                                                    "Z": np.diag([1, -1])}[p] for p in lbl])
                                    for lbl, c in H.items())))
print(f"exact ground energy : {exact:.5f} Ha")
print(f"VQE ideal simulator : {min(e_ideal):.5f} Ha")
print(f"VQE Gemini Mini Pro : {min(e_dev):.5f} Ha (with device noise)")
