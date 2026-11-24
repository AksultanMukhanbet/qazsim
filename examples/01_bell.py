"""Bell state on the ideal simulator and on the Gemini Mini Pro emulator."""
from qazsim import GeminiMiniPro, QazSim, QuantumCircuit, Statevector, state_fidelity

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.save_density_matrix()
qc.measure([0, 1], [0, 1])
print(qc)

# 1. Ideal simulator
result = QazSim().run(qc, shots=1000).result()
print("QazSim counts:", result.get_counts())

# 2. One SpinQ Gemini Mini Pro (noise from calibration)
dev = GeminiMiniPro()
res = dev.run(qc).result()
rho = res.get_density_matrix()
ideal = Statevector.from_instruction(QuantumCircuit(2).h(0).cx(0, 1))
print("Gemini Mini Pro probabilities:", rho.probabilities_dict(decimals=4))
print("Fidelity with ideal Bell state:", round(state_fidelity(rho, ideal), 4))
