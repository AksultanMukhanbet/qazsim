"""VQE for a 10-qubit transverse-field Ising chain with compiled programs (QazSim 0.2).

The circuit is compiled once; every optimisation step then costs one call that
returns the energy and the exact gradient of all parameters (adjoint method).
"""
import time

import numpy as np

import qazsim as qz

n, layers = 10, 4

# H = Σ Z_i Z_{i+1} + 0.7 Σ X_i   (rightmost character of a label is qubit 0)
H = {}
for i in range(n - 1):
    lbl = ["I"] * n
    lbl[n - 1 - i] = lbl[n - 2 - i] = "Z"
    H["".join(lbl)] = 1.0
for i in range(n):
    lbl = ["I"] * n
    lbl[n - 1 - i] = "X"
    H["".join(lbl)] = 0.7

# hardware-efficient ansatz
th = qz.params("θ", 2 * n * layers)
c = qz.Circuit(n)
k = 0
for _ in range(layers):
    for q in range(n):
        c.ry(th[k], q).rz(th[k + 1], q)
        k += 2
    for q in range(n - 1):
        c.cx(q, q + 1)

prog = qz.compile(c)
print(prog)

# Adam on the exact gradient
rng = np.random.default_rng(0)
x = rng.normal(scale=0.1, size=prog.num_parameters)
m = np.zeros_like(x)
v = np.zeros_like(x)
t0 = time.perf_counter()
for step in range(1, 401):
    e, g = prog.value_and_gradient(H, x)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    x -= 0.05 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    if step % 100 == 0:
        print(f"step {step:3d}: E = {e:.6f}")
print(f"400 energy+gradient evaluations: {time.perf_counter() - t0:.2f} s")

# exact ground energy for comparison (dense diagonalisation, fine for 10 qubits)
I2, X, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
Hm = 0
for lbl, coeff in H.items():
    m_ = np.array([[1.0]])
    for ch in lbl:
        m_ = np.kron(m_, {"I": I2, "X": X, "Z": Z}[ch])
    Hm = Hm + coeff * m_
print(f"VQE energy  : {prog.expectation(H, x):.6f}")
print(f"exact energy: {np.linalg.eigvalsh(Hm)[0]:.6f}")

# a batch: energies of 1000 random parameter sets in one call
batch = rng.uniform(0, 2 * np.pi, size=(1000, prog.num_parameters))
t0 = time.perf_counter()
energies = prog.expectation(H, batch)
print(f"1000 energies in one batch: {time.perf_counter() - t0:.3f} s (min {energies.min():.3f})")
