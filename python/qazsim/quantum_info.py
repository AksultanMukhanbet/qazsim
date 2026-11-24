"""Statevector, DensityMatrix, Operator — analogues of qiskit.quantum_info."""

from __future__ import annotations

import numpy as np

from . import _accelerate

_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


# ------------------------------------------------------------------ kernels
def apply_unitary(state: np.ndarray, matrix: np.ndarray, qubits) -> None:
    """In-place application of ``matrix`` to ``qubits`` of a state vector (Rust)."""
    _accelerate.apply_matrix(state, np.ascontiguousarray(matrix, dtype=np.complex128),
                             [int(q) for q in qubits])


def apply_unitary_dm(rho_flat: np.ndarray, matrix: np.ndarray, qubits, n: int) -> None:
    """In-place rho -> U rho U† on a flattened density matrix of n qubits.

    The flattened index is row * 2^n + col, so column bits are qubits 0..n-1
    and row bits are qubits n..2n-1 of a 2n-qubit vector.
    """
    m = np.ascontiguousarray(matrix, dtype=np.complex128)
    _accelerate.apply_matrix(rho_flat, m, [q + n for q in qubits])
    _accelerate.apply_matrix(rho_flat, np.ascontiguousarray(m.conj()), [int(q) for q in qubits])


def apply_kraus_dm(rho_flat: np.ndarray, kraus, qubits, n: int) -> np.ndarray:
    """rho -> sum_k K rho K†. Returns a new flattened array."""
    if len(kraus) == 1:
        apply_unitary_dm(rho_flat, kraus[0], qubits, n)
        return rho_flat
    out = np.zeros_like(rho_flat)
    for k in kraus:
        tmp = rho_flat.copy()
        apply_unitary_dm(tmp, k, qubits, n)  # works for non-unitary K too
        out += tmp
    return out


def _pauli_label_matrix_ops(label: str):
    """'XZ' -> [(matrix X, qubit 1), (matrix Z, qubit 0)] (rightmost = qubit 0)."""
    label = label.upper()
    n = len(label)
    return [(_PAULI[ch], n - 1 - i) for i, ch in enumerate(label) if ch != "I"]


def _normalize_observable(obs):
    """Accept 'ZZ', [('ZZ', 1.0), ('XI', 0.5)] or {'ZZ': 1.0}."""
    if isinstance(obs, str):
        return [(obs, 1.0)]
    if isinstance(obs, dict):
        return list(obs.items())
    return [(lbl, c) for lbl, c in obs]


def _marginal_probs(probs: np.ndarray, n: int, qargs) -> np.ndarray:
    if qargs is None:
        return probs
    qargs = list(qargs)
    t = probs.reshape([2] * n)  # axis i <-> qubit n-1-i
    keep_axes = [n - 1 - q for q in qargs]
    drop = tuple(a for a in range(n) if a not in keep_axes)
    t = t.sum(axis=drop) if drop else t
    remaining = [a for a in range(n) if a in keep_axes]  # ascending axis order
    # want axis order so that qargs[0] is least significant (last axis)
    order = [remaining.index(n - 1 - q) for q in reversed(qargs)]
    return np.transpose(t, order).reshape(-1)


def _probs_dict(probs: np.ndarray, width: int, decimals=None):
    out = {}
    for i in np.nonzero(probs > 1e-15)[0]:
        v = float(probs[i])
        out[format(int(i), f"0{width}b")] = round(v, decimals) if decimals else v
    return out


# --------------------------------------------------------------- Statevector
class Statevector:
    def __init__(self, data):
        d = np.array(data, dtype=np.complex128).reshape(-1)
        n = int(round(np.log2(d.size)))
        if 2 ** n != d.size:
            raise ValueError("length must be a power of two")
        self._data = d
        self.num_qubits = n

    @property
    def data(self) -> np.ndarray:
        return self._data

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        """'01+' style label; rightmost character is qubit 0."""
        single = {"0": [1, 0], "1": [0, 1],
                  "+": [2 ** -0.5, 2 ** -0.5], "-": [2 ** -0.5, -(2 ** -0.5)],
                  "r": [2 ** -0.5, 1j * 2 ** -0.5], "l": [2 ** -0.5, -1j * 2 ** -0.5]}
        v = np.array([1], dtype=np.complex128)
        for ch in label:
            v = np.kron(v, np.array(single[ch], dtype=np.complex128))
        return cls(v)

    @classmethod
    def from_int(cls, i: int, dims: int) -> "Statevector":
        v = np.zeros(dims, dtype=np.complex128)
        v[i] = 1
        return cls(v)

    @classmethod
    def from_instruction(cls, circuit) -> "Statevector":
        v = np.zeros(2 ** circuit.num_qubits, dtype=np.complex128)
        v[0] = 1
        return cls(v).evolve(circuit)

    def evolve(self, circuit) -> "Statevector":
        out = self._data.copy()
        for ins in circuit.data:
            if ins.name in ("barrier",) or ins.name.startswith("save_"):
                continue
            if not ins.is_gate:
                raise ValueError(f"Statevector cannot evolve through '{ins.name}'")
            apply_unitary(out, ins.to_matrix(), ins.qubits)
        if circuit.global_phase:
            out *= np.exp(1j * circuit.global_phase)
        return Statevector(out)

    def probabilities(self, qargs=None) -> np.ndarray:
        probs = _accelerate.probabilities(self._data)
        return _marginal_probs(probs, self.num_qubits, qargs)

    def probabilities_dict(self, qargs=None, decimals=None) -> dict:
        width = self.num_qubits if qargs is None else len(list(qargs))
        return _probs_dict(self.probabilities(qargs), width, decimals)

    def sample_counts(self, shots: int, qargs=None, seed=None) -> dict:
        rng = np.random.default_rng(seed)
        p = self.probabilities(qargs)
        width = self.num_qubits if qargs is None else len(list(qargs))
        idx = rng.choice(p.size, size=shots, p=p / p.sum())
        vals, cnt = np.unique(idx, return_counts=True)
        return {format(int(v), f"0{width}b"): int(c) for v, c in zip(vals, cnt)}

    def expectation_value(self, observable) -> float:
        """<ψ|O|ψ> for a Pauli string or a weighted sum of Pauli strings."""
        total = 0.0
        for label, coeff in _normalize_observable(observable):
            if len(label) != self.num_qubits:
                raise ValueError(f"Pauli '{label}' has wrong length")
            tmp = self._data.copy()
            for m, q in _pauli_label_matrix_ops(label):
                apply_unitary(tmp, m, [q])
            total += coeff * np.vdot(self._data, tmp)
        return float(np.real(total))

    def inner(self, other: "Statevector") -> complex:
        return complex(np.vdot(self._data, other._data))

    def equiv(self, other: "Statevector", atol: float = 1e-8) -> bool:
        """Equal up to a global phase."""
        return abs(abs(self.inner(other)) - 1) < atol

    def to_density_matrix(self) -> "DensityMatrix":
        return DensityMatrix(np.outer(self._data, self._data.conj()))

    def __repr__(self):
        return f"Statevector({np.array2string(self._data, precision=4)}, dims={(2,) * self.num_qubits})"


# ------------------------------------------------------------- DensityMatrix
class DensityMatrix:
    def __init__(self, data):
        d = np.array(data, dtype=np.complex128)
        if d.ndim == 1:
            d = np.outer(d, d.conj())
        n = int(round(np.log2(d.shape[0])))
        if d.shape != (2 ** n, 2 ** n):
            raise ValueError("density matrix must be 2^n x 2^n")
        self._data = d
        self.num_qubits = n

    @property
    def data(self) -> np.ndarray:
        return self._data

    @classmethod
    def from_instruction(cls, circuit) -> "DensityMatrix":
        d = np.zeros((2 ** circuit.num_qubits,) * 2, dtype=np.complex128)
        d[0, 0] = 1
        return cls(d).evolve(circuit)

    def evolve(self, circuit) -> "DensityMatrix":
        n = self.num_qubits
        flat = self._data.reshape(-1).copy()
        for ins in circuit.data:
            if ins.name in ("barrier",) or ins.name.startswith("save_"):
                continue
            if not ins.is_gate:
                raise ValueError(f"DensityMatrix cannot evolve through '{ins.name}'")
            apply_unitary_dm(flat, ins.to_matrix(), ins.qubits, n)
        return DensityMatrix(flat.reshape(2 ** n, 2 ** n))

    def probabilities(self, qargs=None) -> np.ndarray:
        probs = np.clip(np.real(np.diag(self._data)), 0, None)
        return _marginal_probs(probs, self.num_qubits, qargs)

    def probabilities_dict(self, qargs=None, decimals=None) -> dict:
        width = self.num_qubits if qargs is None else len(list(qargs))
        return _probs_dict(self.probabilities(qargs), width, decimals)

    def expectation_value(self, observable) -> float:
        total = 0.0
        n = self.num_qubits
        for label, coeff in _normalize_observable(observable):
            m = np.array([[1]], dtype=np.complex128)
            for ch in label.upper():
                m = np.kron(m, _PAULI[ch])
            total += coeff * np.trace(m @ self._data)
        return float(np.real(total))

    def purity(self) -> float:
        return float(np.real(np.trace(self._data @ self._data)))

    def trace(self) -> float:
        return float(np.real(np.trace(self._data)))

    def __repr__(self):
        return f"DensityMatrix({np.array2string(self._data, precision=4)}, dims={(2,) * self.num_qubits})"


# ------------------------------------------------------------------ Operator
class Operator:
    """Unitary matrix of a circuit (built column by column with the Rust kernel)."""

    def __init__(self, data):
        from .circuit import QuantumCircuit
        if isinstance(data, QuantumCircuit):
            n = data.num_qubits
            if n > 14:
                raise ValueError("Operator is limited to 14 qubits")
            dim = 2 ** n
            cols = []
            for i in range(dim):
                cols.append(Statevector.from_int(i, dim).evolve(data).data)
            self._data = np.array(cols).T
        else:
            self._data = np.array(data, dtype=np.complex128)
        self.num_qubits = int(round(np.log2(self._data.shape[0])))

    @property
    def data(self) -> np.ndarray:
        return self._data

    def equiv(self, other, atol: float = 1e-8) -> bool:
        """Equal up to a global phase."""
        b = other.data if isinstance(other, Operator) else np.asarray(other)
        a = self._data
        idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        if abs(b[idx]) < atol:
            return False
        phase = a[idx] / b[idx]
        return np.allclose(a, phase * b, atol=atol)


# ------------------------------------------------------------------ fidelity
def state_fidelity(a, b) -> float:
    """Fidelity between two states (Statevector or DensityMatrix)."""
    if isinstance(a, Statevector) and isinstance(b, Statevector):
        return abs(a.inner(b)) ** 2
    if isinstance(a, Statevector):
        a, b = b, a
    if isinstance(b, Statevector):
        rho = a.data if isinstance(a, DensityMatrix) else np.asarray(a)
        return float(np.real(np.vdot(b.data, rho @ b.data)))
    # both density matrices: (Tr sqrt(sqrt(r) s sqrt(r)))^2
    r, s = a.data, b.data
    w, v = np.linalg.eigh(r)
    sr = (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T
    m = sr @ s @ sr
    ev = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
    return float(np.sum(np.sqrt(ev)) ** 2)
