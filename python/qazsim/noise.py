"""Noise models (Kraus channels), API modelled on qiskit_aer.noise."""

from __future__ import annotations

import itertools
import math

import numpy as np

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class QuantumError:
    """Quantum channel given by Kraus operators on ``num_qubits`` qubits.

    Index convention of the Kraus matrices is the same as for gates:
    bit b of the index belongs to the b-th qubit the error is applied to.
    """

    def __init__(self, kraus):
        ks = [np.asarray(k, dtype=np.complex128) for k in kraus]
        d = ks[0].shape[0]
        n = int(round(math.log2(d)))
        if 2 ** n != d or any(k.shape != (d, d) for k in ks):
            raise ValueError("Kraus operators must be 2^n x 2^n")
        s = sum(k.conj().T @ k for k in ks)
        if not np.allclose(s, np.eye(d), atol=1e-8):
            raise ValueError("Kraus operators are not trace preserving")
        self.kraus = ks
        self.num_qubits = n

    def compose(self, other: "QuantumError") -> "QuantumError":
        """Channel: first ``self``, then ``other`` (same qubits)."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("errors act on different numbers of qubits")
        return QuantumError([b @ a for a in self.kraus for b in other.kraus])

    def tensor(self, other: "QuantumError") -> "QuantumError":
        """self ⊗ other: ``other`` acts on the lower qubits, ``self`` on the higher."""
        return QuantumError([np.kron(a, b) for a in self.kraus for b in other.kraus])

    def expand(self, other: "QuantumError") -> "QuantumError":
        """other ⊗ self: ``self`` acts on the lower qubits."""
        return other.tensor(self)

    def __repr__(self):
        return f"QuantumError(num_qubits={self.num_qubits}, num_kraus={len(self.kraus)})"


# ------------------------------------------------------------ standard errors
def depolarizing_error(param: float, num_qubits: int = 1) -> QuantumError:
    """E(ρ) = (1 - p) ρ + p I / 2^n."""
    if not 0 <= param <= 4 ** num_qubits / (4 ** num_qubits - 1):
        raise ValueError("depolarizing parameter out of range")
    paulis = []
    for combo in itertools.product([_I, _X, _Y, _Z], repeat=num_qubits):
        m = np.array([[1]], dtype=np.complex128)
        for p in combo:
            m = np.kron(m, p)
        paulis.append(m)
    w_other = param / 4 ** num_qubits
    w_id = 1 - param + w_other
    kraus = [math.sqrt(w_id) * paulis[0]] + [math.sqrt(w_other) * p for p in paulis[1:]]
    return QuantumError([k for k in kraus if np.any(k)])


def pauli_error(probs: dict) -> QuantumError:
    """{'X': 0.1, 'I': 0.9} — one-or-more-qubit Pauli channel."""
    mats = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    kraus = []
    for label, pr in probs.items():
        m = np.array([[1]], dtype=np.complex128)
        for ch in label:
            m = np.kron(m, mats[ch])
        if pr > 0:
            kraus.append(math.sqrt(pr) * m)
    return QuantumError(kraus)


def amplitude_damping_error(gamma: float) -> QuantumError:
    k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return QuantumError([k0, k1])


def phase_damping_error(lam: float) -> QuantumError:
    k0 = np.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, math.sqrt(lam)]], dtype=np.complex128)
    return QuantumError([k0, k1])


def thermal_relaxation_error(t1: float, t2: float, time: float) -> QuantumError:
    """T1/T2 relaxation during ``time`` (same units for all three, T2 <= 2*T1).

    Built as amplitude damping (T1) followed by pure dephasing chosen so that
    coherences decay as exp(-time / T2).
    """
    if t2 > 2 * t1:
        raise ValueError("T2 must be <= 2*T1")
    gamma = 1 - math.exp(-time / t1)
    lam = 1 - math.exp(-2 * time / t2 + time / t1)
    lam = min(max(lam, 0.0), 1.0)
    return amplitude_damping_error(gamma).compose(phase_damping_error(lam))


# --------------------------------------------------------------- readout error
class ReadoutError:
    """probabilities[i][j] = P(read j | prepared i) for one qubit."""

    def __init__(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (2, 2) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("readout matrix must be 2x2 with rows summing to 1")
        self.probabilities = p


# ----------------------------------------------------------------- NoiseModel
class NoiseModel:
    """Collection of errors attached to gates (and readout).

    >>> nm = NoiseModel()
    >>> nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx"])
    >>> nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    """

    def __init__(self, basis_gates=None):
        self.basis_gates = list(basis_gates) if basis_gates else None
        self._all: dict[str, QuantumError] = {}
        self._local: dict[tuple, QuantumError] = {}
        self._readout_all: ReadoutError | None = None
        self._readout_local: dict[int, ReadoutError] = {}

    def add_all_qubit_quantum_error(self, error: QuantumError, instructions):
        for name in ([instructions] if isinstance(instructions, str) else instructions):
            if name in self._all:
                self._all[name] = self._all[name].compose(error)
            else:
                self._all[name] = error

    def add_quantum_error(self, error: QuantumError, instructions, qubits):
        qubits = tuple(qubits)
        if len(qubits) != error.num_qubits:
            raise ValueError("error size does not match number of qubits")
        for name in ([instructions] if isinstance(instructions, str) else instructions):
            key = (name, qubits)
            self._local[key] = self._local[key].compose(error) if key in self._local else error

    def add_all_qubit_readout_error(self, error: ReadoutError):
        self._readout_all = error

    def add_readout_error(self, error: ReadoutError, qubit: int):
        self._readout_local[int(qubit)] = error

    # used by the simulator
    def errors_for(self, name: str, qubits: tuple):
        """List of (QuantumError, qubits) to apply after instruction ``name``."""
        out = []
        local = self._local.get((name, tuple(qubits)))
        if local is not None:
            out.append((local, tuple(qubits)))
            return out
        err = self._all.get(name)
        if err is None:
            return out
        if err.num_qubits == len(qubits):
            out.append((err, tuple(qubits)))
        elif err.num_qubits == 1:  # 1-qubit error on every qubit of the gate
            out.extend((err, (q,)) for q in qubits)
        else:
            raise ValueError(f"error on '{name}' has wrong size")
        return out

    def readout_for(self, qubit: int):
        return self._readout_local.get(qubit, self._readout_all)

    @property
    def has_readout_errors(self) -> bool:
        return self._readout_all is not None or bool(self._readout_local)

    @property
    def noise_instructions(self) -> list:
        return sorted(set(self._all) | {k[0] for k in self._local})

    def __repr__(self):
        return f"NoiseModel(instructions={self.noise_instructions})"
