"""Standard gate matrices.

Convention (same as Qiskit): little-endian. For a gate applied to qubits
(q0, q1, ...), bit b of the matrix index corresponds to q_b. For example
``cx(control, target)`` flips the target (bit 1) when the control (bit 0) is 1.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

_C = np.complex128


def _m(rows) -> np.ndarray:
    return np.array(rows, dtype=_C)


# ---------- one-qubit gates ----------
def id_():
    return np.eye(2, dtype=_C)


def x():
    return _m([[0, 1], [1, 0]])


def y():
    return _m([[0, -1j], [1j, 0]])


def z():
    return _m([[1, 0], [0, -1]])


def h():
    s = 1 / math.sqrt(2)
    return _m([[s, s], [s, -s]])


def s():
    return _m([[1, 0], [0, 1j]])


def sdg():
    return _m([[1, 0], [0, -1j]])


def t():
    return _m([[1, 0], [0, cmath.exp(1j * math.pi / 4)]])


def tdg():
    return _m([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]])


def sx():
    return 0.5 * _m([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


def rx(theta):
    c, s_ = math.cos(theta / 2), math.sin(theta / 2)
    return _m([[c, -1j * s_], [-1j * s_, c]])


def ry(theta):
    c, s_ = math.cos(theta / 2), math.sin(theta / 2)
    return _m([[c, -s_], [s_, c]])


def rz(theta):
    return _m([[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]])


def p(lam):
    return _m([[1, 0], [0, cmath.exp(1j * lam)]])


def u(theta, phi, lam):
    c, s_ = math.cos(theta / 2), math.sin(theta / 2)
    return _m(
        [
            [c, -cmath.exp(1j * lam) * s_],
            [cmath.exp(1j * phi) * s_, cmath.exp(1j * (phi + lam)) * c],
        ]
    )


# ---------- two-qubit gates (qubit order: control/first = bit 0) ----------
def cx():
    return _m([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def cy():
    return _m([[1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1, 0], [0, 1j, 0, 0]])


def cz():
    return np.diag([1, 1, 1, -1]).astype(_C)


def swap():
    return _m([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def cp(lam):
    return np.diag([1, 1, 1, cmath.exp(1j * lam)]).astype(_C)


def crz(theta):
    return np.diag(
        [1, cmath.exp(-1j * theta / 2), 1, cmath.exp(1j * theta / 2)]
    ).astype(_C)


def rzz(theta):
    a, b = cmath.exp(-1j * theta / 2), cmath.exp(1j * theta / 2)
    return np.diag([a, b, b, a]).astype(_C)


# ---------- three-qubit gates ----------
def ccx():
    m = np.eye(8, dtype=_C)
    m[[3, 7]] = m[[7, 3]]
    return m


# name -> (number of qubits, number of parameters, matrix function)
STANDARD_GATES = {
    "id": (1, 0, id_),
    "x": (1, 0, x),
    "y": (1, 0, y),
    "z": (1, 0, z),
    "h": (1, 0, h),
    "s": (1, 0, s),
    "sdg": (1, 0, sdg),
    "t": (1, 0, t),
    "tdg": (1, 0, tdg),
    "sx": (1, 0, sx),
    "rx": (1, 1, rx),
    "ry": (1, 1, ry),
    "rz": (1, 1, rz),
    "p": (1, 1, p),
    "u": (1, 3, u),
    "cx": (2, 0, cx),
    "cy": (2, 0, cy),
    "cz": (2, 0, cz),
    "swap": (2, 0, swap),
    "cp": (2, 1, cp),
    "crz": (2, 1, crz),
    "rzz": (2, 1, rzz),
    "ccx": (3, 0, ccx),
}


def gate_matrix(name: str, params=()) -> np.ndarray:
    """Matrix of a standard gate with numeric parameters."""
    if name not in STANDARD_GATES:
        raise KeyError(f"unknown gate '{name}'")
    _, n_par, fn = STANDARD_GATES[name]
    if len(params) != n_par:
        raise ValueError(f"gate '{name}' takes {n_par} parameter(s), got {len(params)}")
    return fn(*[float(v) for v in params])
