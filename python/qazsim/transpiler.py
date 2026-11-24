"""Transpiler: rewrite a circuit into a device's basis gates and simplify it.

Assumes all-to-all connectivity (no qubit routing yet). That is exact for
SpinQ Gemini Mini Pro, which has 2 coupled qubits.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from .circuit import Instruction, QuantumCircuit
from .gates import STANDARD_GATES, gate_matrix

_EPS = 1e-10
_DEFAULT_BASIS = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx",
                  "rx", "ry", "rz", "p", "u", "cx", "cy", "cz", "swap",
                  "cp", "crz", "rzz", "ccx", "unitary"]
_DIRECTIVES = {"measure", "reset", "barrier", "save_statevector",
               "save_density_matrix", "save_probabilities"}


class TranspilerError(Exception):
    pass


# ------------------------------------------------------- 1-qubit synthesis
def zyz_angles(u: np.ndarray):
    """U = e^{i·phase} Rz(phi) Ry(theta) Rz(lam). Returns (theta, phi, lam, phase)."""
    u = np.asarray(u, dtype=np.complex128)
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    v = u / cmath.sqrt(det)  # SU(2)
    theta = 2 * math.atan2(abs(v[1, 0]), abs(v[0, 0]))
    if abs(v[0, 0]) < _EPS:
        plus, minus = 0.0, 2 * cmath.phase(v[1, 0])
    elif abs(v[1, 0]) < _EPS:
        plus, minus = 2 * cmath.phase(v[1, 1]), 0.0
    else:
        plus, minus = 2 * cmath.phase(v[1, 1]), 2 * cmath.phase(v[1, 0])
    phi, lam = (plus + minus) / 2, (plus - minus) / 2
    m = gate_matrix("rz", (phi,)) @ gate_matrix("ry", (theta,)) @ gate_matrix("rz", (lam,))
    i, j = np.unravel_index(np.argmax(np.abs(u)), u.shape)
    phase = cmath.phase(u[i, j] / m[i, j])
    return theta, phi, lam, phase


def _wrap(a: float):
    """Map a rotation angle to (-π, π]. R(a + 2π) = -R(a), so each 2π shift
    adds π to the global phase. Returns (angle, phase_change)."""
    k = math.floor((a + math.pi) / (2 * math.pi))
    if abs((a + math.pi) - 2 * math.pi * k) < _EPS:  # a == -π (mod 2π) -> use +π
        k -= 1
    return a - 2 * math.pi * k, (math.pi * k) % (2 * math.pi)


def synth_1q(u: np.ndarray, q: int, basis: set):
    """Return ([Instruction...], global_phase) implementing u on qubit q."""
    theta, phi, lam, phase = zyz_angles(u)
    if abs(theta) < 1e-9:  # pure Z rotation
        theta, phi, lam = 0.0, phi + lam, 0.0
    if {"rz", "ry"} <= basis:
        seq = [("rz", lam), ("ry", theta), ("rz", phi)]
    elif {"rz", "rx"} <= basis:  # Ry(t) = Rz(π/2) Rx(t) Rz(-π/2)
        seq = [("rz", lam - math.pi / 2), ("rx", theta), ("rz", phi + math.pi / 2)]
        if abs(theta) < 1e-9:
            seq = [("rz", phi + lam)]
    elif "u" in basis:
        return ([Instruction("u", (q,), (), (theta, phi, lam))],
                phase - (phi + lam) / 2)
    else:
        raise TranspilerError(f"cannot synthesize 1-qubit gates in basis {sorted(basis)}")
    out = []
    for name, ang in seq:
        a, dph = _wrap(ang)
        phase += dph
        if abs(a) < 1e-9:
            continue
        out.append(Instruction(name, (q,), (), (a,)))
    return out, phase


# ------------------------------------------------------ multi-qubit rules
def _rule(ins: Instruction, basis: set):
    """One decomposition step for a gate not in the basis, as (name, qubits, params)."""
    n, q, p = ins.name, ins.qubits, ins.params
    two = "cx" if "cx" in basis else ("cz" if "cz" in basis else None)
    if two is None:
        raise TranspilerError("basis needs 'cx' or 'cz' for multi-qubit gates")
    if n == "cx":
        a, b = q
        return [("h", (b,), ()), ("cz", (a, b), ()), ("h", (b,), ())]
    if n == "cz":
        a, b = q
        return [("h", (b,), ()), ("cx", (a, b), ()), ("h", (b,), ())]
    if n == "cy":
        a, b = q
        return [("sdg", (b,), ()), ("cx", (a, b), ()), ("s", (b,), ())]
    if n == "swap":
        a, b = q
        return [("cx", (a, b), ()), ("cx", (b, a), ()), ("cx", (a, b), ())]
    if n == "cp":
        (lam,), (a, b) = p, q
        return [("p", (a,), (lam / 2,)), ("cx", (a, b), ()), ("p", (b,), (-lam / 2,)),
                ("cx", (a, b), ()), ("p", (b,), (lam / 2,))]
    if n == "crz":
        (th,), (a, b) = p, q
        return [("rz", (b,), (th / 2,)), ("cx", (a, b), ()), ("rz", (b,), (-th / 2,)),
                ("cx", (a, b), ())]
    if n == "rzz":
        (th,), (a, b) = p, q
        return [("cx", (a, b), ()), ("rz", (b,), (th,)), ("cx", (a, b), ())]
    if n == "ccx":
        a, b, c = q
        return [("h", (c,), ()), ("cx", (b, c), ()), ("tdg", (c,), ()), ("cx", (a, c), ()),
                ("t", (c,), ()), ("cx", (b, c), ()), ("tdg", (c,), ()), ("cx", (a, c), ()),
                ("t", (b,), ()), ("t", (c,), ()), ("h", (c,), ()), ("cx", (a, b), ()),
                ("t", (a,), ()), ("tdg", (b,), ()), ("cx", (a, b), ())]
    raise TranspilerError(f"no decomposition rule for '{n}' on {len(q)} qubits")


def _decompose(ins: Instruction, basis: set, out: list, depth=0) -> float:
    """Append basis instructions for ``ins`` to ``out``; returns added global phase."""
    if depth > 10:
        raise TranspilerError(f"decomposition of '{ins.name}' does not terminate")
    if ins.name in _DIRECTIVES or ins.name in basis:
        out.append(ins)
        return 0.0
    if len(ins.qubits) == 1:
        seq, phase = synth_1q(ins.to_matrix(), ins.qubits[0], basis)
        out.extend(seq)
        return phase
    if ins.name == "unitary":
        raise TranspilerError("multi-qubit 'unitary' cannot be decomposed yet")
    phase = 0.0
    for name, qs, ps in _rule(ins, basis):
        phase += _decompose(Instruction(name, qs, (), ps), basis, out, depth + 1)
    return phase


# ----------------------------------------------------------- optimization
def _merge_1q(instrs: list, n: int, basis: set):
    """Fuse runs of 1-qubit gates on each qubit and re-synthesize them."""
    out, phase = [], 0.0
    pending: dict[int, list] = {}

    def flush(q):
        nonlocal phase
        run = pending.pop(q, None)
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
            return
        m = np.eye(2, dtype=np.complex128)
        for g in run:
            m = g.to_matrix() @ m
        seq, ph = synth_1q(m, q, basis)
        if len(seq) <= len(run):
            out.extend(seq)
            phase += ph
        else:
            out.extend(run)

    for ins in instrs:
        if len(ins.qubits) == 1 and ins.is_gate:
            pending.setdefault(ins.qubits[0], []).append(ins)
            continue
        for q in (ins.qubits if ins.name != "barrier" else range(n)):
            flush(q)
        out.append(ins)
    for q in list(pending):
        flush(q)
    return out, phase


_SELF_INVERSE_2Q = {"cx", "cy", "cz", "swap"}
_SYMMETRIC = {"cz", "swap"}


def _cancel_pairs(instrs: list) -> list:
    """Remove adjacent pairs of identical self-inverse 2-qubit gates."""
    out: list = []
    last: dict[int, int] = {}  # qubit -> index in out of the last op on it
    for ins in instrs:
        if ins.name in _SELF_INVERSE_2Q:
            a, b = ins.qubits
            ia, ib = last.get(a), last.get(b)
            if ia is not None and ia == ib and out[ia] is not None:
                prev = out[ia]
                same = prev.name == ins.name and (
                    prev.qubits == ins.qubits
                    or (ins.name in _SYMMETRIC and set(prev.qubits) == set(ins.qubits)))
                if same:
                    out[ia] = None
                    for q in (a, b):  # find previous op on these qubits
                        last.pop(q, None)
                        for k in range(ia - 1, -1, -1):
                            if out[k] is not None and q in out[k].qubits:
                                last[q] = k
                                break
                    continue
        out.append(ins)
        idx = len(out) - 1
        for q in ins.qubits:
            last[q] = idx
    return [i for i in out if i is not None]


# ------------------------------------------------------------- entry point
def transpile(circuits, backend=None, basis_gates=None, optimization_level: int = 1):
    """Rewrite circuit(s) into ``basis_gates`` (or ``backend.basis_gates``).

    optimization_level 0: decomposition only; 1: also fuse 1-qubit gates and
    cancel adjacent CX/CZ/CY/SWAP pairs.
    """
    single = isinstance(circuits, QuantumCircuit)
    circs = [circuits] if single else list(circuits)
    if basis_gates is None and backend is not None:
        basis_gates = getattr(backend, "basis_gates", None)
    basis = set(basis_gates or _DEFAULT_BASIS)
    outs = []
    for qc in circs:
        if backend is not None and hasattr(backend, "num_qubits") and \
                qc.num_qubits > backend.num_qubits:
            raise TranspilerError(f"circuit has {qc.num_qubits} qubits, "
                                  f"device '{backend.name}' has {backend.num_qubits}")
        new = qc.copy()
        instrs, phase = [], qc.global_phase
        for ins in qc.data:
            if ins.is_parameterized:
                raise TranspilerError("bind parameters before transpiling")
            phase += _decompose(ins, basis, instrs)
        if optimization_level >= 1:
            for _ in range(3):
                instrs, ph = _merge_1q(instrs, qc.num_qubits, basis)
                phase += ph
                instrs = _cancel_pairs(instrs)
        new.data = instrs
        new.global_phase = phase
        outs.append(new)
    return outs[0] if single else outs
