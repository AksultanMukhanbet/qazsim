"""QazSim 0.2 execution model: build a ``Circuit``, ``compile`` it once, run it many times.

>>> import qazsim as qz
>>> th = qz.params("θ", 2)
>>> c = qz.Circuit(2).ry(th[0], 0).cx(0, 1).ry(th[1], 1)
>>> prog = qz.compile(c)
>>> prog.expectation("ZZ", [0.1, 0.2])
>>> prog.gradient({"ZZ": 1.0, "XI": 0.5}, [[0.1, 0.2], [0.3, 0.4]])   # batch of 2

Compilation fuses neighbouring gates into blocks of up to ``max_fused`` qubits,
multiplies out every block without free parameters once, and keeps the
structure of parametrised blocks so that new parameter values cost only small
matrix products. All execution happens in the Rust core in one call per batch.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence

import numpy as np

from . import _accelerate

__all__ = ["Param", "params", "Circuit", "compile", "Program", "pauli_terms"]


# ------------------------------------------------------------------ parameters
class Param:
    """Affine expression ``offset + Σ coeff · name``.

    ``Param("θ")`` is a free parameter; ``2 * θ + 0.5`` and ``θ - φ`` stay
    affine. Affinity is what makes exact gradients by the chain rule cheap.
    """

    __slots__ = ("terms", "offset")

    def __init__(self, name: str | None = None, *, terms=None, offset: float = 0.0):
        if name is not None:
            if not isinstance(name, str) or not name:
                raise TypeError("parameter name must be a non-empty string")
            terms = {name: 1.0}
        self.terms: dict[str, float] = dict(terms or {})
        self.offset = float(offset)

    @staticmethod
    def coerce(x) -> "Param | float":
        if isinstance(x, Param):
            return x
        if isinstance(x, str):
            return Param(x)
        if isinstance(x, numbers.Real):
            return float(x)
        raise TypeError(f"gate parameter must be a number, a name or a Param, got {type(x).__name__}")

    @property
    def names(self) -> list[str]:
        return list(self.terms)

    def bind(self, values: Mapping[str, float]) -> float:
        return self.offset + sum(c * float(values[n]) for n, c in self.terms.items())

    def _combine(self, other, sign: float) -> "Param":
        if isinstance(other, str):
            other = Param(other)
        if isinstance(other, Param):
            t = dict(self.terms)
            for n, c in other.terms.items():
                t[n] = t.get(n, 0.0) + sign * c
            return Param(terms={n: c for n, c in t.items() if c != 0.0},
                         offset=self.offset + sign * other.offset)
        if isinstance(other, numbers.Real):
            return Param(terms=self.terms, offset=self.offset + sign * float(other))
        return NotImplemented

    def __add__(self, o):
        return self._combine(o, 1.0)

    __radd__ = __add__

    def __sub__(self, o):
        return self._combine(o, -1.0)

    def __rsub__(self, o):
        r = self._combine(o, -1.0)
        return NotImplemented if r is NotImplemented else -r

    def __mul__(self, o):
        if isinstance(o, numbers.Real):
            o = float(o)
            return Param(terms={n: c * o for n, c in self.terms.items() if c * o != 0.0},
                         offset=self.offset * o)
        if isinstance(o, (Param, str)):
            raise TypeError("product of two parameters is not affine; QazSim gate "
                            "parameters must be affine (a·θ + b·φ + c)")
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, numbers.Real):
            return self * (1.0 / float(o))
        return NotImplemented

    def __neg__(self):
        return self * -1.0

    def __repr__(self):
        parts = [f"{'' if c == 1 else ('-' if c == -1 else f'{c:g}·')}{n}" for n, c in self.terms.items()]
        if self.offset or not parts:
            parts.append(f"{self.offset:g}")
        return " + ".join(parts).replace("+ -", "- ")


def params(name: str, count: int | None = None):
    """``params("θ")`` -> one Param; ``params("θ", 3)`` -> [θ[0], θ[1], θ[2]]."""
    if count is None:
        return Param(name)
    return [Param(f"{name}[{i}]") for i in range(int(count))]


# --------------------------------------------------------------------- circuit
_GATES = {
    # name: (qubits, parameters)
    "id": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0), "h": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0), "sx": (1, 0), "sxdg": (1, 0),
    "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1), "u": (1, 3),
    "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "swap": (2, 0), "iswap": (2, 0),
    "cp": (2, 1), "crx": (2, 1), "cry": (2, 1), "crz": (2, 1),
    "rxx": (2, 1), "ryy": (2, 1), "rzz": (2, 1),
    "ccx": (3, 0), "ccz": (3, 0), "cswap": (3, 0),
}


class Circuit:
    """Gate sequence on ``num_qubits`` qubits. Methods return the circuit, so
    calls chain: ``Circuit(2).h(0).cx(0, 1)``.

    Qubit 0 is the least significant bit of a basis index; in bit strings and
    Pauli labels the rightmost character is qubit 0.
    """

    def __init__(self, num_qubits: int):
        if int(num_qubits) < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.num_qubits = int(num_qubits)
        self.ops: list[tuple] = []  # (name, qubits, params, matrix)

    # ---- construction ----------------------------------------------------
    def append(self, name: str, qubits, params=()) -> "Circuit":
        name = name.lower()
        if name not in _GATES:
            raise KeyError(f"unknown gate '{name}'")
        nq, npar = _GATES[name]
        qubits = (qubits,) if isinstance(qubits, (int, np.integer)) else tuple(qubits)
        qubits = tuple(int(q) for q in qubits)
        if len(qubits) != nq:
            raise ValueError(f"gate '{name}' acts on {nq} qubit(s), got {len(qubits)}")
        if len(params) != npar:
            raise ValueError(f"gate '{name}' takes {npar} parameter(s), got {len(params)}")
        self._check(qubits)
        self.ops.append((name, qubits, tuple(Param.coerce(p) for p in params), None))
        return self

    def _check(self, qubits):
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(f"qubit {q} out of range (circuit has {self.num_qubits})")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"duplicate qubits {qubits}")

    def unitary(self, matrix, qubits) -> "Circuit":
        """Arbitrary unitary; bit b of the matrix index belongs to ``qubits[b]``."""
        qubits = (qubits,) if isinstance(qubits, (int, np.integer)) else tuple(int(q) for q in qubits)
        self._check(qubits)
        m = np.asarray(matrix, dtype=np.complex128)
        d = 1 << len(qubits)
        if m.shape != (d, d):
            raise ValueError(f"matrix must be {d}x{d} for {len(qubits)} qubit(s)")
        if not np.allclose(m.conj().T @ m, np.eye(d), atol=1e-10):
            raise ValueError("matrix is not unitary")
        self.ops.append(("unitary", qubits, (), m))
        return self

    def compose(self, other: "Circuit", qubits=None) -> "Circuit":
        """Append ``other``; ``qubits[i]`` is where its qubit i goes."""
        qmap = list(range(other.num_qubits)) if qubits is None else list(qubits)
        if len(qmap) != other.num_qubits:
            raise ValueError("qubits must map every qubit of the appended circuit")
        for name, qs, ps, m in other.ops:
            mapped = tuple(qmap[q] for q in qs)
            self._check(mapped)
            self.ops.append((name, mapped, ps, m))
        return self

    def copy(self) -> "Circuit":
        c = Circuit(self.num_qubits)
        c.ops = list(self.ops)
        return c

    # ---- inspection -------------------------------------------------------
    @property
    def parameters(self) -> list[str]:
        """Parameter names in order of first appearance."""
        seen: dict[str, None] = {}
        for _, _, ps, _ in self.ops:
            for p in ps:
                if isinstance(p, Param):
                    for n in p.terms:
                        seen.setdefault(n, None)
        return list(seen)

    def count_ops(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, *_ in self.ops:
            out[name] = out.get(name, 0) + 1
        return out

    def __len__(self):
        return len(self.ops)

    def __repr__(self):
        return f"Circuit({self.num_qubits} qubits, {len(self.ops)} gates, {len(self.parameters)} parameters)"

    # ---- conversion -------------------------------------------------------
    @classmethod
    def from_quantum_circuit(cls, qc) -> "Circuit":
        """Convert a 0.1 ``QuantumCircuit`` (gates only; measurements and
        ``save_*`` directives are dropped). Its parameter expressions must be
        affine; they are probed numerically and checked."""
        from .circuit import ParameterExpression

        c = cls(qc.num_qubits)
        for ins in qc.data:
            if ins.name in ("measure", "barrier", "save_statevector",
                            "save_density_matrix", "save_probabilities"):
                continue
            if ins.name == "reset":
                raise ValueError("reset is not supported by compiled programs")
            if ins.name == "unitary":
                c.unitary(ins.matrix, ins.qubits)
                continue
            ps = [_affine_from_expression(p) if isinstance(p, ParameterExpression) else float(p)
                  for p in ins.params]
            c.append(ins.name, ins.qubits, ps)
        return c

    # ---- gates (generated below) ----------------------------------------


def _affine_from_expression(expr) -> Param:
    params_ = sorted(expr.parameters, key=lambda p: p.name)
    zero = {p: 0.0 for p in params_}
    offset = float(expr.bind(zero))
    terms = {}
    for p in params_:
        terms[p.name] = float(expr.bind({**zero, p: 1.0})) - offset
    rng = np.random.default_rng(1234)
    for _ in range(2):
        point = {p: float(rng.normal()) for p in params_}
        expect = offset + sum(terms[p.name] * v for p, v in point.items())
        if abs(float(expr.bind(point)) - expect) > 1e-9 * (1 + abs(expect)):
            raise ValueError(f"{expr!r} is not affine in its parameters")
    return Param(terms={n: c for n, c in terms.items() if c != 0.0}, offset=offset)


def _make_gate(name, nq, npar):
    if npar == 0:
        def gate(self, *qubits):
            return self.append(name, qubits)
        gate.__doc__ = f"{name.upper()} on qubit(s) {', '.join('q%d' % i for i in range(nq))}."
    else:
        def gate(self, *args):
            if len(args) != npar + nq:
                raise TypeError(f"{name}() takes {npar} parameter(s) then {nq} qubit(s)")
            return self.append(name, args[npar:], args[:npar])
        gate.__doc__ = f"{name.upper()}: {npar} parameter(s) then {nq} qubit(s)."
    gate.__name__ = name
    return gate


for _n, (_q, _p) in _GATES.items():
    setattr(Circuit, _n, _make_gate(_n, _q, _p))


# ----------------------------------------------------------------- observables
_PAULI = {"I", "X", "Y", "Z"}


def pauli_terms(observable, num_qubits: int) -> list[tuple[int, int, float]]:
    """Parse ``"ZZ"``, ``{"ZZ": 1.0, "XI": 0.5}`` or ``[("ZZ", 1.0), ...]`` into
    (x_mask, z_mask, coeff) triples. Rightmost character is qubit 0."""
    if isinstance(observable, str):
        items = [(observable, 1.0)]
    elif isinstance(observable, Mapping):
        items = list(observable.items())
    elif isinstance(observable, Sequence):
        items = [tuple(t) for t in observable]
    else:
        raise TypeError("observable must be a Pauli label, a dict or a list of (label, coeff)")
    out = []
    for label, coeff in items:
        label = label.upper().replace(" ", "")
        if len(label) != num_qubits or set(label) - _PAULI:
            raise ValueError(f"Pauli label '{label}' must use I, X, Y, Z and have {num_qubits} characters")
        c = complex(coeff)
        if abs(c.imag) > 1e-12:
            raise ValueError("observable coefficients must be real (Hermitian observable)")
        x = z = 0
        for i, ch in enumerate(reversed(label)):
            if ch in "XY":
                x |= 1 << i
            if ch in "ZY":
                z |= 1 << i
        out.append((x, z, c.real))
    return out


# --------------------------------------------------------------------- program
class Program:
    """A compiled circuit. Parameter values can be given as

    * a dict ``{name: value}`` or a 1-D array in ``program.parameters`` order -> single result;
    * a 2-D array of shape ``(batch, num_parameters)`` or a list of dicts -> batched result.
    """

    def __init__(self, circuit: Circuit, max_fused: int, max_fused_diag: int):
        self.parameters: tuple[str, ...] = tuple(circuit.parameters)
        index = {n: i for i, n in enumerate(self.parameters)}
        specs = []
        for name, qubits, ps, matrix in circuit.ops:
            exprs = []
            for p in ps:
                if isinstance(p, Param):
                    exprs.append((p.offset, [(index[n], c) for n, c in p.terms.items()]))
                else:
                    exprs.append((float(p), []))
            mat = None if matrix is None else [complex(v) for v in np.asarray(matrix).ravel()]
            specs.append((name, list(qubits), exprs, mat))
        self.num_qubits = circuit.num_qubits
        self._core = _accelerate.Program(circuit.num_qubits, specs, len(self.parameters),
                                         max_fused, max_fused_diag)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def stats(self) -> dict:
        """Size of the compiled plan: ops, fused blocks, largest block, ..."""
        return dict(self._core.stats())

    def __repr__(self):
        s = self.stats
        return (f"Program({self.num_qubits} qubits, {s['ops']} gates -> {s['blocks']} blocks, "
                f"{self.num_parameters} parameters)")

    # ---- parameter handling ----------------------------------------------
    def _values(self, values) -> tuple[np.ndarray, bool]:
        """-> (array of shape (batch, P), batched?)"""
        P = self.num_parameters
        if values is None:
            if P:
                raise ValueError(f"program has {P} parameter(s): {', '.join(self.parameters)}")
            return np.zeros((1, 0)), False
        if isinstance(values, Mapping):
            return self._row_from_dict(values)[None, :], False
        if isinstance(values, Sequence) and values and isinstance(values[0], Mapping):
            return np.stack([self._row_from_dict(v) for v in values]), True
        a = np.asarray(values, dtype=np.float64)
        if a.ndim == 1 and a.shape[0] == P:
            return np.ascontiguousarray(a[None, :]), False
        if a.ndim == 2 and a.shape[1] == P:
            return np.ascontiguousarray(a), True
        raise ValueError(f"expected {P} parameter value(s) per row, got shape {a.shape}")

    def _row_from_dict(self, d) -> np.ndarray:
        missing = [n for n in self.parameters if n not in d]
        if missing:
            raise ValueError(f"missing parameter value(s): {', '.join(missing)}")
        return np.array([float(d[n]) for n in self.parameters])

    # ---- execution --------------------------------------------------------
    def statevector(self, values=None) -> np.ndarray:
        a, batched = self._values(values)
        if batched:
            return np.stack([self._core.statevector(row) for row in a])
        return self._core.statevector(a[0])

    def probabilities(self, values=None) -> np.ndarray:
        a, batched = self._values(values)
        out = self._core.probabilities(a)
        return out if batched else out[0]

    def sample(self, shots: int, values=None, seed=None) -> dict[str, int] | list[dict[str, int]]:
        """Measure all qubits ``shots`` times -> counts ``{'01': n, ...}``
        (rightmost character is qubit 0)."""
        probs = self.probabilities(values)
        rng = np.random.default_rng(seed)
        rows = probs if probs.ndim == 2 else probs[None, :]
        res = []
        for p in rows:
            counts = rng.multinomial(int(shots), p / p.sum())
            res.append({format(i, f"0{self.num_qubits}b"): int(counts[i]) for i in np.flatnonzero(counts)})
        return res if probs.ndim == 2 else res[0]

    def expectation(self, observable, values=None):
        """<ψ(θ)| H |ψ(θ)> for a Pauli-sum observable."""
        a, batched = self._values(values)
        out = self._core.expectation(a, pauli_terms(observable, self.num_qubits))
        return out if batched else float(out[0])

    def value_and_gradient(self, observable, values=None):
        """(<H>, ∂<H>/∂θ) by adjoint differentiation: cost ≈ 3 circuit runs,
        independent of the number of parameters."""
        a, batched = self._values(values)
        v, g = self._core.value_and_gradient(a, pauli_terms(observable, self.num_qubits))
        return (v, g) if batched else (float(v[0]), g[0])

    def gradient(self, observable, values=None) -> np.ndarray:
        return self.value_and_gradient(observable, values)[1]


def compile(circuit: Circuit, *, max_fused: int = 5, max_fused_diag: int = 10) -> Program:  # noqa: A001
    """Compile a ``Circuit`` (or a 0.1 ``QuantumCircuit``) into a ``Program``.

    ``max_fused``: largest dense block (1-5 qubits, 1 disables fusion). Within
    that limit a cost model decides whether fusing two blocks pays off;
    ``max_fused_diag``: largest block made only of diagonal gates.
    """
    if not isinstance(circuit, Circuit):
        circuit = Circuit.from_quantum_circuit(circuit)
    return Program(circuit, int(max_fused), int(max_fused_diag))
