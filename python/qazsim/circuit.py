"""QuantumCircuit, Instruction and Parameter — API modelled on Qiskit."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
import numpy as np

from .gates import STANDARD_GATES, gate_matrix

# ---------------------------------------------------------------- parameters


class ParameterExpression:
    """Arithmetic expression of Parameters (supports + - * / and unary -)."""

    def __init__(self, params: frozenset, fn):
        self._params = params
        self._fn = fn  # fn(values: dict[Parameter, float]) -> float

    @property
    def parameters(self) -> frozenset:
        return self._params

    def bind(self, values: dict):
        missing = [p for p in self._params if p not in values]
        if missing:
            return ParameterExpression(
                frozenset(missing),
                lambda v, f=self._fn, b=dict(values): f({**b, **v}),
            )
        return float(self._fn(values))

    # arithmetic
    def _op(self, other, op):
        if isinstance(other, ParameterExpression):
            return ParameterExpression(
                self._params | other._params,
                lambda v, a=self._fn, b=other._fn: op(a(v), b(v)),
            )
        if isinstance(other, numbers.Real):
            return ParameterExpression(self._params, lambda v, a=self._fn: op(a(v), other))
        return NotImplemented

    def _rop(self, other, op):
        if isinstance(other, numbers.Real):
            return ParameterExpression(self._params, lambda v, a=self._fn: op(other, a(v)))
        return NotImplemented

    def __add__(self, o):
        return self._op(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._rop(o, lambda a, b: a + b)

    def __sub__(self, o):
        return self._op(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._rop(o, lambda a, b: a - b)

    def __mul__(self, o):
        return self._op(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return self._rop(o, lambda a, b: a * b)

    def __truediv__(self, o):
        return self._op(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return self._rop(o, lambda a, b: a / b)

    def __neg__(self):
        return ParameterExpression(self._params, lambda v, a=self._fn: -a(v))

    def __repr__(self):
        names = ", ".join(sorted(p.name for p in self._params))
        return f"ParameterExpression({names})"


class Parameter(ParameterExpression):
    """A named free parameter, e.g. ``theta = Parameter("θ")``."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(frozenset([self]), lambda v, me=self: v[me])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"Parameter({self.name})"


class ParameterVector(list):
    """List of Parameters named ``name[0]``, ``name[1]``, ..."""

    def __init__(self, name: str, length: int):
        super().__init__(Parameter(f"{name}[{i}]") for i in range(length))
        self.name = name


def _bind_value(v, values):
    if isinstance(v, ParameterExpression):
        return v.bind(values)
    return v


# --------------------------------------------------------------- instructions


@dataclass
class Instruction:
    name: str
    qubits: tuple
    clbits: tuple = ()
    params: tuple = ()
    matrix: np.ndarray | None = field(default=None, repr=False)  # for "unitary"
    label: str | None = None

    @property
    def is_gate(self) -> bool:
        return self.name in STANDARD_GATES or self.name == "unitary"

    @property
    def is_parameterized(self) -> bool:
        return any(isinstance(p, ParameterExpression) for p in self.params)

    def to_matrix(self) -> np.ndarray:
        if self.name == "unitary":
            return self.matrix
        if self.is_parameterized:
            raise ValueError(f"gate '{self.name}' has unbound parameters")
        return gate_matrix(self.name, self.params)


# Instructions that are not gates but are understood by the simulators.
DIRECTIVES = {"measure", "reset", "barrier", "save_statevector",
              "save_density_matrix", "save_probabilities"}


# -------------------------------------------------------------------- circuit


def _as_list(x) -> list:
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    return [int(i) for i in x]


class QuantumCircuit:
    """Quantum circuit with an API close to ``qiskit.QuantumCircuit``.

    >>> qc = QuantumCircuit(2, 2)
    >>> qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    """

    def __init__(self, num_qubits: int, num_clbits: int = 0, name: str | None = None):
        if num_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.num_qubits = int(num_qubits)
        self.num_clbits = int(num_clbits)
        self.name = name or "circuit"
        self.data: list[Instruction] = []
        self.global_phase = 0.0

    # ---- generic append -------------------------------------------------
    def _check_qubits(self, qubits):
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(f"qubit {q} out of range (circuit has {self.num_qubits})")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"duplicate qubits {qubits}")

    def append(self, name: str, qubits, params=(), clbits=()) -> "QuantumCircuit":
        qubits = tuple(_as_list(qubits))
        self._check_qubits(qubits)
        if name in STANDARD_GATES:
            n_q, n_p, _ = STANDARD_GATES[name]
            if len(qubits) != n_q:
                raise ValueError(f"gate '{name}' acts on {n_q} qubit(s)")
            if len(params) != n_p:
                raise ValueError(f"gate '{name}' takes {n_p} parameter(s)")
        elif name not in DIRECTIVES:
            raise KeyError(f"unknown instruction '{name}'")
        self.data.append(Instruction(name, qubits, tuple(clbits), tuple(params)))
        return self

    # ---- one-qubit gates -------------------------------------------------
    def id(self, q):
        return self.append("id", q)

    def x(self, q):
        return self.append("x", q)

    def y(self, q):
        return self.append("y", q)

    def z(self, q):
        return self.append("z", q)

    def h(self, q):
        return self.append("h", q)

    def s(self, q):
        return self.append("s", q)

    def sdg(self, q):
        return self.append("sdg", q)

    def t(self, q):
        return self.append("t", q)

    def tdg(self, q):
        return self.append("tdg", q)

    def sx(self, q):
        return self.append("sx", q)

    def rx(self, theta, q):
        return self.append("rx", q, (theta,))

    def ry(self, theta, q):
        return self.append("ry", q, (theta,))

    def rz(self, theta, q):
        return self.append("rz", q, (theta,))

    def p(self, lam, q):
        return self.append("p", q, (lam,))

    def u(self, theta, phi, lam, q):
        return self.append("u", q, (theta, phi, lam))

    # ---- multi-qubit gates ----------------------------------------------
    def cx(self, control, target):
        return self.append("cx", (control, target))

    def cy(self, control, target):
        return self.append("cy", (control, target))

    def cz(self, control, target):
        return self.append("cz", (control, target))

    def swap(self, q1, q2):
        return self.append("swap", (q1, q2))

    def cp(self, lam, control, target):
        return self.append("cp", (control, target), (lam,))

    def crz(self, theta, control, target):
        return self.append("crz", (control, target), (theta,))

    def rzz(self, theta, q1, q2):
        return self.append("rzz", (q1, q2), (theta,))

    def ccx(self, c1, c2, target):
        return self.append("ccx", (c1, c2, target))

    def unitary(self, matrix, qubits, label: str | None = None):
        """Arbitrary unitary on ``qubits`` (qubits[0] = least significant bit)."""
        qubits = tuple(_as_list(qubits))
        self._check_qubits(qubits)
        m = np.asarray(matrix, dtype=np.complex128)
        d = 2 ** len(qubits)
        if m.shape != (d, d):
            raise ValueError(f"matrix must be {d}x{d}")
        if not np.allclose(m @ m.conj().T, np.eye(d), atol=1e-8):
            raise ValueError("matrix is not unitary")
        self.data.append(Instruction("unitary", qubits, (), (), m, label))
        return self

    # ---- non-unitary instructions ---------------------------------------
    def measure(self, qubit, clbit):
        qs, cs = _as_list(qubit), _as_list(clbit)
        if len(qs) != len(cs):
            raise ValueError("number of qubits and clbits must match")
        for q, c in zip(qs, cs):
            self._check_qubits((q,))
            if not 0 <= c < self.num_clbits:
                raise IndexError(f"clbit {c} out of range (circuit has {self.num_clbits})")
            self.data.append(Instruction("measure", (q,), (c,)))
        return self

    def measure_all(self):
        """Add one classical bit per qubit and measure every qubit."""
        start = self.num_clbits
        self.num_clbits += self.num_qubits
        self.barrier()
        return self.measure(list(range(self.num_qubits)),
                            list(range(start, start + self.num_qubits)))

    def reset(self, qubit):
        for q in _as_list(qubit):
            self.append("reset", q)
        return self

    def barrier(self, *qubits):
        qs = [q for x in qubits for q in _as_list(x)] or list(range(self.num_qubits))
        self.data.append(Instruction("barrier", tuple(qs)))
        return self

    # Aer-style save instructions: store the state at this point of the circuit
    def save_statevector(self, label: str = "statevector"):
        self.data.append(Instruction("save_statevector", tuple(range(self.num_qubits)), label=label))
        return self

    def save_density_matrix(self, label: str = "density_matrix"):
        self.data.append(Instruction("save_density_matrix", tuple(range(self.num_qubits)), label=label))
        return self

    def save_probabilities(self, label: str = "probabilities"):
        self.data.append(Instruction("save_probabilities", tuple(range(self.num_qubits)), label=label))
        return self

    # ---- parameters -------------------------------------------------------
    @property
    def parameters(self) -> list:
        ps = set()
        for ins in self.data:
            for v in ins.params:
                if isinstance(v, ParameterExpression):
                    ps |= v.parameters
        return sorted(ps, key=lambda p: p.name)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def assign_parameters(self, values, inplace: bool = False) -> "QuantumCircuit":
        """Bind parameters from a dict {Parameter: value} or a list in
        ``self.parameters`` order."""
        if not isinstance(values, dict):
            values = list(values)
            params = self.parameters
            if len(values) != len(params):
                raise ValueError(f"expected {len(params)} values, got {len(values)}")
            values = dict(zip(params, values))
        qc = self if inplace else self.copy()
        for ins in qc.data:
            if ins.params:
                ins.params = tuple(_bind_value(v, values) for v in ins.params)
        return qc

    # ---- utilities --------------------------------------------------------
    def copy(self, name: str | None = None) -> "QuantumCircuit":
        qc = QuantumCircuit(self.num_qubits, self.num_clbits, name or self.name)
        qc.global_phase = self.global_phase
        qc.data = [Instruction(i.name, i.qubits, i.clbits, i.params, i.matrix, i.label)
                   for i in self.data]
        return qc

    def compose(self, other: "QuantumCircuit", qubits=None, clbits=None,
                inplace: bool = False) -> "QuantumCircuit":
        """Append ``other`` mapped onto ``qubits`` / ``clbits`` of this circuit."""
        qmap = _as_list(qubits) if qubits is not None else list(range(other.num_qubits))
        cmap = _as_list(clbits) if clbits is not None else list(range(other.num_clbits))
        if len(qmap) != other.num_qubits or len(cmap) != other.num_clbits:
            raise ValueError("qubit/clbit mapping does not match the other circuit")
        qc = self if inplace else self.copy()
        qc.global_phase += other.global_phase
        for i in other.data:
            qc.data.append(Instruction(i.name, tuple(qmap[q] for q in i.qubits),
                                       tuple(cmap[c] for c in i.clbits),
                                       i.params, i.matrix, i.label))
        return qc

    def inverse(self) -> "QuantumCircuit":
        """Inverse circuit (gates only)."""
        inv = QuantumCircuit(self.num_qubits, self.num_clbits, self.name + "_dg")
        inv.global_phase = -self.global_phase
        self_inverse = {"id", "x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx"}
        neg = {"rx", "ry", "rz", "p", "cp", "crz", "rzz"}
        swap_names = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        for ins in reversed(self.data):
            n = ins.name
            if n == "barrier":
                inv.data.append(ins)
            elif n in self_inverse:
                inv.append(n, ins.qubits)
            elif n in neg:
                inv.append(n, ins.qubits, (-ins.params[0],))
            elif n in swap_names:
                inv.append(swap_names[n], ins.qubits)
            elif n == "u":
                th, ph, la = ins.params
                inv.append("u", ins.qubits, (-th, -la, -ph))
            elif n == "sx":
                inv.unitary(gate_matrix("sx").conj().T, ins.qubits, label="sxdg")
            elif n == "unitary":
                inv.unitary(ins.matrix.conj().T, ins.qubits, label=(ins.label or "U") + "†")
            else:
                raise ValueError(f"cannot invert instruction '{n}'")
        return inv

    def count_ops(self) -> dict:
        out: dict[str, int] = {}
        for ins in self.data:
            out[ins.name] = out.get(ins.name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def size(self) -> int:
        return sum(1 for i in self.data if i.name != "barrier")

    def depth(self) -> int:
        level = [0] * self.num_qubits
        for ins in self.data:
            if ins.name == "barrier":
                continue
            d = max(level[q] for q in ins.qubits) + 1
            for q in ins.qubits:
                level[q] = d
        return max(level) if level else 0

    def has_mid_circuit_measurement(self) -> bool:
        """True if any qubit is acted on after it was measured, or is reset."""
        measured = set()
        for ins in self.data:
            if ins.name == "reset":
                return True
            if ins.name == "measure":
                measured.add(ins.qubits[0])
            elif ins.name not in ("barrier",) and not ins.name.startswith("save_"):
                if measured & set(ins.qubits):
                    return True
        return False

    def to_matrix(self) -> np.ndarray:
        """Full unitary of the circuit (small circuits only)."""
        from .quantum_info import Operator
        return Operator(self).data

    def draw(self) -> str:
        from .visualization import text_drawer
        return text_drawer(self)

    def __str__(self):
        return self.draw()

    def __repr__(self):
        return (f"QuantumCircuit(name='{self.name}', num_qubits={self.num_qubits}, "
                f"num_clbits={self.num_clbits}, size={self.size()})")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


__all__ = ["QuantumCircuit", "Instruction", "Parameter", "ParameterVector",
           "ParameterExpression"]
