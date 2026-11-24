"""OpenQASM 2.0 import/export — the bridge to circuits written in other SDKs.

Supports one quantum and one classical register, the gates of qelib1.inc that
QazSim knows, measure, reset and barrier.
"""

from __future__ import annotations

import ast
import math
import operator
import re

from .circuit import QuantumCircuit

_TO_QASM = {"p": "u1", "u": "u3", "id": "id"}
_FROM_QASM = {"u1": "p", "u3": "u", "u": "u", "cu1": "cp", "CX": "cx"}


def dumps(circuit: QuantumCircuit) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.num_qubits}];"]
    if circuit.num_clbits:
        lines.append(f"creg c[{circuit.num_clbits}];")
    for ins in circuit.data:
        if ins.name.startswith("save_"):
            continue
        if ins.is_parameterized:
            raise ValueError("bind parameters before exporting to QASM")
        qs = ",".join(f"q[{q}]" for q in ins.qubits)
        if ins.name == "measure":
            lines.append(f"measure q[{ins.qubits[0]}] -> c[{ins.clbits[0]}];")
        elif ins.name in ("barrier", "reset"):
            lines.append(f"{ins.name} {qs};")
        elif ins.name == "unitary":
            raise ValueError("'unitary' has no OpenQASM 2 form; transpile first")
        elif ins.name == "cp":
            lines.append(f"cu1({float(ins.params[0])!r}) {qs};")
        else:
            name = _TO_QASM.get(ins.name, ins.name)
            par = "(" + ",".join(repr(float(p)) for p in ins.params) + ")" if ins.params else ""
            lines.append(f"{name}{par} {qs};")
    return "\n".join(lines) + "\n"


# ---- safe evaluation of parameter expressions such as "pi/2" or "-3*pi/4"
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}
_FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
          "ln": math.log, "sqrt": math.sqrt}


def _eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression: {expr}")
    return ev(ast.parse(expr, mode="eval"))


_STMT = re.compile(r"^(\w+)\s*(?:\((.*?)\))?\s+(.*)$")
_ARG = re.compile(r"(\w+)\[(\d+)\]")


def loads(text: str) -> QuantumCircuit:
    text = re.sub(r"//.*", "", text)
    stmts = [s.strip() for s in text.split(";") if s.strip()]
    qreg = creg = None
    nq = nc = 0
    body = []
    for s in stmts:
        if s.startswith("OPENQASM") or s.startswith("include"):
            continue
        m = re.match(r"^qreg\s+(\w+)\[(\d+)\]$", s)
        if m:
            if qreg is not None:
                raise ValueError("only one qreg is supported")
            qreg, nq = m.group(1), int(m.group(2))
            continue
        m = re.match(r"^creg\s+(\w+)\[(\d+)\]$", s)
        if m:
            if creg is not None:
                raise ValueError("only one creg is supported")
            creg, nc = m.group(1), int(m.group(2))
            continue
        body.append(s)
    if qreg is None:
        raise ValueError("no qreg found")
    qc = QuantumCircuit(nq, nc)
    for s in body:
        if s.startswith("measure"):
            m = re.match(r"^measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]$", s)
            if not m:
                raise ValueError(f"cannot parse: {s}")
            qc.measure(int(m.group(2)), int(m.group(4)))
            continue
        m = _STMT.match(s)
        if not m:
            raise ValueError(f"cannot parse: {s}")
        name, par, args = m.group(1), m.group(2), m.group(3)
        qubits = [int(i) for reg, i in _ARG.findall(args)]
        if name == "barrier":
            qc.barrier(*qubits)
            continue
        if name == "reset":
            qc.reset(qubits)
            continue
        params = tuple(_eval(p.strip()) for p in par.split(",")) if par else ()
        name = _FROM_QASM.get(name, name)
        if name == "u2":
            name, params = "u", (math.pi / 2,) + params
        qc.append(name, qubits, params)
    return qc
