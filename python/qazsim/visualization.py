"""Text drawing of circuits (one column per instruction)."""

from __future__ import annotations

from .circuit import ParameterExpression

_SHORT = {"measure": "M", "reset": "|0>", "unitary": "U"}
_CONTROLLED = {"cx": "X", "cy": "Y", "cz": "■", "cp": "P", "crz": "Rz", "ccx": "X"}


def _fmt(v) -> str:
    if isinstance(v, ParameterExpression):
        return getattr(v, "name", "expr")
    return f"{float(v):.3g}"


def _label(ins) -> str:
    if ins.name in _CONTROLLED:
        name = _CONTROLLED[ins.name]
    else:
        name = ins.label or _SHORT.get(ins.name, ins.name.upper())
    if ins.params:
        name += "(" + ",".join(_fmt(p) for p in ins.params) + ")"
    return name


def text_drawer(circuit) -> str:
    n = circuit.num_qubits
    names = [f"q_{i}: " for i in range(n)]
    w0 = max(len(s) for s in names)
    rows = [s.rjust(w0) for s in names]
    for ins in circuit.data:
        if ins.name.startswith("save_"):
            continue
        cells = {}
        if ins.name == "barrier":
            for q in ins.qubits:
                cells[q] = "░"
        elif ins.name in _CONTROLLED:
            *ctrl, tgt = ins.qubits
            for c in ctrl:
                cells[c] = "■"
            cells[tgt] = _label(ins) if ins.name != "cz" else "■"
        elif ins.name == "swap":
            for q in ins.qubits:
                cells[q] = "x"
        elif ins.name == "measure":
            cells[ins.qubits[0]] = f"M{ins.clbits[0]}"
        else:
            lab = _label(ins)
            for q in ins.qubits:
                cells[q] = lab
        width = max(len(c) for c in cells.values()) + 2
        lo, hi = min(cells), max(cells)
        for q in range(n):
            if q in cells:
                rows[q] += "─" + cells[q].center(width - 2, "─") + "─"
            elif lo < q < hi and ins.name != "barrier":
                rows[q] += "─" + "┼".center(width - 2, "─") + "─"
            else:
                rows[q] += "─" * width
    return "\n".join(r + "─" for r in rows)
