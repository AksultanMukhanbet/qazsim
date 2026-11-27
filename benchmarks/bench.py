"""QazSim benchmarks: compiled programs (0.2) vs the 0.1 API vs Qiskit Aer.

    python benchmarks/bench.py                 # everything that is installed
    python benchmarks/bench.py --only state    # state-vector circuits only
    python benchmarks/bench.py --api01         # only the 0.1 API (for an old install)

Every timing is the best of several runs, wall clock. Qiskit Aer is optional and
used only as an external reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import qazsim

HAS_V2 = hasattr(qazsim, "compile")
try:
    import qiskit
    import qiskit_aer
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2

    HAS_AER = True
except ImportError:  # pragma: no cover
    HAS_AER = False


def best(fn, repeat=3):
    t = []
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return min(t)


# ------------------------------------------------------------------ circuits
# A circuit is a list of (gate, qubits, params); builders below turn it into
# each library's own object.

def layers_circuit(n, layers=5, seed=0):
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(layers):
        ops += [("ry", (q,), (float(rng.uniform(0, 2 * np.pi)),)) for q in range(n)]
        ops += [("cx", (q, q + 1), ()) for q in range(n - 1)]
    return ops


def brickwork_circuit(n, depth=20, seed=0):
    rng = np.random.default_rng(seed)
    ops = []
    for d in range(depth):
        ops += [("u", (q,), tuple(float(x) for x in rng.uniform(0, 2 * np.pi, 3))) for q in range(n)]
        ops += [("cx", (q, q + 1), ()) for q in range(d % 2, n - 1, 2)]
    return ops


def qft_circuit(n):
    ops = [("x", (q,), ()) for q in range(0, n, 3)]
    for j in reversed(range(n)):
        ops.append(("h", (j,), ()))
        for k in reversed(range(j)):
            ops.append(("cp", (k, j), (math.pi / 2 ** (j - k),)))
    for q in range(n // 2):
        ops.append(("swap", (q, n - 1 - q), ()))
    return ops


def to_qazsim(n, ops):
    c = qazsim.Circuit(n)
    for g, qs, ps in ops:
        c.append(g, qs, ps)
    return c


def to_qazsim01(n, ops):
    qc = qazsim.QuantumCircuit(n)
    for g, qs, ps in ops:
        getattr(qc, g)(*ps, *qs)
    return qc


def to_qiskit(n, ops):
    qc = qiskit.QuantumCircuit(n)
    for g, qs, ps in ops:
        getattr(qc, g)(*ps, *qs)
    return qc


# ------------------------------------------------------------- variational
def ansatz_ops(n, layers):
    """Hardware-efficient ansatz: (ry, rz on every qubit, cx chain) x layers."""
    ops, k = [], 0
    for _ in range(layers):
        for q in range(n):
            ops.append(("ry", (q,), k))
            ops.append(("rz", (q,), k + 1))
            k += 2
        ops += [("cx", (q, q + 1), None) for q in range(n - 1)]
    return ops, k


def ising(n):
    """Transverse-field Ising chain: Σ Z_i Z_{i+1} + 0.7 Σ X_i."""
    terms = {}
    for i in range(n - 1):
        lbl = ["I"] * n
        lbl[n - 1 - i] = lbl[n - 2 - i] = "Z"
        terms["".join(lbl)] = 1.0
    for i in range(n):
        lbl = ["I"] * n
        lbl[n - 1 - i] = "X"
        terms["".join(lbl)] = 0.7
    return terms


def ansatz_qazsim(n, layers):
    ops, P = ansatz_ops(n, layers)
    th = qazsim.params("θ", P)
    c = qazsim.Circuit(n)
    for g, qs, k in ops:
        c.append(g, qs, () if k is None else (th[k],))
    return c


def ansatz_qiskit(n, layers):
    from qiskit.circuit import ParameterVector
    ops, P = ansatz_ops(n, layers)
    th = ParameterVector("θ", P)
    qc = qiskit.QuantumCircuit(n)
    for g, qs, k in ops:
        getattr(qc, g)(*(() if k is None else (th[k],)), *qs)
    return qc


# ------------------------------------------------------------------ suites
def suite_state(sizes, results):
    for name, build in [("layers", layers_circuit), ("brickwork", brickwork_circuit), ("qft", qft_circuit)]:
        for n in sizes:
            ops = build(n)
            row = {"suite": "state", "circuit": name, "qubits": n, "gates": len(ops)}
            if HAS_V2:
                c = to_qazsim(n, ops)
                s = time.perf_counter()
                prog = qazsim.compile(c)
                row["qazsim02_compile"] = time.perf_counter() - s
                row["qazsim02"] = best(prog.statevector)
                row["blocks"] = prog.stats["blocks"]
            if HAS_AER:
                qc = to_qiskit(n, ops)
                qc.save_statevector()
                # Aer's default (fusion on) and fusion off; on few cores the latter can win
                sim = AerSimulator(method="statevector")
                row["aer"] = best(lambda: sim.run(qc).result())
                sim_nf = AerSimulator(method="statevector", fusion_enable=False)
                row["aer_no_fusion"] = best(lambda: sim_nf.run(qc).result())
            qc01 = to_qazsim01(n, ops)
            qc01.save_statevector()
            row["qazsim01_api"] = best(lambda: qazsim.QazSim().run(qc01).result(), repeat=2)
            print(json.dumps(row), flush=True)
            results.append(row)


def suite_variational(n, layers, batch, results):
    H = ising(n)
    rng = np.random.default_rng(1)
    _, P = ansatz_ops(n, layers)
    X = rng.uniform(0, 2 * np.pi, size=(batch, P))
    row = {"suite": "expectation_batch", "qubits": n, "params": P, "batch": batch, "terms": len(H)}
    if HAS_V2:
        prog = qazsim.compile(ansatz_qazsim(n, layers))
        row["qazsim02"] = best(lambda: prog.expectation(H, X))
        ref = prog.expectation(H, X[:3])
    if HAS_AER:
        qc = ansatz_qiskit(n, layers)
        op = SparsePauliOp.from_list(list(H.items()))
        est = EstimatorV2()
        row["aer_estimator"] = best(lambda: est.run([(qc, op, X)]).result())
        aer_vals = est.run([(qc, op, X[:3])]).result()[0].data.evs
        if HAS_V2:
            row["max_abs_diff_vs_aer"] = float(np.max(np.abs(aer_vals - ref)))
    print(json.dumps(row), flush=True)
    results.append(row)

    # gradient of one parameter set
    x = X[0]
    row = {"suite": "gradient", "qubits": n, "params": P, "terms": len(H)}
    if HAS_V2:
        row["qazsim02_adjoint"] = best(lambda: prog.gradient(H, x))
        g_ad = prog.gradient(H, x)
    if HAS_AER:
        shifts = np.repeat(x[None, :], 2 * P, axis=0)
        for i in range(P):
            shifts[2 * i, i] += np.pi / 2
            shifts[2 * i + 1, i] -= np.pi / 2

        def aer_param_shift():
            ev = est.run([(qc, op, shifts)]).result()[0].data.evs
            return 0.5 * (ev[0::2] - ev[1::2])

        row["aer_parameter_shift"] = best(aer_param_shift)
        if HAS_V2:
            row["max_abs_diff_vs_aer"] = float(np.max(np.abs(aer_param_shift() - g_ad)))
    print(json.dumps(row), flush=True)
    results.append(row)


def suite_api01_only(sizes, n_var, layers, batch, results):
    """For an old install that has only the 0.1 API."""
    for name, build in [("layers", layers_circuit), ("brickwork", brickwork_circuit), ("qft", qft_circuit)]:
        for n in sizes:
            ops = build(n)
            qc01 = to_qazsim01(n, ops)
            qc01.save_statevector()
            t = best(lambda: qazsim.QazSim().run(qc01).result(), repeat=2)
            row = {"suite": "state", "circuit": name, "qubits": n, "qazsim014": t}
            print(json.dumps(row), flush=True)
            results.append(row)
    H = ising(n_var)
    ops, P = ansatz_ops(n_var, layers)
    th = qazsim.ParameterVector("θ", P)
    qc = qazsim.QuantumCircuit(n_var)
    for g, qs, k in ops:
        getattr(qc, g)(*(() if k is None else (th[k],)), *qs)
    X = np.random.default_rng(1).uniform(0, 2 * np.pi, size=(batch, P))

    def run():
        return [qazsim.Statevector.from_instruction(qc.assign_parameters(list(x))).expectation_value(H)
                for x in X]

    row = {"suite": "expectation_batch", "qubits": n_var, "params": P, "batch": batch,
           "qazsim014": best(run, repeat=1)}
    print(json.dumps(row), flush=True)
    results.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,20,22,24")
    ap.add_argument("--only", choices=["state", "variational"])
    ap.add_argument("--api01", action="store_true")
    ap.add_argument("--vqubits", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    results = []
    meta = {"qazsim": qazsim.__version__, "threads": qazsim._accelerate.num_threads()}
    if HAS_AER:
        meta["qiskit_aer"] = qiskit_aer.__version__
    print(json.dumps(meta))
    if a.api01:
        suite_api01_only(sizes, a.vqubits, a.layers, a.batch, results)
    else:
        if a.only in (None, "state"):
            suite_state(sizes, results)
        if a.only in (None, "variational"):
            suite_variational(a.vqubits, a.layers, a.batch, results)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
