"""Tests of the 0.2 compiled-program API against an independent numpy reference."""

import cmath
import math

import numpy as np
import pytest

import qazsim as qz
from qazsim import Parameter, QuantumCircuit

# ------------------------------------------------------------ reference model
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
P0, P1 = np.diag([1.0 + 0j, 0]), np.diag([0j, 1.0])


def rot(G, t):
    return math.cos(t / 2) * np.eye(G.shape[0]) - 1j * math.sin(t / 2) * G


def ctrl(U):  # control = bit 0, target = bit 1  ->  kron(target, control)
    return np.kron(I2, P0) + np.kron(U, P1)


def u3(t, p, l):
    return np.array([[math.cos(t / 2), -cmath.exp(1j * l) * math.sin(t / 2)],
                     [cmath.exp(1j * p) * math.sin(t / 2), cmath.exp(1j * (p + l)) * math.cos(t / 2)]])


REF = {
    "id": lambda: I2, "x": lambda: X, "y": lambda: Y, "z": lambda: Z,
    "h": lambda: np.array([[1, 1], [1, -1]]) / math.sqrt(2),
    "s": lambda: np.diag([1, 1j]), "sdg": lambda: np.diag([1, -1j]),
    "t": lambda: np.diag([1, cmath.exp(1j * math.pi / 4)]),
    "tdg": lambda: np.diag([1, cmath.exp(-1j * math.pi / 4)]),
    "sx": lambda: 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
    "sxdg": lambda: 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]),
    "rx": lambda t: rot(X, t), "ry": lambda t: rot(Y, t), "rz": lambda t: rot(Z, t),
    "p": lambda l: np.diag([1, cmath.exp(1j * l)]), "u": u3,
    "cx": lambda: ctrl(X), "cy": lambda: ctrl(Y), "cz": lambda: ctrl(Z),
    "swap": lambda: np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    "iswap": lambda: np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "cp": lambda l: ctrl(np.diag([1, cmath.exp(1j * l)])),
    "crx": lambda t: ctrl(rot(X, t)), "cry": lambda t: ctrl(rot(Y, t)), "crz": lambda t: ctrl(rot(Z, t)),
    "rxx": lambda t: rot(np.kron(X, X), t), "ryy": lambda t: rot(np.kron(Y, Y), t),
    "rzz": lambda t: rot(np.kron(Z, Z), t),
    # 3 qubits: kron(bit2, bit1, bit0)
    "ccx": lambda: np.kron(np.kron(I2, I2), I2) - np.kron(np.kron(I2, P1), P1)
    + np.kron(np.kron(X, P1), P1),
    "ccz": lambda: np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex),
    "cswap": lambda: np.kron(np.eye(4), P0) + np.kron(np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                                                 [0, 1, 0, 0], [0, 0, 0, 1]]), P1),
}
GATES = {k: v for k, v in qz.program._GATES.items()}


def ref_apply(psi, M, qubits):
    n = int(math.log2(len(psi)))
    out = np.zeros_like(psi)
    mask = sum(1 << q for q in qubits)
    for i in range(len(psi)):
        if psi[i] == 0:
            continue
        local = sum(((i >> q) & 1) << b for b, q in enumerate(qubits))
        base = i & ~mask
        for r in range(1 << len(qubits)):
            j = base | sum(((r >> b) & 1) << q for b, q in enumerate(qubits))
            out[j] += M[r, local] * psi[i]
    assert n >= len(qubits)
    return out


def ref_apply_tensor(psi, M, qubits):
    """Same as ref_apply, via numpy tensordot (fast enough for ~20 qubits)."""
    n = int(math.log2(len(psi)))
    k = len(qubits)
    G = np.asarray(M, dtype=complex).reshape((2,) * (2 * k))  # axes: out bits k-1..0, in bits k-1..0
    t = psi.reshape((2,) * n)  # axis a <-> qubit n-1-a
    in_axes = [k + (k - 1 - b) for b in range(k)]
    res = np.tensordot(G, t, axes=(in_axes, [n - 1 - q for q in qubits]))
    res = np.moveaxis(res, list(range(k)), [n - 1 - qubits[k - 1 - j] for j in range(k)])
    return res.reshape(-1)


def ref_state(n, ops, values):
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1
    for name, qubits, ps in ops:
        vals = [p.bind(values) if isinstance(p, qz.Param) else p for p in ps]
        M = REF[name](*vals) if name != "unitary" else ps
        psi = ref_apply(psi, np.asarray(M, dtype=complex), qubits)
    return psi


def pauli_matrix(label):
    m = np.array([[1.0 + 0j]])
    for ch in label:  # leftmost = highest qubit
        m = np.kron(m, {"I": I2, "X": X, "Y": Y, "Z": Z}[ch])
    return m


def random_circuit(n, depth, rng, param_names=("a", "b", "c")):
    c = qz.Circuit(n)
    ops = []
    names = [g for g in GATES if GATES[g][0] <= n]
    for _ in range(depth):
        g = names[rng.integers(len(names))]
        nq, npar = GATES[g]
        qs = tuple(int(q) for q in rng.choice(n, nq, replace=False))
        ps = []
        for _ in range(npar):
            r = rng.random()
            if r < 0.4:
                ps.append(float(rng.normal()))
            elif r < 0.7:
                ps.append(qz.Param(param_names[rng.integers(len(param_names))]))
            else:
                a, b = rng.choice(param_names, 2, replace=False)
                ps.append(float(rng.normal()) * qz.Param(a) - 0.5 * qz.Param(b) + float(rng.normal()))
        c.append(g, qs, ps)
        ops.append((g, qs, [qz.Param.coerce(p) for p in ps]))
    return c, ops


# ----------------------------------------------------------------- correctness
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("max_fused", [1, 2, 3, 4, 5])
def test_random_circuits_match_reference(n, max_fused):
    rng = np.random.default_rng(100 * n + max_fused)
    for _ in range(4):
        c, ops = random_circuit(n, 40, rng)
        prog = qz.compile(c, max_fused=max_fused, max_fused_diag=max(max_fused, 6))
        vals = {k: float(v) for k, v in zip(prog.parameters, rng.normal(size=prog.num_parameters))}
        np.testing.assert_allclose(prog.statevector(vals), ref_state(n, ops, vals), atol=1e-10)


def test_tensor_reference_agrees_with_loop_reference():
    rng = np.random.default_rng(9)
    psi = rng.normal(size=32) + 1j * rng.normal(size=32)
    M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    np.testing.assert_allclose(ref_apply_tensor(psi, M, (4, 0, 2)), ref_apply(psi, M, (4, 0, 2)))


def test_every_gate_alone():
    rng = np.random.default_rng(7)
    for g, (nq, npar) in GATES.items():
        n = nq + 1
        prep = qz.Circuit(n)
        for q in range(n):
            prep.u(*rng.normal(size=3), q)
        ps = [float(x) for x in rng.normal(size=npar)]
        qs = tuple(range(n))[::-1][:nq]
        c = prep.copy().append(g, qs, ps)
        ops = [("u", (q,), list(prep.ops[q][2])) for q in range(n)] + [(g, qs, ps)]
        np.testing.assert_allclose(qz.compile(c).statevector(), ref_state(n, ops, {}), atol=1e-12,
                                   err_msg=g)


def test_unitary_gate_and_compose():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    U, _ = np.linalg.qr(A)
    c = qz.Circuit(4).h(0).unitary(U, [3, 0, 2]).cx(0, 1)
    other = qz.Circuit(2).ry("t", 0).cz(0, 1)
    c.compose(other, qubits=[2, 1])
    ops = [("h", (0,), []), ("unitary", (3, 0, 2), U), ("cx", (0, 1), []),
           ("ry", (2,), [qz.Param("t")]), ("cz", (2, 1), [])]
    np.testing.assert_allclose(qz.compile(c).statevector({"t": 0.3}), ref_state(4, ops, {"t": 0.3}),
                               atol=1e-12)


@pytest.mark.parametrize("n", [15, 17])
def test_large_states_parallel_kernels(n):
    """High qubits, every block size and the diagonal kernel, on states above the
    parallel threshold: fused results must equal unfused ones and the 0.1 path."""
    rng = np.random.default_rng(n)
    c = qz.Circuit(n)
    for layer in range(3):
        for q in range(n):
            c.u(*rng.normal(size=3), q)
        for q in range(layer % 2, n - 1, 2):
            c.cx(q, q + 1)
        c.rzz(0.3, 0, n - 1).cp(0.7, n - 1, 2).ccx(n - 1, 0, n // 2).cswap(1, n - 1, n - 2)
        for q in range(n):
            c.rz(rng.normal(), q)
    ref = np.zeros(1 << n, dtype=complex)
    ref[0] = 1
    for name, qubits, ps, _ in c.ops:
        ref = ref_apply_tensor(ref, REF[name](*ps), qubits)
    np.testing.assert_allclose(qz.compile(c, max_fused=1, max_fused_diag=1).statevector(), ref, atol=1e-10)
    for k in (2, 3, 4, 5):
        np.testing.assert_allclose(qz.compile(c, max_fused=k).statevector(), ref, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(ref), 1.0, atol=1e-12)


def test_fusion_reduces_blocks_and_folds_static_gates():
    c = qz.Circuit(3)
    for _ in range(10):
        c.h(0).h(0).t(1).tdg(1).cx(0, 2).cx(0, 2)  # identities overall
    prog = qz.compile(c)
    assert prog.stats["ops"] == 60
    assert prog.stats["blocks"] == 0  # everything multiplied out and dropped as identity
    np.testing.assert_allclose(prog.statevector(), np.eye(8)[0], atol=1e-12)


# ------------------------------------------------------------ batches & values
def test_batch_equals_rows_and_parameter_formats():
    rng = np.random.default_rng(11)
    c, _ = random_circuit(4, 30, rng)
    prog = qz.compile(c)
    batch = rng.normal(size=(6, prog.num_parameters))
    probs = prog.probabilities(batch)
    assert probs.shape == (6, 16)
    for row, pr in zip(batch, probs):
        np.testing.assert_allclose(pr, np.abs(prog.statevector(row)) ** 2, atol=1e-12)
    as_dicts = [dict(zip(prog.parameters, r)) for r in batch]
    np.testing.assert_allclose(prog.probabilities(as_dicts), probs, atol=1e-14)
    np.testing.assert_allclose(prog.statevector(batch[:2])[1], prog.statevector(batch[1]), atol=1e-14)


def test_expectation_matches_dense_observable():
    rng = np.random.default_rng(5)
    n = 4
    c, ops = random_circuit(n, 35, rng)
    prog = qz.compile(c)
    obs = {"ZZII": 0.7, "XYZI": -1.3, "IIIY": 0.25, "YYXX": 0.5, "IIII": 2.0}
    H = sum(cf * pauli_matrix(lbl) for lbl, cf in obs.items())
    batch = rng.normal(size=(5, prog.num_parameters))
    got = prog.expectation(obs, batch)
    for row, val in zip(batch, got):
        psi = ref_state(n, ops, dict(zip(prog.parameters, row)))
        assert val == pytest.approx(np.vdot(psi, H @ psi).real, abs=1e-10)


def test_sample_counts():
    prog = qz.compile(qz.Circuit(2).h(0).cx(0, 1))
    counts = prog.sample(4000, seed=1)
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 4000
    assert abs(counts["00"] - 2000) < 200


# ------------------------------------------------------------------- gradients
def finite_diff(prog, obs, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (prog.expectation(obs, x + e) - prog.expectation(obs, x - e)) / (2 * h)
    return g


@pytest.mark.parametrize("seed", range(6))
def test_adjoint_gradient_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 4
    c, _ = random_circuit(n, 45, rng, param_names=("a", "b", "c", "d"))
    prog = qz.compile(c)
    if prog.num_parameters == 0:
        pytest.skip("no parameters drawn")
    labels = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(3)]
    obs = {lbl: float(rng.normal()) for lbl in labels}
    x = rng.normal(size=prog.num_parameters)
    val, grad = prog.value_and_gradient(obs, x)
    assert val == pytest.approx(prog.expectation(obs, x), abs=1e-12)
    np.testing.assert_allclose(grad, finite_diff(prog, obs, x), atol=1e-6)


def test_gradient_batch_and_shared_parameters():
    th = qz.params("θ", 3)
    c = qz.Circuit(3)
    for q in range(3):
        c.ry(th[q], q)
    c.cx(0, 1).cx(1, 2).rzz(2 * th[0] - th[2] + 0.1, 0, 2).u(th[1], th[2], th[0], 1).crx(th[1], 2, 0)
    prog = qz.compile(c)
    xs = np.random.default_rng(0).normal(size=(4, 3))
    v, g = prog.value_and_gradient({"ZZZ": 1.0, "XIX": 0.3}, xs)
    assert v.shape == (4,) and g.shape == (4, 3)
    for x, gi in zip(xs, g):
        np.testing.assert_allclose(gi, finite_diff(prog, {"ZZZ": 1.0, "XIX": 0.3}, x), atol=1e-6)


def test_vqe_h2_converges_with_gradients():
    H = {"II": -1.052373245772859, "IZ": 0.39793742484318045, "ZI": -0.39793742484318045,
         "ZZ": -0.01128010425623538, "XX": 0.18093119978423156}
    prog = qz.compile(qz.Circuit(2).x(0).ry("θ", 1).cx(1, 0))
    x = np.array([0.0])
    for _ in range(200):
        x -= 0.5 * prog.gradient(H, x)
    exact = np.linalg.eigvalsh(sum(c * pauli_matrix(l) for l, c in H.items()))[0]
    assert prog.expectation(H, x) == pytest.approx(exact, abs=1e-6)


# ------------------------------------------------------------- conversion, errors
def test_from_quantum_circuit():
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.ry(2 * a + 0.5, 1)
    qc.crz(a - b, 0, 2)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    prog = qz.compile(qc)
    assert set(prog.parameters) == {"a", "b"}
    vals = {"a": 0.3, "b": -1.1}
    qc_bound = QuantumCircuit(3)
    qc_bound.h(0)
    qc_bound.ry(2 * 0.3 + 0.5, 1)
    qc_bound.crz(0.3 + 1.1, 0, 2)
    qc_bound.cx(1, 2)
    expected = qz.Statevector.from_instruction(qc_bound).data
    np.testing.assert_allclose(prog.statevector(vals), expected, atol=1e-12)

    bad = QuantumCircuit(1)
    bad.rx(a * b, 0)
    with pytest.raises(ValueError, match="not affine"):
        qz.compile(bad)


def test_errors():
    th = qz.Param("θ")
    with pytest.raises(TypeError, match="affine"):
        th * th
    with pytest.raises(ValueError):
        qz.Circuit(2).cx(0, 0)
    with pytest.raises(IndexError):
        qz.Circuit(2).h(2)
    with pytest.raises(TypeError):
        qz.Circuit(2).rx(0.5)  # missing the qubit
    prog = qz.compile(qz.Circuit(2).rx(th, 0))
    with pytest.raises(ValueError, match="parameter"):
        prog.statevector()
    with pytest.raises(ValueError, match="missing"):
        prog.statevector({"φ": 1.0})
    with pytest.raises(ValueError):
        prog.expectation("ZZZ", [0.1])
    with pytest.raises(ValueError, match="real"):
        prog.expectation({"ZZ": 1j}, [0.1])
    with pytest.raises(ValueError):
        qz.compile(qz.Circuit(2), max_fused=6)
