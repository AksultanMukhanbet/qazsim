//! Compiled programs: the 0.2 execution model.
//!
//! A circuit is compiled once into a `Program`:
//!   1. ops are grouped into fused blocks of up to `max_fused` qubits
//!      (`max_fused_diag` for blocks made only of diagonal gates);
//!   2. blocks without free parameters are multiplied out at compile time,
//!      identity blocks are dropped and numerically diagonal ones use the
//!      diagonal kernel;
//!   3. blocks with parameters keep their structure and are multiplied out
//!      when parameter values arrive, which costs O(gates · 4^k), not O(2^n).
//!
//! A second plan, where every parametrised gate stands alone, is used for
//! adjoint differentiation. Batches of parameter sets run in parallel across
//! the batch for small states and across amplitudes for large ones.

use std::collections::HashMap;

use num_complex::Complex64 as C;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::gates::Gate;
use crate::kernels::{self, PauliTerm, PARALLEL_THRESHOLD};

const EPS: f64 = 1e-14;

/// offset + Σ coeff · θ[index]
#[derive(Clone, Debug)]
pub struct ParamExpr {
    pub offset: f64,
    pub terms: Vec<(usize, f64)>,
}

impl ParamExpr {
    #[inline]
    fn eval(&self, p: &[f64]) -> f64 {
        self.terms.iter().fold(self.offset, |a, &(i, c)| a + c * p[i])
    }
}

#[derive(Clone, Debug)]
pub struct Op {
    pub gate: Gate,
    pub qubits: Vec<usize>,
    pub params: Vec<ParamExpr>,
    pub matrix: Option<Vec<C>>,
    diagonal: bool,
}

impl Op {
    /// Qubits on which the gate is "identity unless 1" (for the cost model).
    fn controls(&self) -> Vec<usize> {
        use Gate::*;
        match self.gate {
            Cx | Cy | Crx | Cry | Crz | Cswap => vec![self.qubits[0]],
            Ccx => vec![self.qubits[0], self.qubits[1]],
            Cz | Cp | Ccz => self.qubits.clone(),
            _ => Vec::new(),
        }
    }

    fn parametric(&self) -> bool {
        self.params.iter().any(|e| !e.terms.is_empty())
    }
    fn values(&self, p: &[f64]) -> Vec<f64> {
        self.params.iter().map(|e| e.eval(p)).collect()
    }
    fn matrix(&self, p: &[f64]) -> Vec<C> {
        match &self.matrix {
            Some(m) => m.clone(),
            None => self.gate.matrix(&self.values(p)),
        }
    }
}

fn is_diagonal(m: &[C], dim: usize) -> bool {
    (0..dim).all(|r| (0..dim).all(|c| r == c || m[r * dim + c].norm_sqr() < EPS * EPS))
}

fn dagger(m: &[C], dim: usize) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            out[c * dim + r] = m[r * dim + c].conj();
        }
    }
    out
}

/// A block ready to run. `controls` are qubits on which the block acts as
/// "identity unless this bit is 1": the kernel then touches only the
/// controlled subspace (half the state per control).
#[derive(Clone, Debug)]
enum Kernel {
    Dense { qubits: Vec<usize>, controls: Vec<usize>, mat: Vec<C> },
    Diag { qubits: Vec<usize>, controls: Vec<usize>, d: Vec<C> },
    Skip,
}

#[inline]
fn drop_bit(x: usize, b: usize) -> usize {
    ((x >> (b + 1)) << b) | (x & ((1usize << b) - 1))
}

#[inline]
fn close(a: C, b: C) -> bool {
    (a - b).norm_sqr() < EPS * EPS
}

impl Kernel {
    fn dense(qubits: Vec<usize>, mat: Vec<C>) -> Kernel {
        Kernel::Dense { qubits, controls: Vec::new(), mat }
    }

    #[inline]
    fn apply(&self, state: &mut [C]) {
        match self {
            Kernel::Dense { qubits, controls, mat } => kernels::apply_dense_ctrl(state, mat, qubits, controls),
            Kernel::Diag { qubits, controls, d } => kernels::apply_diag_ctrl(state, d, qubits, controls),
            Kernel::Skip => {}
        }
    }

    fn adjoint(&self) -> Kernel {
        match self {
            Kernel::Dense { qubits, controls, mat } => Kernel::Dense {
                qubits: qubits.clone(),
                controls: controls.clone(),
                mat: dagger(mat, 1 << qubits.len()),
            },
            Kernel::Diag { qubits, controls, d } => Kernel::Diag {
                qubits: qubits.clone(),
                controls: controls.clone(),
                d: d.iter().map(|x| x.conj()).collect(),
            },
            Kernel::Skip => Kernel::Skip,
        }
    }

    /// Exact rewrites: dense -> diagonal, pull out control qubits, drop identities.
    fn simplify(self) -> Kernel {
        let one = C::new(1.0, 0.0);
        let mut k = self;
        // dense -> diagonal
        if let Kernel::Dense { qubits, controls, mat } = k {
            let dim = 1 << qubits.len();
            k = if is_diagonal(&mat, dim) {
                Kernel::Diag { d: (0..dim).map(|i| mat[i * dim + i]).collect(), qubits, controls }
            } else {
                Kernel::Dense { qubits, controls, mat }
            };
        }
        // controls
        k = match k {
            Kernel::Dense { mut qubits, mut controls, mut mat } => {
                let mut b = 0;
                while b < qubits.len() && qubits.len() > 1 {
                    let dim = 1usize << qubits.len();
                    let is_ctrl = (0..dim).all(|r| {
                        (0..dim).all(|c| {
                            ((r >> b) & 1 == 1 && (c >> b) & 1 == 1)
                                || close(mat[r * dim + c], if r == c { one } else { C::new(0.0, 0.0) })
                        })
                    });
                    if is_ctrl {
                        let h = dim / 2;
                        let mut m2 = vec![C::new(0.0, 0.0); h * h];
                        for r in (0..dim).filter(|r| (r >> b) & 1 == 1) {
                            for c in (0..dim).filter(|c| (c >> b) & 1 == 1) {
                                m2[drop_bit(r, b) * h + drop_bit(c, b)] = mat[r * dim + c];
                            }
                        }
                        controls.push(qubits.remove(b));
                        mat = m2;
                    } else {
                        b += 1;
                    }
                }
                Kernel::Dense { qubits, controls, mat }
            }
            Kernel::Diag { mut qubits, mut controls, mut d } => {
                let mut b = 0;
                while b < qubits.len() {
                    let is_ctrl = d.iter().enumerate().all(|(i, &x)| (i >> b) & 1 == 1 || close(x, one));
                    if is_ctrl {
                        d = d.iter().enumerate().filter(|(i, _)| (i >> b) & 1 == 1).map(|(_, &x)| x).collect();
                        controls.push(qubits.remove(b));
                    } else {
                        b += 1;
                    }
                }
                if d.iter().all(|&x| close(x, one)) {
                    Kernel::Skip
                } else {
                    Kernel::Diag { qubits, controls, d }
                }
            }
            Kernel::Skip => Kernel::Skip,
        };
        k
    }
}

#[derive(Clone, Debug)]
struct Block {
    qubits: Vec<usize>,
    ops: Vec<usize>,
    diag: bool,
    parametric: bool,
    fixed: Option<Kernel>,
    fixed_adj: Option<Kernel>,
}

/// Relative cost of running one block on a large state, in units of one
/// memory-bound one-qubit pass (measured on AVX2 hardware). Controlled blocks
/// touch 2^-controls of the state, but strided access keeps a floor.
fn block_cost(k: usize, diag: bool, ctrls: usize) -> f64 {
    const DENSE: [f64; 6] = [1.0, 1.0, 1.8, 3.5, 7.3, 17.0];
    let k_eff = k.saturating_sub(ctrls);
    let base = if diag { 1.1 } else if k_eff <= 5 { DENSE[k_eff] } else { 17.0 * 2f64.powi(k_eff as i32 - 5) };
    (base * 0.5f64.powi(ctrls as i32)).max(0.35)
}

/// Greedy fusion guided by `block_cost`: a gate joins the blocks it touches
/// only when the fused block is cheaper than keeping them apart.
/// `barrier`: every parametrised op becomes its own block.
fn fuse(ops: &[Op], n: usize, max_k: usize, max_k_diag: usize, barrier: bool) -> Vec<Block> {
    struct Open {
        qubits: Vec<usize>,
        ops: Vec<usize>,
        diag: bool,
        parametric: bool,
        alive: bool,
        ctrls: Vec<usize>,
    }
    let cost = |b: &Open| block_cost(b.qubits.len(), b.diag, b.ctrls.len());
    let mut open: Vec<Open> = Vec::new();
    let mut owner: Vec<Option<usize>> = vec![None; n];
    let mut out: Vec<Block> = Vec::new();

    fn close(id: usize, open: &mut [Open], owner: &mut [Option<usize>], out: &mut Vec<Block>) {
        let b = &mut open[id];
        if !b.alive {
            return;
        }
        b.alive = false;
        for &q in &b.qubits {
            owner[q] = None;
        }
        out.push(Block {
            qubits: std::mem::take(&mut b.qubits),
            ops: std::mem::take(&mut b.ops),
            diag: b.diag,
            parametric: b.parametric,
            fixed: None,
            fixed_adj: None,
        });
    }

    for (gi, op) in ops.iter().enumerate() {
        let mut touched: Vec<usize> = op.qubits.iter().filter_map(|&q| owner[q]).collect();
        touched.sort_unstable();
        touched.dedup();
        let alone = barrier && op.parametric();
        if !alone {
            let mut union: Vec<usize> = Vec::new();
            for &t in &touched {
                union.extend_from_slice(&open[t].qubits);
            }
            for &q in &op.qubits {
                if !union.contains(&q) {
                    union.push(q);
                }
            }
            let diag = op.diagonal && touched.iter().all(|&t| open[t].diag);
            let limit = if diag { max_k_diag.max(max_k) } else { max_k };
            // A qubit stays a control of the fused block only if it is a control of
            // every part: a part that acts on other qubits is not identity on c = 0.
            let op_ctrls = op.controls();
            let ctrls: Vec<usize> = op_ctrls
                .iter()
                .copied()
                .filter(|c| touched.iter().all(|&t| open[t].ctrls.contains(c)))
                .collect();
            let merged = block_cost(union.len(), diag, ctrls.len());
            let apart: f64 = touched.iter().map(|&t| cost(&open[t])).sum::<f64>()
                + block_cost(op.qubits.len(), op.diagonal, op_ctrls.len());
            if union.len() <= limit && (touched.is_empty() || merged <= apart + 1e-9) {
                let mut list = Vec::new();
                let mut par = op.parametric();
                for &t in &touched {
                    list.extend_from_slice(&open[t].ops);
                    par |= open[t].parametric;
                    open[t].alive = false;
                }
                list.push(gi);
                let id = open.len();
                for &q in &union {
                    owner[q] = Some(id);
                }
                open.push(Open { qubits: union, ops: list, diag, parametric: par, alive: true, ctrls });
                continue;
            }
        }
        for &t in &touched {
            close(t, &mut open, &mut owner, &mut out);
        }
        let fits = op.qubits.len() <= if op.diagonal { max_k_diag.max(max_k) } else { max_k };
        if alone || !fits {
            out.push(Block {
                qubits: op.qubits.clone(),
                ops: vec![gi],
                diag: op.diagonal,
                parametric: op.parametric(),
                fixed: None,
                fixed_adj: None,
            });
        } else {
            let id = open.len();
            for &q in &op.qubits {
                owner[q] = Some(id);
            }
            open.push(Open {
                qubits: op.qubits.clone(),
                ops: vec![gi],
                diag: op.diagonal,
                parametric: op.parametric(),
                alive: true,
                ctrls: op.controls(),
            });
        }
    }
    for id in 0..open.len() {
        close(id, &mut open, &mut owner, &mut out);
    }
    out
}

/// Multiply out a block for parameters `p`.
fn build_kernel(block: &Block, ops: &[Op], p: &[f64]) -> Kernel {
    let k = block.qubits.len();
    let dim = 1usize << k;
    let local = |op: &Op| -> Vec<usize> {
        op.qubits.iter().map(|q| block.qubits.iter().position(|b| b == q).unwrap()).collect()
    };
    if block.diag {
        let mut d = vec![C::new(1.0, 0.0); dim];
        for &oi in &block.ops {
            let op = &ops[oi];
            let gdim = 1usize << op.qubits.len();
            let m = op.matrix(p);
            let gd: Vec<C> = (0..gdim).map(|j| m[j * gdim + j]).collect();
            let pos = local(op);
            for (i, v) in d.iter_mut().enumerate() {
                let mut l = 0usize;
                for (b, &q) in pos.iter().enumerate() {
                    l |= ((i >> q) & 1) << b;
                }
                *v *= gd[l];
            }
        }
        return Kernel::Diag { qubits: block.qubits.clone(), controls: Vec::new(), d };
    }
    if block.ops.len() == 1 && ops[block.ops[0]].qubits == block.qubits {
        return Kernel::dense(block.qubits.clone(), ops[block.ops[0]].matrix(p));
    }
    // Columns of the block matrix are evolved together as one 2k-qubit "state":
    // flat[r + c·dim] = M[r][c]; gates act on the low k bits.
    let mut flat = vec![C::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        flat[i + i * dim] = C::new(1.0, 0.0);
    }
    for &oi in &block.ops {
        let op = &ops[oi];
        kernels::apply_dense(&mut flat, &op.matrix(p), &local(op));
    }
    let mut mat = vec![C::new(0.0, 0.0); dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            mat[r * dim + c] = flat[r + c * dim];
        }
    }
    Kernel::dense(block.qubits.clone(), mat)
}

fn prepare(mut blocks: Vec<Block>, ops: &[Op], with_adjoint: bool) -> Vec<Block> {
    for b in blocks.iter_mut() {
        if !b.parametric {
            let k = build_kernel(b, ops, &[]).simplify();
            if with_adjoint {
                b.fixed_adj = Some(k.adjoint());
            }
            b.fixed = Some(k);
        }
    }
    blocks.retain(|b| !matches!(b.fixed, Some(Kernel::Skip)));
    blocks
}

#[inline]
fn apply_op(state: &mut [C], op: &Op, m: Vec<C>) {
    Kernel::dense(op.qubits.clone(), m).simplify().apply(state);
}

#[pyclass(frozen, module = "qazsim._accelerate")]
pub struct Program {
    n: usize,
    num_params: usize,
    ops: Vec<Op>,
    fwd: Vec<Block>,
    grad: Vec<Block>,
}

type OpSpec = (String, Vec<usize>, Vec<(f64, Vec<(usize, f64)>)>, Option<Vec<C>>);

impl Program {
    fn run_into(&self, p: &[f64], state: &mut [C]) {
        state.fill(C::new(0.0, 0.0));
        state[0] = C::new(1.0, 0.0);
        for b in &self.fwd {
            match &b.fixed {
                Some(k) => k.apply(state),
                None => build_kernel(b, &self.ops, p).simplify().apply(state),
            }
        }
    }

    fn state(&self, p: &[f64]) -> Vec<C> {
        let mut s = vec![C::new(0.0, 0.0); 1usize << self.n];
        self.run_into(p, &mut s);
        s
    }

    /// Adjoint differentiation: one forward pass plus one backward sweep.
    fn value_and_grad(&self, p: &[f64], terms: &[PauliTerm]) -> (f64, Vec<f64>) {
        let mut phi = self.state(p);
        let mut lam = vec![C::new(0.0, 0.0); phi.len()];
        kernels::apply_pauli_sum(&phi, terms, &mut lam);
        let value = kernels::re_inner(&phi, &lam);
        let mut grad = vec![0.0; self.num_params];
        let mut mu = vec![C::new(0.0, 0.0); phi.len()];
        for b in self.grad.iter().rev() {
            if let Some(adj) = &b.fixed_adj {
                adj.apply(&mut phi);
                adj.apply(&mut lam);
                continue;
            }
            let op = &self.ops[b.ops[0]];
            let vals = op.values(p);
            let dim = 1usize << op.qubits.len();
            let madj = dagger(&op.gate.matrix(&vals), dim);
            apply_op(&mut phi, op, madj.clone());
            for (j, e) in op.params.iter().enumerate() {
                if e.terms.is_empty() {
                    continue;
                }
                mu.copy_from_slice(&phi);
                apply_op(&mut mu, op, op.gate.derivative(&vals, j));
                let g = 2.0 * kernels::re_inner(&lam, &mu);
                for &(i, c) in &e.terms {
                    grad[i] += c * g;
                }
            }
            apply_op(&mut lam, op, madj);
        }
        (value, grad)
    }

    /// Map `f` over the rows of a (batch, num_params) array.
    fn batch<T: Send, F: Fn(&[f64]) -> T + Sync>(&self, flat: &[f64], rows: usize, f: F) -> Vec<T> {
        let pn = self.num_params;
        let row = |b: usize| &flat[b * pn..(b + 1) * pn];
        // Small states, or enough rows to keep every thread busy with whole
        // circuits (states up to 2^20 amplitudes = 16 MB per thread).
        let threads = rayon::current_num_threads();
        let across = rows > 1 && ((1usize << self.n) < PARALLEL_THRESHOLD || (rows >= 2 * threads && self.n <= 20));
        if across {
            (0..rows).into_par_iter().map(|b| f(row(b))).collect()
        } else {
            (0..rows).map(|b| f(row(b))).collect()
        }
    }

    fn check_params(&self, shape: &[usize]) -> PyResult<usize> {
        if shape.len() != 2 || shape[1] != self.num_params {
            return Err(PyValueError::new_err(format!(
                "parameters must have shape (batch, {})", self.num_params
            )));
        }
        Ok(shape[0])
    }

    fn check_terms(&self, terms: Vec<(usize, usize, f64)>) -> PyResult<Vec<PauliTerm>> {
        let lim = 1usize << self.n;
        terms
            .into_iter()
            .map(|(x, z, coeff)| {
                if x >= lim || z >= lim {
                    Err(PyValueError::new_err("Pauli term acts on a qubit outside the circuit"))
                } else {
                    Ok(PauliTerm { x, z, coeff })
                }
            })
            .collect()
    }
}

#[pymethods]
impl Program {
    #[new]
    #[pyo3(signature = (num_qubits, ops, num_params, max_fused=5, max_fused_diag=10))]
    fn new(
        num_qubits: usize,
        ops: Vec<OpSpec>,
        num_params: usize,
        max_fused: usize,
        max_fused_diag: usize,
    ) -> PyResult<Self> {
        if num_qubits == 0 || num_qubits > 34 {
            return Err(PyValueError::new_err(
                "num_qubits must be between 1 and 34 (a 34-qubit state vector takes 256 GB)",
            ));
        }
        if !(1..=5).contains(&max_fused) || max_fused_diag > 12 {
            return Err(PyValueError::new_err("max_fused must be 1..5, max_fused_diag at most 12"));
        }
        let mut parsed = Vec::with_capacity(ops.len());
        for (name, qubits, params, matrix) in ops {
            let gate = Gate::from_name(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown gate '{name}'")))?;
            let k = qubits.len();
            if k == 0 {
                return Err(PyValueError::new_err(format!("gate '{name}' has no qubits")));
            }
            for (i, &q) in qubits.iter().enumerate() {
                if q >= num_qubits {
                    return Err(PyValueError::new_err(format!("qubit {q} out of range")));
                }
                if qubits[..i].contains(&q) {
                    return Err(PyValueError::new_err(format!("duplicate qubit {q} in '{name}'")));
                }
            }
            if gate == Gate::Unitary {
                let m = matrix.as_ref().ok_or_else(|| PyValueError::new_err("unitary needs a matrix"))?;
                if m.len() != 1usize << (2 * k) || k > 12 {
                    return Err(PyValueError::new_err("unitary matrix size does not match its qubits"));
                }
            } else if gate.num_qubits() != k {
                return Err(PyValueError::new_err(format!(
                    "gate '{name}' acts on {} qubit(s), got {k}", gate.num_qubits()
                )));
            }
            if params.len() != gate.num_params() {
                return Err(PyValueError::new_err(format!(
                    "gate '{name}' takes {} parameter(s), got {}", gate.num_params(), params.len()
                )));
            }
            let params: Vec<ParamExpr> = params
                .into_iter()
                .map(|(offset, terms)| ParamExpr { offset, terms })
                .collect();
            if params.iter().flat_map(|e| e.terms.iter()).any(|&(i, _)| i >= num_params) {
                return Err(PyValueError::new_err("parameter index out of range"));
            }
            let diagonal = match &matrix {
                Some(m) if gate == Gate::Unitary => is_diagonal(m, 1 << k),
                _ => gate.is_diagonal(),
            };
            let matrix = if gate == Gate::Unitary { matrix } else { None };
            parsed.push(Op { gate, qubits, params, matrix, diagonal });
        }
        let fwd = prepare(fuse(&parsed, num_qubits, max_fused, max_fused_diag, false), &parsed, false);
        let grad = prepare(fuse(&parsed, num_qubits, max_fused, max_fused_diag, true), &parsed, true);
        Ok(Program { n: num_qubits, num_params, ops: parsed, fwd, grad })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.n
    }

    #[getter]
    fn num_params(&self) -> usize {
        self.num_params
    }

    /// Sizes of the compiled plans.
    fn stats(&self) -> HashMap<String, usize> {
        let mut s = HashMap::new();
        s.insert("ops".into(), self.ops.len());
        s.insert("blocks".into(), self.fwd.len());
        s.insert("gradient_blocks".into(), self.grad.len());
        s.insert("static_blocks".into(), self.fwd.iter().filter(|b| !b.parametric).count());
        s.insert(
            "diagonal_blocks".into(),
            self.fwd
                .iter()
                .filter(|b| b.diag || matches!(b.fixed, Some(Kernel::Diag { .. })))
                .count(),
        );
        s.insert("max_block_qubits".into(), self.fwd.iter().map(|b| b.qubits.len()).max().unwrap_or(0));
        s
    }

    /// Final state for one parameter vector.
    fn statevector<'py>(&self, py: Python<'py>, params: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<C>>> {
        let p = params.as_slice()?.to_vec();
        if p.len() != self.num_params {
            return Err(PyValueError::new_err(format!("expected {} parameter(s)", self.num_params)));
        }
        let s = py.detach(|| self.state(&p));
        Ok(s.into_pyarray(py))
    }

    /// Probabilities for each row of a (batch, num_params) array -> (batch, 2^n).
    fn probabilities<'py>(&self, py: Python<'py>, params: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.check_params(params.shape())?;
        let flat = params.as_array().iter().copied().collect::<Vec<f64>>();
        let out: Vec<Vec<f64>> = py.detach(|| self.batch(&flat, rows, |p| kernels::probabilities(&self.state(p))));
        let dim = 1usize << self.n;
        let arr = Array2::from_shape_vec((rows, dim), out.into_iter().flatten().collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// <H> for each row; terms are (x_mask, z_mask, coeff).
    fn expectation<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray2<'py, f64>,
        terms: Vec<(usize, usize, f64)>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rows = self.check_params(params.shape())?;
        let terms = self.check_terms(terms)?;
        let flat = params.as_array().iter().copied().collect::<Vec<f64>>();
        let out = py.detach(|| self.batch(&flat, rows, |p| kernels::expectation(&self.state(p), &terms)));
        Ok(out.into_pyarray(py))
    }

    /// (<H>, d<H>/dθ) for each row, by adjoint differentiation.
    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray2<'py, f64>,
        terms: Vec<(usize, usize, f64)>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
        let rows = self.check_params(params.shape())?;
        let terms = self.check_terms(terms)?;
        let flat = params.as_array().iter().copied().collect::<Vec<f64>>();
        let out = py.detach(|| self.batch(&flat, rows, |p| self.value_and_grad(p, &terms)));
        let values: Vec<f64> = out.iter().map(|(v, _)| *v).collect();
        let grads: Vec<f64> = out.into_iter().flat_map(|(_, g)| g).collect();
        let g = Array2::from_shape_vec((rows, self.num_params), grads)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((values.into_pyarray(py), g.into_pyarray(py)))
    }
}
