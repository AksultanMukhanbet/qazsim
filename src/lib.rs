//! QazSim Rust core (`qazsim._accelerate`).
//!
//! Same layout idea as Qiskit: the user API is written in Python, the heavy
//! numerical kernels live here and are exposed to Python through PyO3.
//!
//! Conventions (identical to Qiskit):
//! * little-endian qubit order: qubit 0 is the least significant bit of a
//!   basis-state index;
//! * for a k-qubit matrix acting on `qubits = [q0, q1, ...]`, bit b of the
//!   matrix row/column index corresponds to `qubits[b]`.

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Below this many amplitudes the work is done on one thread:
/// spawning threads costs more than it saves for small states.
const PARALLEL_THRESHOLD: usize = 1 << 14;

/// Apply a dense 2^k x 2^k matrix (row-major) to `qubits` of `state` in place.
///
/// All amplitudes touched by one application differ only in the target bits,
/// so they lie inside one block of length 2^(max_qubit + 1). Blocks are
/// independent and are processed in parallel with rayon.
fn apply_matrix_inplace(state: &mut [Complex64], mat: &[Complex64], qubits: &[usize]) {
    match qubits.len() {
        1 => return apply_1q(state, mat, qubits[0]),
        2 => return apply_2q(state, mat, qubits[0], qubits[1]),
        _ => {}
    }
    let k = qubits.len();
    let dim = 1usize << k;
    let max_q = *qubits.iter().max().expect("at least one qubit");
    let block = 1usize << (max_q + 1);

    let mut target_mask = 0usize;
    for &q in qubits {
        target_mask |= 1usize << q;
    }
    // offsets[j] = position of local basis state j relative to the base index
    let offsets: Vec<usize> = (0..dim)
        .map(|j| {
            qubits
                .iter()
                .enumerate()
                .filter(|(b, _)| (j >> b) & 1 == 1)
                .fold(0usize, |acc, (_, &q)| acc | (1usize << q))
        })
        .collect();

    let work = |chunk: &mut [Complex64]| {
        let mut buf = vec![Complex64::new(0.0, 0.0); dim];
        for base in 0..chunk.len() {
            if base & target_mask != 0 {
                continue;
            }
            for j in 0..dim {
                buf[j] = chunk[base + offsets[j]];
            }
            for r in 0..dim {
                let row = &mat[r * dim..(r + 1) * dim];
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..dim {
                    acc += row[j] * buf[j];
                }
                chunk[base + offsets[r]] = acc;
            }
        }
    };

    run_blocks(state, block, work);
}

/// Split `state` into independent pieces (multiples of `block`) and run `work`
/// on each, in parallel for large states. Pieces are at least MIN_TASK long
/// so that gates on low qubits do not create millions of tiny tasks.
const MIN_TASK: usize = 1 << 13;

fn run_blocks<F>(state: &mut [Complex64], block: usize, work: F)
where
    F: Fn(&mut [Complex64]) + Sync + Send,
{
    let task = block.max(MIN_TASK).min(state.len());
    if state.len() >= PARALLEL_THRESHOLD && state.len() > task {
        state.par_chunks_mut(task).for_each(work);
    } else {
        work(state);
    }
}

/// Fast path: one-qubit gate. Pairs (i, i + 2^q) with bit q of i equal to 0.
fn apply_1q(state: &mut [Complex64], m: &[Complex64], q: usize) {
    let (m00, m01, m10, m11) = (m[0], m[1], m[2], m[3]);
    let half = 1usize << q;
    let block = half << 1;
    run_blocks(state, block, |chunk: &mut [Complex64]| {
        for b in chunk.chunks_mut(block) {
            let (lo, hi) = b.split_at_mut(half);
            for (a0, a1) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x0, x1) = (*a0, *a1);
                *a0 = m00 * x0 + m01 * x1;
                *a1 = m10 * x0 + m11 * x1;
            }
        }
    });
}

#[inline(always)]
fn insert_zero(x: usize, bit: usize) -> usize {
    ((x >> bit) << (bit + 1)) | (x & ((1usize << bit) - 1))
}

/// Fast path: two-qubit gate on (q0, q1); bit 0 of the matrix index is q0.
fn apply_2q(state: &mut [Complex64], m: &[Complex64], q0: usize, q1: usize) {
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let block = 1usize << (hi + 1);
    let o = [0usize, 1 << q0, 1 << q1, (1 << q0) | (1 << q1)];
    let mut mm = [Complex64::new(0.0, 0.0); 16];
    mm.copy_from_slice(&m[..16]);
    run_blocks(state, block, move |chunk: &mut [Complex64]| {
        for j in 0..chunk.len() / 4 {
            let base = insert_zero(insert_zero(j, lo), hi);
            let x = [chunk[base], chunk[base + o[1]], chunk[base + o[2]], chunk[base + o[3]]];
            for r in 0..4 {
                chunk[base + o[r]] =
                    mm[4 * r] * x[0] + mm[4 * r + 1] * x[1] + mm[4 * r + 2] * x[2] + mm[4 * r + 3] * x[3];
            }
        }
    });
}

/// apply_matrix(state, matrix, qubits)
///
/// Apply `matrix` (complex128, shape 2^k x 2^k) to `qubits` of the complex128
/// vector `state`, in place. The GIL is released during the computation.
#[pyfunction]
fn apply_matrix(
    py: Python<'_>,
    mut state: PyReadwriteArray1<'_, Complex64>,
    matrix: PyReadonlyArray2<'_, Complex64>,
    qubits: Vec<usize>,
) -> PyResult<()> {
    let n_amp = state.len();
    if n_amp == 0 || !n_amp.is_power_of_two() {
        return Err(PyValueError::new_err("state length must be a power of two"));
    }
    let num_qubits = n_amp.trailing_zeros() as usize;
    if qubits.is_empty() {
        return Err(PyValueError::new_err("qubits must not be empty"));
    }
    for (i, &q) in qubits.iter().enumerate() {
        if q >= num_qubits {
            return Err(PyValueError::new_err(format!(
                "qubit {q} out of range for a {num_qubits}-qubit state"
            )));
        }
        if qubits[..i].contains(&q) {
            return Err(PyValueError::new_err(format!("duplicate qubit {q}")));
        }
    }
    let dim = 1usize << qubits.len();
    let shape = matrix.shape();
    if shape != [dim, dim] {
        return Err(PyValueError::new_err(format!(
            "matrix must have shape ({dim}, {dim}) for {} qubit(s)",
            qubits.len()
        )));
    }
    let mat: Vec<Complex64> = matrix.as_array().iter().copied().collect();
    let slice = state
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("state must be a contiguous array"))?;
    py.detach(|| apply_matrix_inplace(slice, &mat, &qubits));
    Ok(())
}

/// probabilities(state) -> float64 array |amplitude|^2
#[pyfunction]
fn probabilities<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be a contiguous array"))?;
    let out: Vec<f64> = py.detach(|| {
        if slice.len() >= PARALLEL_THRESHOLD {
            slice.par_iter().map(|a| a.norm_sqr()).collect()
        } else {
            slice.iter().map(|a| a.norm_sqr()).collect()
        }
    });
    Ok(out.into_pyarray(py))
}

/// Number of threads rayon uses (set RAYON_NUM_THREADS to change it).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule(gil_used = false)]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
