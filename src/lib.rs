//! QazSim Rust core (`qazsim._accelerate`).
//!
//! * `kernels` — state-vector kernels (dense k-qubit, diagonal, Pauli sums);
//! * `gates`   — gate matrices and their parameter derivatives;
//! * `program` — compiled programs: gate fusion, batched execution and
//!   adjoint gradients (the 0.2 execution model);
//! * `apply_matrix` / `probabilities` — low-level functions used by the 0.1 API.
//!
//! Qubit order is little-endian: qubit 0 is the least significant bit of a
//! basis-state index; for a k-qubit matrix on `[q0, q1, ...]`, bit b of the
//! row/column index belongs to `q_b`.

mod gates;
mod kernels;
mod program;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    py.detach(|| kernels::apply_dense(slice, &mat, &qubits));
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
    let out: Vec<f64> = py.detach(|| kernels::probabilities(slice));
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
    m.add_class::<program::Program>()?;
    Ok(())
}
