//! State-vector kernels.
//!
//! The state is a `[Complex64]` of length 2^n. A k-qubit gate touches groups of
//! 2^k amplitudes that differ only in the target bits; groups are disjoint, so
//! they are processed in parallel through a raw pointer. Unlike splitting the
//! slice into contiguous blocks, this also parallelises gates on the highest
//! qubits.

use num_complex::Complex64 as C;
use rayon::prelude::*;

/// Below this many amplitudes everything runs on the calling thread.
pub const PARALLEL_THRESHOLD: usize = 1 << 14;

#[derive(Clone, Copy)]
struct SendPtr(*mut C);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[inline(always)]
fn insert_zero(x: usize, bit: usize) -> usize {
    ((x >> bit) << (bit + 1)) | (x & ((1usize << bit) - 1))
}

/// Parallelise inside a kernel only for large states, and not when we are
/// already running on a rayon worker (batched execution parallelises across
/// the batch instead: one whole circuit per thread, no synchronisation per gate).
#[inline]
pub fn parallel(len: usize) -> bool {
    len >= PARALLEL_THRESHOLD && rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1
}

/// Run `f(start, end)` over `0..n` work items, in parallel when the state is large.
#[inline]
fn for_ranges<F: Fn(usize, usize) + Sync + Send>(n: usize, state_len: usize, f: F) {
    if !parallel(state_len) {
        f(0, n);
        return;
    }
    let chunk = (n / (rayon::current_num_threads() * 8)).max(1024).min(n.max(1));
    let tasks = n.div_ceil(chunk);
    (0..tasks).into_par_iter().for_each(|t| {
        let s = t * chunk;
        f(s, (s + chunk).min(n));
    });
}

/// Positions of all touched bits (targets and controls), sorted, plus the control mask.
#[inline]
fn layout(targets: &[usize], controls: &[usize]) -> (Vec<usize>, usize) {
    let mut all: Vec<usize> = targets.iter().chain(controls.iter()).copied().collect();
    all.sort_unstable();
    (all, controls.iter().fold(0, |m, &c| m | (1usize << c)))
}

#[inline(always)]
fn group_base(g: usize, sorted: &[usize], ctrl: usize) -> usize {
    sorted.iter().fold(g, |b, &q| insert_zero(b, q)) | ctrl
}

/// Apply a dense 2^K x 2^K matrix (row-major, D = 2^K) to `targets`, only on
/// the subspace where every control bit is 1.
fn apply_dense_k<const K: usize, const D: usize>(state: &mut [C], mat: &[C], targets: &[usize], controls: &[usize]) {
    let (sorted, ctrl) = layout(targets, controls);
    let mut offs = [0usize; D];
    for (j, o) in offs.iter_mut().enumerate() {
        for b in 0..K {
            if (j >> b) & 1 == 1 {
                *o |= 1 << targets[b];
            }
        }
    }
    let mut m = [[C::new(0.0, 0.0); D]; D];
    for r in 0..D {
        m[r].copy_from_slice(&mat[r * D..(r + 1) * D]);
    }
    let len = state.len();
    let ptr = SendPtr(state.as_mut_ptr());
    let simd = simd_available();
    for_ranges(len >> sorted.len(), len, move |s, e| {
        let p = ptr;
        // SAFETY: groups are disjoint; the AVX2 path runs only when the CPU has it.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if simd {
                return dense_groups_avx2::<K, D>(p.0, s, e, &sorted, ctrl, &offs, &m);
            }
            let _ = simd;
            dense_groups::<K, D>(p.0, s, e, &sorted, ctrl, &offs, &m)
        }
    });
}

#[inline(always)]
fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Body of the dense kernel for groups s..e. Kept `inline(always)` so that the
/// AVX2 wrapper below gets its own vectorised copy.
#[inline(always)]
unsafe fn dense_groups<const K: usize, const D: usize>(
    p: *mut C, s: usize, e: usize, sorted: &[usize], ctrl: usize, offs: &[usize; D], m: &[[C; D]; D],
) {
    for g in s..e {
        let base = group_base(g, sorted, ctrl);
        let mut x = [C::new(0.0, 0.0); D];
        for j in 0..D {
            x[j] = *p.add(base + offs[j]);
        }
        for r in 0..D {
            let row = &m[r];
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for j in 0..D {
                re += row[j].re * x[j].re - row[j].im * x[j].im;
                im += row[j].re * x[j].im + row[j].im * x[j].re;
            }
            *p.add(base + offs[r]) = C::new(re, im);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dense_groups_avx2<const K: usize, const D: usize>(
    p: *mut C, s: usize, e: usize, sorted: &[usize], ctrl: usize, offs: &[usize; D], m: &[[C; D]; D],
) {
    dense_groups::<K, D>(p, s, e, sorted, ctrl, offs, m)
}

/// One-qubit gate without controls: contiguous halves, which vectorise well.
fn apply_1q(state: &mut [C], m: &[C], q: usize) {
    let (m00, m01, m10, m11) = (m[0], m[1], m[2], m[3]);
    let half = 1usize << q;
    let len = state.len();
    let block = half << 1;
    let body = |c: &mut [C]| {
        for b in c.chunks_mut(block) {
            let (lo, hi) = b.split_at_mut(half);
            for (a0, a1) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x0, x1) = (*a0, *a1);
                *a0 = m00 * x0 + m01 * x1;
                *a1 = m10 * x0 + m11 * x1;
            }
        }
    };
    if !parallel(len) {
        return body(state);
    }
    if block > len / (2 * rayon::current_num_threads()) {
        // high qubit: too few contiguous blocks, the group form keeps all threads busy
        return apply_dense_k::<1, 2>(state, m, &[q], &[]);
    }
    state.par_chunks_mut(block.max(1 << 12)).for_each(body);
}

/// Dense matrix on `targets` (bit b of the matrix index is targets[b]).
pub fn apply_dense(state: &mut [C], mat: &[C], targets: &[usize]) {
    apply_dense_ctrl(state, mat, targets, &[])
}

/// Dense matrix on `targets`, applied where all `controls` are 1.
pub fn apply_dense_ctrl(state: &mut [C], mat: &[C], targets: &[usize], controls: &[usize]) {
    match targets.len() {
        1 if controls.is_empty() => apply_1q(state, mat, targets[0]),
        1 => apply_dense_k::<1, 2>(state, mat, targets, controls),
        2 => apply_dense_k::<2, 4>(state, mat, targets, controls),
        3 => apply_dense_k::<3, 8>(state, mat, targets, controls),
        4 => apply_dense_k::<4, 16>(state, mat, targets, controls),
        5 => apply_dense_k::<5, 32>(state, mat, targets, controls),
        _ => apply_dense_dyn(state, mat, targets, controls),
    }
}

fn apply_dense_dyn(state: &mut [C], mat: &[C], targets: &[usize], controls: &[usize]) {
    let k = targets.len();
    let d = 1usize << k;
    let (sorted, ctrl) = layout(targets, controls);
    let offs: Vec<usize> = (0..d)
        .map(|j| (0..k).filter(|b| (j >> b) & 1 == 1).fold(0, |a, b| a | (1 << targets[b])))
        .collect();
    let len = state.len();
    let ptr = SendPtr(state.as_mut_ptr());
    for_ranges(len >> sorted.len(), len, |s, e| {
        let p = ptr;
        let mut x = vec![C::new(0.0, 0.0); d];
        for g in s..e {
            let base = group_base(g, &sorted, ctrl);
            unsafe {
                for j in 0..d {
                    x[j] = *p.0.add(base + offs[j]);
                }
                for r in 0..d {
                    let row = &mat[r * d..(r + 1) * d];
                    let acc = row.iter().zip(x.iter()).fold(C::new(0.0, 0.0), |a, (m, v)| a + m * v);
                    *p.0.add(base + offs[r]) = acc;
                }
            }
        }
    });
}

/// Diagonal matrix on `targets`, applied where all `controls` are 1.
pub fn apply_diag_ctrl(state: &mut [C], d: &[C], targets: &[usize], controls: &[usize]) {
    let k = targets.len();
    let len = state.len();
    if k == 0 {
        // pure phase on the controlled subspace
        let (sorted, ctrl) = layout(&[], controls);
        let ph = d[0];
        let ptr = SendPtr(state.as_mut_ptr());
        return for_ranges(len >> sorted.len(), len, move |s, e| {
            let p = ptr;
            for g in s..e {
                unsafe { *p.0.add(group_base(g, &sorted, ctrl)) *= ph };
            }
        });
    }
    if controls.is_empty() && targets.iter().all(|&q| q < 12) {
        // all targets in the low bits: the diagonal repeats with period 2^(max+1)
        let period = 1usize << (targets.iter().max().unwrap() + 1);
        let pattern: Vec<C> = (0..period)
            .map(|i| d[targets.iter().enumerate().fold(0, |l, (b, &q)| l | (((i >> q) & 1) << b))])
            .collect();
        let body = |c: &mut [C]| {
            for blk in c.chunks_mut(period) {
                for (a, f) in blk.iter_mut().zip(pattern.iter()) {
                    *a *= f;
                }
            }
        };
        if !parallel(len) {
            body(state);
        } else {
            state.par_chunks_mut(period.max(1 << 12)).for_each(body);
        }
        return;
    }
    let (sorted, ctrl) = layout(targets, controls);
    let dd = 1usize << k;
    let offs: Vec<usize> = (0..dd)
        .map(|j| (0..k).filter(|b| (j >> b) & 1 == 1).fold(0, |a, b| a | (1 << targets[b])))
        .collect();
    let ptr = SendPtr(state.as_mut_ptr());
    for_ranges(len >> sorted.len(), len, |s, e| {
        let p = ptr;
        for g in s..e {
            let base = group_base(g, &sorted, ctrl);
            for j in 0..dd {
                unsafe { *p.0.add(base + offs[j]) *= d[j] };
            }
        }
    });
}

// ------------------------------------------------------------ observables

/// One Pauli product: X on `x` bits, Z on `z` bits (Y = both), times `coeff`.
#[derive(Clone, Copy, Debug)]
pub struct PauliTerm {
    pub x: usize,
    pub z: usize,
    pub coeff: f64,
}

impl PauliTerm {
    /// P|i> = phase * sign(i) |i ^ x>, phase = i^(number of Y).
    #[inline]
    fn phase(&self) -> C {
        match (self.x & self.z).count_ones() % 4 {
            0 => C::new(1.0, 0.0),
            1 => C::new(0.0, 1.0),
            2 => C::new(-1.0, 0.0),
            _ => C::new(0.0, -1.0),
        }
    }
}

#[inline(always)]
fn sign(i: usize, z: usize) -> f64 {
    if (i & z).count_ones() & 1 == 1 { -1.0 } else { 1.0 }
}

/// <ψ| Σ c P |ψ> (real part; exact for Hermitian sums). Terms with the same
/// X-mask share one pass over the state.
pub fn expectation(state: &[C], terms: &[PauliTerm]) -> f64 {
    let mut groups: Vec<(usize, Vec<(usize, C)>)> = Vec::new();
    for t in terms {
        let w = (t.z, t.phase() * t.coeff);
        match groups.iter_mut().find(|(x, _)| *x == t.x) {
            Some((_, v)) => v.push(w),
            None => groups.push((t.x, vec![w])),
        }
    }
    let mut total = 0.0;
    for (x, zs) in &groups {
        let x = *x;
        let f = |i: usize| -> C {
            let w: C = zs.iter().map(|&(z, c)| c * sign(i, z)).sum();
            state[i ^ x].conj() * state[i] * w
        };
        let s: C = if parallel(state.len()) {
            (0..state.len()).into_par_iter().map(f).sum()
        } else {
            (0..state.len()).map(f).sum()
        };
        total += s.re;
    }
    total
}

/// out = Σ c P |ψ>
pub fn apply_pauli_sum(state: &[C], terms: &[PauliTerm], out: &mut [C]) {
    let phases: Vec<C> = terms.iter().map(|t| t.phase() * t.coeff).collect();
    let f = |(j, o): (usize, &mut C)| {
        let mut acc = C::new(0.0, 0.0);
        for (t, ph) in terms.iter().zip(phases.iter()) {
            let i = j ^ t.x;
            acc += ph * sign(i, t.z) * state[i];
        }
        *o = acc;
    };
    if parallel(state.len()) {
        out.par_iter_mut().enumerate().for_each(f);
    } else {
        out.iter_mut().enumerate().for_each(f);
    }
}

/// Re <a|b>
pub fn re_inner(a: &[C], b: &[C]) -> f64 {
    if parallel(a.len()) {
        a.par_iter().zip(b.par_iter()).map(|(x, y)| (x.conj() * y).re).sum()
    } else {
        a.iter().zip(b.iter()).map(|(x, y)| (x.conj() * y).re).sum()
    }
}

pub fn probabilities(state: &[C]) -> Vec<f64> {
    if parallel(state.len()) {
        state.par_iter().map(|a| a.norm_sqr()).collect()
    } else {
        state.iter().map(|a| a.norm_sqr()).collect()
    }
}
