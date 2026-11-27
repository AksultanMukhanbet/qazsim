//! Gate library of the 0.2 core: matrices and their parameter derivatives.
//!
//! Matrices are row-major, dimension 2^k for a k-qubit gate. Qubit order is
//! little-endian: for a gate on `[q0, q1, ...]`, bit b of the row/column index
//! belongs to `q_b`. Controlled gates take the control(s) first, so
//! `cx(c, t)` flips bit 1 when bit 0 is set.

use num_complex::Complex64 as C;

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);
const I: C = C::new(0.0, 1.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Id, X, Y, Z, H, S, Sdg, T, Tdg, Sx, Sxdg,
    Rx, Ry, Rz, P, U,
    Cx, Cy, Cz, Swap, ISwap, Cp, Crx, Cry, Crz, Rxx, Ryy, Rzz,
    Ccx, Ccz, Cswap,
    /// Arbitrary matrix supplied by the user (no parameters).
    Unitary,
}

impl Gate {
    pub fn from_name(name: &str) -> Option<Gate> {
        use Gate::*;
        Some(match name {
            "id" | "i" => Id, "x" => X, "y" => Y, "z" => Z, "h" => H,
            "s" => S, "sdg" => Sdg, "t" => T, "tdg" => Tdg, "sx" => Sx, "sxdg" => Sxdg,
            "rx" => Rx, "ry" => Ry, "rz" => Rz, "p" => P, "u" => U,
            "cx" | "cnot" => Cx, "cy" => Cy, "cz" => Cz, "swap" => Swap, "iswap" => ISwap,
            "cp" => Cp, "crx" => Crx, "cry" => Cry, "crz" => Crz,
            "rxx" => Rxx, "ryy" => Ryy, "rzz" => Rzz,
            "ccx" | "toffoli" => Ccx, "ccz" => Ccz, "cswap" | "fredkin" => Cswap,
            "unitary" => Unitary,
            _ => return None,
        })
    }

    /// Number of qubits (0 for `Unitary`: taken from the matrix).
    pub fn num_qubits(self) -> usize {
        use Gate::*;
        match self {
            Id | X | Y | Z | H | S | Sdg | T | Tdg | Sx | Sxdg | Rx | Ry | Rz | P | U => 1,
            Cx | Cy | Cz | Swap | ISwap | Cp | Crx | Cry | Crz | Rxx | Ryy | Rzz => 2,
            Ccx | Ccz | Cswap => 3,
            Unitary => 0,
        }
    }

    pub fn num_params(self) -> usize {
        use Gate::*;
        match self {
            Rx | Ry | Rz | P | Cp | Crx | Cry | Crz | Rxx | Ryy | Rzz => 1,
            U => 3,
            _ => 0,
        }
    }

    /// True when the matrix is diagonal for every parameter value.
    pub fn is_diagonal(self) -> bool {
        use Gate::*;
        matches!(self, Id | Z | S | Sdg | T | Tdg | Rz | P | Cz | Cp | Crz | Rzz | Ccz)
    }

    /// Matrix for numeric parameters `p`.
    pub fn matrix(self, p: &[f64]) -> Vec<C> {
        use Gate::*;
        let h = std::f64::consts::FRAC_1_SQRT_2;
        match self {
            Id => vec![ONE, ZERO, ZERO, ONE],
            X => vec![ZERO, ONE, ONE, ZERO],
            Y => vec![ZERO, -I, I, ZERO],
            Z => vec![ONE, ZERO, ZERO, -ONE],
            H => vec![C::new(h, 0.0), C::new(h, 0.0), C::new(h, 0.0), C::new(-h, 0.0)],
            S => vec![ONE, ZERO, ZERO, I],
            Sdg => vec![ONE, ZERO, ZERO, -I],
            T => vec![ONE, ZERO, ZERO, C::from_polar(1.0, std::f64::consts::FRAC_PI_4)],
            Tdg => vec![ONE, ZERO, ZERO, C::from_polar(1.0, -std::f64::consts::FRAC_PI_4)],
            Sx => {
                let (a, b) = (C::new(0.5, 0.5), C::new(0.5, -0.5));
                vec![a, b, b, a]
            }
            Sxdg => {
                let (a, b) = (C::new(0.5, -0.5), C::new(0.5, 0.5));
                vec![a, b, b, a]
            }
            Rx | Ry | Rz | Rxx | Ryy | Rzz => pauli_rotation(self, p[0], false),
            P => vec![ONE, ZERO, ZERO, C::from_polar(1.0, p[0])],
            U => u_matrix(p[0], p[1], p[2], None),
            Cx => controlled(&X.matrix(&[])),
            Cy => controlled(&Y.matrix(&[])),
            Cz => diag(&[ONE, ONE, ONE, -ONE]),
            Cp => diag(&[ONE, ONE, ONE, C::from_polar(1.0, p[0])]),
            Crx => controlled(&Rx.matrix(p)),
            Cry => controlled(&Ry.matrix(p)),
            Crz => controlled(&Rz.matrix(p)),
            Swap => permutation(4, &[(1, 2)]),
            ISwap => {
                let mut m = vec![ZERO; 16];
                m[0] = ONE;
                m[4 + 2] = I;
                m[8 + 1] = I;
                m[15] = ONE;
                m
            }
            Ccx => permutation(8, &[(3, 7)]),
            Ccz => {
                let mut d = vec![ONE; 8];
                d[7] = -ONE;
                diag(&d)
            }
            // controls: bit 0; swapped: bits 1, 2  -> exchange 0b011 <-> 0b101
            Cswap => permutation(8, &[(3, 5)]),
            Unitary => unreachable!("unitary matrices are stored on the op"),
        }
    }

    /// d(matrix)/d(p[j]).
    pub fn derivative(self, p: &[f64], j: usize) -> Vec<C> {
        use Gate::*;
        match self {
            Rx | Ry | Rz | Rxx | Ryy | Rzz => pauli_rotation(self, p[0], true),
            P => vec![ZERO, ZERO, ZERO, I * C::from_polar(1.0, p[0])],
            U => u_matrix(p[0], p[1], p[2], Some(j)),
            Cp => diag(&[ZERO, ZERO, ZERO, I * C::from_polar(1.0, p[0])]),
            Crx => controlled_derivative(&pauli_rotation(Rx, p[0], true)),
            Cry => controlled_derivative(&pauli_rotation(Ry, p[0], true)),
            Crz => controlled_derivative(&pauli_rotation(Rz, p[0], true)),
            _ => panic!("gate {self:?} has no parameters"),
        }
    }
}

fn diag(d: &[C]) -> Vec<C> {
    let n = d.len();
    let mut m = vec![ZERO; n * n];
    for i in 0..n {
        m[i * n + i] = d[i];
    }
    m
}

fn permutation(n: usize, swaps: &[(usize, usize)]) -> Vec<C> {
    let mut perm: Vec<usize> = (0..n).collect();
    for &(a, b) in swaps {
        perm.swap(a, b);
    }
    let mut m = vec![ZERO; n * n];
    for (r, &c) in perm.iter().enumerate() {
        m[r * n + c] = ONE;
    }
    m
}

/// Control on bit 0, target on bit 1: acts with `u` on indices 1 and 3.
fn controlled(u: &[C]) -> Vec<C> {
    let mut m = diag(&[ONE, ONE, ONE, ONE]);
    m[4 + 1] = u[0];
    m[4 + 3] = u[1];
    m[12 + 1] = u[2];
    m[12 + 3] = u[3];
    m
}

fn controlled_derivative(du: &[C]) -> Vec<C> {
    let mut m = vec![ZERO; 16];
    m[4 + 1] = du[0];
    m[4 + 3] = du[1];
    m[12 + 1] = du[2];
    m[12 + 3] = du[3];
    m
}

/// exp(-i θ/2 G) = cos(θ/2) I - i sin(θ/2) G for a Pauli product G (G² = I),
/// or its θ-derivative -sin(θ/2)/2 I - i cos(θ/2)/2 G.
fn pauli_rotation(g: Gate, theta: f64, derivative: bool) -> Vec<C> {
    let gen = match g {
        Gate::Rx => vec![ZERO, ONE, ONE, ZERO],
        Gate::Ry => vec![ZERO, -I, I, ZERO],
        Gate::Rz => vec![ONE, ZERO, ZERO, -ONE],
        Gate::Rxx => permutation(4, &[(0, 3), (1, 2)]),
        Gate::Ryy => {
            // Y⊗Y |b1 b0> = -(-1)^(b0+b1) |~b1 ~b0>
            let mut m = vec![ZERO; 16];
            m[3] = -ONE;
            m[4 + 2] = ONE;
            m[8 + 1] = ONE;
            m[12] = -ONE;
            m
        }
        Gate::Rzz => diag(&[ONE, -ONE, -ONE, ONE]),
        _ => unreachable!(),
    };
    let n = if gen.len() == 4 { 2 } else { 4 };
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let (a, b) = if derivative { (-s / 2.0, c / 2.0) } else { (c, s) };
    let mut m: Vec<C> = gen.iter().map(|&x| -I * b * x).collect();
    for i in 0..n {
        m[i * n + i] += a;
    }
    m
}

/// U(θ, φ, λ) or its derivative with respect to parameter `d` (0 = θ, 1 = φ, 2 = λ).
fn u_matrix(th: f64, ph: f64, la: f64, d: Option<usize>) -> Vec<C> {
    let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
    let (el, ep, epl) = (C::from_polar(1.0, la), C::from_polar(1.0, ph), C::from_polar(1.0, ph + la));
    match d {
        None => vec![C::new(c, 0.0), -el * s, ep * s, epl * c],
        Some(0) => vec![C::new(-s / 2.0, 0.0), -el * c / 2.0, ep * c / 2.0, -epl * s / 2.0],
        Some(1) => vec![ZERO, ZERO, I * ep * s, I * epl * c],
        Some(2) => vec![ZERO, -I * el * s, ZERO, I * epl * c],
        _ => unreachable!(),
    }
}
