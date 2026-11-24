"""QazSim backend: run -> Job -> Result, API modelled on qiskit_aer.AerSimulator."""

from __future__ import annotations

import time
import uuid

import numpy as np

from . import _accelerate
from .circuit import QuantumCircuit
from .quantum_info import (DensityMatrix, Statevector, apply_kraus_dm,
                           apply_unitary, apply_unitary_dm)

_RESET_KRAUS = [np.array([[1, 0], [0, 0]], dtype=np.complex128),
                np.array([[0, 1], [0, 0]], dtype=np.complex128)]


class QazSimError(Exception):
    pass


# --------------------------------------------------------------------- Result
class Result:
    def __init__(self, backend_name, job_id, experiments, time_taken):
        self.backend_name = backend_name
        self.job_id = job_id
        self.results = experiments  # list of dicts
        self.time_taken = time_taken
        self.success = True

    def _exp(self, experiment=None) -> dict:
        if experiment is None:
            if len(self.results) != 1:
                raise QazSimError("several experiments: pass an index or a circuit")
            return self.results[0]
        if isinstance(experiment, int):
            return self.results[experiment]
        if isinstance(experiment, QuantumCircuit):
            for r in self.results:
                if r["circuit"] is experiment:
                    return r
            experiment = experiment.name
        for r in self.results:
            if r["name"] == experiment:
                return r
        raise QazSimError(f"no experiment '{experiment}'")

    def get_counts(self, experiment=None) -> dict:
        exp = self._exp(experiment)
        if "counts" not in exp:
            raise QazSimError("no counts: the circuit has no measurements")
        return dict(exp["counts"])

    def get_memory(self, experiment=None) -> list:
        exp = self._exp(experiment)
        if "memory" not in exp:
            raise QazSimError("run with memory=True to get per-shot results")
        return list(exp["memory"])

    def get_statevector(self, experiment=None, label="statevector") -> Statevector:
        return self._saved(experiment, label)

    def get_density_matrix(self, experiment=None, label="density_matrix") -> DensityMatrix:
        return self._saved(experiment, label)

    def get_probabilities(self, experiment=None, label="probabilities") -> dict:
        return self._saved(experiment, label)

    def _saved(self, experiment, label):
        data = self._exp(experiment)["data"]
        if label not in data:
            raise QazSimError(f"nothing saved under '{label}' (add qc.save_...() to the circuit)")
        return data[label]

    def data(self, experiment=None) -> dict:
        exp = self._exp(experiment)
        out = dict(exp["data"])
        if "counts" in exp:
            out["counts"] = dict(exp["counts"])
        return out

    def __repr__(self):
        return (f"Result(backend_name='{self.backend_name}', job_id='{self.job_id}', "
                f"experiments={len(self.results)}, time_taken={self.time_taken:.4f}s)")


class Job:
    """Execution is synchronous; ``result()`` returns immediately."""

    def __init__(self, backend, result: Result):
        self._backend = backend
        self._result = result
        self._job_id = result.job_id

    def job_id(self) -> str:
        return self._job_id

    def backend(self):
        return self._backend

    def status(self) -> str:
        return "DONE"

    def result(self) -> Result:
        return self._result


# -------------------------------------------------------------------- helpers
def _format_counts(clbit_values: np.ndarray, num_clbits: int) -> tuple[dict, list]:
    vals, cnt = np.unique(clbit_values, return_counts=True)
    counts = {format(int(v), f"0{num_clbits}b"): int(c) for v, c in zip(vals, cnt)}
    return counts, vals


def _apply_readout(bits: np.ndarray, qubits, noise_model, rng) -> np.ndarray:
    """bits[s, i] = measured bit of qubits[i] in shot s; flips per readout error."""
    if noise_model is None or not noise_model.has_readout_errors:
        return bits
    out = bits.copy()
    for i, q in enumerate(qubits):
        ro = noise_model.readout_for(q)
        if ro is None:
            continue
        p_flip = np.where(bits[:, i] == 0, ro.probabilities[0, 1], ro.probabilities[1, 0])
        flip = rng.random(bits.shape[0]) < p_flip
        out[:, i] ^= flip.astype(out.dtype)
    return out


# -------------------------------------------------------------------- backend
class QazSim:
    """QazSim quantum circuit simulator.

    Methods: ``"automatic"`` (default), ``"statevector"``, ``"density_matrix"``.

    >>> sim = QazSim()
    >>> counts = sim.run(qc, shots=1000).result().get_counts()
    """

    METHODS = ("automatic", "statevector", "density_matrix")

    def __init__(self, method: str = "automatic", noise_model=None,
                 seed_simulator: int | None = None, name: str = "qazsim"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.name = name
        self.options = {"method": method, "noise_model": noise_model,
                        "seed_simulator": seed_simulator}

    # Qiskit-like options API
    def set_options(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.options:
                raise KeyError(f"unknown option '{k}'")
            self.options[k] = v

    @property
    def num_threads(self) -> int:
        return _accelerate.num_threads()

    def __repr__(self):
        return f"{type(self).__name__}(name='{self.name}', method='{self.options['method']}')"

    # ------------------------------------------------------------------ run
    def run(self, circuits, shots: int = 1024, memory: bool = False,
            parameter_binds=None, seed_simulator=None, **run_options) -> Job:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = list(circuits)
        if parameter_binds is not None:
            if len(parameter_binds) != len(circuits):
                raise ValueError("parameter_binds must have one entry per circuit")
            circuits = [c.assign_parameters(b) for c, b in zip(circuits, parameter_binds)]
        seed = seed_simulator if seed_simulator is not None else self.options["seed_simulator"]
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        exps = []
        for qc in circuits:
            if qc.parameters:
                raise QazSimError(f"circuit '{qc.name}' has unbound parameters "
                                  f"{[p.name for p in qc.parameters]}")
            self._validate(qc)
            exps.append(self._run_circuit(qc, shots, memory, rng))
        result = Result(self.name, str(uuid.uuid4()), exps, time.perf_counter() - t0)
        return Job(self, result)

    def _validate(self, circuit):
        """Hook for device backends (limits, basis gates)."""

    def _method_for(self, circuit) -> str:
        m = self.options["method"]
        if m != "automatic":
            return m
        nm = self.options["noise_model"]
        return "density_matrix" if nm is not None and nm.noise_instructions else "statevector"

    def _run_circuit(self, qc, shots, memory, rng) -> dict:
        method = self._method_for(qc)
        exp = {"name": qc.name, "circuit": qc, "shots": shots, "method": method, "data": {}}
        nm = self.options["noise_model"]
        if method == "statevector" and nm is not None and nm.noise_instructions:
            raise QazSimError("gate noise requires method='density_matrix' "
                              "(readout errors work with any method)")
        if qc.has_mid_circuit_measurement():
            if method == "density_matrix":
                raise QazSimError("mid-circuit measurement/reset: use method='statevector'")
            self._run_trajectories(qc, shots, memory, rng, exp)
        elif method == "statevector":
            self._run_statevector(qc, shots, memory, rng, exp)
        else:
            self._run_density_matrix(qc, shots, memory, rng, exp)
        return exp

    # ---------------------------------------------------- statevector method
    def _run_statevector(self, qc, shots, memory, rng, exp):
        n = qc.num_qubits
        psi = np.zeros(2 ** n, dtype=np.complex128)
        psi[0] = 1
        measures = []
        for ins in qc.data:
            if ins.is_gate:
                apply_unitary(psi, ins.to_matrix(), ins.qubits)
            elif ins.name == "measure":
                measures.append((ins.qubits[0], ins.clbits[0]))
            elif ins.name.startswith("save_"):
                self._save(exp, ins, Statevector(psi * np.exp(1j * qc.global_phase)))
        if measures:
            probs = _accelerate.probabilities(psi)
            self._sample(probs, measures, qc, shots, memory, rng, exp)

    # ------------------------------------------------- density matrix method
    def _run_density_matrix(self, qc, shots, memory, rng, exp):
        n = qc.num_qubits
        if n > 15:
            raise QazSimError("density_matrix method is limited to 15 qubits")
        nm = self.options["noise_model"]
        rho = np.zeros(4 ** n, dtype=np.complex128)
        rho[0] = 1
        measures = []
        for ins in qc.data:
            if ins.is_gate:
                apply_unitary_dm(rho, ins.to_matrix(), ins.qubits, n)
                if nm is not None:
                    for err, qs in nm.errors_for(ins.name, ins.qubits):
                        rho = apply_kraus_dm(rho, err.kraus, qs, n)
            elif ins.name == "measure":
                measures.append((ins.qubits[0], ins.clbits[0]))
            elif ins.name.startswith("save_"):
                self._save(exp, ins, DensityMatrix(rho.reshape(2 ** n, 2 ** n).copy()))
        if measures:
            probs = np.clip(np.real(rho.reshape(2 ** n, 2 ** n).diagonal()), 0, None)
            self._sample(probs, measures, qc, shots, memory, rng, exp)

    # ------------------------------------------ mid-circuit measurement/reset
    def _run_trajectories(self, qc, shots, memory, rng, exp):
        n = qc.num_qubits
        nm = self.options["noise_model"]
        values = np.zeros(shots, dtype=np.int64)
        for s in range(shots):
            psi = np.zeros(2 ** n, dtype=np.complex128)
            psi[0] = 1
            creg = 0
            for ins in qc.data:
                if ins.is_gate:
                    apply_unitary(psi, ins.to_matrix(), ins.qubits)
                elif ins.name in ("measure", "reset"):
                    q = ins.qubits[0]
                    idx = np.arange(psi.size)
                    one = (idx >> q) & 1 == 1
                    p1 = float(np.sum(np.abs(psi[one]) ** 2))
                    bit = int(rng.random() < p1)
                    keep = one if bit else ~one
                    psi = np.where(keep, psi, 0)
                    psi /= np.sqrt(p1 if bit else 1 - p1)
                    if ins.name == "measure":
                        rbit = bit
                        if nm is not None and nm.readout_for(q) is not None:
                            pr = nm.readout_for(q).probabilities
                            if rng.random() < pr[bit, 1 - bit]:
                                rbit = 1 - bit
                        c = ins.clbits[0]
                        creg = (creg & ~(1 << c)) | (rbit << c)
                    elif bit:  # reset: flip |1> back to |0>
                        apply_unitary(psi, np.array([[0, 1], [1, 0]], dtype=np.complex128), [q])
                elif ins.name.startswith("save_") and s == 0:
                    self._save(exp, ins, Statevector(psi.copy()))
            values[s] = creg
        counts, _ = _format_counts(values, qc.num_clbits)
        exp["counts"] = counts
        if memory:
            exp["memory"] = [format(int(v), f"0{qc.num_clbits}b") for v in values]

    # ------------------------------------------------------------- sampling
    def _sample(self, probs, measures, qc, shots, memory, rng, exp):
        qubits = [q for q, _ in measures]
        clbits = [c for _, c in measures]
        p = probs / probs.sum()
        idx = rng.choice(p.size, size=shots, p=p)
        bits = np.stack([(idx >> q) & 1 for q in qubits], axis=1)
        bits = _apply_readout(bits, qubits, self.options["noise_model"], rng)
        values = np.zeros(shots, dtype=np.int64)
        for i, c in enumerate(clbits):  # later measurement of same clbit wins
            values = (values & ~(1 << c)) | (bits[:, i].astype(np.int64) << c)
        counts, _ = _format_counts(values, qc.num_clbits)
        exp["counts"] = counts
        if memory:
            exp["memory"] = [format(int(v), f"0{qc.num_clbits}b") for v in values]

    @staticmethod
    def _save(exp, ins, state):
        if ins.name == "save_statevector":
            if isinstance(state, DensityMatrix):
                raise QazSimError("save_statevector is not available with density_matrix; "
                                  "use save_density_matrix")
            exp["data"][ins.label] = state
        elif ins.name == "save_density_matrix":
            exp["data"][ins.label] = (state if isinstance(state, DensityMatrix)
                                      else state.to_density_matrix())
        elif ins.name == "save_probabilities":
            exp["data"][ins.label] = state.probabilities_dict()
