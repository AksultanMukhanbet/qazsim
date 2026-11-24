# QazSim

Квантовый симулятор Quantum AI Laboratory КазНУ. Устроен как Qiskit: пользовательский API на Python, вычислительное ядро на Rust, связь через PyO3. Qiskit и Qiskit Aer пакет не использует.

```
qazsim/
├── Cargo.toml              # Rust-ядро (crate qazsim-accelerate)
├── pyproject.toml          # сборка Python-пакета через maturin
├── src/lib.rs              # ядро: применение вентилей, вероятности (rayon, многопоточно)
├── python/qazsim/
│   ├── circuit.py          # QuantumCircuit, Parameter, ParameterVector
│   ├── gates.py            # матрицы вентилей
│   ├── quantum_info.py     # Statevector, DensityMatrix, Operator, state_fidelity
│   ├── simulator.py        # QazSim: run → Job → Result
│   ├── noise.py            # NoiseModel, каналы Крауса, ошибки считывания
│   ├── transpiler.py       # transpile: разложение в базис устройства и оптимизация
│   ├── qasm2.py            # импорт/экспорт OpenQASM 2.0
│   ├── visualization.py    # текстовая отрисовка схем
│   └── devices/gemini.py   # эмулятор SpinQ Gemini Mini Pro
├── tests/test_qazsim.py    # 59 тестов
└── examples/
```

## Сборка

Нужны Python ≥ 3.9 и Rust (https://rustup.rs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release        # собрать ядро и установить пакет в окружение
pytest -q                        # тесты
```

Готовый wheel для установки на другие машины:

```bash
maturin build --release          # wheel появится в target/wheels/
pip install target/wheels/qazsim-*.whl
```

Число потоков ядра задаётся переменной `RAYON_NUM_THREADS`.

## Использование

API повторяет Qiskit, поэтому код переносится почти без изменений.

```python
from qazsim import QuantumCircuit, QazSim

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
print(qc)

result = QazSim().run(qc, shots=1000).result()
print(result.get_counts())          # {'00': 505, '11': 495}
```

Вектор состояния и матрица плотности:

```python
from qazsim import Statevector, DensityMatrix

sv = Statevector.from_instruction(qc_without_measure)
sv.probabilities_dict()
sv.expectation_value({"ZZ": 1.0, "XX": 0.5})

qc.save_statevector()      # как в Qiskit Aer
QazSim().run(qc).result().get_statevector()
```

Параметризованные схемы:

```python
from qazsim import Parameter
theta = Parameter("θ")
qc = QuantumCircuit(1, 1)
qc.ry(theta, 0)
qc.measure(0, 0)
QazSim().run([qc, qc], parameter_binds=[{theta: 0.1}, {theta: 0.2}])
```

Шум:

```python
from qazsim import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx", "ry"])
nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.03, 0.97]]))
QazSim(noise_model=nm).run(qc)     # метод density_matrix выбирается автоматически
```

## SpinQ Gemini Mini Pro

`GeminiMiniPro` ведёт себя как одно устройство лаборатории:
- 2 кубита, нативные вентили `id x y z h rx ry rz cx cz`;
- схема автоматически транспилируется в эти вентили;
- схема отклоняется, если в ней больше 2 кубитов или она превышает лимит 80/40 операций;
- результат — вероятности и матрица плотности (ЯМР работает с ансамблем молекул);
- шум: T1/T2-релаксация на время вентиля + деполяризация.

```python
from qazsim import GeminiMiniPro, GeminiCalibration, gemini_fleet

dev = GeminiMiniPro()                                  # калибровка по умолчанию
dev = GeminiMiniPro(GeminiCalibration(t1=4.2, t2=0.28), name="gemini_3")
fleet = gemini_fleet(7)                                # gemini_1 ... gemini_7

qc.save_density_matrix()
rho = dev.run(qc).result().get_density_matrix()
```

Параметры шума по умолчанию взяты из статьи об исходном SpinQ Gemini ([arXiv:2101.10017](https://arxiv.org/abs/2101.10017)): T1 ≈ 4 с, T2 ≈ 0,3 с, однокубитный вентиль ≈ 20 мкс, CNOT ≈ 800 мкс, точность вентилей ≈ 0,99 и 0,98. **Замените их калибровкой ваших устройств.** Набор нативных вентилей Mini Pro и точный смысл лимита «80/40» нужно сверить с документацией SpinQ. Сейчас одна двухкубитная операция считается за две однокубитные.

## Соглашения (как в Qiskit)

- Порядок кубитов little-endian: кубит 0 — младший бит, в строке `'01'` правый символ — кубит 0.
- `cx(control, target)`; матрица k-кубитного вентиля на кубитах `[q0, q1, ...]`: бит b индекса соответствует `q_b`.
- Ключи `get_counts()`: классический бит 0 справа.

## Производительность

Замер на 2 ядрах CPU: схема из 5 слоёв `ry` + цепочки `cx`. Результаты совпадают с Qiskit Aer до 1e-9.

| Кубиты | Вентилей | QazSim | Qiskit Aer |
|---|---|---|---|
| 20 | 195 | 0,34 с | 0,21–0,35 с |
| 24 | 235 | 7,5 с | 5,2–5,8 с |
| 26 | 255 | 32,8 с | 24–26 с |

Память вектора состояния: 16 байт × 2ⁿ (30 кубитов ≈ 16 ГБ, 34 ≈ 256 ГБ). Метод `density_matrix` ограничен 15 кубитами.

## Что дальше

1. **Слияние вентилей (gate fusion).** Склеивать соседние вентили в одну матрицу перед запуском: главный источник отставания от Aer.
2. **Отдельные ядра для диагональных вентилей** (rz, cz, cp, rzz) — без умножения полной матрицы.
3. **GPU:** ядро на CUDA (через крейт `cudarc`) или CuPy на стороне Python.
4. **MPI:** распределение вектора состояния по узлам суперкомпьютера КазНУ.
5. **Маршрутизация кубитов в транспиляторе** — для устройств, где не все кубиты связаны. Для Gemini Mini Pro она не нужна.
6. **OpenQASM 3** и импорт схем из PennyLane и Cirq.
7. **REST API** для подключения QazSim к облачной платформе.

## Лицензия

Apache 2.0, см. `LICENSE`.
# qazsim
