# QazSim

Квантовый симулятор Quantum AI Laboratory. Пользовательский API на Python, вычислительное ядро на Rust, связь через PyO3.

```
qazsim/
├── Cargo.toml              # Rust-ядро (crate qazsim-accelerate)
├── pyproject.toml          # сборка Python-пакета через maturin
├── src/
│   ├── lib.rs              # модуль qazsim._accelerate
│   ├── kernels.rs          # ядра вектора состояния: плотные, диагональные, управляемые (AVX2, rayon)
│   ├── gates.rs            # матрицы вентилей и их производные по параметрам
│   └── program.rs          # компиляция: слияние вентилей, пакетный запуск, adjoint-градиенты
├── python/qazsim/
│   ├── program.py          # Circuit, Param, compile, Program (API 0.2)
│   ├── circuit.py          # QuantumCircuit, Parameter, ParameterVector
│   ├── gates.py            # матрицы вентилей
│   ├── quantum_info.py     # Statevector, DensityMatrix, Operator, state_fidelity
│   ├── simulator.py        # QazSim: run → Job → Result
│   ├── noise.py            # NoiseModel, каналы Крауса, ошибки считывания
│   ├── transpiler.py       # transpile: разложение в базис устройства и оптимизация
│   ├── qasm2.py            # импорт/экспорт OpenQASM 2.0
│   ├── visualization.py    # текстовая отрисовка схем
│   └── devices/gemini.py   # эмулятор SpinQ Gemini Mini Pro
├── tests/                  # 103 теста
├── benchmarks/bench.py     # замеры производительности
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

qc.save_statevector()
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

## Компилируемые программы (0.2)

Схема компилируется один раз в программу `Program`, после чего запускается с любыми значениями параметров, в том числе целым пакетом за один вызов. Всё исполнение идёт в ядре на Rust.

```python
import numpy as np
import qazsim as qz

th = qz.params("θ", 2)                         # θ[0], θ[1]
c = qz.Circuit(2).ry(th[0], 0).cx(0, 1).ry(2 * th[1] + 0.1, 1)
prog = qz.compile(c)

prog.statevector([0.3, 0.5])                   # вектор состояния
prog.probabilities({"θ[0]": 0.3, "θ[1]": 0.5}) # вероятности
prog.sample(1000, [0.3, 0.5], seed=1)          # {'00': ..., '11': ...}

H = {"ZZ": 1.0, "XI": 0.5}                     # наблюдаемая — сумма Паули
prog.expectation(H, [0.3, 0.5])                # <H>
prog.value_and_gradient(H, [0.3, 0.5])         # (<H>, градиент по всем параметрам)

batch = np.random.rand(1000, 2)                # 1000 наборов параметров
prog.expectation(H, batch)                     # массив из 1000 значений, один вызов
```

Что происходит при компиляции:

- **Слияние вентилей.** Соседние вентили объединяются в блоки до 5 кубитов. Модель стоимости решает, выгодно ли слияние: блок, который почти весь вектор состояния не трогает, сливать с плотным вентилем невыгодно.
- **Блоки без свободных параметров** перемножаются один раз при компиляции. Тождественные блоки удаляются, диагональные считаются диагональным ядром.
- **Управляемые кубиты** находятся автоматически, в том числе внутри слитых блоков. Такой блок трогает только ту часть вектора, где управляющие кубиты равны 1: половину для `cx`, четверть для `cp`.
- **Блоки с параметрами** сохраняют структуру и перемножаются при подстановке значений. Это стоит O(вентилей · 4ᵏ), а не O(2ⁿ).

Градиенты считаются сопряжённым методом (adjoint): примерно три прогона схемы на все параметры сразу, а не два прогона на каждый параметр. Пакеты параметров распределяются по потокам целыми схемами.

Параметры вентилей — аффинные выражения: `2 * θ - φ + 0.5`. Ограничение позволяет считать градиент точно по цепному правилу. Схему API 0.1 можно скомпилировать напрямую: `qz.compile(quantum_circuit)`, измерения при этом отбрасываются.

Вентили: `id x y z h s sdg t tdg sx sxdg rx ry rz p u cx cy cz swap iswap cp crx cry crz rxx ryy rzz ccx ccz cswap`, а также произвольный `unitary(matrix, qubits)`.

API 0.1 (`QuantumCircuit` → `QazSim().run`) сохраняется: он нужен для измерений в середине схемы, моделей шума и эмулятора Gemini Mini Pro. Он тоже работает на новых ядрах.

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

## Лицензия

Apache 2.0, см. `LICENSE`.
# qazsim
