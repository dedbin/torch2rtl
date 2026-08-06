# Iris MLP: обучение → Q8.4 → RTL → generic Yosys

Этот пример воспроизводит полный board-independent маршрут для классического
Iris dataset:

```text
sklearn Iris + pandas
  -> stratified train/validation/test
  -> PyTorch IrisMLP(4 -> 8 -> 3)
  -> checkpoint
  -> GraphIR
  -> signed Q8.4 fixed-point reference
  -> SystemVerilog + 30 настоящих test vectors
  -> Icarus Verilog / Verilator
  -> generic Yosys prep/stat
```

Dataset входит в scikit-learn и из сети не загружается. FPGA-flow здесь
намеренно отсутствует. Числа Yosys ниже относятся к generic cells, а не к LUT,
DSP, BRAM или ресурсам конкретной FPGA.

## Воспроизведение с нуля

Все команды выполняются из корня репозитория. Python должен быть версии 3.12.

```bash
uv --cache-dir temp/uv-cache sync --dev --extra iris --frozen

uv --cache-dir temp/uv-cache run --frozen --extra iris \
  python examples/iris_mlp/train.py

uv --cache-dir temp/uv-cache run --frozen --extra iris \
  python examples/iris_mlp/compile.py

uv --cache-dir temp/uv-cache run --frozen \
  torch2rtl verify build/iris_mlp/rtl_q8_4

uv --cache-dir temp/uv-cache run --frozen \
  torch2rtl synth build/iris_mlp/rtl_q8_4
```

Для `verify` нужен либо `iverilog` вместе с `vvp`, либо `verilator`. Для
последней команды нужен `yosys`. Отсутствующий EDA-инструмент не загружается
автоматически: соответствующая явная команда завершится с ненулевым кодом и
понятным сообщением.

## Что фиксировано

- seed: `42`;
- split: train `90`, validation `30`, test `30`;
- каждый split содержит одинаковое число samples трёх классов;
- min-max normalization в `[-1, 1]` fit только по train;
- модель: `Linear(4, 8) -> ReLU -> Linear(8, 3)`;
- обучаемых параметров: `67`;
- loss: `CrossEntropyLoss`;
- optimizer: Adam, `lr=0.01`;
- максимум `300` эпох, сохраняется минимум validation loss;
- RTL-формат: `bits=8`, `frac_bits=4`, `acc_bits=32`.

При проверенной среде и seed 42 test accuracy равна `93.33%`. Regression-тест
требует не менее `90%`, чтобы не привязывать корректность примера к мелким
различиям версий PyTorch/pandas/scikit-learn.

`train.py` сохраняет в `build/iris_mlp/iris_mlp.pt` веса, train-only
normalization, имена признаков/классов, размеры модели, seed, индексы всех трёх
split, best epoch и best validation loss. Затем checkpoint загружается через:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

Скрипт проверяет новый экземпляр модели, пустые missing/unexpected keys, exact
совпадение параметров, нормализованных test inputs, logits и predicted classes.

`compile.py` заново восстанавливает split и normalization только из bundled Iris
dataset и checkpoint, а затем проверяет все 30 test samples:

- PyTorch float и GraphIR float имеют logits shape `(30, 3)` и одинаковые классы;
- quantized inputs имеют shape `(30, 4)`;
- Q8.4 logits имеют shape `(30, 3)`, classes — `(30,)`;
- PyTorch и Q8.4 выбирают один класс на всех 30 samples;
- значения лежат в signed int8;
- saturation на выходах `fc1` и `fc2` отсутствует.

## Пользовательские verification-векторы

Iris test inputs передаются генератору напрямую, без замены файлов после
компиляции:

```python
emit_systemverilog(
    graph=graph,
    cfg=cfg,
    out_dir=out_dir,
    input_vectors=quantized_test_inputs,
    vector_source="iris_test_split",
)
```

Пользователь задаёт только quantized inputs и provenance. `torch2rtl` сам
вычисляет expected fixed-point logits/classes через `QuantizedGraph` и одним
вызовом создаёт согласованные vector-файлы, `vectors.json`, `report.json`,
`visualization.json` и testbench. Передать готовый oracle API не позволяет.

Старый flow остаётся доступен:

```python
emit_systemverilog(
    graph=graph,
    cfg=cfg,
    out_dir=out_dir,
    vector_count=16,
    seed=0,
)
```

`vector_count`/`seed` и `input_vectors` конфликтуют явно и не игнорируются.

## Артефакты

После компиляции каталог `build/iris_mlp/rtl_q8_4/` сразу готов для `verify` и
`synth`. В нём находятся `top.sv`, вспомогательные RTL-модули, `tb_top.sv`, три
vector-файла, `vectors.json`, `report.json`, `visualization.json` и автономный
`visualization.html`. Никакой из этих файлов вручную исправлять не нужно.

Ограничения совпадают с текущим MVP: одна последовательная FX-цепочка,
статические формы, полностью комбинационный RTL, встроенные в `top.sv` веса,
нет clock/reset/valid/ready и нет target-specific FPGA synthesis/place-and-route.
