# AGENTS.md — правила работы с репозиторием torch2rtl

## Назначение проекта

`torch2rtl` — учебно-исследовательский компилятор небольшого статического
поднабора PyTorch в fixed-point SystemVerilog. Это воспроизводимый
board-independent вертикальный срез, а не готовый промышленный HLS-компилятор и
не законченный FPGA-ускоритель.

Версия пакета в `pyproject.toml` — главный источник номера версии. Текущий
публичный контракт v0.2 описан в `README.md` и
`docs/v0.2_semantic_correctness.md`. Для изменений frontend дополнительно
обязателен `docs/frontend_architecture.md`.

Не использовать календарные или понедельные планы как источник требований.
Направление следующей работы определяется текущей задачей пользователя и
фактическим состоянием кода.

## Формат совместной работы

- Общаться с пользователем по-русски, если он не попросил другой язык.
- Объяснять решения через путь данных, небольшие числовые примеры и конкретные
  файлы проекта. Пользователь лучше знает Python, чем SystemVerilog, цифровую
  схемотехнику и устройство компиляторов.
- В учебной задаче сначала предлагать пользователю написать основную часть
  самостоятельно. Готовую реализацию писать по прямой просьбе «сделай» или
  «напиши за меня».
- Тесты и testbench можно писать по просьбе пользователя, но нужно объяснять,
  какую ошибку они обнаруживают и почему являются достаточной проверкой.
- После пользовательской попытки проверять реальный файл и давать конкретный
  разбор: что верно, где нарушен контракт, как воспроизвести и проверить
  исправление.
- Не растягивать одну идею на серию однотипных упражнений. Предпочитать короткие
  вертикальные срезы: идея → код → запуск → контролируемая ошибка → диагностика.
- Если пользователь просит реализацию, выполнять её полностью, а не превращать
  запрос в обязательное учебное упражнение.

## Фактический pipeline

```text
PyTorch nn.Module в eval-режиме, CPU float32/float64
  -> проверка model/input contract и controlled deepcopy
  -> torch.fx symbolic trace
  -> опциональный fusion точной пары Conv2d -> BatchNorm2d
  -> semantic GraphIR
  -> QuantizedGraph и fixed-point reference
  -> Jinja2 SystemVerilog + testbench + verification vectors
  -> Icarus Verilog или Verilator
  -> generic Yosys prep/stat
  -> report.json + автономная HTML-визуализация
```

Независимая dependency-free C11-реализация в `c_reference/` используется в
тестах как отдельный oracle fixed-point арифметики. Она не является скрытой
стадией обычной команды компиляции.

## Карта репозитория

- `torch2rtl/frontend/pytorch_fx.py` — стабильный публичный facade:
  `parse_model()`, загрузка модели и runtime-integrity checks;
- `torch2rtl/frontend/_pipeline.py` — порядок стадий frontend;
- `torch2rtl/frontend/_model_contract.py`, `_state_guard.py`, `_fusion.py`,
  `_lowering.py`, `_semantics.py` — проверка модели, безопасная копия, fusion,
  lowering и семантические границы;
- `torch2rtl/ir/` — `TensorIR`, операции и `GraphIR`;
- `torch2rtl/quant/fixed_point.py` — конфигурация, квантование и saturation;
- `torch2rtl/quant/reference.py` — float и integer reference, quantized IR и
  проверка ширины аккумулятора;
- `torch2rtl/backend/systemverilog/` — emitter и Jinja2-шаблоны RTL/testbench;
- `torch2rtl/verify/` — векторы, provenance, сравнение, запуск симулятора и
  проверка целостности generated sources;
- `torch2rtl/synth/` — валидация отчёта и generic Yosys flow;
- `torch2rtl/visualization/` — manifest, trace и автономный Hardware Explorer;
- `torch2rtl/cli.py`, `torch2rtl/demo.py` — команды и board-free demo-flow;
- `c_reference/` — независимый C11 fixed-point oracle;
- `examples/` — модели, обучение/компиляция и самостоятельные RTL-упражнения;
- `tests/` — unit, regression, adversarial, release-contract и EDA-тесты;
- `docs/DEMO.md` — короткий сценарий демонстрации;
- `docs/torch2rtl_guide.md` и PDF — расширенное руководство.

## Контракт frontend и GraphIR

- Поддерживается одна последовательная FX-цепочка, не произвольный DAG.
- Поддержаны точные стандартные типы `nn.Linear`, `nn.ReLU`, `nn.Flatten` и
  `nn.Conv2d`; `BatchNorm2d` поддержан только как соседняя fusible-пара после
  `Conv2d`. Финальный явный argmax допускается только как глобальный
  `argmax(dim=None, keepdim=False)`.
- Входы и формы статические. `Conv2d` принимает unbatched CHW, `Linear` — один
  вектор, `Flatten` должен быть полным. Input rank ограничен 64, а input и
  Conv2d output — 1 000 000 элементов.
- `Conv2d` не поддерживает groups/depthwise и dilation; structural/index
  arithmetic должна помещаться в signed 32-bit SystemVerilog `int`.
- Модель должна быть в `eval`, использовать однородный `float32` или `float64`
  и обычные contiguous CPU tensors. Training, autograd и GPU не входят в
  контракт.
- Hooks, subclasses, custom metaclasses/call paths, monkeypatch используемых
  PyTorch API, mutation во время trace, динамический Python control flow и
  другие эффекты, не представленные в GraphIR, должны отклоняться явно через
  `UnsupportedOpError`.
- Исходная модель не должна изменяться ни на успешном, ни на ошибочном пути.
  Trace и semantic probes выполняются на контролируемых копиях.
- Точная пара `Conv2d -> BatchNorm2d` сворачивается в новый `Conv2d` только в
  eval-режиме, с накопленной корректной статистикой и без fan-out. Преобразование
  сохраняется в `graph.metadata` и `report.json`.
- `GraphIR.output` — настоящий PyTorch output. Если модель возвращает логиты,
  семантический GraphIR не получает выдуманный `ArgmaxIR`; `class_id` добавляется
  позже как производный hardware/reference output. Если модель явно возвращает
  допустимый argmax, он остаётся частью GraphIR.
- Любое расширение operator set требует согласованных изменений model contract,
  lowering, IR, float/fixed references, RTL emitter, report/visualization и
  тестов. Нельзя добавлять узел только в одном слое pipeline.

Frontend разбит на ацикличные компоненты и защищён architecture/complexity
ratchet-тестами. Перед его рефакторингом прочитать
`docs/frontend_architecture.md`; не обходить manifest и runtime consistency
checks ради локального прохождения одного теста.

## Fixed-point контракт

Для конфигурации `bits=B`, `frac_bits=F`, `acc_bits=A`:

```text
2 <= B <= 32
0 <= F < B
B < A <= 64
```

Значения по умолчанию: `B=8`, `F=6`, `A=32`, то есть signed Q8.6 с масштабом
64 и целочисленным диапазоном `[-128, 127]`.

```text
quantize: finite float * 2**F -> np.rint (ties-to-even) -> B-bit saturation
MAC:      (bias_q << F) + sum(input_q * weight_q)
requant:  arithmetic MAC >> F
output:   B-bit saturation
argmax:   индекс первого максимума
```

Между fixed-point слоями передаются целые квантизованные значения. Деление на
`2**F` применяется только при деквантовании для человека или float-метрик.
Переполнение аккумулятора не является допустимой wraparound-семантикой:
компилятор статически проверяет каждый product/MAC-prefix, а Python reference
повторяет runtime-проверки.

## RTL, векторы и синтез

- Backend полностью комбинационный: без `clock`, `reset`, `valid` и `ready`.
  Циклы в `always_comb` описывают разворачиваемое параллельное железо.
- Веса встроены как константы в `top.sv`. Сгенерированные `*_weights.mem` и
  `*_bias.mem` являются отчётными артефактами; текущий RTL их не загружает.
- Testbench bit-exact проверяет все logits и `class_id` относительно Python
  fixed-point reference.
- Пользовательские quantized input vectors допустимы только вместе с непустым
  `vector_source`; ожидаемые результаты всегда вычисляет внутренний oracle.
- `report.json`, `vectors.json`, testbench и SHA-256 RTL входят в проверяемый
  build-контракт. Не править generated build вручную как способ «исправить»
  компилятор.
- Yosys выполняет generic `read_verilog -sv`, `prep -top top`, `stat`. Это не
  technology mapping, place-and-route и не доказательство работы на плате.
- Не заявлять latency в тактах, throughput, Fmax, LUT/DSP/BRAM конкретной FPGA
  или board readiness без выбранной платы, toolchain и воспроизводимых данных.

## Примеры

Встроенные demo CLI: `tiny-mlp`, `grid-classifier`, `tiny-conv`, `image-cnn`.

- `tiny_conv` — минимальный Conv2d → ReLU → Flatten → Linear вертикальный срез;
- `tiny_mlp` — небольшой обучаемый MLP;
- `grid_classifier` — более длинная compile-only MLP-цепочка;
- `image_cnn` — train/evaluate/compile пример для маленьких изображений;
- `iris_mlp` — optional пример с зависимостями `pandas` и `scikit-learn`;
- `fixed_point_mac` и `sv_basics` — самостоятельные RTL-примеры с testbench.

Не выбирать случайную прикладную сеть или FPGA-плату только ради видимости
определённости. Board-specific работа начинается после появления требований к
устройству, интерфейсу, latency/throughput, accuracy и ресурсам.

## Правила изменения файлов

- Перед изменениями выполнять `git status --short` и сохранять все
  пользовательские незакоммиченные правки.
- Для поиска использовать `rg`, `rg --files`, для просмотра — `sed` и
  `git diff`.
- Не коммитить, не пушить и не удалять файлы без прямой просьбы пользователя.
- Не редактировать checkpoints, generated RTL/build и PDF как замену изменению
  исходников. Генерируемые артефакты пересобирать штатной командой.
- При изменении публичного поведения синхронно обновлять README, точный контракт,
  demo guide и тесты, которые подтверждают новое поведение.
- Не фиксировать в документации старое число passed-тестов или метрики Yosys как
  вечную истину. Перед утверждением числа повторить соответствующую команду и
  указать окружение.

## Проверка изменений

Из корня репозитория предпочтителен lockfile-driven запуск:

```bash
uv --cache-dir temp/uv-cache run pytest -q
uv --cache-dir temp/uv-cache run torch2rtl --help
uv --cache-dir temp/uv-cache run torch2rtl demo --name tiny-conv --out build/demo
```

Если актуальная `.venv` уже создана, допустим быстрый локальный вариант:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m torch2rtl.cli --help
```

При конфликте pytest с локальными временными каталогами:

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/torch2rtl-pytest
```

Сначала запускать минимально релевантные тесты, затем полный набор, если риск
изменения этого требует. Для затронутых областей полезны:

```bash
.venv/bin/python -m pytest -q tests/test_fx_parser.py
.venv/bin/python -m pytest -q tests/test_c_reference.py
.venv/bin/python -m pytest -q tests/test_sv_generation.py
bash examples/sv_basics/run_tests.sh
```

EDA-этапы проверяются отдельно:

```bash
uv --cache-dir temp/uv-cache run torch2rtl verify build/demo
uv --cache-dir temp/uv-cache run torch2rtl synth build/demo
```

Отсутствие Icarus/Verilator или Yosys допустимо только как явно указанный
`skipped` для объединённого `demo`. Явные `verify` и `synth` должны завершаться
ошибкой, если нужного инструмента нет.

## Начало новой задачи

1. Полностью прочитать этот `AGENTS.md`.
2. Выполнить `git status --short`, ничего не меняя.
3. Прочитать `README.md`, `pyproject.toml` и документы, относящиеся к задаче.
4. Проверить фактический код и соответствующие тесты; не полагаться на старые
   отчёты или память о состоянии репозитория.
5. Сформулировать пользователю текущий контракт и только затем менять код.
6. После изменений показать diff, выполнить достаточную проверку и честно
   отделить подтверждённое от непроверенного.
