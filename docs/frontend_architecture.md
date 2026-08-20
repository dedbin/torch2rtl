# Архитектура frontend после behaviour-preserving refactor

Frontend по-прежнему реализует тот же публичный контракт `parse_model`, но его
внутренние причины изменения разделены на ацикличные компоненты. Стрелка ниже
означает «модуль слева импортирует модуль справа»:

```text
pytorch_fx -> _errors, _pipeline
           -> public API, loader, runtime consistency checks,
              explicit component manifest and immutable public wrapper

_state_guard    -> _errors
_model_contract -> _errors, _state_guard
_fusion         -> _errors, _state_guard
_lowering       -> _errors, IR
_semantics      -> _errors, _state_guard, IR, quant.reference
_pipeline       -> _errors, _state_guard, _model_contract,
                   _fusion, _lowering, _semantics, IR
```

Внутренние модули не импортируют `pytorch_fx` или
`torch2rtl.frontend.__init__`. Тест архитектуры разбирает обе формы импорта —
`import ...` и `from ... import ...`, включая относительные импорты — и
сопоставляет найденные рёбра с этой точной схемой. `pytorch_fx.py` формирует
явный статический manifest: для каждого компонента фиксируются точный
module/namespace, функции в месте определения и их состояние, реальные
межкомпонентные bindings и фактические consumers в facade. Динамический
registry или механизм plugins для операторов не используется.

## Неизменяемые compiler invariants

- Порядок pipeline фиксирован: input shape → model contract → controlled copy
  → FX trace → Conv/BatchNorm fusion → GraphIR lowering → условная Boundary A
  → обязательная Boundary B.
- Boundary A выполняется только когда fusion действительно изменила traced
  graph; она сохраняет прежние допуски float32/float64 и отдельно проверяет
  итоговый класс. Boundary B выполняется всегда и остаётся exact
  `array_equal`.
- Concrete probes выполняют пользовательский `forward()` с отключёнными
  градиентами. Исходный grad mode восстанавливается до немедленной проверки
  согласованности runtime-компонентов.
- Source model не меняется ни на успешном, ни на ошибочном пути.
- Public identity, signature, exception metadata/pickle contract и GraphIR
  schema/имена/параметры/metadata не изменяются.

## Границы проверок согласованности

Проверки поддерживают согласованность собственного compiler runtime в одном
Python-процессе. После исходного импорта обнаруживаются изменения объявленных
bindings, состояния compiler-owned функций и классов, а также используемых
runtime dependencies. После FX trace и concrete probes согласованность
проверяется до следующего обращения compiler к module globals.

Это не sandbox для произвольного Python. Вне контракта остаются изменения до
canonical import, native-memory/`ctypes`, debugger/frame manipulation,
конкурентный TOCTOU из другого потока, изменённый checkout и изоляция выполнения
произвольного model file. Когда нужна такая изоляция, компилятор следует
запускать в отдельном процессе.

## Полнота component manifest

Manifest задаётся в production-коде явными неизменяемыми tuples. Отдельный
статический regression-тест начинает обход с `_pipeline._parse_model_impl`,
рекурсивно просматривает вложенные code objects и проверяет каждый фактический
Python bytecode lookup `LOAD_GLOBAL`/`LOAD_FROM_DICT_OR_GLOBALS`. Для каждого
lookup проверяется не только имя, но и identity ожидаемого объекта в namespace
места определения. Runtime dependencies самого facade проверяются отдельными
facade/integrity regressions.

Статический обход намеренно ограничен реально достижимым Python bytecode. Он не
предсказывает имена, собранные динамически для `getattr`, и не раскрывает
внутренние зависимости native или внешнего кода. При появлении такого lookup
его контракт нужно добавить как отдельный явный regression.

## Complexity ratchet

Новые и обычные component functions ограничены 75 LOC и приблизительной
цикломатической сложностью 15. Сохранённые крупные policy/translation функции
зафиксированы тестом как явные исключения без права роста:

- `_fusion._fuse_conv_batchnorm_eval` и
  `_fusion._validate_conv_batchnorm_pair`;
- `_lowering.lower_fx_graph` и `_lowering._lower_conv2d_module`;
- `_model_contract._validate_model_semantics` и
  `_model_contract._validate_reachable_python_function`;
- `_pipeline._parse_model_impl`;
- `_semantics._validate_fusion_boundary`;
- `_state_guard._trusted_named_tensors` и `_state_guard._freeze_state_value`.

Исключения отражают связные policy blocks, перенесённые без semantic rewrite.
Их дальнейшее увеличение запрещено ratchet-тестом; последующее уменьшение не
требует сохранять прежний лимит.

Отдельный non-growth ratchet удерживает facade `pytorch_fx.py` не больше 2301
строки. Все новые функции в нём ограничены 75 LOC; четыре уже существующих
крупных функции имеют собственные верхние границы: 76 LOC для
`_definition_fingerprint`, 215 для `_validate_framework_integrity_impl`, 261
для `_make_integrity_validator` и 138 для `_make_parse_model`. Уменьшение файла
или функции разрешено. Эти метрики не требуют искусственно дробить связную
логику только ради меньшего числа строк.
