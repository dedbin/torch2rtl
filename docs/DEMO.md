# Сценарий демонстрации torch2rtl

Этот сценарий рассчитан примерно на 5-10 минут. Основная команда показа:

```bash
uv --cache-dir temp/uv-cache run torch2rtl demo --name tiny-conv --out build/demo
```

## 1. Что представляет собой проект

`torch2rtl` - учебный маршрут от небольшой модели PyTorch до читаемого SystemVerilog. Он не заменяет промышленный HLS-инструмент: цель проекта - показать цепочку преобразований, проверочные данные, отчет и визуализацию так, чтобы каждый артефакт можно было открыть и объяснить.

## 2. Исходная модель PyTorch

Покажите `examples/tiny_conv/model.py`. Модель принимает тензор формы `(1, 3, 3)` и проходит через:

```text
Conv2d -> ReLU -> Flatten -> Linear -> Argmax
```

Веса заданы вручную, поэтому пример воспроизводимый и не требует обучения.

## 3. Разбор через torch.fx

Откройте `torch2rtl/frontend/pytorch_fx.py`. Функция `parse_model` вызывает `torch.fx.symbolic_trace`, идет по узлам графа и поддерживает только явные операции: `Linear`, `ReLU`, `Flatten`, `Conv2d` и `argmax`. Неподдержанный узел приводит к `UnsupportedOpError`.

## 4. Внутреннее представление GraphIR

Покажите `torch2rtl/ir/graph.py`, `torch2rtl/ir/ops.py` и `torch2rtl/ir/tensor.py`. `GraphIR` хранит входной тензор, выходной тензор и последовательность операций. Для `tiny-conv` в `report.json` должны быть операции:

```text
Conv2dIR, ReluIR, FlattenIR, LinearIR, ArgmaxIR
```

## 5. Фиксированная точка

Покажите `torch2rtl/quant/fixed_point.py`. По умолчанию используется 8 бит и 6 дробных битов: значение умножается на `2**6`, округляется и ограничивается диапазоном от `-128` до `127`. Затем покажите `torch2rtl/quant/reference.py`, где эталонная модель на Python повторяет целочисленную арифметику.

## 6. Генерация SystemVerilog

Откройте `torch2rtl/backend/systemverilog/emit.py` и шаблоны в `torch2rtl/backend/systemverilog/templates/`. После demo-команды покажите:

```text
build/demo/top.sv
build/demo/conv2d_comb.sv
build/demo/linear_comb.sv
build/demo/relu.sv
build/demo/argmax.sv
```

`top.sv` соединяет сгенерированные блоки. Текущая схема комбинационная: тактового входа и потокового интерфейса пока нет.

## 7. Проверочные данные и проверочный модуль

Покажите файлы:

```text
build/demo/input_vectors.txt
build/demo/expected_classes.txt
build/demo/tb_top.sv
```

`input_vectors.txt` содержит квантованные входы, `expected_classes.txt` - ожидаемые классы по фиксированно-точечной эталонной модели, а `tb_top.sv` читает эти файлы и сравнивает выход `class_id`.

## 8. Моделирование

Команда demo сама вызывает моделирование. Если установлены `iverilog` и `vvp`, используется Icarus Verilog; если их нет, но есть `verilator`, используется Verilator. Если симулятор не найден, проект не падает, а выводит:

```text
verification: skipped (simulator not found: install Icarus Verilog or Verilator)
```

Это честный пропуск необязательного этапа, а не скрытая ошибка компиляции.

## 9. Синтез через Yosys

Команда demo также пытается запустить Yosys. Если `yosys` установлен, создается `build/demo/yosys.log`, а метрики попадают в `report.json` и `visualization.html`. Если Yosys отсутствует, вывод должен быть таким:

```text
synthesis: skipped (yosys not found: install Yosys to run synthesis)
```

## 10. Что показывать в visualization.html

Откройте `build/demo/visualization.html`. На странице удобно показать:

- блоки входа, `Conv2d`, `ReLU`, `Flatten`, `Linear`, `Argmax` и выхода;
- связи между сигналами;
- файлы весов и смещений;
- выбранный проверочный вектор;
- статус моделирования и синтеза;
- ссылки на созданные артефакты.

## 11. Честные ограничения

Проговорите ограничения прямо:

- поддержан небольшой поднабор PyTorch;
- входные формы должны быть статическими;
- пакетная размерность для `Conv2d` не поддержана;
- нет обучения, autograd и GPU-логики в компиляторе;
- нет нормализации, attention и больших архитектур;
- RTL сейчас в основном комбинационный;
- Yosys-запуск является простым `read_verilog`, `prep -top top`, `stat`, а не полным маршрутом до платы.

## 12. Быстрый порядок показа

1. Открыть `examples/tiny_conv/model.py`.
2. Запустить `uv --cache-dir temp/uv-cache run torch2rtl demo --name tiny-conv --out build/demo`.
3. Прочитать итоговую сводку команды.
4. Открыть `build/demo/report.json`.
5. Открыть `build/demo/top.sv` и `build/demo/tb_top.sv`.
6. Открыть `build/demo/visualization.html`.
7. Завершить ограничениями и направлениями развития.
