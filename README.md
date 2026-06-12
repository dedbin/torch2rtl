# torch2rtl

`torch2rtl` is a small, honest compiler flow for a narrow PyTorch subset:

```text
PyTorch model -> torch.fx graph -> internal IR -> fixed-point quantization
-> SystemVerilog RTL -> testbench -> simulation verification -> optional Yosys report
```

The project is not a universal PyTorch-to-RTL converter. It intentionally supports only a tiny set of operations so the generated RTL, fixed-point reference, and verification artifacts can be inspected and trusted.

## What It Supports

MVP v0.1 supports:

- `torch.nn.Linear`
- `torch.nn.ReLU`
- `torch.nn.Flatten`
- final `argmax`
- signed fixed-point int8 quantization by default
- combinational SystemVerilog modules
- Python fixed-point reference inference
- optional Icarus Verilog, Verilator, and Yosys integration when installed

The first supported model shape is:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)
```

## What It Does Not Do

`torch2rtl` does not try to support arbitrary PyTorch models, dynamic control flow, tensors with unknown shapes, training graphs, GPUs, autograd, convolutions, normalization, attention, or production timing closure. The MVP backend is combinational and intended for clarity, not performance.

Unsupported FX nodes raise `UnsupportedOpError` with the operation that failed.

## Install

Recommended with `uv`:

```bash
uv --cache-dir temp/uv-cache sync --dev
```

The explicit cache directory is useful on restricted Windows setups where the default `uv` cache may be outside the writable workspace.

## Compile

```bash
uv --cache-dir temp/uv-cache run torch2rtl compile examples/tiny_mlp/model.py \
  --input-shape 16 \
  --bits 8 \
  --frac-bits 6 \
  --out build
```

This creates:

- `build/top.sv`
- `build/linear_comb.sv`
- `build/relu.sv`
- `build/argmax.sv`
- `build/tb_top.sv`
- `build/input_vectors.txt`
- `build/expected_classes.txt`
- layer weight and bias `.mem` files
- `build/report.json`
- `build/visualization.json`
- `build/visualization.html`

Откройте `build/visualization.html`, чтобы посмотреть сгенерированную схему как
интерактивную микросхему. Там показаны RTL-блоки, сигналы пути данных, файлы
памяти с параметрами, статус инструментов и эталонный проход fixed-point
модели для одного тестового вектора.

Чтобы перерисовать схему после `verify` или `synth`, либо выбрать другой
вектор:

```bash
uv --cache-dir temp/uv-cache run torch2rtl visualize build --vector-index 0
```

## Verify

```bash
uv --cache-dir temp/uv-cache run torch2rtl verify build
```

If `iverilog` + `vvp` are available, the command builds and runs the generated testbench. If Icarus is not available but `verilator` is present, it attempts a Verilator binary run. If no simulator is installed, it reports `simulator not found` and exits cleanly.

## Synthesize

```bash
uv --cache-dir temp/uv-cache run torch2rtl synth build
```

If `yosys` is installed, the command runs a simple synthesis/stat pass and writes `build/yosys.log`. If not, it reports `yosys not found`.

## Tiny MLP Example

The example task classifies a 16-element vector by the block of four elements with the largest sum:

- class 0: elements `0..3`
- class 1: elements `4..7`
- class 2: elements `8..11`
- class 3: elements `12..15`

Train the example model:

```bash
uv --cache-dir temp/uv-cache run python examples/tiny_mlp/train.py --epochs 30
```

Compile it from Python:

```bash
uv --cache-dir temp/uv-cache run python examples/tiny_mlp/compile.py --out build
```

## Architecture

- `frontend/pytorch_fx.py`: traces a PyTorch module with `torch.fx.symbolic_trace` and converts supported nodes to IR.
- `ir/`: PyTorch-independent dataclasses for tensors, ops, and graphs.
- `quant/fixed_point.py`: fixed-point config, quantization, dequantization, and saturation helpers.
- `quant/reference.py`: fixed-point inference without PyTorch; this is the RTL oracle.
- `backend/systemverilog/`: Jinja2 templates and emitter for readable combinational SystemVerilog.
- `verify/`: random vector generation, simulator discovery, and comparison helpers.
- `synth/`: optional Yosys integration and report update helpers.

## Tests

```bash
uv --cache-dir temp/uv-cache run pytest -q
```

Tests cover quantization saturation, manual fixed-point arithmetic, FX parsing, generated SystemVerilog, and stable reference classes.

## Roadmap

- v0.1: Linear/ReLU/Flatten/Argmax
- v0.2: Conv1d/Conv2d
- v0.3: sequential MAC backend
- v0.4: ONNX frontend
- v0.5: streaming interface / AXI-like interface
