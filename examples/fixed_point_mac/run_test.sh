#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${EXAMPLE_DIR}/../.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/torch2rtl-fixed-point-mac"

PYTHON="${REPO_ROOT}/.venv/bin/python"
VECTORS_FILE="${BUILD_DIR}/vectors.txt"
SIMULATION="${BUILD_DIR}/fixed_point_mac.vvp"

command -v iverilog >/dev/null 2>&1 || {
    echo "ERROR: iverilog was not found in PATH" >&2
    exit 127
}

command -v vvp >/dev/null 2>&1 || {
    echo "ERROR: vvp was not found in PATH" >&2
    exit 127
}

command -v yosys >/dev/null 2>&1 || {
    echo "ERROR: yosys was not found in PATH" >&2
    exit 127
}

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: project Python was not found: ${PYTHON}" >&2
    exit 127
fi

mkdir -p "${BUILD_DIR}"

echo "==> Generating vectors with Python reference"
"${PYTHON}" -B "${EXAMPLE_DIR}/generate_vectors.py" > "${VECTORS_FILE}"

echo "==> Compiling fixed-point MAC"
iverilog \
    -g2012 \
    -Wall \
    -s fixed_point_mac_tb \
    -o "${SIMULATION}" \
    "${EXAMPLE_DIR}/fixed_point_mac.sv" \
    "${EXAMPLE_DIR}/fixed_point_mac_tb.sv"

echo "==> Comparing RTL with Python reference"
vvp "${SIMULATION}" "+VECTORS=${VECTORS_FILE}"

echo "==> Checking synthesizeability with Yosys"
yosys -q -p \
    "read_verilog -sv ${EXAMPLE_DIR}/fixed_point_mac.sv; \
     prep -top fixed_point_mac; \
     check"

echo "ALL FIXED-POINT MAC TESTS PASSED"