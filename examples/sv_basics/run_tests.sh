#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/torch2rtl-sv-basics"
PASSED=0
TOTAL=7

command -v iverilog >/dev/null 2>&1 || {
    echo "ERROR: iverilog was not found in PATH" >&2
    exit 127
}
command -v vvp >/dev/null 2>&1 || {
    echo "ERROR: vvp was not found in PATH" >&2
    exit 127
}

mkdir -p "$BUILD_DIR"

run_test() {
    local name="$1"
    local top="$2"
    shift 2

    local executable="$BUILD_DIR/${name}.vvp"
    local simulation_output

    echo "==> ${name}"
    if ! iverilog -g2012 -s "$top" -o "$executable" "$@"; then
        echo "FAIL: ${name} compilation failed" >&2
        return 1
    fi

    if ! simulation_output="$(cd "$BUILD_DIR" && vvp "$executable" 2>&1)"; then
        printf '%s\n' "$simulation_output"
        echo "FAIL: ${name} simulation process failed" >&2
        return 1
    fi
    printf '%s\n' "$simulation_output"

    case "$simulation_output" in
        *FAIL*|*ERROR*)
            echo "FAIL: ${name} reported a self-checking error" >&2
            return 1
            ;;
        *PASS*|*pass*)
            ;;
        *)
            echo "FAIL: ${name} reported no success marker" >&2
            return 1
            ;;
    esac

    PASSED=$((PASSED + 1))
    echo "<== PASS: ${name}"
    echo
}

run_test \
    full_adder \
    summator_tb \
    "$ROOT_DIR/summator.v" \
    "$ROOT_DIR/summator_tb.sv"

run_test \
    adder_4bit \
    summator_4b_tb \
    "$ROOT_DIR/summator.v" \
    "$ROOT_DIR/4b_summator.v" \
    "$ROOT_DIR/summator_4b_tb.v"

run_test \
    signed_multiplier \
    signed_multiplier_tb \
    "$ROOT_DIR/signed_multiplier.sv" \
    "$ROOT_DIR/signed_multiplier_tb.sv"

run_test \
    signed_register \
    signed_register_tb \
    "$ROOT_DIR/signed_register.sv" \
    "$ROOT_DIR/signed_register_tb.sv"

run_test \
    multiplexer_2_1 \
    multiplexer_2_1_tb \
    "$ROOT_DIR/multiplexer_2_1.v" \
    "$ROOT_DIR/multiplexer_2_1_tb.sv"

run_test \
    signed_comparator \
    signed_comparator_tb \
    "$ROOT_DIR/signed_comparator.sv" \
    "$ROOT_DIR/signed_comparator_tb.sv"

run_test \
    relu_signed \
    relu_signed_tb \
    "$ROOT_DIR/relu_signed.sv" \
    "$ROOT_DIR/relu_signed_tb.sv"

echo "ALL SYSTEMVERILOG BASIC TESTS PASSED (${PASSED}/${TOTAL})"
