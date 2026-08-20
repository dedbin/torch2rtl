# SystemVerilog basics

This directory contains reproducible foundational RTL exercises used before
building larger torch2rtl datapaths.

The examples cover:

- one-bit full adder (`summator`);
- four-bit ripple-carry adder (`summator_4b`);
- signed multiplier;
- signed register with asynchronous active-high reset;
- 2:1 multiplexer;
- signed comparator;
- signed ReLU.

Every module has a self-checking testbench. From the repository root, run the
complete set with Icarus Verilog:

```bash
bash examples/sv_basics/run_tests.sh
```

The runner compiles simulation executables and writes any VCD files under
`${TMPDIR:-/tmp}/torch2rtl-sv-basics`; it does not write generated artifacts
into the source tree.

These checks establish RTL functional behavior only. They are not FPGA
technology mapping and do not report LUT, DSP, BRAM, timing, or power for a
specific device.
