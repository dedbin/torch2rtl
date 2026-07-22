`timescale 1ns / 1ps

module relu_signed_tb;

  localparam int DATA_BITS = 8;

  logic signed [DATA_BITS-1:0] in_data;
  logic signed [DATA_BITS-1:0] out_data;

  relu_signed dut (
    .in_data(in_data),
    .out_data(out_data)
  );

  task automatic check_value(input logic signed [DATA_BITS-1:0] value,
                              input logic signed [DATA_BITS-1:0] expected);
    begin
      in_data = value;
      #1;
      if (out_data !== expected) begin
        $display("[ERROR] in_data=%0d expected=%0d got=%0d", value, expected, out_data);
        $fatal(1);
      end else begin
        $display("[OK] in_data=%0d out_data=%0d", value, out_data);
      end
    end
  endtask

  initial begin
    $display("Starting relu_signed testbench...");

    check_value(-8'sd5, 8'sd0);
    check_value(-8'sd1, 8'sd0);
    check_value(8'sd0, 8'sd0);
    check_value(8'sd1, 8'sd1);
    check_value(8'sd42, 8'sd42);
    check_value(8'sd127, 8'sd127);
    check_value(-8'sd128, 8'sd0);

    $display("All relu_signed tests passed.");
    $finish;
  end

endmodule
