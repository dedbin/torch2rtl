`timescale 1ns / 1ps

module signed_comparator_tb;

  localparam int DATA_BITS = 8;

  logic signed [DATA_BITS-1:0] a;
  logic signed [DATA_BITS-1:0] b;
  logic gt;

  signed_comparator #(
    .DATA_BITS(DATA_BITS)
  ) dut (
    .a(a),
    .b(b),
    .gt(gt)
  );

    task automatic check_value(input logic signed [DATA_BITS-1:0] a_value,
                                input logic signed [DATA_BITS-1:0] b_value,
                                input logic expected);
        begin
        a = a_value;
        b = b_value;
        #1;
        if (gt !== expected) begin
            $display("[ERROR] a=%0d b=%0d expected=%0b got=%0b", a_value, b_value, expected, gt);
            $fatal(1);
        end else begin
            $display("[OK] a=%0d b=%0d gt=%0b", a_value, b_value, gt);
        end
        end
    endtask

    initial begin
        $display("Starting signed_comparator testbench...");
        check_value(-8'sd5, -8'sd10, 1'b1);
        check_value(-8'sd10, -8'sd5, 1'b0);
        check_value(8'sd0, 8'sd0, 1'b0);
        check_value(8'sd5, 8'sd10, 1'b0);
        check_value(8'sd10, 8'sd5, 1'b1);
        check_value( 8'sd1, -8'sd1, 1'b1);
        check_value(-8'sd1,  8'sd1, 1'b0);
        check_value(8'sd127, 8'sh80, 1'b1);
        check_value(8'sh80, 8'sd127, 1'b0);

        $display("All signed_comparator tests passed.");
        $finish;
    end

endmodule