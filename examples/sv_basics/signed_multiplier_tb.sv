`timescale 1ns / 1ps

module signed_multiplier_tb;

  localparam int DATA_BITS = 8;

  logic signed [DATA_BITS-1:0] a;
  logic signed [DATA_BITS-1:0] b;
  logic signed [2*DATA_BITS-1:0] product;

  signed_multiplier #(
    .DATA_BITS(DATA_BITS)
  ) dut (
    .a(a),
    .b(b),
    .product(product)
  );

    task automatic check_value(input logic signed [DATA_BITS-1:0] a_value,
                                input logic signed [DATA_BITS-1:0] b_value,
                                input logic signed [2*DATA_BITS-1:0] expected);
        begin
        a = a_value;
        b = b_value;
        #1;
        if (product !== expected) begin
            $display("[ERROR] a=%0d b=%0d expected=%0d got=%0d", a_value, b_value, expected, product);
            $fatal(1);
        end else begin
            $display("[OK] a=%0d b=%0d product=%0d", a_value, b_value, product);
        end
        end
    endtask

    initial begin
        $display("Starting signed_multiplier testbench...");
        check_value(-8'sd5,  -8'sd10,  16'sd50);
        check_value(-8'sd10, -8'sd5,   16'sd50);
        check_value( 8'sd0,   8'sd0,   16'sd0);
        check_value( 8'sd5,   8'sd10,  16'sd50);
        check_value( 8'sd10,  8'sd5,   16'sd50);
        check_value( 8'sd1,  -8'sd1,  -16'sd1);
        check_value(-8'sd1,   8'sd1,  -16'sd1);

        check_value(8'sd127, 8'sh80,  -16'sd16256);
        check_value(8'sh80, 8'sd127, -16'sd16256);

        check_value(8'sd127, 8'sd127, 16'sd16129);
        check_value(8'sh80,  8'sh80,  16'sd16384);

        $display("All signed_multiplier tests passed.");
        $finish;
    end

endmodule