`timescale 1ns/1ps

module signed_register_tb;

    localparam int DATA_BITS = 8;

    logic clk;
    logic reset;
    logic signed [DATA_BITS-1:0] d;
    logic signed [DATA_BITS-1:0] q;

    signed_register #(
        .DATA_BITS(DATA_BITS)
    ) dut (
        .clk(clk),
        .reset(reset),
        .d(d),
        .q(q)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 0;
        d = '0;
        #1;

        d = 8'sd42;
        @(posedge clk);
        #1;
        assert(q == 8'sd42) else $fatal(1, "Test case 1 failed: q=%0d", q);

        d = -8'sd15;
        #1;
        assert(q == 8'sd42) else $fatal(1, "Hold test failed: q changed before posedge clk, q=%0d", q);

        @(posedge clk);
        #1;
        assert(q == -8'sd15) else $fatal(1, "Test case 2 failed: q=%0d", q);

        reset = 1;
        #1;
        assert(q == '0) else $fatal(1, "Async reset test failed: q=%0d", q);

        $display("All test cases passed.");
        $finish;
    end

endmodule
