`timescale 1ns/100ps

module multiplexer_2_1_tb;

    logic a;
    logic b;
    logic select;
    logic y;

    multiplexer_2_1 dut (
        .a(a),
        .b(b),
        .select(select),
        .y(y)
    );

    task automatic check_mux(
        input logic test_a,
        input logic test_b,
        input logic test_select,
        input logic expected
    );
        begin
            a = test_a;
            b = test_b;
            select = test_select;
            #1;

            if (y !== expected) begin
                $display(
                    "[ERROR] a=%0b b=%0b select=%0b expected=%0b got=%0b",
                    test_a, test_b, test_select, expected, y
                );
                $fatal(1);
            end else begin
                $display(
                    "[OK] a=%0b b=%0b select=%0b y=%0b",
                    test_a, test_b, test_select, y
                );
            end
        end
    endtask

    initial begin
        $display("Starting multiplexer_2_1 testbench...");

        check_mux(1'b0, 1'b0, 1'b0, 1'b0);
        check_mux(1'b0, 1'b1, 1'b0, 1'b0);
        check_mux(1'b1, 1'b0, 1'b0, 1'b1);
        check_mux(1'b1, 1'b1, 1'b0, 1'b1);

        check_mux(1'b0, 1'b0, 1'b1, 1'b0);
        check_mux(1'b0, 1'b1, 1'b1, 1'b1);
        check_mux(1'b1, 1'b0, 1'b1, 1'b0);
        check_mux(1'b1, 1'b1, 1'b1, 1'b1);

        $display("All multiplexer_2_1 tests passed.");
        $finish;
    end

endmodule
