`timescale 1ns/1ps

module summator_4b_tb;

    // 4х битные регистры входных данных
    reg [3:0] a;
    reg [3:0] b;

    // 4x битный вывод sum и перенос
    wire [3:0] sum;
    wire       cout;

    reg [4:0] expected;
    integer errors;

    // Device Under Test (dut): что мы тестируем
    summator_4b dut (
        .a    (a),
        .b    (b),
        .sum  (sum),
        .cout (cout)
    );

    task check;
        input [3:0] test_a;
        input [3:0] test_b;
        begin
            a = test_a;
            b = test_b;
            expected = {1'b0, test_a} + {1'b0, test_b};

            #10;

            if ({cout, sum} !== expected) begin
                $display(
                    "FAIL: %0d + %0d: expected %05b, got %b_%04b",
                    test_a, test_b, expected, cout, sum
                );
                errors = errors + 1;
            end else begin
                $display(
                    "PASS: %0d + %0d = %0d (binary %b_%04b)",
                    test_a, test_b, {cout, sum}, cout, sum
                );
            end
        end
    endtask

    initial begin
        $dumpfile("summator_4b.vcd");
        $dumpvars(0, summator_4b_tb);

        errors = 0;
        a = 4'b0000;
        b = 4'b0000;

        check(4'd0,  4'd0);   // 0  + 0  = 0
        check(4'd1,  4'd1);   // 1  + 1  = 2
        check(4'd3,  4'd5);   // 3  + 5  = 8
        check(4'd10, 4'd5);   // 10 + 5  = 15
        check(4'd7,  4'd9);   // 7  + 9  = 16: carry propagation
        check(4'd15, 4'd15);  // 15 + 15 = 30: largest result

        if (errors == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $fatal(1, "TESTS FAILED: %0d error(s)", errors);
        end

        $finish;
    end

endmodule
