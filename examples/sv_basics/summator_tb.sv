`timescale 1ns/100ps

module summator_tb;

    logic a;
    logic b;
    logic cin;

    logic sum;
    logic cout;

    integer errors;

    summator dut (
        .a(a),
        .b(b),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );

    task automatic check_summator(
        input logic test_a,
        input logic test_b,
        input logic test_cin
    );
        logic [1:0] expected;
        begin
            a = test_a;
            b = test_b;
            cin = test_cin;

            expected = {1'b0, test_a}
                     + {1'b0, test_b}
                     + {1'b0, test_cin};

            #1;

            if ({cout, sum} !== expected) begin
                $display(
                    "[ERROR] a=%0b b=%0b cin=%0b expected=%0b_%0b got=%0b_%0b",
                    test_a, test_b, test_cin,
                    expected[1], expected[0], cout, sum
                );
                errors = errors + 1;
            end else begin
                $display(
                    "[OK] a=%0b b=%0b cin=%0b -> cout=%0b sum=%0b",
                    test_a, test_b, test_cin, cout, sum
                );
            end
        end
    endtask

    initial begin
        $dumpfile("summator.vcd");
        $dumpvars(0, summator_tb);

        $display("Starting summator testbench...");
        errors = 0;

        check_summator(1'b0, 1'b0, 1'b0);
        check_summator(1'b0, 1'b0, 1'b1);
        check_summator(1'b0, 1'b1, 1'b0);
        check_summator(1'b0, 1'b1, 1'b1);
        check_summator(1'b1, 1'b0, 1'b0);
        check_summator(1'b1, 1'b0, 1'b1);
        check_summator(1'b1, 1'b1, 1'b0);
        check_summator(1'b1, 1'b1, 1'b1);

        if (errors == 0) begin
            $display("ALL SUMMATOR TESTS PASSED");
        end else begin
            $fatal(1, "SUMMATOR TESTS FAILED: %0d error(s)", errors);
        end

        $finish;
    end

endmodule
