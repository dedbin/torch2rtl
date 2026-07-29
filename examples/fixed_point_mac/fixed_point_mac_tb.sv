`timescale 1ns / 1ps

module fixed_point_mac_tb;
    localparam int TAPS = 4;
    localparam int DATA_BITS = 8;
    localparam int FRAC_BITS = 6;
    localparam int ACC_BITS = 18;
    localparam int EXPECTED_TESTS = 6;

    logic signed [TAPS * DATA_BITS - 1:0] in_data;
    logic signed [TAPS * DATA_BITS - 1:0] weights;
    logic signed [DATA_BITS - 1:0] bias;
    logic signed [DATA_BITS - 1:0] out_data;

    fixed_point_mac #(
        .TAPS(TAPS),
        .DATA_BITS(DATA_BITS),
        .FRAC_BITS(FRAC_BITS),
        .ACC_BITS(ACC_BITS)
    ) dut (
        .in_data(in_data),
        .weights(weights),
        .bias(bias),
        .out_data(out_data)
    );

    task automatic check_value(
        input logic signed [TAPS * DATA_BITS - 1:0] in_value,
        input logic signed [TAPS * DATA_BITS - 1:0] weight_value,
        input logic signed [DATA_BITS - 1:0] bias_value,
        input logic signed [DATA_BITS - 1:0] expected
    );
        begin
            in_data = in_value;
            weights = weight_value;
            bias = bias_value;
            #1;

            if (out_data !== expected) begin
                $display(
                    "[ERROR] bias=%0d expected=%0d got=%0d",
                    bias_value,
                    expected,
                    out_data
                );
                $fatal(1);
            end else begin
                $display(
                    "[OK] bias=%0d expected=%0d got=%0d",
                    bias_value,
                    expected,
                    out_data
                );
            end
        end
    endtask

    initial begin
        integer vectors_file;
        integer fields_read;
        integer tests_run;
        string vectors_path;

        logic signed [TAPS * DATA_BITS - 1:0] vector_in_data;
        logic signed [TAPS * DATA_BITS - 1:0] vector_weights;
        logic signed [DATA_BITS - 1:0] vector_bias;
        logic signed [DATA_BITS - 1:0] vector_expected;

        if (!$value$plusargs("VECTORS=%s", vectors_path)) begin
            $fatal(1, "Missing required +VECTORS=<path>");
        end

        vectors_file = $fopen(vectors_path, "r");

        if (vectors_file == 0) begin
            $fatal(1, "Cannot open vectors file: %s", vectors_path);
        end

        tests_run = 0;

        while (!$feof(vectors_file)) begin
            fields_read = $fscanf(
                vectors_file,
                "%h %h %h %h\n",
                vector_in_data,
                vector_weights,
                vector_bias,
                vector_expected
            );

            if (fields_read == 4) begin
                check_value(
                    vector_in_data,
                    vector_weights,
                    vector_bias,
                    vector_expected
                );
                tests_run = tests_run + 1;
            end else if (fields_read != -1) begin
                $fatal(
                    1,
                    "Malformed vectors file: expected 4 hex fields"
                );
            end
        end

        $fclose(vectors_file);

        if (tests_run != EXPECTED_TESTS) begin
            $fatal(
                1,
                "Expected %0d vectors, got %0d",
                EXPECTED_TESTS,
                tests_run
            );
        end

        $display("PASS fixed_point_mac: %0d vectors", tests_run);
        $finish;
    end

endmodule
