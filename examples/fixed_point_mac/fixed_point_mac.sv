`timescale 1ns / 1ps

module fixed_point_mac #(
    parameter int TAPS = 4,
    parameter int DATA_BITS = 8,
    parameter int FRAC_BITS = 6,
    parameter int ACC_BITS = 18
) (
    input logic signed [TAPS * DATA_BITS - 1:0] in_data,
    input logic signed [TAPS * DATA_BITS - 1:0] weights,

    input logic signed [DATA_BITS - 1:0] bias,
    output logic signed [DATA_BITS - 1:0] out_data
);
    function automatic logic signed [DATA_BITS - 1:0] lane_at(
        input logic [TAPS * DATA_BITS - 1:0] values,
        input integer index
    );
        begin
            lane_at = values[index * DATA_BITS +: DATA_BITS];
        end
    endfunction

    logic signed [2 * DATA_BITS - 1:0] product;
    integer tap;

    logic signed [ACC_BITS - 1:0] bias_extended;
    logic signed [ACC_BITS - 1:0] acc;
    logic signed [ACC_BITS - 1:0] shifted;
    logic signed [ACC_BITS - 1:0] saturated;

    assign bias_extended = bias;

    always_comb begin
        acc = bias_extended <<< FRAC_BITS;
        product = '0;

        for (tap = 0; tap < TAPS; tap = tap + 1) begin
            product = lane_at(in_data, tap) * lane_at(weights, tap);
            acc = acc + product;
        end
    end

    assign shifted = acc >>> FRAC_BITS;

    localparam logic signed [ACC_BITS - 1:0] MAX_Q =
        (1 <<< (DATA_BITS - 1)) - 1;
    localparam logic signed [ACC_BITS - 1:0] MIN_Q =
        -(1 <<< (DATA_BITS - 1));

    always_comb begin
        if (shifted > MAX_Q) begin
            saturated = MAX_Q;
        end else if (shifted < MIN_Q) begin
            saturated = MIN_Q;
        end else begin
            saturated = shifted;
        end
    end

    assign out_data = saturated[DATA_BITS - 1:0];
endmodule
