module signed_multiplier #(
    parameter int DATA_BITS = 8
) (
    input logic signed [DATA_BITS -1: 0] a,
    input logic signed [DATA_BITS -1: 0] b,
    output logic signed [2*DATA_BITS -1: 0] product
);

    assign product = a * b;
endmodule