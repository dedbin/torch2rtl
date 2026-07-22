module signed_comparator #(
    parameter int DATA_BITS = 8
) (
    input logic signed [DATA_BITS -1: 0] a,
    input logic signed [DATA_BITS -1: 0] b,
    output logic gt
);

    assign gt = (a > b);

endmodule