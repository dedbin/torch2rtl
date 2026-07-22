module relu_signed #(
    parameter int DATA_BITS = 8
) (
    input logic signed [DATA_BITS -1: 0] in_data,
    output logic signed [DATA_BITS -1: 0] out_data
);
    logic signed [DATA_BITS - 1: 0] value;
    localparam logic signed [DATA_BITS-1:0] ZERO_Q = '0;

    always_comb begin
        value = in_data;
        if (value < ZERO_Q) begin
            out_data = ZERO_Q;
        end else begin
            out_data = value;
        end
    end


endmodule