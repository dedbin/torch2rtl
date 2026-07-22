module signed_register #(
    parameter int DATA_BITS = 8
)(
    input logic clk,
    input logic reset,
    input logic signed [DATA_BITS -1: 0] d,
    output logic signed [DATA_BITS -1: 0] q
);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            q <= '0;
        end else begin
            q <= d;
        end
    end

endmodule
