module summator_4b(
    input wire[3:0] a,
    input wire[3:0] b,

    output wire[3:0] sum,
    output wire cout
);
    // Провода переноса
    wire c1;
    wire c2;
    wire c3;

    summator u0(
        .a (a[0]),
        .b (b[0]),
        .cin(1'b0),
        .sum(sum[0]),
        .cout (c1)
    );

    summator u1(
        .a (a[1]),
        .b (b[1]),
        .cin(c1),
        .sum(sum[1]),
        .cout (c2)
    );

    summator u2(
        .a (a[2]),
        .b (b[2]),
        .cin(c2),
        .sum(sum[2]),
        .cout (c3)
    );

    summator u3(
        .a (a[3]),
        .b (b[3]),
        .cin(c3),
        .sum(sum[3]),
        .cout (cout)
    );

endmodule
