#ifndef TORCH2RTL_FIXED_REFERENCE_H
#define TORCH2RTL_FIXED_REFERENCE_H

/*
 * Dependency-free C11 oracle for torch2rtl fixed-point semantics.
 *
 * Every tensor element is an already-quantized int64_t.  The library owns no
 * buffers, performs no allocation, and retains no hidden state.  A non-NULL
 * pointer accompanied by count/capacity N is a caller promise that at least N
 * elements are accessible through that pointer; the C API cannot discover an
 * allocation's true size.
 *
 * Fixed-point arithmetic is:
 *
 *   accumulator = bias_q * 2^frac_bits + sum(input_q * weight_q)
 *   requantized = floor(accumulator / 2^frac_bits)
 *   output_q    = saturation_to_signed_bits(requantized)
 *
 * The scaled bias, every product, and every MAC prefix must fit the signed
 * acc_bits range.  All tensor operands must fit the signed bits range.
 *
 * Validation precedence is stable across the public API:
 *   1. mandatory pointers and basic argument/alias rules;
 *   2. t2r_config;
 *   3. structural dimensions and forbidden zero sizes;
 *   4. checked size/dimension arithmetic;
 *   5. input/output/weight capacities;
 *   6. the optional-bias contract;
 *   7. operand ranges;
 *   8. product, scaled-bias, and MAC-prefix accumulator checks.
 * A check that does not apply to a function is skipped without changing the
 * order of the remaining checks.
 *
 * On any status other than T2R_OK, every output of the call is invalid and
 * must not be used.  A tensor output may already contain a computed prefix.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    T2R_OK = 0,
    T2R_INVALID_ARGUMENT,
    T2R_DIMENSION_OVERFLOW,
    T2R_BUFFER_TOO_SMALL,
    T2R_VALUE_OUT_OF_RANGE,
    T2R_ACCUMULATOR_OVERFLOW
} t2r_status;

typedef struct {
    unsigned bits;
    unsigned frac_bits;
    unsigned acc_bits;
} t2r_config;

/* Parameters for one unbatched CHW convolution with OIHW weights. */
typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t input_height;
    size_t input_width;
    size_t kernel_height;
    size_t kernel_width;
    size_t stride_height;
    size_t stride_width;
    size_t padding_height;
    size_t padding_width;
} t2r_conv2d_params;

/*
 * A valid configuration satisfies:
 *   2 <= bits <= 32
 *   frac_bits < bits
 *   bits < acc_bits <= 64
 */

/* Saturate an int64_t to the signed config->bits range. */
t2r_status t2r_saturate(
    const t2r_config *config,
    int64_t value,
    int64_t *output
);

/*
 * Portable floor division of an in-range accumulator by 2^frac_bits.
 * An accumulator outside the signed acc_bits range is rejected.
 */
t2r_status t2r_requantize(
    const t2r_config *config,
    int64_t accumulator,
    int64_t *output
);

/* Checked bias_q * 2^frac_bits.  bias must fit signed bits. */
t2r_status t2r_scale_bias_checked(
    const t2r_config *config,
    int64_t bias,
    int64_t *result
);

/* Checked multiplication of two signed-bits tensor operands. */
t2r_status t2r_multiply_checked(
    const t2r_config *config,
    int64_t left,
    int64_t right,
    int64_t *result
);

/* Checked addition of two signed-acc_bits values. */
t2r_status t2r_accumulate_checked(
    const t2r_config *config,
    int64_t accumulator,
    int64_t term,
    int64_t *result
);

/*
 * Compute one Linear output from [in_features] input and weight row.
 * input_count and weights_count are readable capacities and may exceed
 * in_features.  A scalar zero bias represents an absent bias for this helper.
 * Exact output alias with input or weights is rejected; arbitrary overlap is
 * otherwise a caller precondition.
 */
t2r_status t2r_linear_one(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    const int64_t *weights,
    size_t weights_count,
    int64_t bias,
    size_t in_features,
    int64_t *output
);

/*
 * Linear layouts are input [in], weights [out][in], bias [out], output [out].
 * Counts for input and weights are readable capacities; output_capacity is a
 * writable capacity.  Extra capacity is ignored.
 *
 * Optional bias contract:
 *   bias == NULL && bias_count == 0              -> zero bias
 *   bias != NULL && bias_count == out_features   -> supplied bias
 * Every other pointer/count combination returns T2R_INVALID_ARGUMENT.
 *
 * Output must not overlap input, weights, or bias.  Exact pointer aliases are
 * rejected; arbitrary partial overlap is a caller precondition.  No restrict
 * qualification is imposed by this API.
 */
t2r_status t2r_linear(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    const int64_t *weights,
    size_t weights_count,
    const int64_t *bias,
    size_t bias_count,
    size_t in_features,
    size_t out_features,
    int64_t *output,
    size_t output_capacity
);

/*
 * Validate convolution structure and checked size arithmetic, then return
 *   floor((input + 2 * padding - kernel) / stride) + 1
 * for both spatial dimensions.  All channel/input/kernel/stride dimensions
 * must be positive.  A kernel producing no output is invalid.  The routine
 * also verifies that CHW input, OIHW weights, and CHW output element counts
 * are representable as size_t.  output_height and output_width must be
 * distinct writable pointers.
 */
t2r_status t2r_conv2d_output_shape(
    const t2r_conv2d_params *params,
    size_t *output_height,
    size_t *output_width
);

/*
 * Conv2d layouts are input CHW, weights OIHW, bias O, and output CHW.
 * Counts for input and weights are readable capacities; output_capacity is a
 * writable capacity.  Extra capacity is ignored.  Groups, dilation, and batch
 * dimensions are not supported.
 *
 * Optional bias contract:
 *   bias == NULL && bias_count == 0                -> zero bias
 *   bias != NULL && bias_count == out_channels     -> supplied bias
 * Every other pointer/count combination returns T2R_INVALID_ARGUMENT.
 *
 * Output must not overlap input, weights, or bias.  Exact pointer aliases are
 * rejected; arbitrary partial overlap is a caller precondition.
 */
t2r_status t2r_conv2d(
    const t2r_config *config,
    const t2r_conv2d_params *params,
    const int64_t *input,
    size_t input_count,
    const int64_t *weights,
    size_t weights_count,
    const int64_t *bias,
    size_t bias_count,
    int64_t *output,
    size_t output_capacity
);

/*
 * Elementwise ReLU.  Exact in-place operation (input == output) is allowed.
 * Other partial overlap is outside the supported contract.  input_count is
 * both the nonzero tensor size and required output size.
 */
t2r_status t2r_relu(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *output,
    size_t output_capacity
);

/*
 * Checked linear-order copy of a nonempty contiguous tensor.  memmove
 * semantics apply: disjoint, exact in-place, and arbitrary partial overlap
 * are all supported.  input_count is the required output size.
 */
t2r_status t2r_flatten(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *output,
    size_t output_capacity
);

/*
 * Global first-maximum over a nonempty tensor.  Strict > comparison preserves
 * the first index on ties.  The resulting index is int64_t and input_count - 1
 * must be representable as int64_t.
 */
t2r_status t2r_argmax(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *index
);

#ifdef __cplusplus
}
#endif

#endif
