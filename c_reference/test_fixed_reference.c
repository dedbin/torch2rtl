#include "fixed_reference.h"

#include <stdint.h>
#include <stdio.h>

static unsigned failures = 0u;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            (void)fprintf(                                                      \
                stderr,                                                         \
                "FAIL %s:%d: %s\n",                                           \
                __FILE__,                                                       \
                __LINE__,                                                       \
                #condition                                                      \
            );                                                                  \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

#define CHECK_STATUS(expression, expected_status)                               \
    do {                                                                        \
        const t2r_status actual_status_ = (expression);                         \
        const t2r_status expected_status_ = (expected_status);                  \
        if (actual_status_ != expected_status_) {                               \
            (void)fprintf(                                                      \
                stderr,                                                         \
                "FAIL %s:%d: %s returned %d, expected %d\n",                  \
                __FILE__,                                                       \
                __LINE__,                                                       \
                #expression,                                                    \
                (int)actual_status_,                                             \
                (int)expected_status_                                            \
            );                                                                  \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static void check_array(
    const int64_t *actual,
    const int64_t *expected,
    size_t count,
    const char *name
)
{
    size_t index = 0u;

    for (index = 0u; index < count; ++index) {
        if (actual[index] != expected[index]) {
            (void)fprintf(
                stderr,
                "FAIL %s[%zu]: got %lld, expected %lld\n",
                name,
                index,
                (long long)actual[index],
                (long long)expected[index]
            );
            ++failures;
        }
    }
}

static void test_config_and_primitives(void)
{
    const t2r_config cfg = {8u, 6u, 12u};
    const t2r_config cfg_frac_zero = {8u, 0u, 32u};
    const t2r_config cfg_acc64 = {32u, 31u, 64u};
    const t2r_config invalid_bits_low = {1u, 0u, 2u};
    const t2r_config invalid_bits_high = {33u, 0u, 64u};
    const t2r_config invalid_frac = {8u, 8u, 32u};
    const t2r_config invalid_acc_low = {8u, 6u, 8u};
    const t2r_config invalid_acc_high = {8u, 6u, 65u};
    int64_t result = 0;

    CHECK_STATUS(t2r_saturate(&cfg, 126, &result), T2R_OK);
    CHECK(result == 126);
    CHECK_STATUS(t2r_saturate(&cfg, 128, &result), T2R_OK);
    CHECK(result == 127);
    CHECK_STATUS(t2r_saturate(&cfg, -129, &result), T2R_OK);
    CHECK(result == -128);
    CHECK_STATUS(t2r_saturate(NULL, 0, &result), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_saturate(&cfg, 0, NULL), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(
        t2r_saturate(&invalid_bits_low, 0, &result),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_saturate(&invalid_bits_high, 0, &result),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_saturate(&invalid_frac, 0, &result),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_saturate(&invalid_acc_low, 0, &result),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_saturate(&invalid_acc_high, 0, &result),
        T2R_INVALID_ARGUMENT
    );

    CHECK_STATUS(t2r_requantize(&cfg, 4480, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_requantize(&cfg, 2047, &result), T2R_OK);
    CHECK(result == 31);
    CHECK_STATUS(t2r_requantize(&cfg, 63, &result), T2R_OK);
    CHECK(result == 0);
    CHECK_STATUS(t2r_requantize(&cfg, 64, &result), T2R_OK);
    CHECK(result == 1);
    CHECK_STATUS(t2r_requantize(&cfg, -63, &result), T2R_OK);
    CHECK(result == -1);
    CHECK_STATUS(t2r_requantize(&cfg, -64, &result), T2R_OK);
    CHECK(result == -1);
    CHECK_STATUS(t2r_requantize(&cfg, -65, &result), T2R_OK);
    CHECK(result == -2);
    CHECK_STATUS(t2r_requantize(&cfg_frac_zero, -123, &result), T2R_OK);
    CHECK(result == -123);

    CHECK_STATUS(t2r_scale_bias_checked(&cfg, 31, &result), T2R_OK);
    CHECK(result == 1984);
    CHECK_STATUS(t2r_scale_bias_checked(&cfg, 32, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_scale_bias_checked(&cfg, -32, &result), T2R_OK);
    CHECK(result == -2048);
    CHECK_STATUS(t2r_scale_bias_checked(&cfg, -33, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_scale_bias_checked(&cfg, 128, &result), T2R_VALUE_OUT_OF_RANGE);

    CHECK_STATUS(t2r_multiply_checked(&cfg, 89, 23, &result), T2R_OK);
    CHECK(result == 2047);
    CHECK_STATUS(t2r_multiply_checked(&cfg, 64, 32, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_multiply_checked(&cfg, -128, 16, &result), T2R_OK);
    CHECK(result == -2048);
    CHECK_STATUS(t2r_multiply_checked(&cfg, -128, 17, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_multiply_checked(&cfg, 128, 1, &result), T2R_VALUE_OUT_OF_RANGE);

    CHECK_STATUS(t2r_accumulate_checked(&cfg, 2000, 47, &result), T2R_OK);
    CHECK(result == 2047);
    CHECK_STATUS(t2r_accumulate_checked(&cfg, 2000, 48, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(t2r_accumulate_checked(&cfg, -2000, -48, &result), T2R_OK);
    CHECK(result == -2048);
    CHECK_STATUS(t2r_accumulate_checked(&cfg, -2000, -49, &result), T2R_ACCUMULATOR_OVERFLOW);
    CHECK_STATUS(
        t2r_accumulate_checked(&cfg_acc64, INT64_MAX, 0, &result),
        T2R_OK
    );
    CHECK(result == INT64_MAX);
    CHECK_STATUS(
        t2r_accumulate_checked(&cfg_acc64, INT64_MAX, 1, &result),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_accumulate_checked(&cfg_acc64, INT64_MIN, -1, &result),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_scale_bias_checked(&cfg_acc64, INT32_MIN, &result),
        T2R_OK
    );
    CHECK(result == -INT64_C(4611686018427387904));
    CHECK_STATUS(
        t2r_multiply_checked(&cfg_acc64, INT32_MIN, INT32_MIN, &result),
        T2R_OK
    );
    CHECK(result == INT64_C(4611686018427387904));
    CHECK_STATUS(
        t2r_requantize(&cfg_acc64, INT64_MIN, &result),
        T2R_OK
    );
    CHECK(result == -INT64_C(4294967296));
}

static void test_null_pointer_contracts(void)
{
    const t2r_config cfg = {8u, 6u, 32u};
    const t2r_conv2d_params params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u
    };
    const int64_t input[] = {1};
    const int64_t weights[] = {1};
    int64_t output[] = {0};
    int64_t scalar = 0;
    size_t height = 0u;
    size_t width = 0u;

    CHECK_STATUS(t2r_requantize(NULL, 0, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_requantize(&cfg, 0, NULL), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_scale_bias_checked(NULL, 0, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_scale_bias_checked(&cfg, 0, NULL), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_multiply_checked(NULL, 0, 0, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_multiply_checked(&cfg, 0, 0, NULL), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_accumulate_checked(NULL, 0, 0, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_accumulate_checked(&cfg, 0, 0, NULL), T2R_INVALID_ARGUMENT);

    CHECK_STATUS(
        t2r_linear_one(NULL, input, 1u, weights, 1u, 0, 1u, &scalar),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, NULL, 1u, weights, 1u, 0, 1u, &scalar),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 1u, NULL, 1u, 0, 1u, &scalar),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 1u, weights, 1u, 0, 1u, NULL),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(NULL, input, 1u, weights, 1u, NULL, 0u, 1u, 1u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(&cfg, NULL, 1u, weights, 1u, NULL, 0u, 1u, 1u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(&cfg, input, 1u, NULL, 1u, NULL, 0u, 1u, 1u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(&cfg, input, 1u, weights, 1u, NULL, 0u, 1u, 1u, NULL, 1u),
        T2R_INVALID_ARGUMENT
    );

    CHECK_STATUS(
        t2r_conv2d_output_shape(NULL, &height, &width),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, NULL, &width),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &height, NULL),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(NULL, &params, input, 1u, weights, 1u, NULL, 0u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(&cfg, NULL, input, 1u, weights, 1u, NULL, 0u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(&cfg, &params, NULL, 1u, weights, 1u, NULL, 0u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(&cfg, &params, input, 1u, NULL, 1u, NULL, 0u, output, 1u),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(&cfg, &params, input, 1u, weights, 1u, NULL, 0u, NULL, 1u),
        T2R_INVALID_ARGUMENT
    );

    CHECK_STATUS(t2r_relu(NULL, input, 1u, output, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_relu(&cfg, NULL, 1u, output, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_relu(&cfg, input, 1u, NULL, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_flatten(NULL, input, 1u, output, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_flatten(&cfg, NULL, 1u, output, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_flatten(&cfg, input, 1u, NULL, 1u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_argmax(NULL, input, 1u, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_argmax(&cfg, NULL, 1u, &scalar), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_argmax(&cfg, input, 1u, NULL), T2R_INVALID_ARGUMENT);
}

static void test_validation_precedence(void)
{
    const t2r_config cfg = {8u, 6u, 32u};
    const t2r_config narrow_cfg = {8u, 6u, 9u};
    const t2r_config invalid_cfg = {1u, 0u, 2u};
    const int64_t valid_input[] = {127};
    const int64_t invalid_input[] = {128};
    const int64_t weight[] = {127};
    const int64_t bias[] = {0};
    int64_t output[] = {0};
    t2r_conv2d_params params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u
    };

    CHECK_STATUS(
        t2r_linear(
            &invalid_cfg, valid_input, 0u, weight, 0u, NULL, 1u,
            1u, 1u, output, 0u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, valid_input, 0u, weight, 0u, NULL, 1u,
            0u, 1u, output, 0u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, valid_input, 1u, weight, 1u, NULL, 1u,
            SIZE_MAX, 2u, output, 1u
        ),
        T2R_DIMENSION_OVERFLOW
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, valid_input, 0u, weight, 0u, NULL, 1u,
            1u, 1u, output, 0u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, invalid_input, 1u, weight, 1u, NULL, 1u,
            1u, 1u, output, 1u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(
            &narrow_cfg, invalid_input, 1u, weight, 1u, bias, 1u,
            1u, 1u, output, 1u
        ),
        T2R_VALUE_OUT_OF_RANGE
    );
    CHECK_STATUS(
        t2r_linear(
            &narrow_cfg, valid_input, 1u, weight, 1u, bias, 1u,
            1u, 1u, output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );

    CHECK_STATUS(
        t2r_conv2d(
            &invalid_cfg, &params, valid_input, 0u, weight, 0u,
            NULL, 1u, output, 0u
        ),
        T2R_INVALID_ARGUMENT
    );
    params.stride_height = 0u;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, valid_input, 0u, weight, 0u,
            NULL, 1u, output, 0u
        ),
        T2R_INVALID_ARGUMENT
    );
    params.stride_height = 1u;
    params.padding_height = SIZE_MAX;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, valid_input, 0u, weight, 0u,
            NULL, 1u, output, 0u
        ),
        T2R_DIMENSION_OVERFLOW
    );
    params.padding_height = 0u;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, valid_input, 0u, weight, 0u,
            NULL, 1u, output, 0u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, invalid_input, 1u, weight, 1u,
            NULL, 1u, output, 1u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(
            &narrow_cfg, &params, invalid_input, 1u, weight, 1u,
            bias, 1u, output, 1u
        ),
        T2R_VALUE_OUT_OF_RANGE
    );
    CHECK_STATUS(
        t2r_conv2d(
            &narrow_cfg, &params, valid_input, 1u, weight, 1u,
            bias, 1u, output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
}

static void test_full_mac_acc64_boundaries(void)
{
    const t2r_config cfg = {32u, 31u, 64u};
    const int64_t max_input[] = {INT32_MAX, 1, 1, 1, 1, 1};
    const int64_t max_weights[] = {
        INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, 1, 1
    };
    const int64_t min_input[] = {INT32_MIN, -1, -1, -1};
    const int64_t min_weights[] = {INT32_MAX, INT32_MAX, 1, 1};
    const int64_t max_bias[] = {INT32_MAX};
    const int64_t min_bias[] = {INT32_MIN};
    int64_t output = 0;
    t2r_conv2d_params params = {
        1u, 1u, 1u, 5u, 1u, 5u, 1u, 1u, 0u, 0u
    };

    CHECK_STATUS(
        t2r_linear(
            &cfg, max_input, 5u, max_weights, 5u, max_bias, 1u,
            5u, 1u, &output, 1u
        ),
        T2R_OK
    );
    CHECK(output == INT32_MAX);
    CHECK_STATUS(
        t2r_linear(
            &cfg, max_input, 6u, max_weights, 6u, max_bias, 1u,
            6u, 1u, &output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, min_input, 3u, min_weights, 3u, min_bias, 1u,
            3u, 1u, &output, 1u
        ),
        T2R_OK
    );
    CHECK(output == INT32_MIN);
    CHECK_STATUS(
        t2r_linear(
            &cfg, min_input, 4u, min_weights, 4u, min_bias, 1u,
            4u, 1u, &output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );

    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, max_input, 5u, max_weights, 5u,
            max_bias, 1u, &output, 1u
        ),
        T2R_OK
    );
    CHECK(output == INT32_MAX);
    params.input_width = 6u;
    params.kernel_width = 6u;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, max_input, 6u, max_weights, 6u,
            max_bias, 1u, &output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    params.input_width = 3u;
    params.kernel_width = 3u;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, min_input, 3u, min_weights, 3u,
            min_bias, 1u, &output, 1u
        ),
        T2R_OK
    );
    CHECK(output == INT32_MIN);
    params.input_width = 4u;
    params.kernel_width = 4u;
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, min_input, 4u, min_weights, 4u,
            min_bias, 1u, &output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
}

static void test_cancellation_cannot_hide_overflow(void)
{
    const t2r_config cfg = {8u, 0u, 9u};
    const int64_t product_input[] = {16};
    const int64_t product_weights[] = {16};
    const int64_t product_bias[] = {-1};
    const int64_t prefix_input[] = {20, 10, -10};
    const int64_t prefix_weights[] = {10, 10, 10};
    const int64_t prefix_bias[] = {0};
    const int64_t negative_prefix_input[] = {-20, -10, 10};
    int64_t output = 0;
    t2r_conv2d_params params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u
    };

    /* product=256 is invalid even though bias=-1 would make final sum 255. */
    CHECK_STATUS(
        t2r_linear(
            &cfg,
            product_input,
            1u,
            product_weights,
            1u,
            product_bias,
            1u,
            1u,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg,
            &params,
            product_input,
            1u,
            product_weights,
            1u,
            product_bias,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );

    params.input_width = 3u;
    params.kernel_width = 3u;
    /* Prefix 200 + 100 overflows before the later -100 cancellation. */
    CHECK_STATUS(
        t2r_linear(
            &cfg,
            prefix_input,
            3u,
            prefix_weights,
            3u,
            prefix_bias,
            1u,
            3u,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg,
            &params,
            prefix_input,
            3u,
            prefix_weights,
            3u,
            prefix_bias,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    /* Prefix -200 - 100 symmetrically underflows before +100. */
    CHECK_STATUS(
        t2r_linear(
            &cfg,
            negative_prefix_input,
            3u,
            prefix_weights,
            3u,
            prefix_bias,
            1u,
            3u,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg,
            &params,
            negative_prefix_input,
            3u,
            prefix_weights,
            3u,
            prefix_bias,
            1u,
            &output,
            1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
}

static void test_linear(void)
{
    const t2r_config cfg = {8u, 6u, 32u};
    const t2r_config narrow_cfg = {8u, 6u, 9u};
    const int64_t input[] = {64, 32, 16};
    const int64_t weights[] = {16, 64, 48, 64, 32, 0};
    const int64_t bias[] = {13, -10};
    const int64_t expected[] = {73, 70};
    const int64_t expected_without_bias[] = {60, 80};
    const int64_t bad_input[] = {128};
    const int64_t overflow_input[] = {127};
    const int64_t overflow_weight[] = {127};
    int64_t output[2] = {0, 0};
    int64_t scalar = 0;

    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 3u, weights, 3u, 13, 3u, &scalar),
        T2R_OK
    );
    CHECK(scalar == 73);
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 2u, weights, 3u, 13, 3u, &scalar),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 3u, weights, 3u, 128, 3u, &scalar),
        T2R_VALUE_OUT_OF_RANGE
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 3u, weights, 3u, 13, 0u, &scalar),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear_one(&cfg, input, 3u, weights, 3u, 13, 3u, (int64_t *)input),
        T2R_INVALID_ARGUMENT
    );

    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, bias, 2u,
            3u, 2u, output, 2u
        ),
        T2R_OK
    );
    check_array(output, expected, 2u, "linear output");
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, NULL, 0u,
            3u, 2u, output, 2u
        ),
        T2R_OK
    );
    check_array(output, expected_without_bias, 2u, "linear no-bias output");
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 2u, weights, 6u, bias, 2u,
            3u, 2u, output, 2u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 5u, bias, 2u,
            3u, 2u, output, 2u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, bias, 2u,
            3u, 2u, output, 1u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 0u, weights, 6u, bias, 1u,
            3u, 2u, output, 0u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, NULL, 1u,
            3u, 2u, output, 2u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, bias, 1u,
            3u, 2u, output, 2u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, bad_input, 1u, overflow_weight, 1u, NULL, 0u,
            1u, 1u, output, 1u
        ),
        T2R_VALUE_OUT_OF_RANGE
    );
    CHECK_STATUS(
        t2r_linear(
            &narrow_cfg, overflow_input, 1u, overflow_weight, 1u, NULL, 0u,
            1u, 1u, output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_linear(
            &cfg, input, 3u, weights, 6u, bias, 2u,
            SIZE_MAX, 2u, output, 2u
        ),
        T2R_DIMENSION_OVERFLOW
    );

    {
        const t2r_config cfg32 = {32u, 0u, 64u};
        const int64_t endpoint_input[] = {INT32_MIN, INT32_MAX};
        const int64_t identity_weights[] = {1, 0, 0, 1};
        const int64_t endpoint_expected[] = {INT32_MIN, INT32_MAX};

        CHECK_STATUS(
            t2r_linear(
                &cfg32,
                endpoint_input,
                2u,
                identity_weights,
                4u,
                NULL,
                0u,
                2u,
                2u,
                output,
                2u
            ),
            T2R_OK
        );
        check_array(output, endpoint_expected, 2u, "linear signed-32 endpoints");
    }
}

static void test_conv2d_shape_and_basic(void)
{
    const t2r_config cfg = {8u, 0u, 32u};
    t2r_conv2d_params params = {
        1u, 1u, 3u, 3u, 2u, 2u, 1u, 1u, 0u, 0u
    };
    const int64_t input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int64_t weights[] = {1, 0, 0, 1};
    const int64_t bias[] = {1};
    const int64_t expected[] = {7, 9, 13, 15};
    const int64_t expected_no_bias[] = {6, 8, 12, 14};
    int64_t output[4] = {0, 0, 0, 0};
    size_t output_height = 0u;
    size_t output_width = 0u;

    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &output_height, &output_width),
        T2R_OK
    );
    CHECK(output_height == 2u);
    CHECK(output_width == 2u);
    CHECK_STATUS(
        t2r_conv2d_output_shape(NULL, &output_height, &output_width),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &output_height, &output_height),
        T2R_INVALID_ARGUMENT
    );

    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 9u, weights, 4u,
            bias, 1u, output, 4u
        ),
        T2R_OK
    );
    check_array(output, expected, 4u, "conv basic output");
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 9u, weights, 4u,
            NULL, 0u, output, 4u
        ),
        T2R_OK
    );
    check_array(output, expected_no_bias, 4u, "conv no-bias output");
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 8u, weights, 4u,
            bias, 1u, output, 4u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 9u, weights, 4u,
            bias, 1u, output, 3u
        ),
        T2R_BUFFER_TOO_SMALL
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 9u, weights, 4u,
            NULL, 1u, output, 4u
        ),
        T2R_INVALID_ARGUMENT
    );
    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &params, input, 9u, weights, 4u,
            bias, 0u, output, 4u
        ),
        T2R_INVALID_ARGUMENT
    );

    params.input_height = 0u;
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &output_height, &output_width),
        T2R_INVALID_ARGUMENT
    );
    params.input_height = 1u;
    params.input_width = 1u;
    params.kernel_height = 2u;
    params.kernel_width = 2u;
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &output_height, &output_width),
        T2R_INVALID_ARGUMENT
    );
    params.kernel_height = 1u;
    params.kernel_width = 1u;
    params.padding_height = SIZE_MAX;
    CHECK_STATUS(
        t2r_conv2d_output_shape(&params, &output_height, &output_width),
        T2R_DIMENSION_OVERFLOW
    );
}

static void test_conv2d_channels_stride_padding_and_overflow(void)
{
    const t2r_config cfg = {8u, 0u, 32u};
    const t2r_config q_cfg = {8u, 6u, 32u};
    const t2r_config narrow_cfg = {8u, 6u, 14u};
    const int64_t channel_input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int64_t channel_weights[] = {1, 1, 1, -1};
    const int64_t channel_expected[] = {6, 8, 10, 12, -4, -4, -4, -4};
    t2r_conv2d_params channel_params = {
        2u, 2u, 2u, 2u, 1u, 1u, 1u, 1u, 0u, 0u
    };
    int64_t channel_output[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    const int64_t spatial_input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int64_t spatial_weights[] = {1, 1, 1, 1};
    const int64_t spatial_bias[] = {1};
    const int64_t spatial_expected[] = {2, 6, 12, 29};
    t2r_conv2d_params spatial_params = {
        1u, 1u, 3u, 3u, 2u, 2u, 2u, 2u, 1u, 1u
    };
    int64_t spatial_output[4] = {0, 0, 0, 0};

    const int64_t max_value[] = {127};
    const int64_t min_value[] = {-128};
    const int64_t max_weight[] = {127};
    t2r_conv2d_params scalar_params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u
    };
    int64_t scalar_output = 0;

    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &channel_params, channel_input, 8u,
            channel_weights, 4u, NULL, 0u,
            channel_output, 8u
        ),
        T2R_OK
    );
    check_array(channel_output, channel_expected, 8u, "conv channel output");

    CHECK_STATUS(
        t2r_conv2d(
            &cfg, &spatial_params, spatial_input, 9u,
            spatial_weights, 4u, spatial_bias, 1u,
            spatial_output, 4u
        ),
        T2R_OK
    );
    check_array(spatial_output, spatial_expected, 4u, "conv stride/pad output");

    CHECK_STATUS(
        t2r_conv2d(
            &q_cfg, &scalar_params, max_value, 1u,
            max_weight, 1u, NULL, 0u, &scalar_output, 1u
        ),
        T2R_OK
    );
    CHECK(scalar_output == 127);
    CHECK_STATUS(
        t2r_conv2d(
            &q_cfg, &scalar_params, min_value, 1u,
            max_weight, 1u, NULL, 0u, &scalar_output, 1u
        ),
        T2R_OK
    );
    CHECK(scalar_output == -128);
    CHECK_STATUS(
        t2r_conv2d(
            &narrow_cfg, &scalar_params, max_value, 1u,
            max_weight, 1u, NULL, 0u, &scalar_output, 1u
        ),
        T2R_ACCUMULATOR_OVERFLOW
    );
    CHECK_STATUS(
        t2r_conv2d(
            &q_cfg, &scalar_params, max_value, 1u,
            max_weight, 1u, NULL, 0u,
            (int64_t *)max_value, 1u
        ),
        T2R_INVALID_ARGUMENT
    );
}

static void test_relu_flatten_argmax(void)
{
    const t2r_config cfg = {8u, 6u, 32u};
    const int64_t values[] = {-128, -1, 0, 1, 127};
    const int64_t relu_expected[] = {0, 0, 0, 1, 127};
    const int64_t invalid_value[] = {128};
    int64_t relu_output[5] = {0, 0, 0, 0, 0};
    int64_t relu_in_place[] = {-128, -1, 0, 1, 127};
    int64_t flatten_output[5] = {0, 0, 0, 0, 0};
    int64_t forward_overlap[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int64_t forward_expected[] = {0, 1, 0, 1, 2, 3, 4, 5};
    int64_t backward_overlap[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int64_t backward_expected[] = {2, 3, 4, 5, 6, 7, 6, 7};
    const int64_t tied_logits[] = {-5, -2, -2, -3};
    const int64_t first_logits[] = {7, 6, 7, -3};
    const int64_t last_logits[] = {-5, -4, -3, -2};
    int64_t index = -1;

    CHECK_STATUS(t2r_relu(&cfg, values, 5u, relu_output, 5u), T2R_OK);
    check_array(relu_output, relu_expected, 5u, "relu output");
    CHECK_STATUS(
        t2r_relu(&cfg, relu_in_place, 5u, relu_in_place, 5u),
        T2R_OK
    );
    check_array(relu_in_place, relu_expected, 5u, "relu in-place");
    CHECK_STATUS(t2r_relu(&cfg, values, 0u, relu_output, 5u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_relu(&cfg, values, 5u, relu_output, 4u), T2R_BUFFER_TOO_SMALL);
    CHECK_STATUS(t2r_relu(&cfg, invalid_value, 1u, relu_output, 1u), T2R_VALUE_OUT_OF_RANGE);

    CHECK_STATUS(t2r_flatten(&cfg, values, 5u, flatten_output, 5u), T2R_OK);
    check_array(flatten_output, values, 5u, "flatten output");
    CHECK_STATUS(
        t2r_flatten(&cfg, flatten_output, 5u, flatten_output, 5u),
        T2R_OK
    );
    check_array(flatten_output, values, 5u, "flatten in-place");
    CHECK_STATUS(
        t2r_flatten(&cfg, forward_overlap, 6u, forward_overlap + 2, 6u),
        T2R_OK
    );
    check_array(forward_overlap, forward_expected, 8u, "flatten forward overlap");
    CHECK_STATUS(
        t2r_flatten(&cfg, backward_overlap + 2, 6u, backward_overlap, 6u),
        T2R_OK
    );
    check_array(backward_overlap, backward_expected, 8u, "flatten backward overlap");
    CHECK_STATUS(t2r_flatten(&cfg, values, 0u, flatten_output, 5u), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_flatten(&cfg, values, 5u, flatten_output, 4u), T2R_BUFFER_TOO_SMALL);

    CHECK_STATUS(t2r_argmax(&cfg, tied_logits, 4u, &index), T2R_OK);
    CHECK(index == 1);
    CHECK_STATUS(t2r_argmax(&cfg, first_logits, 4u, &index), T2R_OK);
    CHECK(index == 0);
    CHECK_STATUS(t2r_argmax(&cfg, last_logits, 4u, &index), T2R_OK);
    CHECK(index == 3);
    CHECK_STATUS(t2r_argmax(&cfg, tied_logits, 0u, &index), T2R_INVALID_ARGUMENT);
    CHECK_STATUS(t2r_argmax(&cfg, invalid_value, 1u, &index), T2R_VALUE_OUT_OF_RANGE);
}

int main(void)
{
    test_config_and_primitives();
    test_null_pointer_contracts();
    test_validation_precedence();
    test_full_mac_acc64_boundaries();
    test_cancellation_cannot_hide_overflow();
    test_linear();
    test_conv2d_shape_and_basic();
    test_conv2d_channels_stride_padding_and_overflow();
    test_relu_flatten_argmax();

    if (failures != 0u) {
        (void)fprintf(stderr, "FAIL fixed reference tests=%u\n", failures);
        return 1;
    }
    (void)printf("PASS fixed reference tests\n");
    return 0;
}
