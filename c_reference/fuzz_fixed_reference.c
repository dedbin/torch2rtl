#include "fixed_reference.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define FUZZ_ITERATIONS 5000u
#define MAX_DATA 128u

static uint32_t rng_state = UINT32_C(0x6d2b79f5);
static unsigned failures = 0u;

static uint32_t next_random(void)
{
    uint32_t value = rng_state;

    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    rng_state = value;
    return value;
}

static size_t random_size(size_t limit)
{
    return (size_t)(next_random() % (uint32_t)limit);
}

static void fail_iteration(
    unsigned iteration,
    const char *expression,
    int line
)
{
    (void)fprintf(
        stderr,
        "FUZZ FAIL iteration=%u line=%d: %s\n",
        iteration,
        line,
        expression
    );
    ++failures;
}

#define FUZZ_CHECK(iteration, condition)                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fail_iteration((iteration), #condition, __LINE__);                 \
        }                                                                       \
    } while (0)

static int64_t minimum_for_bits(unsigned bits)
{
    return -(INT64_C(1) << (bits - 1u));
}

static int64_t maximum_for_bits(unsigned bits)
{
    return (INT64_C(1) << (bits - 1u)) - INT64_C(1);
}

static int64_t random_tensor_value(const t2r_config *config)
{
    const int64_t minimum = minimum_for_bits(config->bits);
    const uint32_t span = UINT32_C(1) << config->bits;

    return minimum + (int64_t)(next_random() % span);
}

static int64_t floor_divide_power_of_two(int64_t value, unsigned frac_bits)
{
    const int64_t scale = INT64_C(1) << frac_bits;
    int64_t quotient = value / scale;
    const int64_t remainder = value % scale;

    if (value < 0 && remainder != 0) {
        --quotient;
    }
    return quotient;
}

static int64_t saturate_expected(int64_t value, unsigned bits)
{
    const int64_t minimum = minimum_for_bits(bits);
    const int64_t maximum = maximum_for_bits(bits);

    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

static bool in_accumulator_range(int64_t value, unsigned bits)
{
    const int64_t minimum = -(INT64_C(1) << (bits - 1u));
    const int64_t maximum = (INT64_C(1) << (bits - 1u)) - INT64_C(1);

    return value >= minimum && value <= maximum;
}

static t2r_config random_valid_config(void)
{
    t2r_config config;

    config.bits = 2u + next_random() % 7u;
    config.frac_bits = next_random() % config.bits;
    config.acc_bits = config.bits + 1u + next_random() % (16u - config.bits);
    return config;
}

static t2r_config random_invalid_config(void)
{
    t2r_config config = {8u, 6u, 32u};

    switch (next_random() % 5u) {
    case 0u:
        config.bits = 1u;
        break;
    case 1u:
        config.bits = 33u;
        break;
    case 2u:
        config.frac_bits = config.bits;
        break;
    case 3u:
        config.acc_bits = config.bits;
        break;
    default:
        config.acc_bits = 65u;
        break;
    }
    return config;
}

static void fuzz_invalid_config(unsigned iteration)
{
    const t2r_config config = random_invalid_config();
    const t2r_conv2d_params params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u
    };
    int64_t data[2] = {0, 0};
    int64_t output[2] = {0, 0};
    int64_t scalar = 0;

    FUZZ_CHECK(iteration, t2r_saturate(&config, 0, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_requantize(&config, 0, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_scale_bias_checked(&config, 0, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_multiply_checked(&config, 0, 0, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_accumulate_checked(&config, 0, 0, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_linear_one(&config, data, 1u, data, 1u, 0, 1u, &scalar) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_linear(&config, data, 1u, data, 1u, NULL, 0u, 1u, 1u, output, 1u) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_conv2d(&config, &params, data, 1u, data, 1u, NULL, 0u, output, 1u) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_relu(&config, data, 1u, output, 1u) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_flatten(&config, data, 1u, output, 1u) == T2R_INVALID_ARGUMENT);
    FUZZ_CHECK(iteration, t2r_argmax(&config, data, 1u, &scalar) == T2R_INVALID_ARGUMENT);
}

static void fuzz_primitives(unsigned iteration, const t2r_config *config)
{
    const int64_t value = (int64_t)(int32_t)next_random();
    const int64_t left = random_tensor_value(config);
    const int64_t right = random_tensor_value(config);
    const int64_t bias = random_tensor_value(config);
    const int64_t scale = INT64_C(1) << config->frac_bits;
    const int64_t product = left * right;
    const int64_t scaled_bias = bias * scale;
    const int64_t acc_min = -(INT64_C(1) << (config->acc_bits - 1u));
    const int64_t acc_max = (INT64_C(1) << (config->acc_bits - 1u)) - INT64_C(1);
    const int64_t accumulator = acc_min
        + (int64_t)(next_random() % (uint32_t)(acc_max - acc_min + 1));
    const int64_t term = acc_min
        + (int64_t)(next_random() % (uint32_t)(acc_max - acc_min + 1));
    int64_t result = 0;
    t2r_status status = T2R_OK;

    FUZZ_CHECK(iteration, t2r_saturate(config, value, &result) == T2R_OK);
    FUZZ_CHECK(iteration, result == saturate_expected(value, config->bits));

    FUZZ_CHECK(iteration, t2r_requantize(config, accumulator, &result) == T2R_OK);
    FUZZ_CHECK(
        iteration,
        result == floor_divide_power_of_two(accumulator, config->frac_bits)
    );

    status = t2r_scale_bias_checked(config, bias, &result);
    FUZZ_CHECK(
        iteration,
        status == (in_accumulator_range(scaled_bias, config->acc_bits)
            ? T2R_OK
            : T2R_ACCUMULATOR_OVERFLOW)
    );
    if (status == T2R_OK) {
        FUZZ_CHECK(iteration, result == scaled_bias);
    }

    status = t2r_multiply_checked(config, left, right, &result);
    FUZZ_CHECK(
        iteration,
        status == (in_accumulator_range(product, config->acc_bits)
            ? T2R_OK
            : T2R_ACCUMULATOR_OVERFLOW)
    );
    if (status == T2R_OK) {
        FUZZ_CHECK(iteration, result == product);
    }

    status = t2r_accumulate_checked(config, accumulator, term, &result);
    if (term > 0 && accumulator > acc_max - term) {
        FUZZ_CHECK(iteration, status == T2R_ACCUMULATOR_OVERFLOW);
    } else if (term < 0 && accumulator < acc_min - term) {
        FUZZ_CHECK(iteration, status == T2R_ACCUMULATOR_OVERFLOW);
    } else {
        FUZZ_CHECK(iteration, status == T2R_OK);
        if (status == T2R_OK) {
            FUZZ_CHECK(iteration, result == accumulator + term);
        }
    }
}

static void fuzz_elementwise(unsigned iteration, const t2r_config *config)
{
    int64_t input[MAX_DATA];
    int64_t relu_output[MAX_DATA];
    int64_t flatten_output[MAX_DATA];
    size_t count = 1u + random_size(32u);
    size_t output_capacity = (next_random() % 5u == 0u) ? count - 1u : count;
    size_t position = 0u;
    bool invalid_value = false;
    t2r_status expected = T2R_OK;
    t2r_status status = T2R_OK;
    int64_t index = -1;

    for (position = 0u; position < count; ++position) {
        input[position] = random_tensor_value(config);
    }
    if (next_random() % 7u == 0u) {
        input[random_size(count)] = maximum_for_bits(config->bits) + INT64_C(1);
        invalid_value = true;
    }
    if (output_capacity < count) {
        expected = T2R_BUFFER_TOO_SMALL;
    } else if (invalid_value) {
        expected = T2R_VALUE_OUT_OF_RANGE;
    }

    status = t2r_relu(config, input, count, relu_output, output_capacity);
    FUZZ_CHECK(iteration, status == expected);
    status = t2r_flatten(config, input, count, flatten_output, output_capacity);
    FUZZ_CHECK(iteration, status == expected);
    if (expected == T2R_OK) {
        for (position = 0u; position < count; ++position) {
            const int64_t relu_expected = input[position] < 0 ? 0 : input[position];
            FUZZ_CHECK(iteration, relu_output[position] == relu_expected);
            FUZZ_CHECK(iteration, flatten_output[position] == input[position]);
        }
    }

    status = t2r_argmax(config, input, count, &index);
    FUZZ_CHECK(
        iteration,
        status == (invalid_value ? T2R_VALUE_OUT_OF_RANGE : T2R_OK)
    );
    if (status == T2R_OK) {
        size_t best = 0u;
        for (position = 1u; position < count; ++position) {
            if (input[position] > input[best]) {
                best = position;
            }
        }
        FUZZ_CHECK(iteration, index == (int64_t)best);
    }
}

static void fuzz_linear(unsigned iteration, const t2r_config *config)
{
    int64_t input[MAX_DATA];
    int64_t weights[MAX_DATA];
    int64_t bias[MAX_DATA];
    int64_t output[MAX_DATA];
    const size_t in_features = 1u + random_size(8u);
    const size_t out_features = 1u + random_size(4u);
    const size_t weight_count = in_features * out_features;
    size_t input_capacity = in_features;
    size_t weights_capacity = weight_count;
    size_t output_capacity = out_features;
    size_t bias_count = out_features;
    const int64_t *bias_pointer = bias;
    bool invalid_value = false;
    size_t index = 0u;
    t2r_status expected = T2R_OK;
    t2r_status status = T2R_OK;
    int64_t one_output = 0;

    for (index = 0u; index < in_features; ++index) {
        input[index] = random_tensor_value(config);
    }
    for (index = 0u; index < weight_count; ++index) {
        weights[index] = random_tensor_value(config);
    }
    for (index = 0u; index < out_features; ++index) {
        bias[index] = random_tensor_value(config);
    }
    if (next_random() % 11u == 0u) {
        weights[random_size(weight_count)] = maximum_for_bits(config->bits) + 1;
        invalid_value = true;
    }

    {
        const size_t one_input_capacity = next_random() % 5u == 0u
            ? in_features - 1u
            : in_features;
        const size_t one_weights_capacity = next_random() % 5u == 0u
            ? in_features - 1u
            : in_features;
        bool one_invalid_value = false;
        size_t one_index = 0u;
        t2r_status one_expected = T2R_OK;
        const int64_t minimum = minimum_for_bits(config->bits);
        const int64_t maximum = maximum_for_bits(config->bits);

        for (one_index = 0u; one_index < in_features; ++one_index) {
            if (input[one_index] < minimum || input[one_index] > maximum
                || weights[one_index] < minimum
                || weights[one_index] > maximum) {
                one_invalid_value = true;
            }
        }
        if (one_input_capacity < in_features
            || one_weights_capacity < in_features) {
            one_expected = T2R_BUFFER_TOO_SMALL;
        } else if (one_invalid_value) {
            one_expected = T2R_VALUE_OUT_OF_RANGE;
        }
        status = t2r_linear_one(
            config,
            input,
            one_input_capacity,
            weights,
            one_weights_capacity,
            bias[0],
            in_features,
            &one_output
        );
        FUZZ_CHECK(
            iteration,
            status == one_expected || (one_expected == T2R_OK
                && status == T2R_ACCUMULATOR_OVERFLOW)
        );
        if (status == T2R_OK) {
            FUZZ_CHECK(
                iteration,
                one_output >= minimum && one_output <= maximum
            );
        }
    }

    switch (next_random() % 8u) {
    case 0u:
        input_capacity = in_features - 1u;
        break;
    case 1u:
        weights_capacity = weight_count - 1u;
        break;
    case 2u:
        output_capacity = out_features - 1u;
        break;
    case 3u:
        bias_pointer = NULL;
        bias_count = 0u;
        break;
    case 4u:
        bias_pointer = NULL;
        bias_count = 1u;
        break;
    case 5u:
        bias_count = out_features - 1u;
        break;
    default:
        break;
    }

    if (input_capacity < in_features
        || weights_capacity < weight_count
        || output_capacity < out_features) {
        expected = T2R_BUFFER_TOO_SMALL;
    } else if ((bias_pointer == NULL && bias_count != 0u)
               || (bias_pointer != NULL && bias_count != out_features)) {
        expected = T2R_INVALID_ARGUMENT;
    } else if (invalid_value) {
        expected = T2R_VALUE_OUT_OF_RANGE;
    }

    status = t2r_linear(
        config,
        input,
        input_capacity,
        weights,
        weights_capacity,
        bias_pointer,
        bias_count,
        in_features,
        out_features,
        output,
        output_capacity
    );
    FUZZ_CHECK(iteration, status == expected || (expected == T2R_OK
        && status == T2R_ACCUMULATOR_OVERFLOW));
    if (expected == T2R_OK) {
        t2r_config safe_config = *config;
        t2r_status safe_status = T2R_OK;

        safe_config.acc_bits = 32u;
        safe_status = t2r_linear(
            &safe_config,
            input,
            input_capacity,
            weights,
            weights_capacity,
            bias_pointer,
            bias_count,
            in_features,
            out_features,
            output,
            output_capacity
        );
        FUZZ_CHECK(iteration, safe_status == T2R_OK);
        if (safe_status == T2R_OK) {
            for (index = 0u; index < out_features; ++index) {
                FUZZ_CHECK(
                    iteration,
                    output[index] >= minimum_for_bits(config->bits)
                        && output[index] <= maximum_for_bits(config->bits)
                );
            }
        }
    }
    if (status == T2R_OK) {
        for (index = 0u; index < out_features; ++index) {
            FUZZ_CHECK(
                iteration,
                output[index] >= minimum_for_bits(config->bits)
                    && output[index] <= maximum_for_bits(config->bits)
            );
        }
    }
}

static void fuzz_conv2d(unsigned iteration, const t2r_config *config)
{
    int64_t input[MAX_DATA];
    int64_t weights[MAX_DATA];
    int64_t bias[MAX_DATA];
    int64_t output[MAX_DATA];
    int64_t repeated[MAX_DATA];
    t2r_conv2d_params params;
    size_t output_height = 0u;
    size_t output_width = 0u;
    size_t input_count = 0u;
    size_t weight_count = 0u;
    size_t output_count = 0u;
    size_t input_capacity = 0u;
    size_t weights_capacity = 0u;
    size_t output_capacity = 0u;
    size_t bias_count = 0u;
    const int64_t *bias_pointer = bias;
    size_t index = 0u;
    bool invalid_value = false;
    t2r_status shape_status = T2R_OK;
    t2r_status expected = T2R_OK;
    t2r_status status = T2R_OK;

    params.in_channels = 1u + random_size(2u);
    params.out_channels = 1u + random_size(2u);
    params.input_height = 1u + random_size(4u);
    params.input_width = 1u + random_size(4u);
    params.kernel_height = 1u + random_size(3u);
    params.kernel_width = 1u + random_size(3u);
    params.stride_height = 1u + random_size(2u);
    params.stride_width = 1u + random_size(2u);
    params.padding_height = random_size(2u);
    params.padding_width = random_size(2u);
    if (next_random() % 19u == 0u) {
        params.stride_height = 0u;
    }

    shape_status = t2r_conv2d_output_shape(
        &params,
        &output_height,
        &output_width
    );
    if (shape_status != T2R_OK) {
        FUZZ_CHECK(iteration, shape_status == T2R_INVALID_ARGUMENT);
        return;
    }

    input_count = params.in_channels * params.input_height * params.input_width;
    weight_count = params.out_channels * params.in_channels
        * params.kernel_height * params.kernel_width;
    output_count = params.out_channels * output_height * output_width;
    FUZZ_CHECK(iteration, input_count <= MAX_DATA);
    FUZZ_CHECK(iteration, weight_count <= MAX_DATA);
    FUZZ_CHECK(iteration, output_count <= MAX_DATA);
    if (input_count > MAX_DATA || weight_count > MAX_DATA
        || output_count > MAX_DATA) {
        return;
    }

    for (index = 0u; index < input_count; ++index) {
        input[index] = random_tensor_value(config);
    }
    for (index = 0u; index < weight_count; ++index) {
        weights[index] = random_tensor_value(config);
    }
    for (index = 0u; index < params.out_channels; ++index) {
        bias[index] = random_tensor_value(config);
    }
    if (next_random() % 13u == 0u) {
        input[random_size(input_count)] = minimum_for_bits(config->bits) - 1;
        invalid_value = true;
    }

    input_capacity = input_count;
    weights_capacity = weight_count;
    output_capacity = output_count;
    bias_count = params.out_channels;
    switch (next_random() % 8u) {
    case 0u:
        input_capacity = input_count - 1u;
        break;
    case 1u:
        weights_capacity = weight_count - 1u;
        break;
    case 2u:
        output_capacity = output_count - 1u;
        break;
    case 3u:
        bias_pointer = NULL;
        bias_count = 0u;
        break;
    case 4u:
        bias_pointer = NULL;
        bias_count = 1u;
        break;
    case 5u:
        bias_count = params.out_channels - 1u;
        break;
    default:
        break;
    }
    if (input_capacity < input_count
        || weights_capacity < weight_count
        || output_capacity < output_count) {
        expected = T2R_BUFFER_TOO_SMALL;
    } else if ((bias_pointer == NULL && bias_count != 0u)
               || (bias_pointer != NULL && bias_count != params.out_channels)) {
        expected = T2R_INVALID_ARGUMENT;
    } else if (invalid_value) {
        expected = T2R_VALUE_OUT_OF_RANGE;
    }

    status = t2r_conv2d(
        config,
        &params,
        input,
        input_capacity,
        weights,
        weights_capacity,
        bias_pointer,
        bias_count,
        output,
        output_capacity
    );
    FUZZ_CHECK(iteration, status == expected || (expected == T2R_OK
        && status == T2R_ACCUMULATOR_OVERFLOW));
    if (expected == T2R_OK) {
        t2r_config safe_config = *config;
        t2r_status safe_status = T2R_OK;

        safe_config.acc_bits = 32u;
        safe_status = t2r_conv2d(
            &safe_config,
            &params,
            input,
            input_capacity,
            weights,
            weights_capacity,
            bias_pointer,
            bias_count,
            output,
            output_capacity
        );
        FUZZ_CHECK(iteration, safe_status == T2R_OK);
        if (safe_status == T2R_OK) {
            for (index = 0u; index < output_count; ++index) {
                FUZZ_CHECK(
                    iteration,
                    output[index] >= minimum_for_bits(config->bits)
                        && output[index] <= maximum_for_bits(config->bits)
                );
            }
        }
    }
    if (status == T2R_OK) {
        const t2r_status repeated_status = t2r_conv2d(
            config,
            &params,
            input,
            input_capacity,
            weights,
            weights_capacity,
            bias_pointer,
            bias_count,
            repeated,
            output_capacity
        );
        FUZZ_CHECK(iteration, repeated_status == T2R_OK);
        for (index = 0u; index < output_count; ++index) {
            FUZZ_CHECK(iteration, repeated[index] == output[index]);
            FUZZ_CHECK(
                iteration,
                output[index] >= minimum_for_bits(config->bits)
                    && output[index] <= maximum_for_bits(config->bits)
            );
        }
    }
}

static void check_adversarial_shapes(void)
{
    t2r_conv2d_params params = {
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, SIZE_MAX, 0u
    };
    size_t output_height = 0u;
    size_t output_width = 0u;

    FUZZ_CHECK(
        FUZZ_ITERATIONS,
        t2r_conv2d_output_shape(&params, &output_height, &output_width)
            == T2R_DIMENSION_OVERFLOW
    );
    params.padding_height = 0u;
    params.in_channels = SIZE_MAX;
    params.input_width = 2u;
    FUZZ_CHECK(
        FUZZ_ITERATIONS,
        t2r_conv2d_output_shape(&params, &output_height, &output_width)
            == T2R_DIMENSION_OVERFLOW
    );
}

int main(void)
{
    unsigned iteration = 0u;

    for (iteration = 0u; iteration < FUZZ_ITERATIONS; ++iteration) {
        if (iteration % 17u == 0u) {
            fuzz_invalid_config(iteration);
        } else {
            const t2r_config config = random_valid_config();
            fuzz_primitives(iteration, &config);
            fuzz_elementwise(iteration, &config);
            fuzz_linear(iteration, &config);
            fuzz_conv2d(iteration, &config);
        }
    }
    check_adversarial_shapes();

    if (failures != 0u) {
        (void)fprintf(stderr, "FAIL deterministic API fuzz checks=%u\n", failures);
        return 1;
    }
    (void)printf(
        "PASS deterministic API fuzz iterations=%u seed=0x%08x\n",
        FUZZ_ITERATIONS,
        UINT32_C(0x6d2b79f5)
    );
    return 0;
}
