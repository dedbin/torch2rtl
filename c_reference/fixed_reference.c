#include "fixed_reference.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    size_t output_height;
    size_t output_width;
    size_t input_elements;
    size_t weight_elements;
    size_t output_elements;
} t2r_conv2d_shape_info;

static bool t2r_config_is_valid(const t2r_config *config)
{
    return config->bits >= 2u
        && config->bits <= 32u
        && config->frac_bits < config->bits
        && config->acc_bits > config->bits
        && config->acc_bits <= 64u;
}

static void t2r_signed_bounds(
    unsigned bits,
    int64_t *minimum,
    int64_t *maximum
)
{
    if (bits == 64u) {
        *minimum = INT64_MIN;
        *maximum = INT64_MAX;
    } else {
        const int64_t magnitude = INT64_C(1) << (bits - 1u);
        *minimum = -magnitude;
        *maximum = magnitude - INT64_C(1);
    }
}

static bool t2r_in_signed_range(int64_t value, unsigned bits)
{
    int64_t minimum = 0;
    int64_t maximum = 0;

    t2r_signed_bounds(bits, &minimum, &maximum);
    return value >= minimum && value <= maximum;
}

static bool t2r_size_add(size_t left, size_t right, size_t *result)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool t2r_size_multiply(size_t left, size_t right, size_t *result)
{
    if (right != 0u && left > SIZE_MAX / right) {
        return false;
    }
    *result = left * right;
    return true;
}

static bool t2r_element_count_is_representable(size_t count)
{
    return count <= SIZE_MAX / sizeof(int64_t);
}

static uint64_t t2r_magnitude(int64_t value)
{
    if (value < 0) {
        return (uint64_t)(-(value + INT64_C(1))) + UINT64_C(1);
    }
    return (uint64_t)value;
}

static bool t2r_int64_multiply(
    int64_t left,
    int64_t right,
    int64_t *result
)
{
    const bool negative = (left < 0) != (right < 0);
    const uint64_t left_magnitude = t2r_magnitude(left);
    const uint64_t right_magnitude = t2r_magnitude(right);
    const uint64_t negative_limit = UINT64_C(1) << 63u;
    const uint64_t limit = negative ? negative_limit : (uint64_t)INT64_MAX;
    uint64_t product_magnitude = 0u;

    if (right_magnitude != 0u
        && left_magnitude > limit / right_magnitude) {
        return false;
    }

    product_magnitude = left_magnitude * right_magnitude;
    if (!negative) {
        *result = (int64_t)product_magnitude;
    } else if (product_magnitude == negative_limit) {
        *result = INT64_MIN;
    } else {
        *result = -(int64_t)product_magnitude;
    }
    return true;
}

static bool t2r_accumulator_add(
    const t2r_config *config,
    int64_t accumulator,
    int64_t term,
    int64_t *result
)
{
    int64_t minimum = 0;
    int64_t maximum = 0;

    t2r_signed_bounds(config->acc_bits, &minimum, &maximum);
    if (accumulator < minimum || accumulator > maximum
        || term < minimum || term > maximum) {
        return false;
    }
    if (term > 0 && accumulator > maximum - term) {
        return false;
    }
    if (term < 0 && accumulator < minimum - term) {
        return false;
    }

    *result = accumulator + term;
    return true;
}

static bool t2r_accumulator_multiply(
    const t2r_config *config,
    int64_t left,
    int64_t right,
    int64_t *result
)
{
    int64_t minimum = 0;
    int64_t maximum = 0;
    int64_t product = 0;

    if (!t2r_int64_multiply(left, right, &product)) {
        return false;
    }
    t2r_signed_bounds(config->acc_bits, &minimum, &maximum);
    if (product < minimum || product > maximum) {
        return false;
    }

    *result = product;
    return true;
}

static bool t2r_scale_bias(
    const t2r_config *config,
    int64_t bias,
    int64_t *result
)
{
    const int64_t scale = INT64_C(1) << config->frac_bits;
    return t2r_accumulator_multiply(config, bias, scale, result);
}

static int64_t t2r_floor_requantize(
    int64_t accumulator,
    unsigned frac_bits
)
{
    const int64_t scale = INT64_C(1) << frac_bits;
    int64_t quotient = accumulator / scale;
    const int64_t remainder = accumulator % scale;

    if (accumulator < 0 && remainder != 0) {
        quotient -= INT64_C(1);
    }
    return quotient;
}

static int64_t t2r_saturate_value(int64_t value, unsigned bits)
{
    int64_t minimum = 0;
    int64_t maximum = 0;

    t2r_signed_bounds(bits, &minimum, &maximum);
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

static bool t2r_values_in_range(
    const int64_t *values,
    size_t count,
    unsigned bits
)
{
    size_t index = 0u;

    for (index = 0u; index < count; ++index) {
        if (!t2r_in_signed_range(values[index], bits)) {
            return false;
        }
    }
    return true;
}

static t2r_status t2r_conv2d_shape(
    const t2r_conv2d_params *params,
    t2r_conv2d_shape_info *shape
)
{
    size_t double_padding_height = 0u;
    size_t double_padding_width = 0u;
    size_t padded_height = 0u;
    size_t padded_width = 0u;
    size_t height_numerator = 0u;
    size_t width_numerator = 0u;
    size_t temporary = 0u;

    if (params->in_channels == 0u
        || params->out_channels == 0u
        || params->input_height == 0u
        || params->input_width == 0u
        || params->kernel_height == 0u
        || params->kernel_width == 0u
        || params->stride_height == 0u
        || params->stride_width == 0u) {
        return T2R_INVALID_ARGUMENT;
    }

    if (!t2r_size_multiply(params->padding_height, 2u,
                           &double_padding_height)
        || !t2r_size_multiply(params->padding_width, 2u,
                              &double_padding_width)
        || !t2r_size_add(params->input_height, double_padding_height,
                         &padded_height)
        || !t2r_size_add(params->input_width, double_padding_width,
                         &padded_width)) {
        return T2R_DIMENSION_OVERFLOW;
    }

    if (params->kernel_height > padded_height
        || params->kernel_width > padded_width) {
        return T2R_INVALID_ARGUMENT;
    }

    height_numerator = padded_height - params->kernel_height;
    width_numerator = padded_width - params->kernel_width;
    if (!t2r_size_add(height_numerator / params->stride_height, 1u,
                      &shape->output_height)
        || !t2r_size_add(width_numerator / params->stride_width, 1u,
                         &shape->output_width)) {
        return T2R_DIMENSION_OVERFLOW;
    }

    if (!t2r_size_multiply(params->in_channels, params->input_height,
                           &temporary)
        || !t2r_size_multiply(temporary, params->input_width,
                              &shape->input_elements)
        || !t2r_size_multiply(params->out_channels, params->in_channels,
                              &temporary)
        || !t2r_size_multiply(temporary, params->kernel_height, &temporary)
        || !t2r_size_multiply(temporary, params->kernel_width,
                              &shape->weight_elements)
        || !t2r_size_multiply(params->out_channels, shape->output_height,
                              &temporary)
        || !t2r_size_multiply(temporary, shape->output_width,
                              &shape->output_elements)) {
        return T2R_DIMENSION_OVERFLOW;
    }

    if (!t2r_element_count_is_representable(shape->input_elements)
        || !t2r_element_count_is_representable(shape->weight_elements)
        || !t2r_element_count_is_representable(shape->output_elements)) {
        return T2R_DIMENSION_OVERFLOW;
    }

    return T2R_OK;
}

static t2r_status t2r_compute_linear_one(
    const t2r_config *config,
    const int64_t *input,
    const int64_t *weights,
    int64_t bias,
    size_t in_features,
    int64_t *output
)
{
    int64_t accumulator = 0;
    size_t index = 0u;

    if (!t2r_scale_bias(config, bias, &accumulator)) {
        return T2R_ACCUMULATOR_OVERFLOW;
    }
    for (index = 0u; index < in_features; ++index) {
        int64_t product = 0;

        if (!t2r_accumulator_multiply(
                config,
                input[index],
                weights[index],
                &product
            )
            || !t2r_accumulator_add(
                config,
                accumulator,
                product,
                &accumulator
            )) {
            return T2R_ACCUMULATOR_OVERFLOW;
        }
    }

    *output = t2r_saturate_value(
        t2r_floor_requantize(accumulator, config->frac_bits),
        config->bits
    );
    return T2R_OK;
}

t2r_status t2r_saturate(
    const t2r_config *config,
    int64_t value,
    int64_t *output
)
{
    if (config == NULL || output == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }

    *output = t2r_saturate_value(value, config->bits);
    return T2R_OK;
}

t2r_status t2r_requantize(
    const t2r_config *config,
    int64_t accumulator,
    int64_t *output
)
{
    if (config == NULL || output == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_in_signed_range(accumulator, config->acc_bits)) {
        return T2R_ACCUMULATOR_OVERFLOW;
    }

    *output = t2r_floor_requantize(accumulator, config->frac_bits);
    return T2R_OK;
}

t2r_status t2r_scale_bias_checked(
    const t2r_config *config,
    int64_t bias,
    int64_t *result
)
{
    if (config == NULL || result == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_in_signed_range(bias, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }
    if (!t2r_scale_bias(config, bias, result)) {
        return T2R_ACCUMULATOR_OVERFLOW;
    }
    return T2R_OK;
}

t2r_status t2r_multiply_checked(
    const t2r_config *config,
    int64_t left,
    int64_t right,
    int64_t *result
)
{
    if (config == NULL || result == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_in_signed_range(left, config->bits)
        || !t2r_in_signed_range(right, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }
    if (!t2r_accumulator_multiply(config, left, right, result)) {
        return T2R_ACCUMULATOR_OVERFLOW;
    }
    return T2R_OK;
}

t2r_status t2r_accumulate_checked(
    const t2r_config *config,
    int64_t accumulator,
    int64_t term,
    int64_t *result
)
{
    if (config == NULL || result == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_accumulator_add(config, accumulator, term, result)) {
        return T2R_ACCUMULATOR_OVERFLOW;
    }
    return T2R_OK;
}

t2r_status t2r_linear_one(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    const int64_t *weights,
    size_t weights_count,
    int64_t bias,
    size_t in_features,
    int64_t *output
)
{
    if (config == NULL || input == NULL || weights == NULL || output == NULL
        || output == input || output == weights) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (in_features == 0u) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_element_count_is_representable(in_features)) {
        return T2R_DIMENSION_OVERFLOW;
    }
    if (input_count < in_features || weights_count < in_features) {
        return T2R_BUFFER_TOO_SMALL;
    }
    if (!t2r_values_in_range(input, in_features, config->bits)
        || !t2r_values_in_range(weights, in_features, config->bits)
        || !t2r_in_signed_range(bias, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    return t2r_compute_linear_one(
        config,
        input,
        weights,
        bias,
        in_features,
        output
    );
}

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
)
{
    size_t required_weights = 0u;
    size_t output_index = 0u;

    if (config == NULL || input == NULL || weights == NULL || output == NULL
        || output == input || output == weights
        || (bias != NULL && output == bias)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (in_features == 0u || out_features == 0u) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_size_multiply(in_features, out_features, &required_weights)
        || !t2r_element_count_is_representable(in_features)
        || !t2r_element_count_is_representable(required_weights)
        || !t2r_element_count_is_representable(out_features)) {
        return T2R_DIMENSION_OVERFLOW;
    }
    if (input_count < in_features
        || weights_count < required_weights
        || output_capacity < out_features) {
        return T2R_BUFFER_TOO_SMALL;
    }
    if ((bias == NULL && bias_count != 0u)
        || (bias != NULL && bias_count != out_features)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_values_in_range(input, in_features, config->bits)
        || !t2r_values_in_range(weights, required_weights, config->bits)
        || (bias != NULL
            && !t2r_values_in_range(bias, out_features, config->bits))) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    for (output_index = 0u; output_index < out_features; ++output_index) {
        const size_t weight_offset = output_index * in_features;
        const int64_t bias_value = bias == NULL ? INT64_C(0) : bias[output_index];
        const t2r_status status = t2r_compute_linear_one(
            config,
            input,
            weights + weight_offset,
            bias_value,
            in_features,
            output + output_index
        );

        if (status != T2R_OK) {
            return status;
        }
    }
    return T2R_OK;
}

t2r_status t2r_conv2d_output_shape(
    const t2r_conv2d_params *params,
    size_t *output_height,
    size_t *output_width
)
{
    t2r_conv2d_shape_info shape = {0u, 0u, 0u, 0u, 0u};
    t2r_status status = T2R_OK;

    if (params == NULL || output_height == NULL || output_width == NULL
        || output_height == output_width) {
        return T2R_INVALID_ARGUMENT;
    }

    status = t2r_conv2d_shape(params, &shape);
    if (status != T2R_OK) {
        return status;
    }

    *output_height = shape.output_height;
    *output_width = shape.output_width;
    return T2R_OK;
}

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
)
{
    t2r_conv2d_shape_info shape = {0u, 0u, 0u, 0u, 0u};
    t2r_status status = T2R_OK;
    size_t output_channel = 0u;
    size_t output_y = 0u;
    size_t output_x = 0u;

    if (config == NULL || params == NULL || input == NULL || weights == NULL
        || output == NULL || output == input || output == weights
        || (bias != NULL && output == bias)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }

    status = t2r_conv2d_shape(params, &shape);
    if (status != T2R_OK) {
        return status;
    }
    if (input_count < shape.input_elements
        || weights_count < shape.weight_elements
        || output_capacity < shape.output_elements) {
        return T2R_BUFFER_TOO_SMALL;
    }
    if ((bias == NULL && bias_count != 0u)
        || (bias != NULL && bias_count != params->out_channels)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_values_in_range(input, shape.input_elements, config->bits)
        || !t2r_values_in_range(weights, shape.weight_elements, config->bits)
        || (bias != NULL
            && !t2r_values_in_range(
                bias,
                params->out_channels,
                config->bits
            ))) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    for (output_channel = 0u;
         output_channel < params->out_channels;
         ++output_channel) {
        for (output_y = 0u; output_y < shape.output_height; ++output_y) {
            for (output_x = 0u; output_x < shape.output_width; ++output_x) {
                const int64_t bias_value = bias == NULL
                    ? INT64_C(0)
                    : bias[output_channel];
                int64_t accumulator = 0;
                size_t input_channel = 0u;
                const size_t output_index =
                    (output_channel * shape.output_height + output_y)
                    * shape.output_width + output_x;

                if (!t2r_scale_bias(config, bias_value, &accumulator)) {
                    return T2R_ACCUMULATOR_OVERFLOW;
                }

                for (input_channel = 0u;
                     input_channel < params->in_channels;
                     ++input_channel) {
                    size_t kernel_y = 0u;

                    for (kernel_y = 0u;
                         kernel_y < params->kernel_height;
                         ++kernel_y) {
                        const size_t padded_y =
                            output_y * params->stride_height + kernel_y;
                        size_t input_y = 0u;
                        size_t kernel_x = 0u;

                        if (padded_y < params->padding_height) {
                            continue;
                        }
                        input_y = padded_y - params->padding_height;
                        if (input_y >= params->input_height) {
                            continue;
                        }

                        for (kernel_x = 0u;
                             kernel_x < params->kernel_width;
                             ++kernel_x) {
                            const size_t padded_x =
                                output_x * params->stride_width + kernel_x;
                            size_t input_x = 0u;
                            size_t input_index = 0u;
                            size_t weight_index = 0u;
                            int64_t product = 0;

                            if (padded_x < params->padding_width) {
                                continue;
                            }
                            input_x = padded_x - params->padding_width;
                            if (input_x >= params->input_width) {
                                continue;
                            }

                            input_index =
                                (input_channel * params->input_height + input_y)
                                * params->input_width + input_x;
                            weight_index =
                                ((output_channel * params->in_channels
                                  + input_channel)
                                 * params->kernel_height + kernel_y)
                                * params->kernel_width + kernel_x;

                            if (!t2r_accumulator_multiply(
                                    config,
                                    input[input_index],
                                    weights[weight_index],
                                    &product
                                )
                                || !t2r_accumulator_add(
                                    config,
                                    accumulator,
                                    product,
                                    &accumulator
                                )) {
                                return T2R_ACCUMULATOR_OVERFLOW;
                            }
                        }
                    }
                }

                output[output_index] = t2r_saturate_value(
                    t2r_floor_requantize(accumulator, config->frac_bits),
                    config->bits
                );
            }
        }
    }

    return T2R_OK;
}

t2r_status t2r_relu(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *output,
    size_t output_capacity
)
{
    size_t index = 0u;

    if (config == NULL || input == NULL || output == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (input_count == 0u) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_element_count_is_representable(input_count)) {
        return T2R_DIMENSION_OVERFLOW;
    }
    if (output_capacity < input_count) {
        return T2R_BUFFER_TOO_SMALL;
    }
    if (!t2r_values_in_range(input, input_count, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    for (index = 0u; index < input_count; ++index) {
        output[index] = input[index] < 0 ? INT64_C(0) : input[index];
    }
    return T2R_OK;
}

t2r_status t2r_flatten(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *output,
    size_t output_capacity
)
{
    if (config == NULL || input == NULL || output == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (input_count == 0u) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_element_count_is_representable(input_count)) {
        return T2R_DIMENSION_OVERFLOW;
    }
    if (output_capacity < input_count) {
        return T2R_BUFFER_TOO_SMALL;
    }
    if (!t2r_values_in_range(input, input_count, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    memmove(output, input, input_count * sizeof(int64_t));
    return T2R_OK;
}

t2r_status t2r_argmax(
    const t2r_config *config,
    const int64_t *input,
    size_t input_count,
    int64_t *index
)
{
    size_t current = 0u;
    size_t best = 0u;

    if (config == NULL || input == NULL || index == NULL) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_config_is_valid(config)) {
        return T2R_INVALID_ARGUMENT;
    }
    if (input_count == 0u) {
        return T2R_INVALID_ARGUMENT;
    }
    if (!t2r_element_count_is_representable(input_count)
        || (uintmax_t)(input_count - 1u) > (uintmax_t)INT64_MAX) {
        return T2R_DIMENSION_OVERFLOW;
    }
    if (!t2r_values_in_range(input, input_count, config->bits)) {
        return T2R_VALUE_OUT_OF_RANGE;
    }

    for (current = 1u; current < input_count; ++current) {
        if (input[current] > input[best]) {
            best = current;
        }
    }
    *index = (int64_t)best;
    return T2R_OK;
}
