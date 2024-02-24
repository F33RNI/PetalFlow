/**
 * @file forward.c
 * @author Fern Lane
 * @brief Petal forward propagation
 * @version 1.0.0
 * @date 2023-11-17
 *
 * @copyright Copyright (c) 2023-2024 Fern Lane
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * long with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdbool.h>
#include <stdint.h>

#include "dropout.h"
#include "errors.h"
#include "logger.h"
#include "petal.h"

/**
 * @brief Petal forward propagation
 *
 * @param petal pointer to petal struct
 * @param input pointer to 1D array of input data (must have petal->input_shape shape)
 * @param training true for training (to apply dropouts) or false for inference mode
 */
void petal_forward(petal_s *petal, float *input, bool training) {
    // Calculate dropout
    bool dropout_enabled = false;
    if (training && petal->dropout > 0.f && petal->bit_array) {
        // Clear previous dropout
        bit_array_clear(petal->bit_array);

        // Generate new dropout
        dropout_generate_indices(petal->bit_array, petal->dropout);
        if (petal->bit_array->error_code != ERROR_NONE) {
            logger(LOG_E, "petal_forward", "Error generating dropout indices: %s",
                   error_to_str[petal->bit_array->error_code]);
            petal->error_code = petal->bit_array->error_code;
            return;
        }
        dropout_enabled = true;
    }

    // Direct (no weights, input and output are the same size)
    if (petal->petal_type == PETAL_TYPE_DIRECT) {
        // Copy input to the output and apply dropout if needed
        for (uint32_t i = 0; i < petal->output_shape->length; ++i) {
            if (dropout_enabled && bit_array_get_bit(petal->bit_array, i))
                petal->output[i] = 0.f;
            else
                petal->output[i] = input[i];
        }
    }

    // Normalizes all input data using "center" and "deviation" regardless of the number of dimensions
    else if (petal->petal_type == PETAL_TYPE_NORMALIZE_ALL) {
        // Find min and max values
        float min_value = input[0];
        float max_value = input[0];
        for (uint32_t i = 1; i < petal->input_shape->length; ++i) {
            if (input[i] < min_value)
                min_value = input[i];
            else if (input[i] > max_value)
                max_value = input[i];
        }

        // Normalize and apply dropout if needed
        for (uint32_t i = 0; i < petal->output_shape->length; ++i) {
            if (dropout_enabled && bit_array_get_bit(petal->bit_array, i))
                petal->output[i] = 0.f;
            else {
                petal->output[i] = ((input[i] - min_value) / (max_value - min_value + EPSILON));
                petal->output[i] = petal->output[i] * 2.f * petal->deviation + petal->center - petal->deviation;
            }
        }
    }

    // Normalizes each row of input data using "center" and "deviation" independently
    else if (petal->petal_type == PETAL_TYPE_NORMALIZE_IN_ROWS) {
        float min_value, max_value;
        uint32_t row_index, index;
        for (uint32_t row_i = 0; row_i < petal->output_shape->rows; ++row_i) {
            // Calculate current raw index
            row_index = row_i * petal->output_shape->cols;

            // Find min and max values
            min_value = input[row_index];
            max_value = input[row_index];
            for (uint32_t col_i = 1; col_i < petal->output_shape->cols; ++col_i) {
                index = row_index + col_i;
                if (input[index] < min_value)
                    min_value = input[index];
                else if (input[index] > max_value)
                    max_value = input[index];
            }

            // Normalize and apply dropout if needed
            for (uint32_t col_i = 0; col_i < petal->output_shape->cols; ++col_i) {
                index = row_index + col_i;
                if (dropout_enabled && bit_array_get_bit(petal->bit_array, index))
                    petal->output[index] = 0.f;
                else {
                    petal->output[index] = ((input[index] - min_value) / (max_value - min_value + EPSILON));
                    petal->output[index] =
                        petal->output[index] * 2.f * petal->deviation + petal->center - petal->deviation;
                }
            }
        }
    }

    // Normalizes each channel of input data using "center" and "deviation" independently
    else if (petal->petal_type == PETAL_TYPE_NORMALIZE_IN_CHANNELS) {
        float min_value, max_value;
        uint32_t index;
        for (uint32_t channel_i = 0; channel_i < petal->output_shape->depth; ++channel_i) {
            // Find min and max values
            min_value = input[channel_i];
            max_value = input[channel_i];
            for (uint32_t i = 0; i < petal->output_shape->length; i += petal->output_shape->depth) {
                index = channel_i + i;
                if (input[index] < min_value)
                    min_value = input[index];
                else if (input[index] > max_value)
                    max_value = input[index];
            }

            // Normalize and apply dropout if needed
            for (uint32_t i = 0; i < petal->output_shape->length; i += petal->output_shape->depth) {
                index = channel_i + i;
                if (dropout_enabled && bit_array_get_bit(petal->bit_array, index))
                    petal->output[index] = 0.f;
                else {
                    petal->output[index] = ((input[index] - min_value) / (max_value - min_value + EPSILON));
                    petal->output[index] =
                        petal->output[index] * 2.f * petal->deviation + petal->center - petal->deviation;
                }
            }
        }
    }

    // 1D dense petal (fully-connected 1D petal with 2D weights and 1D bias weights)
    else if (petal->petal_type == PETAL_TYPE_DENSE_1D) {
        // Calculate dot
        uint32_t input_row;
        for (uint32_t output_i = 0; output_i < petal->output_shape->length; ++output_i) {
            // Reset output
            petal->output[output_i] = 0.f;

            // Row index
            input_row = output_i * petal->input_shape->length;

            // Don't calculate dot if we need to drop this output
            if (!dropout_enabled || !bit_array_get_bit(petal->bit_array, output_i)) {
                // Dot with weights
                if (petal->weights && petal->weights->weights)
                    for (uint32_t input_i = 0; input_i < petal->input_shape->length; ++input_i)
                        petal->output[output_i] += petal->weights->weights[input_row + input_i] * input[input_i];

                // Sums without weights
                else
                    for (uint32_t input_i = 0; input_i < petal->input_shape->length; ++input_i)
                        petal->output[output_i] += input[input_i];

                // Add bias weights
                if (petal->bias_weights && petal->bias_weights->weights)
                    petal->output[output_i] += petal->bias_weights->weights[output_i];
            }
        }
    }

    // Wrong type
    else {
        logger(LOG_E, "petal_forward", "Wrong petal type: %u", petal->petal_type);
        petal->error_code = ERROR_PETAL_WRONG_TYPE;
        return;
    }

    // Activate output if needed
    uint8_t activation_error = ERROR_NONE;
    if (petal->activation)
        activation_error =
            activation_forward(petal->activation, petal->output, petal->output_shape->length, petal->bit_array);

    // Normalize sum after activation if dropout is enabled
    if (dropout_enabled) {
        float dropout_scaling = 1.f / (1.f - petal->dropout + EPSILON);
        for (uint32_t i = 0; i < petal->output_shape->length; ++i)
            if (petal->output[i] != 0.f)
                petal->output[i] *= dropout_scaling;
    }

    // Check errors (just in case)
    uint8_t bit_array_error = petal->bit_array ? petal->bit_array->error_code : ERROR_NONE;
    if (activation_error != ERROR_NONE || bit_array_error != ERROR_NONE) {
        if (activation_error != ERROR_NONE)
            logger(LOG_E, "petal_forward", "Activation error: %s", error_to_str[activation_error]);
        if (bit_array_error != ERROR_NONE)
            logger(LOG_E, "petal_forward", "Bit array error: %s", error_to_str[bit_array_error]);
        petal->error_code = activation_error > bit_array_error ? activation_error : bit_array_error;
    }
}
