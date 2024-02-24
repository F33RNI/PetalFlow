/**
 * @file backward.c
 * @author Fern Lane
 * @brief Petal backward propagation
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
#include <stdlib.h>
#include <string.h>

#include "dropout.h"
#include "errors.h"
#include "logger.h"
#include "petal.h"

/**
 * @brief
 *
 * @param petal pointer to current petal to which calculate "error_on_input" and weights gradients
 * @param error_right pointer to "error_on_input" array from next (right) petal or array of loss function derivatives
 * NOTE: error_right must have size of current "petal" output (size of right petal's input)
 * @param output_left pointer to output of previous (left) petal or input data in case of first petal
 * NOTE: output_left must have size of current "petal" input (size of left petal's output)
 * (ex. true for GD and false for SGD)
 */
void petal_backward(petal_s *petal, float *error_right, float *output_left) {
    // Nothing to do during backpropagation in this petals
    if (petal->petal_type == PETAL_TYPE_DIRECT || petal->petal_type == PETAL_TYPE_NORMALIZE_ALL ||
        petal->petal_type == PETAL_TYPE_NORMALIZE_IN_ROWS || petal->petal_type == PETAL_TYPE_NORMALIZE_IN_CHANNELS) {
        // Just copy temp error (input and output sizes must match)
        if (!petal->first)
            for (uint32_t i = 0; i < petal->output_shape->length; ++i)
                petal->error_on_input[i] = error_right[i];
    }

    // 1D dense petal (fully-connected 1D petal with 2D weights and 1D bias weights)
    // We need to calculate activation derivative, next error and gradients
    else if (petal->petal_type == PETAL_TYPE_DENSE_1D) {

        // TODO
        // printf("Activated output:\n");
        // print_array(petal->output, 1U, petal->output_shape->length, 1U);

        // Calculate activation derivatives
        uint8_t activation_error =
            activation_backward(petal->activation, petal->output, petal->output_shape->length, petal->bit_array);

        // printf("Activation derivatives:\n");
        // print_array(petal->output, 1U, petal->output_shape->length, 1U);

        if (activation_error != ERROR_NONE) {
            logger(LOG_E, "petal_backward", "Error calculating activation derivatives: %s",
                   error_to_str[activation_error]);
            petal->error_code = activation_error;
            return;
        }

        // Dot error_right with activation derivatives
        uint32_t row_index;
        for (uint32_t jacobian_col = 0; jacobian_col < petal->output_shape->length; ++jacobian_col) {
            // Dot if softmax
            if (petal->activation && petal->activation->type == ACTIVATION_SOFTMAX) {
                for (uint32_t jacobian_row = 0; jacobian_row < petal->output_shape->length; ++jacobian_row) {
                    row_index = jacobian_row * petal->output_shape->length;
                    if (jacobian_row == 0)
                        petal->output[jacobian_col] =
                            petal->output[row_index + jacobian_col] * error_right[jacobian_row];
                    else
                        petal->output[jacobian_col] +=
                            petal->output[row_index + jacobian_col] * error_right[jacobian_row];
                }
            }

            // Otherwise just multiplication
            else
                petal->output[jacobian_col] *= error_right[jacobian_col];
        }

        // Reset temp errors in petal's input
        if (!petal->first)
            for (uint32_t grad_left_i = 0; grad_left_i < petal->input_shape->length; ++grad_left_i)
                petal->error_on_input[grad_left_i] = 0.f;

        // Backpropagate errors, calculate and apply gradients
        uint32_t grad_right_index;
        for (uint32_t grad_right_i = 0; grad_right_i < petal->output_shape->length; ++grad_right_i) {
            grad_right_index = grad_right_i * petal->input_shape->length;
            for (uint32_t grad_left_i = 0; grad_left_i < petal->input_shape->length; ++grad_left_i) {
                // Backpropagate error for next left petal
                if (!petal->first)
                    petal->error_on_input[grad_left_i] +=
                        petal->weights->weights[grad_right_index + grad_left_i] * petal->output[grad_right_i];

                // Calculate gradient for each weight as backward activation * previous petal's forward output
                // Calculate as sum because of batch processing
                if (petal->weights && petal->weights->trainable)
                    petal->weights->gradients[grad_right_index + grad_left_i] +=
                        petal->output[grad_right_i] * output_left[grad_left_i];
            }

            // Calculate gradients for bias weights
            // Calculate as sum because of batch processing
            if (petal->bias_weights && petal->bias_weights->trainable)
                petal->bias_weights->gradients[grad_right_i] += petal->output[grad_right_i];
        }
    }

    // Wrong type
    else {
        logger(LOG_E, "petal_backward", "Wrong petal type: %u", petal->petal_type);
        petal->error_code = ERROR_PETAL_WRONG_TYPE;
        return;
    }
}
