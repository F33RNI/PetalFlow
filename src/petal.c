/**
 * @file petal.c
 * @author Fern Lane
 * @brief Petal initialization and size estimation
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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "activation.h"
#include "dropout.h"
#include "errors.h"
#include "logger.h"
#include "petal.h"
#include "weights.h"

/**
 * @brief Initializes petal's struct and petal's weights if needed
 * Sets petal->error_code in case of error
 *
 * @param petal_type type of the petal (PETAL_TYPE_...)
 * @param first true if it's the first petal (output_left is input data) to prevent error_on_input calculation
 * @param input_shape pointer to petal_shape_s struct:
 * rows - height of input data,
 * cols - width (or size for 1D) of input data,
 * depth - number of channels of input data,
 * length - calculates internally
 * @param output_shape pointer to petal_shape_s struct:
 * rows - height of output data,
 * cols - width (or size for 1D) of output data,
 * depth - number of channels of output data,
 * length - calculates internally
 * @param weights pointer to weights_s struct (for PETAL_TYPE_DENSE_1D) or NULL for other types:
 * trainable - 1 if weights will be trained or 0 if not,
 * initializer - weights initializer (WEIGHTS_INIT_...),
 * weights - pass NULL to initialize weights or pointer to previously initialized weights,
 * center - constant for WEIGHTS_INIT_CONSTANT or center of distribution for other initializers,
 * deviation - deviation of distribution (ignored for WEIGHTS_INIT_CONSTANT),
 * @param bias_weights pointer to weights_s struct (for PETAL_TYPE_DENSE_1D) or NULL for other types:
 * trainable - 1 if bias weights will be trained or 0 if not,
 * initializer - bias weights initializer (WEIGHTS_INIT_...),
 * weights - pass NULL to initialize bias weights or pointer to previously initialized bias weights,
 * center - constant for WEIGHTS_INIT_CONSTANT or center of distribution for other initializers,
 * deviation - deviation of distribution (ignored for WEIGHTS_INIT_CONSTANT)
 * @param activation pointer to activation_s struct or NULL to disable activation:
 * type - activation function (ACTIVATION_...),
 * linear_alpha - factor for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 1.0,
 * linear_const - constant for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 0.0,
 * relu_leak - leak amount (for ACTIVATION_RELU only). Default = 0.01,
 * elu_alpha - the value to which an ELU saturates for negative net inputs (for ACTIVATION_ELU only). Default = 0.01,
 * swish_beta - beta for turning Swish into E-Swish (for ACTIVATION_SWISH only). Default = 1.0
 * @param dropout ratio of dropped outputs (0 to 1)
 * @param center center of normalization for PETAL_TYPE_NORMALIZE_... Default: 0.0
 * @param deviation deviation of normalization for PETAL_TYPE_NORMALIZE_... Default: 1.0
 * @return petal_s* petal's struct
 */
petal_s *petal_init(uint8_t petal_type, bool first, petal_shape_s *input_shape, petal_shape_s *output_shape,
                    weights_s *weights, weights_s *bias_weights, activation_s *activation, float dropout, float center,
                    float deviation) {
    // Log
    logger(LOG_I, "petal_init", "Initializing petal with type: %u", petal_type);

    // Allocate struct
    petal_s *petal = calloc(1U, sizeof(petal_s));
    if (!petal) {
        logger(LOG_E, "petal_init", "Error allocating memory for petal_s struct");
        return NULL;
    }

    // Reset error
    petal->error_code = ERROR_NONE;

    // Copy fields
    petal->petal_type = petal_type;
    petal->first = first;
    petal->input_shape = input_shape;
    petal->output_shape = output_shape;
    petal->weights = weights;
    petal->bias_weights = bias_weights;
    petal->activation = activation;
    petal->dropout = dropout;
    petal->center = center;
    petal->deviation = deviation;
    petal->bit_array = NULL;
    if (petal->activation)
        petal->activation->_derivatives_temp = NULL;

    // Check petal type
    if (petal_type > PETAL_TYPE_MAX) {
        logger(LOG_E, "petal_init", "Wrong petal type: %u", petal_type);
        petal->error_code = ERROR_PETAL_WRONG_TYPE;
        return petal;
    }

    // Check weights initializers
    if (weights && weights->initializer > WEIGHTS_INIT_MAX) {
        logger(LOG_E, "petal_init", "Wrong weights initializer: %u", weights->initializer);
        petal->error_code = ERROR_PETAL_WRONG_WEIGHTS_INIT;
        return petal;
    }
    if (bias_weights && bias_weights->initializer > WEIGHTS_INIT_MAX) {
        logger(LOG_E, "petal_init", "Wrong bias weights initializer: %u", bias_weights->initializer);
        petal->error_code = ERROR_PETAL_WRONG_WEIGHTS_INIT;
        return petal;
    }

    // Check activation
    if (activation && activation->type > ACTIVATION_MAX) {
        logger(LOG_E, "petal_init", "Wrong activation type: %u", activation->type);
        petal->error_code = ERROR_PETAL_WRONG_ACTIVATION;
        return petal;
    }

    // Calculate total input and output size
    input_shape->length = input_shape->rows * input_shape->cols * input_shape->depth;
    output_shape->length = output_shape->rows * output_shape->cols * output_shape->depth;

    // Check input and output shapes for zero
    if (input_shape->length == 0 || output_shape->length == 0) {
        logger(LOG_E, "petal_init", "Zero input or output shape");
        petal->error_code = ERROR_PETAL_SHAPE_ZERO;
        return petal;
    }

    // Check if sizes match each other for some types
    if (petal_type == PETAL_TYPE_DIRECT || petal_type == PETAL_TYPE_NORMALIZE_ALL ||
        petal_type == PETAL_TYPE_NORMALIZE_IN_ROWS || petal_type == PETAL_TYPE_NORMALIZE_IN_CHANNELS) {
        if (input_shape->cols != output_shape->cols || input_shape->rows != output_shape->rows ||
            input_shape->depth != output_shape->depth) {
            logger(LOG_E, "petal_init", "Input and output shapes are not equal");
            petal->error_code = ERROR_PETAL_SHAPES_NOT_EQUAL;
            return petal;
        }
    }

    // Initialize dropout
    if (dropout > 0.f) {
        petal->bit_array = bit_array_init(output_shape->length);
        if (petal->bit_array->error_code != ERROR_NONE) {
            logger(LOG_E, "petal_init", "Dropout bit array initialization error: %s",
                   error_to_str[petal->bit_array->error_code]);
            petal->error_code = petal->bit_array->error_code;
            return petal;
        }
    }

    // Initialize output
    if (activation && activation->type == ACTIVATION_SOFTMAX)
        petal->output = (float *) calloc(output_shape->length * output_shape->length, sizeof(float));
    else
        petal->output = (float *) calloc(output_shape->length, sizeof(float));
    if (!petal->output) {
        logger(LOG_E, "petal_init", "Error allocating memory for petal->output array");
        petal->error_code = ERROR_MALLOC;
        return petal;
    }

    // Initialize gradients error temp (errors for backpropagation)
    if (!petal->first) {
        petal->error_on_input = (float *) calloc(output_shape->length, sizeof(float));
        if (!petal->error_on_input) {
            logger(LOG_E, "petal_init", "Error allocating memory for petal->error_on_input");
            petal->error_code = ERROR_MALLOC;
            return petal;
        }
    } else
        petal->error_on_input = NULL;

    // Weights and errors initialization
    if (petal->petal_type == PETAL_TYPE_DENSE_1D) {
        // Check and initialize weights and bias weights
        uint8_t error_temp = weights_check_init(weights, petal->input_shape->length * petal->output_shape->length);
        if (error_temp == ERROR_NONE) {
            error_temp = weights_check_init(bias_weights, petal->output_shape->length);
            if (error_temp != ERROR_NONE) {
                logger(LOG_E, "petal_init", "Error checking and initializing bias_weights: %s",
                       error_to_str[error_temp]);
                petal->error_code = error_temp;
                return petal;
            }
        } else {
            logger(LOG_E, "petal_init", "Error checking and initializing weights: %s", error_to_str[error_temp]);
            petal->error_code = error_temp;
            return petal;
        }
    }

    return petal;
}

/**
 * @brief Estimates minimum size allocated by petal
 *
 * @param petal pointer to petal struct
 * @return size_t memory size in bytes
 */
size_t petal_estimate_min_size(petal_s *petal) {
    size_t min_size = 0U;
    if (petal) {

        // Struct itself
        min_size += sizeof(petal_s);

        // input_shape
        min_size += sizeof(petal_shape_s);

        // output_shape
        min_size += sizeof(petal_shape_s);

        // weights
        min_size += weights_estimate_min_size(petal->weights);

        // bias_weights
        min_size += weights_estimate_min_size(petal->bias_weights);

        // activation
        if (petal->activation) {
            min_size += sizeof(activation_s);
            if (petal->activation->_derivatives_temp)
                min_size += petal->output_shape->length * sizeof(float);
        }

        // bit_array
        if (petal->bit_array) {
            min_size += sizeof(bit_array_s);
            if (petal->bit_array->data)
                min_size += petal->output_shape->length * sizeof(BIT_ARRAY_TYPE);
        }

        // output
        if (petal->output) {
            if (petal->activation && petal->activation->type == ACTIVATION_SOFTMAX)
                min_size += petal->output_shape->length * petal->output_shape->length * sizeof(float);
            else
                min_size += petal->output_shape->length * sizeof(float);
        }

        // error_on_input
        if (petal->error_on_input)
            min_size += petal->output_shape->length * sizeof(float);
    }
    return min_size;
}

/**
 * @brief Frees memory allocated by petal struct
 *
 * @param petal pointer to petal_s struct
 * @param destroy_weights_structs true to also destroy weights struct
 * @param destroy_weights_array true to also destroy weights->weights array
 * @param destroy_bias_weights_array true to also destroy bias_weights->weights array
 */
void petal_destroy(petal_s *petal, bool destroy_weights_structs, bool destroy_weights_array,
                   bool destroy_bias_weights_array) {
    logger(LOG_I, "petal_destroy", "Destroying petal struct with address: %p", petal);
    weights_destroy(petal->weights, destroy_weights_structs, destroy_weights_array);
    weights_destroy(petal->bias_weights, destroy_weights_structs, destroy_bias_weights_array);
    activation_destroy(petal->activation);
    if (petal->output)
        free(petal->output);
    if (petal->error_on_input)
        free(petal->error_on_input);
    bit_array_destroy(petal->bit_array);
    free(petal);
}
