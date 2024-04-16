/**
 * @file activation.c
 * @author Fern Lane
 * @brief Activation functions and their derivatives
 *
 * @copyright Copyright (c) 2023-2024 Fern Lane
 *
 * This file is part of the PetalFlow distribution <https://github.com/F33RNI/PetalFlow>.
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
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "dropout.h"
#include "errors.h"
#include "logger.h"
#include "petal.h"

/**
 * @brief Applies activation to 1D array
 *
 * Output data (after activation) will be written to the layer array
 * Also, some temp data will be written to the activation->_derivatives_temp
 *
 * @param activation pointer to activation_s struct:
 * type - activation function (ACTIVATION_...),
 * linear_alpha - factor for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 1.0,
 * linear_const - constant for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 0.0,
 * relu_leak - leak amount (for ACTIVATION_RELU only). Default = 0.01,
 * elu_alpha - the value to which an ELU saturates for negative net inputs (for ACTIVATION_ELU only). Default = 0.01,
 * swish_beta - beta for turning Swish into E-Swish (for ACTIVATION_SWISH only). Default = 1.0
 * @param layer pointer to 1D array of data to activate
 * @param layer_length size of 1D array of data to activate
 * @param bit_array pointer to bit array for dropout (indices with set bits (1) will be ignored) or NULL
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t activation_forward(activation_s *activation, float *layer, uint32_t layer_length, bit_array_s *bit_array) {
    // Allocate temp array for activation functions derivatives
    if (!activation->_derivatives_temp) {
        activation->_derivatives_temp = calloc(layer_length, sizeof(float));
        if (!activation->_derivatives_temp) {
            logger(LOG_E, "activation_forward", "Error allocating memory for activation->_derivatives_temp");
            return ERROR_MALLOC;
        }
    }

    // if (!bit_array || !bit_array_get_bit(bit_array, i)) will ignore activation and derivatives for some indices

    // Linear
    // f(x) = ax + c
    if (activation->type == ACTIVATION_LINEAR) {
        if (activation->linear_alpha != 1.f)
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    layer[i] *= activation->linear_alpha;
        if (activation->linear_const != 0.f)
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    layer[i] += activation->linear_const;
    }

    // Leaky ReLU
    // f(x) = [ax {x < 0}, x {x >= 0}]
    else if (activation->type == ACTIVATION_RELU) {
        // Save x for differentiation
        memcpy(activation->_derivatives_temp, layer, layer_length * sizeof(float));

        if (activation->relu_leak != 0.f) {
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    if (layer[i] < 0.f)
                        layer[i] *= activation->relu_leak;
        } else {
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    if (layer[i] < 0.f)
                        layer[i] = 0.f;
        }
    }

    // Exponential Linear Unit
    // f(x) = [a(e^x - 1) {x < 0}, x {x >= 0}]
    else if (activation->type == ACTIVATION_ELU) {
        // Save x for differentiation
        memcpy(activation->_derivatives_temp, layer, layer_length * sizeof(float));

        if (activation->elu_alpha != 0.f) {
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    if (layer[i] < 0.f)
                        layer[i] = activation->elu_alpha * (expf(layer[i]) - 1.f);
        } else {
            for (uint32_t i = 0; i < layer_length; ++i)
                if (!bit_array || !bit_array_get_bit(bit_array, i))
                    if (layer[i] < 0.f)
                        layer[i] = 0.f;
        }
    }

    // Softsign
    // f(x) = x / (|x| + 1)
    else if (activation->type == ACTIVATION_SOFTSIGN) {
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                // Save |x| + 1 for differentiation
                activation->_derivatives_temp[i] = fabsf(layer[i]) + 1.f;

                layer[i] /= activation->_derivatives_temp[i] + EPSILON;
            }
    }

    // Sigmoid
    // f(x) = 1 / (1 + e^(-x))
    else if (activation->type == ACTIVATION_SIGMOID) {
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer[i] = 1.f / (1.f + expf(-layer[i]));
    }

    // Hard sigmoid
    // f(x) = [0 {x < -2.5}, 1 {x > 2.5}, 0.2 * x + 0.5 {-2.5 <= x <= 2.5}]
    else if (activation->type == ACTIVATION_HARD_SIGMOID) {
        // Save x for differentiation
        memcpy(activation->_derivatives_temp, layer, layer_length * sizeof(float));

        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                if (layer[i] < -2.5f)
                    layer[i] = 0.f;
                else if (layer[i] > 2.5f)
                    layer[i] = 1.f;
                else
                    layer[i] = 0.2f * layer[i] + 0.5f;
            }
    }

    // Swish, E-Swish
    // f(x) = Bx * sigmoid(x)
    else if (activation->type == ACTIVATION_SWISH) {
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                // Save 1 + exp(-x) for differentiation
                activation->_derivatives_temp[i] = 1.f + expf(-layer[i]);

                layer[i] *= activation->swish_beta / (activation->_derivatives_temp[i] + +EPSILON);
            }
    }

    // Softmax
    // f(x)[i] = exp(x[i]) / sum(exp(x[0-N]))
    else if (activation->type == ACTIVATION_SOFTMAX) {
        // Find max value for safe expf()
        float layer_max = layer[0];
        for (uint32_t i = 0; i < layer_length; ++i)
            if (layer[i] > layer_max)
                layer_max = layer[i];

        // Calculate sum of exponents
        float exp_sum = 0.f;
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                layer[i] = expf(layer[i] - layer_max);
                exp_sum += layer[i];
            }

        // Divide each exponent by sum
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer[i] /= exp_sum;
    }

    // tanh
    // f(x) = tanh(x)
    else if (activation->type == ACTIVATION_TANH) {
        for (uint32_t i = 0; i < layer_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer[i] = tanhf(layer[i]);
    }

    // Wrong type
    else {
        logger(LOG_E, "activation_forward", "Wrong activation type: %u", activation->type);
        return ERROR_PETAL_WRONG_ACTIVATION;
    }

    // No error
    return ERROR_NONE;
}

/**
 * @brief Applies derivative of activation function to 1D array of previously activated layer
 * This increases performance because we don't need to calculate activation functions again
 * (TODO: implement automatic differentiation)
 * Requires activation->_derivatives_temp for some activation functions
 *
 * Output data (derivatives) will be written to the layer_activated array
 *
 * @param activation pointer to activation_s struct:
 * type - activation function (ACTIVATION_...),
 * linear_alpha - factor for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 1.0,
 * linear_const - constant for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 0.0,
 * relu_leak - leak amount (for ACTIVATION_RELU only). Default = 0.01,
 * elu_alpha - the value to which an ELU saturates for negative net inputs (for ACTIVATION_ELU only). Default = 0.01,
 * swish_beta - beta for turning Swish into E-Swish (for ACTIVATION_SWISH only). Default = 1.0
 * @param layer_activated pointer to 1D array of activated data:
 * NOTE: (for softmax allocated size must be layer_activated_length * layer_activated_length)
 * @param layer_activated_length size of 1D array of activated data
 * @param bit_array pointer to bit array for dropout (indices with set bits (1) will be ignored) or NULL
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t activation_backward(activation_s *activation, float *layer_activated, uint32_t layer_activated_length,
                            bit_array_s *bit_array) {
    // Check temp array
    if (!activation->_derivatives_temp) {
        logger(LOG_E, "activation_backward", "activation->_derivatives_temp is NULL");
        return ERROR_ACTIVATION_NO_TEMP;
    }

    // Linear derivative
    // f'(x) = a
    if (activation->type == ACTIVATION_LINEAR) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer_activated[i] = activation->linear_alpha;
    }

    // Leaky ReLU derivative
    // f'(x) = [a {x < 0}, 1 {x >= 0}]
    else if (activation->type == ACTIVATION_RELU) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                if (activation->_derivatives_temp[i] < 0.f)
                    layer_activated[i] = activation->relu_leak;
                else
                    layer_activated[i] = 1.f;
            }
    }

    // Exponential Linear Unit derivative
    // f'(x) = [f(x) + a {x < 0}, 1 {x >= 0}]
    else if (activation->type == ACTIVATION_ELU) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                if (activation->_derivatives_temp[i] < 0.f)
                    layer_activated[i] += activation->elu_alpha;
                else
                    layer_activated[i] = 1.f;
            }
    }

    // Softsign derivative
    // f'(x) = 1 / (|x| + 1)^2
    else if (activation->type == ACTIVATION_SOFTSIGN) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                layer_activated[i] =
                    1.f / (activation->_derivatives_temp[i] * activation->_derivatives_temp[i] + EPSILON);
            }
    }

    // Sigmoid derivative
    // f'(x) = f(x) * (1 - f(x))
    else if (activation->type == ACTIVATION_SIGMOID) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer_activated[i] *= (1.f - layer_activated[i]);

        // TODO
        // layer_activated[i] = (1.f / (1.f + expf(-layer_activated[i]))) * (1.f - (1.f / (1.f +
        // expf(-layer_activated[i]))));
    }

    // Hard sigmoid derivative
    // f(x) = [0 {x < -2.5}, 0 {x > 2.5}, 0.2 {-2.5 <= x <= 2.5}]
    else if (activation->type == ACTIVATION_HARD_SIGMOID) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i)) {
                if (activation->_derivatives_temp[i] < -2.5f)
                    layer_activated[i] = 0.f;
                else if (activation->_derivatives_temp[i] > 2.5f)
                    layer_activated[i] = 0.f;
                else
                    layer_activated[i] = 0.2f;
            }
    }

    // Swish derivative
    // f'(x) = f(x) + sigmoid(x) * (B - f(x))
    else if (activation->type == ACTIVATION_SWISH) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer_activated[i] = layer_activated[i] + (1.f / (activation->_derivatives_temp[i] + EPSILON)) *
                                                              (activation->swish_beta - layer_activated[i]);
    }

    // Softmax derivative (2D output as jacobian matrix)
    // f'(x)[i,j] = [f(x)[i] * (1 - f(x)[i]) {i = j}, -f(x)[i] * f(x)[i] {i != j}]
    else if (activation->type == ACTIVATION_SOFTMAX) {
        // Copy activated part
        memcpy(activation->_derivatives_temp, layer_activated, layer_activated_length * sizeof(float));

        // Calculate derivative as jacobian matrix
        for (uint32_t row = 0; row < layer_activated_length; ++row)
            for (uint32_t col = 0; col < layer_activated_length; ++col)
                layer_activated[row * layer_activated_length + col] =
                    activation->_derivatives_temp[row] *
                    ((row == col ? 1.0f : 0.0f) - activation->_derivatives_temp[col]);
    }

    // tanh derivative
    // f'(x) = 1 - f(x)^2
    else if (activation->type == ACTIVATION_TANH) {
        for (uint32_t i = 0; i < layer_activated_length; ++i)
            if (!bit_array || !bit_array_get_bit(bit_array, i))
                layer_activated[i] = 1. - layer_activated[i] * layer_activated[i];
    }

    // Wrong type
    else {
        logger(LOG_E, "activation_backward", "Wrong activation type: %u", activation->type);
        return ERROR_PETAL_WRONG_ACTIVATION;
    }

    // No error
    return ERROR_NONE;
}

/**
 * @brief Frees memory allocated by activation struct
 *
 * @param activation pointer to activation_s struct
 */
void activation_destroy(activation_s *activation) {
    if (activation) {
        logger(LOG_I, "activation_destroy", "Destroying activation struct with address: %p", activation);
        if (activation->_derivatives_temp)
            free(activation->_derivatives_temp);
        free(activation);
    }
}
