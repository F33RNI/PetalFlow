/**
 * @file activation.h
 * @author Fern Lane
 * @brief Activation functions and their derivatives
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
#ifndef ACTIVATION_H__
#define ACTIVATION_H__

#include <stdint.h>

#include "dropout.h"

#define ACTIVATION_LINEAR       0U
#define ACTIVATION_RELU         1U
#define ACTIVATION_ELU          2U
#define ACTIVATION_SOFTSIGN     3U
#define ACTIVATION_SIGMOID      4U
#define ACTIVATION_HARD_SIGMOID 5U
#define ACTIVATION_SWISH        6U
#define ACTIVATION_SOFTMAX      7U
#define ACTIVATION_TANH         8U

// For error check and tests
#define ACTIVATION_MAX ACTIVATION_TANH

/**
 * @brief Stores activation function data
 *
 * @param type activation function (ACTIVATION_...)
 * @param linear_alpha factor for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 1.0
 * @param linear_const constant for linear activation (ax + c) (for ACTIVATION_LINEAR only). Default = 0.0
 * @param relu_leak leak amount (for ACTIVATION_RELU only). Default = 0.01
 * @param elu_alpha the value to which an ELU saturates for negative net inputs
 * (for ACTIVATION_ELU only). Default = 0.01
 * @param swish_beta beta for turning Swish into E-Swish (for ACTIVATION_SWISH only). Default = 1.0
 * @param _derivatives_temp internal array for storing data during forward pass for future derivation
 */
typedef struct {
    uint8_t type;
    float linear_alpha, linear_const, relu_leak, elu_alpha, swish_beta;
    float *_derivatives_temp;
} activation_s;

uint8_t activation_forward(activation_s *activation, float *layer, uint32_t layer_length, bit_array_s *bit_array);

uint8_t activation_backward(activation_s *activation, float *layer_activated, uint32_t layer_activated_length,
                            bit_array_s *bit_array);

void activation_destroy(activation_s *activation);

#endif
