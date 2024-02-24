/**
 * @file petal.h
 * @author Fern Lane
 * @brief Stores petal's data and definitions
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
#ifndef PETAL_H__
#define PETAL_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "activation.h"
#include "dropout.h"
#include "weights.h"

// Petal types
#define PETAL_TYPE_DIRECT                0U
#define PETAL_TYPE_NORMALIZE_ALL         1U
#define PETAL_TYPE_NORMALIZE_IN_ROWS     2U
#define PETAL_TYPE_NORMALIZE_IN_CHANNELS 3U
#define PETAL_TYPE_DENSE_1D              4U

// For error check and tests
#define PETAL_TYPE_MAX PETAL_TYPE_DENSE_1D

// Epsilon to prevent division by zero and other undefined states
#ifndef EPSILON
#define EPSILON 1e-15f
#endif

/**
 * @struct petal_shape_s
 * Stores shape of input / output data
 *
 * @param rows height of data
 * @param cols width (or size for 1D) of data
 * @param depth number of channels of data
 * @param length total length (size) of data
 */
typedef struct {
    uint32_t rows, cols, depth;
    uint32_t length;
} petal_shape_s;

/**
 * @struct petal_s
 * Stores petal's data
 *
 * @param petal_type type of the petal (PETAL_TYPE_...)
 * @param first true if it's the first petal (output_left is input data) to prevent error_on_input calculation
 * @param weights - pointers to weights_s structs
 * @param bias_weights - pointers to weights_s structs
 * @param activation - pointer to activation_s struct
 * @param dropout - ratio of dropped outputs (0 to 1)
 * @param bit_array_s - pointer to bit_array_s struct that stores indices to drop
 * @param output - petal outputs
 * @param error_on_input - petal input state during backpropagation
 * @param error_code - initialization or runtime error code
 */
typedef struct {
    uint8_t petal_type;
    bool first;
    petal_shape_s *input_shape, *output_shape;
    weights_s *weights, *bias_weights;
    activation_s *activation;
    float dropout, center, deviation;

    bit_array_s *bit_array;
    float *output, *error_on_input;
    uint8_t error_code;
} petal_s;

petal_s *petal_init(uint8_t petal_type, bool first, petal_shape_s *input_shape, petal_shape_s *output_shape,
                    weights_s *weights, weights_s *bias_weights, activation_s *activation, float dropout, float center,
                    float deviation);

void petal_forward(petal_s *petal, float *input, bool training);

void petal_backward(petal_s *petal, float *error_right, float *output_left);

size_t petal_estimate_min_size(petal_s *petal);

void petal_destroy(petal_s *petal, bool destroy_weights_structs, bool destroy_weights_array,
                   bool destroy_bias_weights_array);

#endif
