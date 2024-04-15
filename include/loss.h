/**
 * @file loss.h
 * @author Fern Lane
 * @brief Stores loss functions data and types definitions
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
#ifndef LOSS_H__
#define LOSS_H__

#include <stddef.h>
#include <stdint.h>

#define LOSS_MEAN_SQUARED_ERROR          0U
#define LOSS_MEAN_SQUARED_LOG_ERROR      1U
#define LOSS_ROOT_MEAN_SQUARED_LOG_ERROR 2U
#define LOSS_MEAN_ABS_ERROR              3U
#define LOSS_BINARY_CROSSENTROPY         4U
#define LOSS_CATEGORICAL_CROSSENTROPY    5U

// For error check and tests
#define LOSS_MAX LOSS_CATEGORICAL_CROSSENTROPY

// Epsilon to prevent division by zero and other undefined states
#ifndef EPSILON
#define EPSILON 1e-15f
#endif

// Size of _derivatives_temp
#define DERIVATIVE_TEMP_ROWS 3U

/**
 * @struct loss_s
 * Stores loss function data
 *
 * @param type loss function (LOSS_...)
 * @param loss output array with the same size as petal's length that stores loss and loss derivatives
 * @param _derivatives_temp_1 internal array for storing data during forward pass for future derivation
 * @param _derivatives_temp_2 internal array for storing data during forward pass for future derivation
 */
typedef struct {
    uint8_t type;
    float *loss;
    float *_derivatives_temp_1, *_derivatives_temp_2;
} loss_s;

uint8_t loss_forward(loss_s *loss, float *predicted, float *expected, uint32_t length);

uint8_t loss_backward(loss_s *loss, uint32_t length);

size_t loss_estimate_min_size(loss_s *loss, uint32_t output_length);

void loss_destroy(loss_s *loss);

#endif
