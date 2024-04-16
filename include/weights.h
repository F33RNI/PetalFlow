/**
 * @file weights.h
 * @author Fern Lane
 * @brief Stores weight's data and definitions
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
#ifndef WEIGHTS_H__
#define WEIGHTS_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "optimizers.h"

// Weights (and bias weights) initializers
#define WEIGHTS_INIT_CONSTANT               0U
#define WEIGHTS_INIT_RANDOM_UNIFORM         1U
#define WEIGHTS_INIT_RANDOM_GAUSSIAN        2U
#define WEIGHTS_INIT_XAVIER_GLOROT_UNIFORM  3U
#define WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN 4U
#define WEIGHTS_INIT_KAIMING_HE_UNIFORM     5U
#define WEIGHTS_INIT_KAIMING_HE_GAUSSIAN    6U

// For error check and tests
#define WEIGHTS_INIT_MAX WEIGHTS_INIT_KAIMING_HE_GAUSSIAN

// Epsilon to prevent division by zero and other undefined states
#ifndef EPSILON
#define EPSILON 1e-15f
#endif

/**
 * @struct weights_s
 * Stores petal's weights
 *
 * @param trainable true if weights will be trained or false if not
 * @param initializer weights initializer (WEIGHTS_INIT_...)
 * @param length_total total length of weights (input * outut length for regular weights, output length for bias
 * weights)
 * @param weights pointer to 1D array of weights
 * @param gradients pointer to 1D array of gradients of weights (initialized internally)
 * @param center constant for WEIGHTS_INIT_CONSTANT or center of distribution for other initializers
 * @param deviation deviation of distribution (for initialization) (ignored for WEIGHTS_INIT_CONSTANT)
 * @param moments pointer to 1D internal temp array of moments
 * @param velocities_or_cache pointer to 1D internal temp array of velocities or gradients cache (for
 * OPTIMIZER_ADA_GRAD)
 * @param _learning_step index of weights update from start of training (for OPTIMIZER_ADAM)
 */
typedef struct {
    bool trainable;
    uint8_t initializer;
    uint32_t length_total;
    float *weights, *gradients;
    float center, deviation;
    float *moments, *velocities_or_cache;
    uint64_t _learning_step;
} weights_s;

uint8_t weights_check_init(weights_s *weights, uint32_t length_total);

uint8_t weights_init(weights_s *weights, bool from_self);

uint8_t weights_update(weights_s *weights, optimizer_s *optimizer);

size_t weights_estimate_min_size(weights_s *weights);

void weights_destroy(weights_s *weights, bool destroy_struct, bool destroy_internal_array);

#endif
