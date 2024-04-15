/**
 * @file weights.c
 * @author Fern Lane
 * @brief Weights initialization, correction (weights update) and size estimation
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
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "errors.h"
#include "logger.h"
#include "optimizers.h"
#include "petal.h"
#include "weights.h"

/**
 * @brief Checks and initializers weights and gradients if needed
 *
 * @param weights pointer to weights_s struct
 * @param length_total total length of weights (input * output for regular weights, output length for bias weights)
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t weights_check_init(weights_s *weights, uint32_t length_total) {
    // Skip if NULL pointer
    if (!weights)
        return ERROR_NONE;

    // Set weights length
    weights->length_total = length_total;

    // Initialize weights if enabled and not initialized previously
    if (!weights->weights) {
        uint8_t error_temp = weights_init(weights, false);
        if (error_temp != ERROR_NONE) {
            logger(LOG_E, "weights_check_init", "Error initializing weights");
            return error_temp;
        }
    }

    // Allocate memory for gradients
    if (weights->trainable && !weights->gradients) {
        weights->gradients = (float *) calloc(length_total, sizeof(float));
        if (!weights->gradients) {
            logger(LOG_E, "weights_check_init", "Error allocating memory for weights->gradients");
            return ERROR_MALLOC;
        }
    }

    // No error
    return ERROR_NONE;
}

/**
 * @brief Initializes array of weights
 *
 * @param weights pointer to weights_s struct
 * @param from_self sets to true internally in case of recursion (to disable logging). So you need to set it to false
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t weights_init(weights_s *weights, bool from_self) {
    // Nothing to initialize
    if (!weights)
        return ERROR_NONE;

    if (!from_self)
        logger(LOG_I, "weights_init", "Initializing weights using %u initializer", weights->initializer);

    // Allocate memory for weights
    if (!weights->weights) {
        weights->weights = (float *) calloc(weights->length_total, sizeof(float));
        if (!weights->weights) {
            logger(LOG_E, "weights_init", "Error allocating memory for weights->weights array");
            return ERROR_MALLOC;
        }
    }

    // Initialize weights
    // All elemets = center (zeros / ones / constant)
    if (weights->initializer == WEIGHTS_INIT_CONSTANT) {
        for (uint32_t i = 0; i < weights->length_total; ++i)
            weights->weights[i] = weights->center;
    }

    // Uniform random distribution
    else if (weights->initializer == WEIGHTS_INIT_RANDOM_UNIFORM) {
        for (uint32_t i = 0; i < weights->length_total; ++i)
            weights->weights[i] =
                (((float) rand() / (float) RAND_MAX) * 2.0 * weights->deviation) + weights->center - weights->deviation;
    }

    // Gaussian normal distribution
    else if (weights->initializer == WEIGHTS_INIT_RANDOM_GAUSSIAN) {
        float x, y, rsq, f;
        for (uint32_t i = 0; i < weights->length_total; i += 2) {
            do {
                // x and y: -1 to 1
                x = ((float) rand() / (float) RAND_MAX) * 2.f - 1.0;
                y = ((float) rand() / (float) RAND_MAX) * 2.f - 1.0;
                rsq = x * x + y * y;
            } while (rsq >= 1.f || rsq == 0.f);

            f = sqrtf(-2.f * logf(rsq) / rsq);

            // Assign two elements at ones
            weights->weights[i] = x * f * weights->deviation + weights->center;
            if (i < weights->length_total - 1)
                weights->weights[i + 1] = y * f * weights->deviation + weights->center;
        }
    }

    // Xavier or Kaiming uniform or normal distribution
    else if (weights->initializer == WEIGHTS_INIT_XAVIER_GLOROT_UNIFORM ||
             weights->initializer == WEIGHTS_INIT_KAIMING_HE_UNIFORM ||
             weights->initializer == WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN ||
             weights->initializer == WEIGHTS_INIT_KAIMING_HE_GAUSSIAN) {
        // Calculate limit
        float limit = sqrtf(((weights->initializer == WEIGHTS_INIT_XAVIER_GLOROT_UNIFORM ||
                              weights->initializer == WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN)
                                 ? 6.f
                                 : 2.f) /
                            (float) weights->length_total);

        // Start from random uniform or normal distribution
        uint8_t initializer_temp = weights->initializer;
        if (weights->initializer == WEIGHTS_INIT_XAVIER_GLOROT_UNIFORM ||
            weights->initializer == WEIGHTS_INIT_KAIMING_HE_UNIFORM)
            weights->initializer = WEIGHTS_INIT_RANDOM_UNIFORM;
        else
            weights->initializer = WEIGHTS_INIT_RANDOM_GAUSSIAN;
        uint8_t error_temp = weights_init(weights, true);
        weights->initializer = initializer_temp;

        // Exit in case of error
        if (error_temp != ERROR_NONE)
            return error_temp;

        // Convert to Xavier or Kaiming
        for (uint32_t i = 0; i < weights->length_total; ++i)
            weights->weights[i] *= limit;
    }

    // Wrong initializer
    else
        return ERROR_PETAL_WRONG_WEIGHTS_INIT;

    // No error
    return ERROR_NONE;
}

/**
 * @brief Updates weights (learning)
 *
 * @param weights pointer to weights struct with calculated gradients
 * @param optimizer pointer to optimizer_s struct
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t weights_update(weights_s *weights, optimizer_s *optimizer) {
    // Ignore if weights are non-trainable
    if (!weights || !weights->trainable)
        return ERROR_NONE;

    // Allocate temp arrays for learning optimizers
    bool first_run = false;
    if (!weights->velocities_or_cache) {
        first_run = true;
        weights->velocities_or_cache = calloc(weights->length_total, sizeof(float));
        if (!weights->velocities_or_cache) {
            logger(LOG_E, "weights_update", "Error allocating memory for weights->velocities_or_cache");
            return ERROR_MALLOC;
        }
    }
    if (optimizer->type == OPTIMIZER_ADAM) {
        if (!weights->moments) {
            first_run = true;
            weights->moments = calloc(weights->length_total, sizeof(float));
            if (!weights->moments) {
                logger(LOG_E, "weights_update", "Error allocating memory for weights->moments");
                return ERROR_MALLOC;
            }
        }
    }

    // Stochastic / regular gradient descend with momentum
    if (optimizer->type == OPTIMIZER_SGD_MOMENTUM) {
        if (optimizer->momentum > 0.f)
            for (uint32_t i = 0; i < weights->length_total; ++i) {
                // Calculate velocities
                weights->velocities_or_cache[i] = optimizer->momentum * weights->velocities_or_cache[i] -
                                                  optimizer->learning_rate * weights->gradients[i];

                // Update weights
                weights->weights[i] += weights->velocities_or_cache[i];
            }
        else
            for (uint32_t i = 0; i < weights->length_total; ++i)
                // Update weights
                weights->weights[i] -= optimizer->learning_rate * weights->gradients[i];
    }

    // RMS Prop
    else if (optimizer->type == OPTIMIZER_RMS_PROP) {
        for (uint32_t i = 0; i < weights->length_total; ++i) {
            // Update velocities
            weights->velocities_or_cache[i] = optimizer->beta_1 * weights->velocities_or_cache[i] +
                                              (1.f - optimizer->beta_1) * weights->gradients[i] * weights->gradients[i];

            // Update weights
            weights->weights[i] -=
                (optimizer->learning_rate / (sqrtf(weights->velocities_or_cache[i]) + EPSILON)) * weights->gradients[i];
        }
    }

    // AdaGrad
    else if (optimizer->type == OPTIMIZER_ADA_GRAD) {
        for (uint32_t i = 0; i < weights->length_total; ++i) {
            // Update cache
            weights->velocities_or_cache[i] += weights->gradients[i] * weights->gradients[i];

            // Update weights
            weights->weights[i] -=
                optimizer->learning_rate * weights->gradients[i] / (sqrtf(weights->velocities_or_cache[i]) + EPSILON);
        }
    }

    // Adam
    else if (optimizer->type == OPTIMIZER_ADAM) {
        float moment_hat, velocity_hat;
        for (uint32_t i = 0; i < weights->length_total; ++i) {
            // Update moments
            weights->moments[i] =
                optimizer->beta_1 * weights->moments[i] + (1.f - optimizer->beta_1) * weights->gradients[i];

            // Update velocities
            weights->velocities_or_cache[i] = optimizer->beta_2 * weights->velocities_or_cache[i] +
                                              (1.f - optimizer->beta_2) * weights->gradients[i] * weights->gradients[i];

            // Weights correction
            moment_hat = weights->moments[i] / (1.f - powf(optimizer->beta_1, (float) weights->_learning_step + 1.f));
            velocity_hat =
                weights->velocities_or_cache[i] / (1.f - powf(optimizer->beta_2, (float) weights->_learning_step + 1.f));

            // Update weights
            weights->weights[i] -= optimizer->learning_rate * moment_hat / (sqrtf(velocity_hat) + EPSILON);

            // Increment step
            weights->_learning_step++;
        }
    }

    // Wrong type
    else {
        logger(LOG_E, "weights_update", "Wrong optimizer type: %u", optimizer->type);
        return ERROR_OPTIMIZER_WRONG_TYPE;
    }

    // Reset gradient sums
    memset(weights->gradients, 0, weights->length_total * sizeof(float));

    // No error
    return ERROR_NONE;
}

/**
 * @brief Estimates minimum size allocated by weights struct
 *
 * @param weights pointer to weights struct
 * @return size_t memory size in bytes
 */
size_t weights_estimate_min_size(weights_s *weights) {
    size_t min_size = 0U;
    if (weights) {
        // Struct itself
        min_size += sizeof(weights_s);

        // weights
        if (weights->weights)
            min_size += weights->length_total * sizeof(float);

        // gradients
        if (weights->gradients)
            min_size += weights->length_total * sizeof(float);

        // moments
        if (weights->moments)
            min_size += weights->length_total * sizeof(float);

        // velocities_or_cache
        if (weights->velocities_or_cache)
            min_size += weights->length_total * sizeof(float);
    }
    return min_size;
}

/**
 * @brief Frees memory allocated by weights struct
 *
 * @param weights pointer to weights_s struct or NULL
 * @param destroy_struct true to destroy struct itself (set to false if struct was defined manually)
 * @param destroy_internal_array true to also destroy weights->weights array
 */
void weights_destroy(weights_s *weights, bool destroy_struct, bool destroy_internal_array) {
    if (weights) {
        if (!destroy_struct)
            logger(LOG_I, "weights_destroy", "Destroying weights struct's (address: %p) internal data", weights);
        else
            logger(LOG_I, "weights_destroy", "Destroying weights struct with address: %p", weights);

        if (destroy_internal_array && weights->weights)
            free(weights->weights);
        if (weights->gradients)
            free(weights->gradients);
        if (weights->moments)
            free(weights->moments);
        if (weights->velocities_or_cache)
            free(weights->velocities_or_cache);
        if (destroy_struct)
            free(weights);
    }
}
