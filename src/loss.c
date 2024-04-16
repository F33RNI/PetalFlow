/**
 * @file loss.c
 * @author Fern Lane
 * @brief Loss functions and their derivatives
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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "errors.h"
#include "logger.h"
#include "loss.h"

/**
 * @brief Calculates loss function
 *
 * Output will be written to the first element of loss->loss array (loss->loss[0])
 * Also, some temp data will be written to the loss->_derivatives_temp
 *
 * @param loss pointer to loss struct:
 * type - loss function (LOSS_...)
 * @param predicted array of predicted y data
 * @param expected true data
 * @param length length of each array
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t loss_forward(loss_s *loss, float *predicted, float *expected, uint32_t length) {
    // Allocate current loss
    if (!loss->loss) {
        loss->loss = calloc(length, sizeof(float));
        if (!loss->loss) {
            logger(LOG_E, "loss_forward", "Error allocating memory for loss->loss array");
            return ERROR_MALLOC;
        }
    }
    // Reset current loss
    memset(loss->loss, 0, length * sizeof(float));

    // Allocate temp arrays for loss functions derivatives
    if (!loss->_derivatives_temp_1) {
        loss->_derivatives_temp_1 = malloc(length * sizeof(float));
        if (!loss->_derivatives_temp_1) {
            logger(LOG_E, "loss_forward", "Error allocating memory for loss->_derivatives_temp_1 array");
            return ERROR_MALLOC;
        }
    }
    if (!loss->_derivatives_temp_2) {
        loss->_derivatives_temp_2 = malloc(length * sizeof(float));
        if (!loss->_derivatives_temp_2) {
            logger(LOG_E, "loss_forward", "Error allocating memory for loss->_derivatives_temp_2 array");
            return ERROR_MALLOC;
        }
    }

    // Mean squared error
    // MSE = 1/n * sum((y_true_i - y_pred_i)^2)
    if (loss->type == LOSS_MEAN_SQUARED_ERROR) {
        for (uint32_t i = 0; i < length; ++i) {
            // Save (y_true - y_predicted) for differentiation
            loss->_derivatives_temp_1[i] = expected[i] - predicted[i];

            // Calculate sum
            loss->loss[0] += loss->_derivatives_temp_1[i] * loss->_derivatives_temp_1[i];
        }

        // Calculate mean
        loss->loss[0] /= (float) length;
    }

    // Mean squared logarithmic error
    // MSLE = 1/n * sum((ln(y_true_i + 1.) - ln(y_pred_i + 1.))^2))
    else if (loss->type == LOSS_MEAN_SQUARED_LOG_ERROR) {
        for (uint32_t i = 0; i < length; ++i) {
            // Save (y_pred + 1) and ln(y_true + 1) - ln(y_pred + 1) for differentiation
            loss->_derivatives_temp_1[i] = predicted[i] + 1.f;
            loss->_derivatives_temp_2[i] = logf(expected[i] + 1.f) - logf(loss->_derivatives_temp_1[i]);

            // Calculate sum
            loss->loss[0] += loss->_derivatives_temp_2[i] * loss->_derivatives_temp_2[i];
        }

        // Calculate mean
        loss->loss[0] /= (float) length;
    }

    // Root mean squared logarithmic error
    // RMSLE = sqrt(1/n * sum((ln(y_true_i + 1.) - ln(y_pred_i + 1.))^2))
    else if (loss->type == LOSS_ROOT_MEAN_SQUARED_LOG_ERROR) {
        for (uint32_t i = 0; i < length; ++i) {
            // Save (y_pred + 1) and ln(y_true + 1) - ln(y_pred + 1) for differentiation
            loss->_derivatives_temp_1[i] = predicted[i] + 1.f;
            loss->_derivatives_temp_2[i] = logf(expected[i] + 1.f) - logf(loss->_derivatives_temp_1[i]);

            // Calculate sum
            loss->loss[0] += loss->_derivatives_temp_2[i] * loss->_derivatives_temp_2[i];
        }

        // Calculate root mean
        loss->loss[0] /= (float) length;
        loss->loss[0] = sqrtf(loss->loss[0]);
    }

    // Mean absolute error
    // MAE = 1/n * sum(|y_true_i - y_pred_i|)
    else if (loss->type == LOSS_MEAN_ABS_ERROR) {
        for (uint32_t i = 0; i < length; ++i) {
            // Save (y_true_i - y_pred_i) and |y_true_i - y_pred_i| for differentiation
            loss->_derivatives_temp_1[i] = expected[i] - predicted[i];
            loss->_derivatives_temp_2[i] = fabsf(loss->_derivatives_temp_1[i]);

            // Calculate sum
            loss->loss[0] += loss->_derivatives_temp_2[i];
        }

        // Calculate mean
        loss->loss[0] /= (float) length;
    }

    // Binary cross-entropy
    // BCE = -1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    else if (loss->type == LOSS_BINARY_CROSSENTROPY) {
        // Calculate sum
        for (uint32_t i = 0; i < length; ++i) {
            // Copy predicted and expected for derivatives
            loss->_derivatives_temp_1[i] = predicted[i];
            loss->_derivatives_temp_2[i] = expected[i];

            loss->loss[0] -=
                expected[i] * logf(predicted[i] + EPSILON) + (1.f - expected[i]) * logf((1.f - predicted[i]) + EPSILON);
        }

        // Calculate mean
        loss->loss[0] /= (float) length;
    }

    // Categorical cross-entropy
    // CCE = -1 * sum(y_true * ln(y_pred))
    else if (loss->type == LOSS_CATEGORICAL_CROSSENTROPY) {
        // Calculate sum
        for (uint32_t i = 0; i < length; ++i) {
            // Copy predicted and expected for derivatives
            loss->_derivatives_temp_1[i] = predicted[i];
            loss->_derivatives_temp_2[i] = expected[i];

            loss->loss[0] -= expected[i] * logf(predicted[i] + EPSILON);
        }
    }

    return ERROR_NONE;
}

/**
 * @brief Applies derivative of loss function
 * (TODO: implement automatic differentiation)
 * Requires activation->_derivatives_temp for some activation functions
 *
 * Output data (derivatives) will be written to the loss->loss array
 *
 * @param loss pointer to loss struct:
 * type - loss function (LOSS_...),
 * loss - stores calculated loss
 * @param length length of predicted and true data (same as for loss_forward)
 * @return uint8_t ERROR_NONE or error code in case of error
 */
uint8_t loss_backward(loss_s *loss, uint32_t length) {
    // Check temp array that stores (y_true - y_pred)
    if (!loss->_derivatives_temp_1 || !loss->_derivatives_temp_2) {
        logger(LOG_E, "loss_backward", "loss->_derivatives_temp_1 or loss->_derivatives_temp_2 is NULL");
        return ERROR_LOSS_NO_TEMP;
    }

    // Mean squared error derivative
    // MSE' = -2(y_true_i - y_pred_i) / n
    if (loss->type == LOSS_MEAN_SQUARED_ERROR) {
        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] = -2.f * loss->_derivatives_temp_1[i] / (float) length;
    }

    // Mean squared logarithmic error derivative
    // MSLE' = -2/n * (ln(y_true_i + 1.) - ln(y_pred_i + 1.)) / (y_pred_i + 1.)
    else if (loss->type == LOSS_MEAN_SQUARED_LOG_ERROR) {
        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] = -2.f / (float) length * loss->_derivatives_temp_2[i] / loss->_derivatives_temp_1[i];
    }

    // Root mean squared logarithmic error derivative
    // RMSLE' = MSLE' / (2 * RMSLE) = -2/n * (ln(y_true_i + 1.) - ln(y_pred_i + 1.)) / (y_pred_i + 1.) / (2 * RMSLE)
    else if (loss->type == LOSS_ROOT_MEAN_SQUARED_LOG_ERROR) {
        // Copy RMSLE value
        float rmsle = loss->loss[0];

        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] = -2.f / (float) length * loss->_derivatives_temp_2[i] / loss->_derivatives_temp_1[i] /
                            (2.f * rmsle + EPSILON);
    }

    // Mean absolute error derivative
    // MAE' = -1/n * (y_true_i - y_pred_i) / |y_true_i - y_pred_i|
    else if (loss->type == LOSS_MEAN_ABS_ERROR) {
        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] =
                -1.f / (float) length * loss->_derivatives_temp_1[i] / (loss->_derivatives_temp_2[i] + EPSILON);
    }

    // Binary cross-entropy derivative
    // BCE' = 1/n * (y_pred_i - y_true_i) / (y_pred_i - y_pred_i * y_pred_i)
    else if (loss->type == LOSS_BINARY_CROSSENTROPY) {
        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] =
                1.f / (float) length * (loss->_derivatives_temp_1[i] - loss->_derivatives_temp_2[i]) /
                (loss->_derivatives_temp_1[i] - loss->_derivatives_temp_1[i] * loss->_derivatives_temp_1[i] + EPSILON);
    }

    // Categorical cross-entropy derivative
    // CCE' = -1 * (y_true / y_pre)
    else if (loss->type == LOSS_CATEGORICAL_CROSSENTROPY) {
        // Calculate derivative
        for (uint32_t i = 0; i < length; ++i)
            loss->loss[i] = -loss->_derivatives_temp_2[i] / (loss->_derivatives_temp_1[i] + EPSILON);
    }

    return ERROR_NONE;
}

/**
 * @brief Estimates minimum size allocated by loss
 *
 * @param loss pointer to loss struct
 * @param output_length length of final petal's output
 * @return size_t memory size in bytes
 */
size_t loss_estimate_min_size(loss_s *loss, uint32_t output_length) {
    size_t min_size = 0U;
    if (loss) {
        // Struct itself
        min_size += sizeof(loss_s);

        // _derivatives_temp_1
        if (loss->_derivatives_temp_1)
            min_size += output_length * sizeof(float);

        // _derivatives_temp_2
        if (loss->_derivatives_temp_2)
            min_size += output_length * sizeof(float);
    }
    return min_size;
}

/**
 * @brief Frees memory allocated by loss struct
 *
 * @param loss pointer to loss_s struct
 */
void loss_destroy(loss_s *loss) {
    if (loss) {
        logger(LOG_I, "loss_destroy", "Destroying loss struct with address: %p", loss);
        if (loss->loss)
            free(loss->loss);
        if (loss->_derivatives_temp_1)
            free(loss->_derivatives_temp_1);
        if (loss->_derivatives_temp_2)
            free(loss->_derivatives_temp_2);
        free(loss);
    }
}
