/**
 * @file optimizers.h
 * @author Fern Lane
 * @brief Optimizers data and definitions
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
#ifndef OPTIMIZERS_H__
#define OPTIMIZERS_H__

#include <stdbool.h>
#include <stdint.h>

#define OPTIMIZER_SGD_MOMENTUM 0U
#define OPTIMIZER_RMS_PROP     1U
#define OPTIMIZER_ADA_GRAD     2U
#define OPTIMIZER_ADAM         3U

// For error check and tests
#define OPTIMIZER_MAX OPTIMIZER_ADAM

/**
 * @struct optimizer_s
 * Stores optimizer's data
 *
 * @param type optimizer type (OPTIMIZER_...)
 * @param learning_rate learning rate (required for all optimizer types) Default: 0.01
 * @param momentum accelerates gradient descent and dampens oscillations (for OPTIMIZER_SGD_MOMENTUM)
 * @param beta_1 hyperparameter (for OPTIMIZER_RMS_PROP and OPTIMIZER_ADAM) Default: 0.9
 * @param beta_2 hyperparameter (for OPTIMIZER_ADAM) Default: 0.999
 */
typedef struct {
    uint8_t type;
    float learning_rate, momentum, beta_1, beta_2;
} optimizer_s;

#endif
