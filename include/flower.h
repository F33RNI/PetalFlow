/**
 * @file flower.h
 * @author Fern Lane
 * @brief Flower struct and high-level functions definitions
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
#ifndef FLOWER_H__
#define FLOWER_H__

#include <stddef.h>
#include <stdint.h>

#include "labeling.h"
#include "loss.h"
#include "metrics.h"
#include "optimizers.h"
#include "petal.h"

/**
 * @struct flower_s
 * Stores flower's petals and other flower's data
 *
 * @param petals pointer to array of pointers of petals
 * @param petals_length number of petals (length of petals array)
 * @param _loss internal pointer to _loss struct
 * @param error_code initialization or runtime error code
 */
typedef struct {
    petal_s **petals;
    uint32_t petals_length;

    loss_s *_loss;
    uint8_t error_code;
} flower_s;

flower_s *flower_init(petal_s **petals, uint32_t petals_length);

float *flower_predict(flower_s *flower, float *input);

float *flower_forward(flower_s *flower, float *input, bool training);

void flower_train(flower_s *flower, uint8_t loss_type, optimizer_s *optimizer, metrics_s *metrics, float **inputs_train,
                  float **outputs_true_train, labels_s **outputs_true_train_sparse, uint32_t train_length,
                  float **inputs_validation, float **outputs_true_validation, labels_s **outputs_true_validation_sparse,
                  uint32_t validation_length, uint32_t batch_size, uint32_t epochs);

size_t flower_estimate_min_size(flower_s *flower);

void flower_destroy(flower_s *flower, bool destroy_petals, bool destroy_weights_array, bool destroy_bias_weights_array);

#endif
