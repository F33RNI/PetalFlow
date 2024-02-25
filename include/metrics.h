/**
 * @file metrics.h
 * @author Fern Lane
 * @brief Stores metrics functions data and types definitions
 * @version 1.0.0
 * @date 2024-01-26
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
#ifndef METRICS_H__
#define METRICS_H__

#include <stdint.h>
#include <time.h>

#define METRICS_TIME_ELAPSED        0U
#define METRICS_LOSS_TRAIN          1U
#define METRICS_ACCURACY_TRAIN      2U
#define METRICS_LOSS_VALIDATION     3U
#define METRICS_ACCURACY_VALIDATION 4U

#ifndef METRICS_PROGRESS_BAR_WIDTH
#define METRICS_PROGRESS_BAR_WIDTH 20U
#endif

/**
 * @struct metrics_s
 * Stores metrics data
 *
 * @param *metrics array of enabled metrics
 * @param metrics_length size of metrics array
 * @param _epoch_index_prev index of last epoch (for checking if epoch was changed)
 * @param _time_now internal variable to store current batch time
 * @param _epoch_time_start internal variable to store each epoch start time
 */
typedef struct {
    uint8_t *metrics;
    uint8_t metrics_length;

    int32_t _epoch_index_prev;
    time_t _time_now, _epoch_time_start, _training_time_start;
} metrics_s;

metrics_s *metrics_init();

void metrics_add(metrics_s *metrics, uint8_t metric);

void metrics_remove(metrics_s *metrics, uint8_t metric);

void metrics_calculate_batch(metrics_s *metrics, uint32_t epoch_index, uint32_t epochs_total, uint32_t batch_index,
                             uint32_t batches_per_epoch, float loss_train, float loss_validation, float accuracy_train,
                             float accuracy_validation);

float metrics_calculate_accuracy(metrics_s *metrics, float *predicted, float *expected, uint32_t length,
                                 float threshold);

void metrics_destroy(metrics_s *metrics);

#endif
