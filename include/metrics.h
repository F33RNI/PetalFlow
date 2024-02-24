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

#define METRICS_LOSS     0U
#define METRICS_ACCURACY 1U

/**
 * @struct metrics_s
 * Stores metrics data
 *
 * @param *metrics array of enabled metrics
 * @param metrics_length size of metrics array
 */
typedef struct {
    uint8_t *metrics;
    uint32_t metrics_length;
} metrics_s;

#endif