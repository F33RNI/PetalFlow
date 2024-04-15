/**
 * @file labeling.h
 * @author Fern Lane
 * @brief Defines methods to convert labels between argmax and arrays
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
#ifndef LABELING_H__
#define LABELING_H__

#include <stdint.h>

/**
 * @struct labels_s
 * Stores labels and length of labels array
 *
 * @param labels pointer to array of labels
 * @param labels_length length of array of labels
 */
typedef struct {
    uint32_t *labels;
    uint32_t labels_length;
} labels_s;

uint32_t petal_output_to_label(float *petal_output, uint32_t petal_output_length);

labels_s *label_to_labels(uint32_t label_index);

void label_to_petal_output(uint32_t label_index, float *petal_output, uint32_t petal_output_length, float low,
                           float upper);

labels_s *petal_output_to_labels(float *petal_output, uint32_t petal_output_length, float threshold);

void labels_to_petal_output(labels_s *labels, float *petal_output, uint32_t petal_output_length, float low,
                            float upper);

void labels_destroy(labels_s *labels);

#endif
