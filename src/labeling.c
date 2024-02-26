/**
 * @file labeling.c
 * @author Fern Lane
 * @brief Converts labels between argmax and arrays
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
#include <stdint.h>
#include <stdlib.h>

#include "errors.h"
#include "labeling.h"
#include "logger.h"

/**
 * @brief Converts petal's output layer into single label index (aka argmax)
 *
 * @param petal_output pointer to array of petal's output layer
 * @param petal_output_length length of petal's output (number of classes)
 * @return uint32_t label index (0 to petal_output_length - 1)
 */
uint32_t petal_output_to_label(float *petal_output, uint32_t petal_output_length) {
    float max_value = petal_output[0];
    uint32_t label_index = 0;
    for (uint32_t i = 0; i < petal_output_length; ++i)
        if (petal_output[i] > max_value) {
            max_value = petal_output[i];
            label_index = i;
        }

    return label_index;
}

/**
 * @brief Converts single label index (aka argmax) into array of labels (see metrics_calculate_accuracy())
 *
 * @return labels_s* pointer to struct containing labels indices and number of them
 */
labels_s *label_to_labels(uint32_t label_index) {
    // Allocate labels_s struct
    labels_s *labels = (labels_s *) calloc(1U, sizeof(labels_s));
    if (!labels) {
        logger(LOG_E, "label_to_labels", "Error allocating memory for labels_s struct");
        return NULL;
    }

    // Only 1 label
    labels->labels_length = 1;

    // Allocate array
    labels->labels = (uint32_t *) realloc(labels->labels, labels->labels_length * sizeof(uint32_t));
    if (!labels->labels) {
        logger(LOG_E, "petal_output_to_labels", "Error reallocating memory for labels->labels array");
        return NULL;
    }

    // Set label index
    labels->labels[0] = label_index;

    return labels;
}

/**
 * @brief Converts label index (single one) into array. ex.: 2 = [0, 0, 1, 0, ..., 0]
 *
 * @param label_index label index (0 to petal_output_length - 1)
 * @param petal_output pointer to target array to store data
 * @param petal_output_length length of target array (number of classes)
 * @param low default output value. Default: 0.0
 * @param upper value at label index. Default: 1.0
 */
void label_to_petal_output(uint32_t label_index, float *petal_output, uint32_t petal_output_length, float low,
                           float upper) {
    // Fill entire array with low values
    for (uint32_t i = 0; i < petal_output_length; ++i)
        petal_output[i] = low;

    // Check index and write upper value
    if (label_index < petal_output_length)
        petal_output[label_index] = upper;

    // Log error
    else
        logger(LOG_E, "label_to_petal_output", "Index %u is out of bounds for array with size %u", label_index,
               petal_output_length);
}

/**
 * @brief Converts petal's output layer into multiple label indexes
 *
 * @param petal_output pointer to array of petal's output layer
 * @param petal_output_length length of petal's output (number of classes)
 * @param threshold threshold above which (or equal to) a class is considered true. Default: 0.5
 * @return labels_s* pointer to struct containing labels indices and number of them
 */
labels_s *petal_output_to_labels(float *petal_output, uint32_t petal_output_length, float threshold) {
    // Allocate labels_s struct
    labels_s *labels = (labels_s *) calloc(1U, sizeof(labels_s));
    if (!labels) {
        logger(LOG_E, "petal_output_to_labels", "Error allocating memory for labels_s struct");
        return NULL;
    }

    // Dynamically relocate arrays and add labels
    for (uint32_t i = 0; i < petal_output_length; ++i) {
        if (petal_output[i] >= threshold) {
            // Increment size and reallocate array
            labels->labels_length++;
            labels->labels = (uint32_t *) realloc(labels->labels, labels->labels_length * sizeof(uint32_t));
            if (!labels->labels) {
                logger(LOG_E, "petal_output_to_labels", "Error reallocating memory for labels->labels array");
                return NULL;
            }

            // Append index
            labels->labels[labels->labels_length - 1] = i;
        }
    }

    return labels;
}

/**
 * @brief Converts multiple label indices into array. ex.: [0, 2] = [1, 0, 1, 0, ..., 0]
 *
 * @param labels pointer to struct containing labels indices and number of them
 * @param petal_output pointer to target array to store data
 * @param petal_output_length length of target array (number of classes)
 * @param low default output value. Default: 0.0
 * @param upper value at label index. Default: 1.0
 */
void labels_to_petal_output(labels_s *labels, float *petal_output, uint32_t petal_output_length, float low,
                            float upper) {
    // Fill entire array with low values
    for (uint32_t i = 0; i < petal_output_length; ++i)
        petal_output[i] = low;

    // Write upper value
    for (uint32_t i = 0; i < labels->labels_length; ++i) {
        // Check index
        if (labels->labels[i] < petal_output_length)
            petal_output[labels->labels[i]] = upper;

        // Log error
        else
            logger(LOG_E, "label_to_petal_output", "Index %u is out of bounds for array with size %u",
                   petal_output[labels->labels[i]], petal_output_length);
    }
}

/**
 * @brief Frees memory allocated by labels struct
 *
 * @param labels pointer to labels_s struct
 */
void labels_destroy(labels_s *labels) {
    if (labels) {
        // logger(LOG_D, "labels_destroy", "Destroying labels struct with address: %p", labels);
        if (labels->labels)
            free(labels->labels);
        free(labels);
    }
}
