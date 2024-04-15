/**
 * @file shuffle.c
 * @author Fern Lane
 * @brief Shuffle functions
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
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "errors.h"
#include "logger.h"
#include "shuffle.h"

/**
 * @brief Shuffles internal arrays of 2d arrays
 *
 * @param array_1 2D array pointer
 * @param array_2 2D array pointer
 * @param array_length number of internal arrays in each array (rows)
 * @param element_size_1 size of each internal array inside array_1 in bytes (cols * sizeof(type of element))
 * @param element_size_2 size of each internal array inside array_2 in bytes (cols * sizeof(type of element))
 * @return true shuffled successfully
 * @return false memory allocation error
 */
bool shuffle_2d(float **array_1, float **array_2, uint32_t array_length, uint32_t element_size_1,
                uint32_t element_size_2) {
    // Allocate buffer with size of internal data
    float *buffer_1 = malloc(element_size_1);
    if (!buffer_1) {
        logger(LOG_E, "shuffle_2d", "Error allocating memory for *buffer_1 array");
        return false;
    }
    float *buffer_2 = malloc(element_size_2);
    if (!buffer_2) {
        logger(LOG_E, "shuffle_2d", "Error allocating memory for *buffer_2 array");
        return false;
    }

    // This is useful if array_length > RAND_MAX to properly generate random numbers from 0 to array_length
    uint32_t rand_multiplier = array_length / RAND_MAX + 1U;

    // Randomly swap elements
    for (uint32_t i = 0; i < array_length; i++) {
        // Generate random index
        uint32_t move_to_index = ((uint32_t) rand() * (uint32_t) rand_multiplier) % array_length;

        // Swap elements in array_1
        memcpy(buffer_1, array_1[move_to_index], element_size_1);
        memcpy(array_1[move_to_index], array_1[i], element_size_1);
        memcpy(array_1[i], buffer_1, element_size_1);

        // Swap elements in array_2
        memcpy(buffer_2, array_2[move_to_index], element_size_2);
        memcpy(array_2[move_to_index], array_2[i], element_size_2);
        memcpy(array_2[i], buffer_2, element_size_2);
    }

    // Clear memory
    free(buffer_1);
    free(buffer_2);

    // No errors
    return true;
}
