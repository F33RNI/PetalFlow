/**
 * @file dropout.c
 * @author Fern Lane
 * @brief Handles petal's dropout as bit map
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

#include "bit_array.h"
#include "dropout.h"
#include "errors.h"
#include "logger.h"

/**
 * @brief Sets bits to 1 on indices to drop
 *
 * @param bit_array pointer to bit_array struct
 * @param dropout_ratio 0 to 1
 */
void dropout_generate_indices(bit_array_s *bit_array, float dropout_ratio) {
    // Calculate how many indices we need to drop
    uint32_t indices_n_to_drop = (float) bit_array->length * dropout_ratio;

    // Handle out-of-bounds access
    if (indices_n_to_drop > bit_array->length) {
        logger(LOG_E, "dropout_generate_indices", "Cannot drop %u indices. Out of bounds for bit array with size %u",
               indices_n_to_drop, bit_array->length);
        bit_array->error_code = ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS;
        return;
    }

    // This is useful if bit_array->length > RAND_MAX to properly generate random numbers from 0 to bit_array->length
    uint32_t rand_multiplier = bit_array->length / RAND_MAX + 1U;

    // Drop random indexes
    uint32_t drop_counter = 0U;
    uint32_t index_to_drop;
    while (drop_counter < indices_n_to_drop) {
        // Generate random index
        index_to_drop = ((uint32_t) rand() * (uint32_t) rand_multiplier) % bit_array->length;

        // Ignore if already dropped
        if (!bit_array_get_bit(bit_array, index_to_drop)) {
            bit_array_set_bit(bit_array, index_to_drop);
            drop_counter++;
        }
    }
}
