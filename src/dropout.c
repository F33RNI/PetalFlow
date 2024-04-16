/**
 * @file dropout.c
 * @author Fern Lane
 * @brief Handles petal's dropout as bit map
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
#include <stdint.h>
#include <stdlib.h>

#include "bit_array.h"
#include "dropout.h"
#include "errors.h"
#include "logger.h"
#include "random.h"

/**
 * @brief Sets bits to 1 on indices to drop
 *
 * @param bit_array pointer to bit_array struct
 * @param dropout_ratio 0 to 1
 */
void dropout_generate_indices(bit_array_s *bit_array, float dropout_ratio) {
    uint32_t indices_n_to_drop_or_keep = 0U;

    // Calculate how many indices we need to drop for [0.0, 0.5] interval
    if (dropout_ratio >= 0.f && dropout_ratio <= .5f)
        indices_n_to_drop_or_keep = (float) bit_array->length * dropout_ratio;

    // Calculate how many indices we need to keep for [0.5, 1.0] interval (because in this case it's easer to keep)
    else if (dropout_ratio <= 1.f && dropout_ratio >= .5f)
        indices_n_to_drop_or_keep = bit_array->length - (uint32_t) ((float) bit_array->length * dropout_ratio);

    // Handle out-of-bounds access
    if (indices_n_to_drop_or_keep > bit_array->length) {
        logger(LOG_E, "dropout_generate_indices",
               "Cannot drop or keep %u indices. Out of bounds for bit array with size %u", indices_n_to_drop_or_keep,
               bit_array->length);
        bit_array->error_code = ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS;
        return;
    }

    // Handle 100% keep / drop
    if (indices_n_to_drop_or_keep == bit_array->length)
        for (uint32_t i = 0; i < bit_array->length; ++i)
            bit_array_set_bit(bit_array, i);

    // Set random indexes
    else {
        uint32_t set_counter = 0U;
        uint32_t index_to_set;
        while (set_counter < indices_n_to_drop_or_keep) {
            // Generate random index
            index_to_set = rk_random_() % bit_array->length;

            // Ignore if already set
            if (bit_array_get_bit(bit_array, index_to_set))
                continue;

            // Set bit and increment counter
            bit_array_set_bit(bit_array, index_to_set);
            set_counter++;
        }
    }

    // Invert if we use keep mode
    if (dropout_ratio <= 1.f && dropout_ratio >= .5f)
        bit_array_not(bit_array);
}
