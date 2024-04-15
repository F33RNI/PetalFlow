/**
 * @file bit_array.c
 * @author Fern Lane
 * @brief Provides array of bits based on array of words
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

#include "bit_array.h"
#include "errors.h"
#include "logger.h"

/**
 * @brief Initializes bit array struct
 *
 * @param size_bits required size of data in bits
 * @return bit_array_s* pointer to initialized bit_array_s struct
 */
bit_array_s *bit_array_init(uint32_t size_bits) {
    logger(LOG_I, "bit_array_init", "Initializing bit array with size: %u bits", size_bits);

    bit_array_s *bit_array = (bit_array_s *) calloc(1U, sizeof(bit_array_s));
    if (!bit_array) {
        logger(LOG_E, "bit_array_init", "Error allocating memory for bit_array");
        bit_array->error_code = ERROR_MALLOC;
        return bit_array;
    }

    // Reset error
    bit_array->error_code = ERROR_NONE;

    // Calculate the number of uint32_t needed to represent size_bits and initialize array with zeros
    bit_array->data = (uint32_t *) calloc((size_bits + (uint32_t) 31) / (uint32_t) 32, sizeof(uint32_t));
    if (!bit_array->data) {
        logger(LOG_E, "bit_array_init", "Error allocating memory for bit_array->data");
        bit_array->error_code = ERROR_MALLOC;
        return bit_array;
    }

    bit_array->length = size_bits;
    return bit_array;
}

/**
 * @brief Sets bit to 1 in bit_array
 *
 * @param bit_array pointer to bit_array_s struct
 * @param index index of bit
 */
void bit_array_set_bit(bit_array_s *bit_array, uint32_t index) {
    // Handle out-of-bounds access
    if (index >= bit_array->length) {
        logger(LOG_E, "bit_array_set_bit", "Index %u is out of bounds for bit array with size %u", index,
               bit_array->length);
        bit_array->error_code = ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS;
        return;
    }

    // Set bit
    bit_array->data[index / (uint32_t) 32] |= (uint32_t) 1 << index % (uint32_t) 32;
}

/**
 * @brief Sets bit to 0 in bit_array
 *
 * @param bit_array pointer to bit_array_s struct
 * @param index index of bit
 */
void bit_array_clear_bit(bit_array_s *bit_array, uint32_t index) {
    // Handle out-of-bounds access
    if (index >= bit_array->length) {
        logger(LOG_E, "bit_array_clear_bit", "Index %u is out of bounds for bit array with size %u", index,
               bit_array->length);
        bit_array->error_code = ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS;
        return;
    }

    // Clear bit
    bit_array->data[index / (uint32_t) 32] &= ~((uint32_t) 1 << index % (uint32_t) 32);
}

/**
 * @brief Gets bit value from bit_array
 *
 * @param bit_array pointer to bit_array_s struct
 * @param index index of bit
 * @return true bit is 1
 * @return false bit is 0
 */
bool bit_array_get_bit(bit_array_s *bit_array, uint32_t index) {
    // Handle out-of-bounds access
    if (index >= bit_array->length) {
        logger(LOG_E, "bit_array_get_bit", "Index %u is out of bounds for bit array with size %u", index,
               bit_array->length);
        bit_array->error_code = ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS;
        return false;
    }

    // Return bit state
    return ((bit_array->data[index / (uint32_t) 32] >> index % (uint32_t) 32) & (uint32_t) 1) != (uint32_t) 0;
}

/**
 * @brief Clears bit array entirely (sets all bits to 0)
 *
 * @param bit_array pointer to bit_array_s struct
 */
void bit_array_clear(bit_array_s *bit_array) { memset(bit_array->data, 0, bit_array->length * sizeof(uint32_t)); }

/**
 * @brief Frees memory allocated by bit_array struct
 *
 * @param bit_array pointer to bit_array_s struct
 */
void bit_array_destroy(bit_array_s *bit_array) {
    if (bit_array) {
        logger(LOG_I, "bit_array_destroy", "Destroying bit array struct with address: %p", bit_array);
        if (bit_array->data)
            free(bit_array->data);
        free(bit_array);
    }
}
