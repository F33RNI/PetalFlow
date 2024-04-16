/**
 * @file bit_array.h
 * @author Fern Lane
 * @brief Defines and stores bit array data
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
#ifndef BIT_ARRAY_H__
#define BIT_ARRAY_H__

#include <stdbool.h>
#include <stdint.h>

#define BIT_ARRAY_TYPE uint8_t
#define BIT_ARRAY_BITS 8U

/**
 * @struct bit_array_s
 * Stores "array of bits"
 *
 * @param data pointer to array of BIT_ARRAY_TYPE numbers
 * @param length length of array in BIT_ARRAY_TYPE
 * @param error_code initialization or runtime error code
 */
typedef struct {
    BIT_ARRAY_TYPE *data;
    uint32_t length;
    uint8_t error_code;
} bit_array_s;

bit_array_s *bit_array_init(uint32_t size_bits);

void bit_array_set_bit(bit_array_s *bit_array, uint32_t index);

void bit_array_clear_bit(bit_array_s *bit_array, uint32_t index);

bool bit_array_get_bit(bit_array_s *bit_array, uint32_t index);

void bit_array_clear(bit_array_s *bit_array);

void bit_array_destroy(bit_array_s *bit_array);

#endif
