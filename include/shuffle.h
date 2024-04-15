/**
 * @file shuffle.h
 * @author Fern Lane
 * @brief Shuffle functions definitions
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
#ifndef SHUFFLE_H__
#define SHUFFLE_H__

#include <stdbool.h>
#include <stdint.h>

bool shuffle_2d(float **array_1, float **array_2, uint32_t array_length, uint32_t element_size_1,
                uint32_t element_size_2);

#endif
