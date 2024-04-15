/**
 * @file errors.c
 * @author Fern Lane
 * @brief Errors conversion to string
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

#include "errors.h"

/**
 * @brief Maps each error to string
 *
 */
const char *error_to_str[15] = {
    "No error",                                                         // 0 (ERROR_NONE)
    "Memory allocation error",                                          // 1 (ERROR_MALLOC)
    "Wrong petal type",                                                 // 2 (ERROR_PETAL_WRONG_TYPE)
    "Wrong weights initializer",                                        // 3 (ERROR_PETAL_WRONG_WEIGHTS_INIT)
    "Wrong activation function",                                        // 4 (ERROR_PETAL_WRONG_ACTIVATION)
    "Zero input or output shape",                                       // 5 (ERROR_PETAL_SHAPE_ZERO)
    "Petal shape in some dimension is too big",                         // 6 (ERROR_PETAL_SHAPE_TOO_BIG)
    "Input and output shapes are not equal",                            // 7 (ERROR_PETAL_SHAPES_NOT_EQUAL)
    "activation->_derivatives_temp is NULL",                            // 8 (ERROR_ACTIVATION_NO_TEMP)
    "loss->_derivatives_temp_1 or loss->_derivatives_temp_2 is NULL",   // 9 (ERROR_LOSS_NO_TEMP)
    "Index is out of bounds for bit array",                             // 10 (ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS)
    "Wrong optimizer type",                                             // 11 (ERROR_OPTIMIZER_WRONG_TYPE)
    "No petals in flower",                                              // 12 (ERROR_FLOWER_NO_PETALS)
    "Wrong loss type",                                                  // 13 (ERROR_LOSS_WRONG_TYPE)
    "Wrong number of batches / length of train dataset"                 // 14 (ERROR_WRONG_BATCH_SIZE)
};
