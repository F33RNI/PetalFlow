/**
 * @file errors.h
 * @author Fern Lane
 * @brief Errors definitions and conversion to string
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
#ifndef ERRORS_H__
#define ERRORS_H__

#include <stdint.h>

#define ERROR_NONE                        0U
#define ERROR_MALLOC                      1U
#define ERROR_PETAL_WRONG_TYPE            2U
#define ERROR_PETAL_WRONG_WEIGHTS_INIT    3U
#define ERROR_PETAL_WRONG_ACTIVATION      4U
#define ERROR_PETAL_SHAPE_ZERO            5U
#define ERROR_PETAL_SHAPE_TOO_BIG         6U
#define ERROR_PETAL_SHAPES_NOT_EQUAL      7U
#define ERROR_ACTIVATION_NO_TEMP          8U
#define ERROR_LOSS_NO_TEMP                9U
#define ERROR_BITMAP_ACCESS_OUT_OF_BOUNDS 10U
#define ERROR_OPTIMIZER_WRONG_TYPE        11U
#define ERROR_FLOWER_NO_PETALS            12U
#define ERROR_LOSS_WRONG_TYPE             13U
#define ERROR_WRONG_BATCH_SIZE            14U

extern const char *error_to_str[15];

#endif
