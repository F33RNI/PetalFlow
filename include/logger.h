/**
 * @file logger.h
 * @author Fern Lane
 * @brief Defines methods and default config for logging
 * @version 1.0.0
 * @date 2024-02-23
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
#ifndef LOGGER_H
#define LOGGER_H

#include <stdbool.h>
#include <stdint.h>

// Logging levels
#define LOG_D    0U
#define LOG_I    1U
#define LOG_W    2U
#define LOG_E    3U
#define LOG_NONE 255U

// Default time formatter
#ifndef LOGGER_TIME_FORMAT
#define LOGGER_TIME_FORMAT "[%Y-%m-%d %H:%M:%S]"
#endif

// Default level fixed formatter
#ifndef LOGGER_LEVEL_FIXED_FORMAT
#define LOGGER_LEVEL_FIXED_FORMAT "[%-7s]"
#endif

// Default level
#ifndef LOGGER_LEVEL
#define LOGGER_LEVEL LOG_I
#endif

#ifdef LOGGING
extern const char *logger_level_to_str[4];
#endif

void logger(uint8_t level, const char *tag, const char *message_or_format, ...);

#endif
