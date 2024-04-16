/**
 * @file logger.с
 * @author Fern Lane
 * @brief Logging implementation
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
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "logger.h"

#ifdef LOGGING
/**
 * @brief Maps each logging level to string
 *
 */
const char *logger_level_to_str[4] = {
    "DEBUG",     // 0 - DEBUG
    "INFO",      // 1 - INFO
    "WARNING",   // 2 - WARNING
    "ERROR",     // 3 - ERROR
};
#endif

/**
 * @brief Formats and prints logging entry
 *
 * @param level logging level (LOG_D, LOG_I, LOG_W, LOG_E or LOG_NONE)
 * @param tag logging tag (for ex. name of void)
 * @param message_or_format (message to log or format for other arguments)
 * @param ... (other logger arguments)
 * ex. logger(LOG_I, "TAG", "Hello world %d, %.2f", 123, 4.5678f);
 * [YYYY-MM-DD HH:MM:SS] [INFO] [TAG] Hello world 123, 4.57
 */
void logger(uint8_t level, const char *tag, const char *message_or_format, ...) {
    // Ignore everything if logging not enabled
#ifdef LOGGING
    // Ignore if level is below specified in config, NONE or > 3 (3 = ERROR)
    if (level < LOGGER_LEVEL || level == LOG_NONE || level > 3)
        return;

// Time
#ifndef LOGGER_DISABLE_TIME
    // Get current time
    time_t time_now;
    time(&time_now);
    struct tm *local_time = localtime(&time_now);

    // Format time
    char time_str[100];
    strftime(time_str, sizeof(time_str), LOGGER_TIME_FORMAT, local_time);

    // Print timestamp
    printf("%s", time_str);
#endif

// Level
#ifndef LOGGER_DISABLE_LEVEL
    printf(" ");
#ifdef LOGGER_LEVEL_FIXED
    printf(LOGGER_LEVEL_FIXED_FORMAT, logger_level_to_str[level]);
#else
    printf("[%s]", logger_level_to_str[level]);
#endif

#endif

// Tag
#ifndef LOGGER_DISABLE_TAG
    printf(" [%s]", tag);
#endif

    // Formatted message
    printf(" ");
    va_list args;
    va_start(args, message_or_format);
    vprintf(message_or_format, args);
    va_end(args);
    printf("\n");
#endif
}
