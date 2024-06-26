/**
 * @file metrics.c
 * @author Fern Lane
 * @brief Computes some metrics that allows you to track training progress
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
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "labeling.h"
#include "logger.h"
#include "metrics.h"

/**
 * @brief Initializes empty metrics_s struct
 *
 * @return metrics_s* pointer to initialized metrics_s struct
 */
metrics_s *metrics_init(uint32_t log_interval) {
    // Log
    logger(LOG_I, "metrics_init", "Initializing metrics with log_interval: %u", log_interval);

    // Allocate struct
    metrics_s *metrics = calloc(1U, sizeof(metrics_s));
    if (!metrics) {
        logger(LOG_E, "metrics_init", "Error allocating memory for metrics_s struct");
        return NULL;
    }

    // Default epoch index
    metrics->_epoch_index_prev = -1;

    return metrics;
}

/**
 * @brief Adds new metric
 *
 * @param metrics pointer to metrics_s struct or NULL
 * @param metric METRICS_...
 */
void metrics_add(metrics_s *metrics, uint8_t metric) {
    if (!metrics)
        return;

    // Check if already exists
    for (uint8_t i = 0; i < metrics->metrics_length; ++i)
        if (metrics->metrics[i] == metric) {
            logger(LOG_W, "metrics_add", "Metric %u already exists", metric);
            return;
        }

    // Reallocate array and add a new metric
    metrics->metrics_length++;
    metrics->metrics = (uint8_t *) realloc(metrics->metrics, metrics->metrics_length * sizeof(uint8_t));
    if (!metrics->metrics) {
        logger(LOG_E, "metrics_add", "Error reallocating memory for metrics->metrics");
        return;
    }
    metrics->metrics[metrics->metrics_length - 1] = metric;
    logger(LOG_I, "metrics_add", "Added metric: %u", metric);
}

/**
 * @brief Removes metric
 *
 * @param metrics pointer to metrics_s struct or NULL
 * @param metric METRICS_...
 */
void metrics_remove(metrics_s *metrics, uint8_t metric) {
    if (!metrics)
        return;

    // Check if not exists
    int16_t metric_index = -1;
    for (uint8_t i = 0; i < metrics->metrics_length; ++i)
        if (metrics->metrics[i] == metric) {
            metric_index = i;
            break;
        }
    if (metric_index == -1) {
        logger(LOG_W, "metrics_remove", "Metric %u not exists", metric);
        return;
    }

    // Remove metric and reallocate array
    metrics->metrics[metric_index] = 0;
    metrics->metrics_length--;
    metrics->metrics = (uint8_t *) realloc(metrics->metrics, metrics->metrics_length * sizeof(uint8_t));
    if (!metrics->metrics) {
        logger(LOG_E, "metrics_remove", "Error reallocating memory for metrics->metrics");
        return;
    }
    logger(LOG_I, "metrics_remove", "Removed metric: %u", metric);
}

/**
 * @brief Calculates and prints (if needed) metrics for each batch
 *
 * @param metrics pointer to metrics_s struct or NULL
 * @param epoch_index index of current epoch (from 0 to epochs_total - 1)
 * @param epochs_total number of epochs
 * @param batch_index index of current batch (from 0 to batches_per_epoch - 1)
 * @param batches_per_epoch number of batches per epoch
 * @param loss_train average training loss of this batch
 * @param loss_validation average validation loss of this batch
 * @param accuracy_train average training accuracy of this batch
 * @param accuracy_validation average validation accuracy of this batch
 */
void metrics_calculate_batch(metrics_s *metrics, uint32_t epoch_index, uint32_t epochs_total, uint32_t batch_index,
                             uint32_t batches_per_epoch, float loss_train, float loss_validation, float accuracy_train,
                             float accuracy_validation) {
    // Ignore everything if no metrics were specified
    if (!metrics || metrics->metrics_length == 0)
        return;

    // New epoch
    if (epoch_index != metrics->_epoch_index_prev) {
        // Save current time
        time(&metrics->_epoch_time_start);

        // First start / new training
        if (epoch_index == 0)
            time(&metrics->_training_time_start);

        // Save epoch index
        metrics->_epoch_index_prev = epoch_index;
    }

    // Last batch in epoch -> print and reset epoch index to make sure new epoch will be recorded correctly next time
    bool last_batch = batch_index >= batches_per_epoch - 1;
    if (last_batch)
        metrics->_epoch_index_prev = -1;

    // Get current time
    time(&metrics->_time_now);

    // Reset line to print progress bar on top
    printf("\r");

    // Draw progress bar on top of previous one
    float progress = ((float) batch_index + 1.f) / (float) batches_per_epoch;
    uint16_t progress_bar_position = progress * METRICS_PROGRESS_BAR_WIDTH;
    printf("[");
    for (uint16_t i = 0; i < METRICS_PROGRESS_BAR_WIDTH; ++i) {
        if (i < progress_bar_position)
            printf("=");
        else if (i == progress_bar_position)
            printf(">");
        else
            printf(" ");
    }
    int16_t batch_index_digits = floorf(log10f(batches_per_epoch) + 1);
    printf("] %*u/%*u", batch_index_digits, batch_index + 1, batch_index_digits, batches_per_epoch);

    // Iterate all metrics
    for (uint8_t i = 0; i < metrics->metrics_length; ++i) {
        // Time
        if (metrics->metrics[i] == METRICS_TIME_ELAPSED) {
            // Calculate time difference between first epoch and now
            int32_t epoch_time_diff = difftime(metrics->_time_now, metrics->_epoch_time_start);

            // Formatted time
            uint16_t epoch_hours;
            uint8_t epoch_minutes, epoch_seconds;

            // Check time (just in case) and format epoch time
            if (epoch_time_diff >= 0) {
                epoch_hours = epoch_time_diff / 3600;
                epoch_minutes = (epoch_time_diff % 3600) / 60;
                epoch_seconds = epoch_time_diff % 60;
            }

            // Log epoch elapsed time
            printf(" | %02u:%02u:%02u", epoch_hours, epoch_minutes, epoch_seconds);
        }

        // Train loss
        else if (metrics->metrics[i] == METRICS_LOSS_TRAIN)
            printf(" | Tloss: %8.4f", loss_train);

        // Train accuracy
        else if (metrics->metrics[i] == METRICS_ACCURACY_TRAIN)
            printf(" | Tacc: %6.2f%%", accuracy_train * 100.f);

        // Validation loss
        else if (metrics->metrics[i] == METRICS_LOSS_VALIDATION)
            printf(" | Vloss: %8.4f", loss_validation);

        // Validation accuracy
        else if (metrics->metrics[i] == METRICS_ACCURACY_VALIDATION)
            printf(" | Vacc: %6.2f%%", accuracy_validation * 100.f);
    }

    // Flush progress bar and metrics
    fflush(stdout);

    // Print new line if it's the last batch in epoch
    if (last_batch)
        printf("\n");

    if (last_batch && epoch_index == epochs_total - 1) {
        // Calculate time difference between first epoch and now
        int32_t train_time_diff = difftime(metrics->_time_now, metrics->_training_time_start);

        // Formatted time
        uint16_t train_hours;
        uint8_t train_minutes, train_seconds;

        // Check time (just in case) and format epoch time
        if (train_time_diff >= 0) {
            train_hours = train_time_diff / 3600;
            train_minutes = (train_time_diff % 3600) / 60;
            train_seconds = train_time_diff % 60;
            logger(LOG_I, "Metrics", "Training finished in %02u:%02u:%02u", train_hours, train_minutes, train_seconds);
        } else
            logger(LOG_I, "Metrics", "Training finished");
    }

    // sleep(1);
}

/**
 * @brief Calculates categorical / binary accuracy
 *
 * @param metrics pointer to metrics_s struct or NULL
 * @param predicted pointer to petal's output layer
 * @param expected pointer to array of "true" petal's output layer
 * @param length length of petal's output layer
 * @param threshold threshold above which (or equal to) a class is considered true. Default: 0.5
 * @return float accuracy
 */
float metrics_calculate_accuracy(metrics_s *metrics, float *predicted, float *expected, uint32_t length,
                                 float threshold) {
    // Ignore everything if no metrics were specified
    if (!metrics || metrics->metrics_length == 0)
        return 0.f;

    // Handle length=0
    if (length == 0) {
        logger(LOG_E, "flower_calculate_accuracy", "length is 0");
        return 0.f;
    }

    // Convert expected labels
    labels_s *expected_labels = petal_output_to_labels(expected, length, threshold);

    // Check labels
    if (!expected_labels) {
        logger(LOG_E, "flower_calculate_accuracy", "Error converting to expected_labels");
        return 0.f;
    }

    // Check if we have multiple labels
    bool multiple = expected_labels->labels_length > 1;

    // Convert to predicted labels
    labels_s *predicted_labels = NULL;
    if (multiple)
        predicted_labels = petal_output_to_labels(predicted, length, threshold);

    // Use as single label
    else
        predicted_labels = label_to_labels(petal_output_to_label(predicted, length));

    // Check labels
    if (!predicted_labels) {
        logger(LOG_E, "flower_calculate_accuracy", "Error converting to predicted_labels");
        labels_destroy(expected_labels);
        return 0.f;
    }

    // Calculate number of matches
    uint32_t matches_counter = 0;
    for (uint32_t i = 0; i < length; ++i) {
        // Check if this index must be true or false
        bool index_expected = false;
        for (uint32_t expected_i = 0; expected_i < expected_labels->labels_length; ++expected_i)
            if (expected_labels->labels[expected_i] == i) {
                index_expected = true;
                break;
            }

        // Get predicted value
        bool index_predicted = false;
        for (uint32_t predicted_i = 0; predicted_i < predicted_labels->labels_length; ++predicted_i)
            if (predicted_labels->labels[predicted_i] == i) {
                index_predicted = true;
                break;
            }

        // Check match
        if (index_expected == index_predicted)
            matches_counter++;
    }

    // Clean up allocated memory
    labels_destroy(predicted_labels);
    labels_destroy(expected_labels);

    // Calculate and return accuracy
    return (float) matches_counter / (float) length;
}

/**
 * @brief Frees memory allocated by metrics_s struct
 *
 * @param metrics pointer to metrics_s struct or NULL
 */
void metrics_destroy(metrics_s *metrics) {
    if (!metrics)
        return;
    logger(LOG_I, "metrics_destroy", "Destroying metrics struct with address: %p", metrics);
    free(metrics);
}
