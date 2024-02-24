/**
 * @file flower.c
 * @author Fern Lane
 * @brief
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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef LOGGING
#include <stdio.h>
#include <time.h>
#endif

#include "errors.h"
#include "flower.h"
#include "labeling.h"
#include "logger.h"
#include "loss.h"
#include "shuffle.h"

/**
 * @brief Initializes flower using array of petals
 *
 * @param petals pointer to an array of pointers of petals
 * @param petals_length number of petals
 * @return flower_s* initialized flower
 */
flower_s *flower_init(petal_s **petals, uint32_t petals_length) {
    // Log
    logger(LOG_I, "flower_init", "Initializing flower with %u petals", petals_length);

    // Allocate struct
    flower_s *flower = calloc(1U, sizeof(flower_s));
    if (!flower) {
        logger(LOG_E, "flower_init", "Error allocating memory for flower_s struct");
        return NULL;
    }

    // Check length of array of petals
    if (petals_length < 1) {
        logger(LOG_E, "flower_init", "A flower cannot have zero petals");
        flower->error_code = ERROR_FLOWER_NO_PETALS;
        return flower;
    }

    // Reset error
    flower->error_code = ERROR_NONE;

    // Copy petals
    flower->petals = petals;
    flower->petals_length = petals_length;

    return flower;
}

float *flower_predict(flower_s *flower, float *input) { return flower_forward(flower, input, false); }

float *flower_forward(flower_s *flower, float *input, bool training) {
    for (uint32_t i = 0; i < flower->petals_length; ++i) {
        // Forward propagation thought each petal
        if (i == 0)
            petal_forward(flower->petals[i], input, training);
        else
            petal_forward(flower->petals[i], flower->petals[i - 1U]->output, training);

        // Check for error
        if (flower->petals[i]->error_code != ERROR_NONE) {
            logger(LOG_E, "flower_forward", "Error during forward propagation: %s",
                   error_to_str[flower->petals[i]->error_code]);
            flower->error_code = flower->petals[i]->error_code;
            return NULL;
        }
    }

    // Return last petal's output layer
    return flower->petals[flower->petals_length - 1]->output;
}

float flower_train_batch(flower_s *flower, uint8_t loss_type, optimizer_s *optimizer, float **inputs_train,
                         float **outputs_true_train, labels_s **outputs_true_train_sparse, uint32_t train_length,
                         float **inputs_test, float **outputs_true_test, labels_s **outputs_true_test_sparse,
                         uint32_t test_length, bool log_accuracy) {
    // Check train length
    if (train_length == 0) {
        logger(LOG_E, "flower_train_batch", "No training data");
        return -1.f;
    }

    // Initialize _loss
    if (!flower->_loss) {
        // Check _loss type
        if (loss_type > LOSS_MAX) {
            logger(LOG_E, "flower_train_batch", "Wrong _loss type: %u", loss_type);
            flower->error_code = ERROR_LOSS_WRONG_TYPE;
            return -1.f;
        }

        flower->_loss = (loss_s *) calloc(1U, sizeof(loss_s));
        if (!flower->_loss) {
            logger(LOG_E, "flower_train_batch", "Error allocating memory for loss_s struct");
            flower->error_code = ERROR_MALLOC;
            return -1.f;
        }
        flower->_loss->type = loss_type;
    }

    // Initialize array for storing true output data in case of sparse labels
    float *output_temp = NULL;
    if (outputs_true_train_sparse) {
        output_temp = malloc(flower->petals[flower->petals_length - 1]->output_shape->length * sizeof(float));
        if (!output_temp) {
            logger(LOG_E, "flower_train_batch", "Error allocating memory for output_temp array");
            flower->error_code = ERROR_MALLOC;
            return -1.f;
        }
    }

#ifdef LOGGING
    // Store batch start time
    time_t time_started;
    if (time(&time_started) == -1)
        time_started = 0;

    // Variables to store accuracy
    float accuracy_train = 0.f;
    float accuracy_test = 0.f;
#endif

    // Variables to store losses
    float loss_train = 0.f;
    float loss_test = 0.f;

    // Variable to check for error
    uint8_t error_temp;

    // Training stage
    for (uint32_t sample_i = 0; sample_i < train_length; ++sample_i) {
        // Shuffle train dataset
        shuffle_2d(inputs_train, outputs_true_train, train_length, flower->petals[0]->input_shape->length,
                   flower->petals[flower->petals_length - 1]->output_shape->length);

        // Forward propagation
        float *predicted = flower_forward(flower, inputs_train[sample_i], true);

        // Check for error
        if (!predicted)
            return -1.f;

        // Use temp output array in case of sparse labels
        if (outputs_true_train_sparse) {
            // Convert into output data
            labels_to_petal_output(outputs_true_train_sparse[sample_i], output_temp,
                                   flower->petals[flower->petals_length - 1]->output_shape->length, 0.f, 1.f);

            // Calculate _loss using sparse labeling
            error_temp = loss_forward(flower->_loss, predicted, output_temp,
                                      flower->petals[flower->petals_length - 1]->output_shape->length);
        }

        // Calculate _loss using array labeling
        else
            error_temp = loss_forward(flower->_loss, predicted, outputs_true_train[sample_i],
                                      flower->petals[flower->petals_length - 1]->output_shape->length);

        // Check _loss calculation error
        if (error_temp != ERROR_NONE) {
            logger(LOG_E, "flower_train_batch", "Error calculating _loss: %s", error_to_str[error_temp]);
            flower->error_code = error_temp;
            return -1.f;
        }

        // Store last _loss
        loss_train = flower->_loss->loss[0];

        // Calculate accuracy sum (for mean)
#ifdef LOGGING
        if (log_accuracy) {
            if (outputs_true_train_sparse)
                accuracy_train += flower_calculate_accuracy(
                    predicted, output_temp, flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
            else
                accuracy_train +=
                    flower_calculate_accuracy(predicted, outputs_true_train[sample_i],
                                              flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
        }
#endif
        // Backpropagate _loss
        loss_backward(flower->_loss, flower->petals[flower->petals_length - 1]->output_shape->length);

        // Backpropagate petals
        for (int32_t petal_i = flower->petals_length - 1; petal_i >= 0; --petal_i) {
            // Backpropagate and calculate gradients
            if (petal_i == flower->petals_length - 1)
                petal_backward(flower->petals[petal_i], flower->_loss->loss, flower->petals[petal_i - 1]->output);
            else if (petal_i == 0)
                petal_backward(flower->petals[petal_i], flower->petals[petal_i + 1]->error_on_input,
                               inputs_train[sample_i]);
            else
                petal_backward(flower->petals[petal_i], flower->petals[petal_i + 1]->error_on_input,
                               flower->petals[petal_i - 1]->output);

            // Check for error
            if (flower->petals[petal_i]->error_code != ERROR_NONE) {
                logger(LOG_E, "flower_train_batch", "Error during backpropagation: %s",
                       error_to_str[flower->petals[petal_i]->error_code]);
                flower->error_code = flower->petals[petal_i]->error_code;
                return -1.f;
            }
        }

        // Update weights after each sample in non-batch mode
        // !optimizer->batch
        if (false)
            for (uint32_t petal_i = 0; petal_i < flower->petals_length; ++petal_i) {
                error_temp = weights_update(flower->petals[petal_i]->weights, optimizer);
                if (error_temp == ERROR_NONE)
                    error_temp = weights_update(flower->petals[petal_i]->bias_weights, optimizer);
                if (error_temp != ERROR_NONE) {
                    logger(LOG_E, "flower_train_batch", "Error updating weights: %s", error_to_str[error_temp]);
                    flower->error_code = error_temp;
                    return -1.f;
                }
            }
    }

    // Update weights after entire batch in batch mode
    // optimizer->batch
    if (true)
        for (uint32_t petal_i = 0; petal_i < flower->petals_length; ++petal_i) {
            error_temp = weights_update(flower->petals[petal_i]->weights, optimizer);
            if (error_temp == ERROR_NONE)
                error_temp = weights_update(flower->petals[petal_i]->bias_weights, optimizer);
            if (error_temp != ERROR_NONE) {
                logger(LOG_E, "flower_train_batch", "Error updating weights: %s", error_to_str[error_temp]);
                flower->error_code = error_temp;
                return -1.f;
            }
        }

    // Testing stage
    if (test_length != 0) {
        for (uint32_t sample_i = 0; sample_i < test_length; ++sample_i) {
            // Shuffle test dataset
            shuffle_2d(inputs_test, outputs_true_test, test_length, flower->petals[0]->input_shape->length,
                       flower->petals[flower->petals_length - 1]->output_shape->length);

            // Forward propagation
            float *predicted = flower_forward(flower, inputs_test[sample_i], false);

            // Check for error
            if (!predicted)
                return -1.f;

            // Use temp output array in case of sparse labels
            if (outputs_true_test_sparse) {
                // Convert into output data
                labels_to_petal_output(outputs_true_test_sparse[sample_i], output_temp,
                                       flower->petals[flower->petals_length - 1]->output_shape->length, 0.f, 1.f);

                // Calculate _loss using sparse labeling
                error_temp = loss_forward(flower->_loss, predicted, output_temp,
                                          flower->petals[flower->petals_length - 1]->output_shape->length);
            }

            // Calculate _loss using array labeling
            else
                error_temp = loss_forward(flower->_loss, predicted, outputs_true_test[sample_i],
                                          flower->petals[flower->petals_length - 1]->output_shape->length);

            // Check _loss calculation error
            if (error_temp != ERROR_NONE) {
                logger(LOG_E, "flower_train_batch", "Error calculating _loss during test: %s",
                       error_to_str[error_temp]);
                flower->error_code = error_temp;
                return -1.f;
            }

#ifdef LOGGING
            // Calculate _loss sum (for mean)
            loss_test += flower->_loss->loss[0];

            // Calculate accuracy sum (for mean)
            if (log_accuracy) {
                if (outputs_true_test_sparse)
                    accuracy_test += flower_calculate_accuracy(
                        predicted, output_temp, flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
                else
                    accuracy_test += flower_calculate_accuracy(
                        predicted, outputs_true_test[sample_i],
                        flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
            }
#endif
        }
    }

#ifdef LOGGING
    // Calculate means
    if (test_length != 0)
        loss_test /= (float) test_length;
    if (log_accuracy) {
        if (train_length != 0)
            accuracy_train /= (float) train_length;
        if (test_length != 0)
            accuracy_test /= (float) test_length;
    }

    // Calculate time
    time_t time_stopped;
    int32_t seconds_passed = -1;
    if (time(&time_stopped) != -1 && time_started > 0)
        seconds_passed = (time_stopped - time_started);

    if (log_accuracy)
        printf("[Train] _loss: %9.6f, accuracy: %6.2f%% avg | [Test] _loss: %9.6f avg, accuracy: %6.2f%% avg\n",
               loss_train, accuracy_train * 100.f, loss_test, accuracy_test * 100.f);

#endif
}

/**
 * @brief Calculates categorical / binary accuracy
 * TODO: Move it to the metrics calculation and add other metrics
 *
 * @param predicted pointer to petal's output layer
 * @param expected pointer to array of "true" petal's output layer
 * @param length length of petal's output layer
 * @param threshold threshold above which (or equal to) a class is considered true. Default: 0.5
 * @return float accuracy
 */
float flower_calculate_accuracy(float *predicted, float *expected, uint32_t length, float threshold) {
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
    labels_s *predicted_labels;
    if (multiple)
        predicted_labels = petal_output_to_labels(predicted, length, threshold);

    // Use as single label
    else {
        uint32_t predicted_labels_labels[] = {petal_output_to_label(predicted, length)};
        predicted_labels = &(labels_s){predicted_labels_labels, 1U};
    }

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
    if (multiple)
        labels_destroy(predicted_labels);
    labels_destroy(expected_labels);

    // Calculate and return accuracy
    return (float) matches_counter / (float) length;
}

/**
 * @brief Estimates minimum size allocated by flower
 *
 * @param flower pointer to flower struct
 * @return size_t memory size in bytes
 */
size_t flower_estimate_min_size(flower_s *flower) {
    size_t min_size = 0U;
    if (flower) {
        // Struct itself
        min_size += sizeof(flower_s);

        // Each petal
        for (uint32_t i = 0; i < flower->petals_length; ++i)
            min_size += petal_estimate_min_size(flower->petals[i]);

        // _loss
        if (flower->petals_length > 0)
            min_size +=
                loss_estimate_min_size(flower->_loss, flower->petals[flower->petals_length - 1]->output_shape->length);
    }
    return min_size;
}

/**
 * @brief Frees memory allocated by flower struct
 *
 * @param flower pointer to flower_s struct
 * @param destroy_petals true to also destroy each petal
 * @param destroy_weights_array true to also destroy weights->weights array for each petal false to not
 * @param destroy_bias_weights_array true to also destroy bias_weights->weights array for each petal false to not
 */
void flower_destroy(flower_s *flower, bool destroy_petals, bool destroy_weights_array,
                    bool destroy_bias_weights_array) {
    logger(LOG_I, "flower_destroy", "Destroying flower struct with address: %p", flower);
    if (destroy_petals)
        for (uint32_t i = 0; i < flower->petals_length; ++i)
            petal_destroy(flower->petals[i], true, destroy_weights_array, destroy_bias_weights_array);
    loss_destroy(flower->_loss);
    free(flower);
}
