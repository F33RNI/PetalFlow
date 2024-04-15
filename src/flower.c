/**
 * @file flower.c
 * @author Fern Lane
 * @brief Main file that combines everything and allows higher-level access to train and predict functions
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

#include "errors.h"
#include "flower.h"
#include "labeling.h"
#include "logger.h"
#include "loss.h"
#include "metrics.h"
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

/**
 * @brief Alias for flower_forward(flower, input, false)
 *
 * @param flower pointer to initialized flower_s struct
 * @param input pointer to array of input data (must be the same size as 1st petal's input)
 * @return float* pointer to the last petal's output layer
 */
float *flower_predict(flower_s *flower, float *input) { return flower_forward(flower, input, false); }

/**
 * @brief Forward propagation through each petal
 *
 * @param flower pointer to initialized flower_s struct
 * @param input pointer to array of input data (must be the same size as 1st petal's input)
 * @param training true to enable training mode (to apply dropout)
 * @return float* pointer to the last petal's output layer
 */
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

/**
 * @brief Early implementation of backpropagation learning
 *
 * @param flower pointer to initialized flower_s struct
 * @param loss_type loss function (LOSS_...)
 * @param optimizer pointer to initialized optimizer_s struct
 * type - optimizer type (OPTIMIZER_...)
 * learning_rate - learning rate (required for all optimizer types) Default: 0.01
 * momentum - accelerates gradient descent and dampens oscillations (for OPTIMIZER_SGD_MOMENTUM)
 * beta_1 - hyperparameter (for OPTIMIZER_RMS_PROP and OPTIMIZER_ADAM) Default: 0.9
 * beta_2 - hyperparameter (for OPTIMIZER_ADAM) Default: 0.999
 * @param metrics pointer to initialized metrics_s struct
 * @param inputs_train pointer to array of arrays of training input data (train dataset)
 * @param outputs_true_train pointer to array of arrays of training output data (train dataset)
 * @param outputs_true_train_sparse pointer to array of label_s arrays of sparse training output data (1 = [0, 1, ...])
 * @param train_length number of training samples (size of training dataset)
 * @param inputs_validation pointer to array of arrays of validation input data (validation dataset)
 * @param outputs_true_validation pointer to array of arrays of validation output data (train dataset)
 * @param outputs_true_validation_sparse pointer to array of label_s arrays of sparse validation output data
 * @param validation_length number of validation samples (size of validation dataset)
 * @param batch_size samples per batch
 * @param epochs total number of training epochs
 */
void flower_train(flower_s *flower, uint8_t loss_type, optimizer_s *optimizer, metrics_s *metrics, float **inputs_train,
                  float **outputs_true_train, labels_s **outputs_true_train_sparse, uint32_t train_length,
                  float **inputs_validation, float **outputs_true_validation, labels_s **outputs_true_validation_sparse,
                  uint32_t validation_length, uint32_t batch_size, uint32_t epochs) {
    // Check train length
    if (train_length == 0) {
        logger(LOG_E, "flower_train", "No training data");
        return;
    }

    // Initialize _loss
    if (!flower->_loss) {
        // Check _loss type
        if (loss_type > LOSS_MAX) {
            logger(LOG_E, "flower_train", "Wrong _loss type: %u", loss_type);
            flower->error_code = ERROR_LOSS_WRONG_TYPE;
            return;
        }

        flower->_loss = (loss_s *) calloc(1U, sizeof(loss_s));
        if (!flower->_loss) {
            logger(LOG_E, "flower_train", "Error allocating memory for loss_s struct");
            flower->error_code = ERROR_MALLOC;
            return;
        }
        flower->_loss->type = loss_type;
    }

    // Initialize array for storing true output data in case of sparse labels
    float *output_temp = NULL;
    if (outputs_true_train_sparse) {
        output_temp = malloc(flower->petals[flower->petals_length - 1]->output_shape->length * sizeof(float));
        if (!output_temp) {
            logger(LOG_E, "flower_train", "Error allocating memory for output_temp array");
            flower->error_code = ERROR_MALLOC;
            return;
        }
    }

    // Calculate number of batches
    uint32_t batches_per_epoch = train_length / batch_size;
    if (batches_per_epoch * batch_size < train_length)
        batches_per_epoch++;

    // Check it
    if (batches_per_epoch == 0) {
        logger(LOG_E, "flower_train", "Batch size (%u) must be less then dataset length (%u) that should be not 0",
               batch_size, train_length);
        flower->error_code = ERROR_WRONG_BATCH_SIZE;
        return;
    }

    // Log
    logger(LOG_I, "flower_train", "Training started");

    // Iterate each epoch
    for (uint32_t epoch_index = 0; epoch_index < epochs; ++epoch_index) {
        // Log epoch number
        logger(LOG_I, "flower_train", "Epoch: %u/%u", epoch_index + 1, epochs);

        // Shuffle train dataset
        shuffle_2d(inputs_train, outputs_true_train, train_length, flower->petals[0]->input_shape->length,
                   flower->petals[flower->petals_length - 1]->output_shape->length);

        // Iterate each batch
        for (uint32_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
            // Calculate train dataset position and make sure we have at least 1 sample to train on
            uint32_t sample_index_from = batch_index * batch_size;
            uint32_t sample_index_to = batch_index * batch_size + batch_size;
            if (sample_index_to > train_length)
                sample_index_to = train_length;
            if (sample_index_to <= sample_index_from)
                continue;

            // Variables to store accuracy and losses
            float accuracy_train_batch_avg = 0.f;
            float accuracy_validation_avg = 0.f;
            float loss_train_batch_avg = 0.f;
            float loss_validation_avg = 0.f;

            // Variable to check for error
            uint8_t error_temp;

            // --------------------------- //
            // -----  TRAINING STAGE ----- //
            // --------------------------- //
            for (uint32_t sample_index = sample_index_from; sample_index < sample_index_to; ++sample_index) {
                // ----- FORWARD PROPAGATION ----- //
                float *predicted = flower_forward(flower, inputs_train[sample_index], true);

                // Check for error
                if (!predicted)
                    return;

                // Use temp output array in case of sparse labels
                if (outputs_true_train_sparse) {
                    // Convert into output data
                    labels_to_petal_output(outputs_true_train_sparse[sample_index], output_temp,
                                           flower->petals[flower->petals_length - 1]->output_shape->length, 0.f, 1.f);

                    // Calculate _loss using sparse labeling
                    error_temp = loss_forward(flower->_loss, predicted, output_temp,
                                              flower->petals[flower->petals_length - 1]->output_shape->length);
                }

                // Calculate _loss using array labeling
                else
                    error_temp = loss_forward(flower->_loss, predicted, outputs_true_train[sample_index],
                                              flower->petals[flower->petals_length - 1]->output_shape->length);

                // Check _loss calculation error
                if (error_temp != ERROR_NONE) {
                    logger(LOG_E, "flower_train", "Error calculating _loss: %s", error_to_str[error_temp]);
                    flower->error_code = error_temp;
                    return;
                }

                // Add to sum to calculate mean
                loss_train_batch_avg += flower->_loss->loss[0];

                // Calculate accuracy and add to sum to calculate mean
                if (outputs_true_train_sparse)
                    accuracy_train_batch_avg += metrics_calculate_accuracy(
                        metrics, predicted, output_temp,
                        flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
                else
                    accuracy_train_batch_avg += metrics_calculate_accuracy(
                        metrics, predicted, outputs_true_train[sample_index],
                        flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);

                // ----- BACKWARD PROPAGATION ----- //
                loss_backward(flower->_loss, flower->petals[flower->petals_length - 1]->output_shape->length);

                // Backpropagate petals
                for (int32_t petal_i = flower->petals_length - 1; petal_i >= 0; --petal_i) {
                    // Backpropagate and calculate gradients
                    if (petal_i == flower->petals_length - 1)
                        petal_backward(flower->petals[petal_i], flower->_loss->loss,
                                       flower->petals[petal_i - 1]->output);
                    else if (petal_i == 0)
                        petal_backward(flower->petals[petal_i], flower->petals[petal_i + 1]->error_on_input,
                                       inputs_train[sample_index]);
                    else
                        petal_backward(flower->petals[petal_i], flower->petals[petal_i + 1]->error_on_input,
                                       flower->petals[petal_i - 1]->output);

                    // Check for error
                    if (flower->petals[petal_i]->error_code != ERROR_NONE) {
                        logger(LOG_E, "flower_train", "Error during backpropagation: %s",
                               error_to_str[flower->petals[petal_i]->error_code]);
                        flower->error_code = flower->petals[petal_i]->error_code;
                        return;
                    }
                }
            }

            // Calculate mean stats
            loss_train_batch_avg /= (float) (sample_index_to - sample_index_from);
            accuracy_train_batch_avg /= (float) (sample_index_to - sample_index_from);

            // --------------------------- //
            // -----  WEIGHTS UPDATE ----- //
            // --------------------------- //
            for (uint32_t petal_i = 0; petal_i < flower->petals_length; ++petal_i) {
                error_temp = weights_update(flower->petals[petal_i]->weights, optimizer);
                if (error_temp == ERROR_NONE)
                    error_temp = weights_update(flower->petals[petal_i]->bias_weights, optimizer);
                if (error_temp != ERROR_NONE) {
                    logger(LOG_E, "flower_train", "Error updating weights: %s", error_to_str[error_temp]);
                    flower->error_code = error_temp;
                    return;
                }
            }

            // ----------------------------- //
            // -----  VALIDATION STAGE ----- //
            // ----------------------------- //
            if (validation_length > 0) {
                for (uint32_t sample_index = 0; sample_index < validation_length; ++sample_index) {
                    // Forward propagation
                    float *predicted = flower_forward(flower, inputs_validation[sample_index], false);

                    // Check for error
                    if (!predicted)
                        return;

                    // Use temp output array in case of sparse labels
                    if (outputs_true_validation_sparse) {
                        // Convert into output data
                        labels_to_petal_output(outputs_true_validation_sparse[sample_index], output_temp,
                                               flower->petals[flower->petals_length - 1]->output_shape->length, 0.f,
                                               1.f);

                        // Calculate _loss using sparse labeling
                        error_temp = loss_forward(flower->_loss, predicted, output_temp,
                                                  flower->petals[flower->petals_length - 1]->output_shape->length);
                    }

                    // Calculate _loss using array labeling
                    else
                        error_temp = loss_forward(flower->_loss, predicted, outputs_true_validation[sample_index],
                                                  flower->petals[flower->petals_length - 1]->output_shape->length);

                    // Check _loss calculation error
                    if (error_temp != ERROR_NONE) {
                        logger(LOG_E, "flower_train", "Error calculating _loss during validation: %s",
                               error_to_str[error_temp]);
                        flower->error_code = error_temp;
                        return;
                    }

                    // Add to sum to calculate mean
                    loss_validation_avg += flower->_loss->loss[0];

                    // Calculate accuracy and add to sum to calculate mean
                    if (outputs_true_validation_sparse)
                        accuracy_validation_avg += metrics_calculate_accuracy(
                            metrics, predicted, output_temp,
                            flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
                    else
                        accuracy_validation_avg += metrics_calculate_accuracy(
                            metrics, predicted, outputs_true_validation[sample_index],
                            flower->petals[flower->petals_length - 1]->output_shape->length, 0.5f);
                }

                // Calculate mean stats
                loss_validation_avg /= (float) validation_length;
                accuracy_validation_avg /= (float) validation_length;
            }

            // -------------------- //
            // -----  METRICS ----- //
            // -------------------- //
            metrics_calculate_batch(metrics, epoch_index, epochs, batch_index, batches_per_epoch, loss_train_batch_avg,
                                    loss_validation_avg, accuracy_train_batch_avg, accuracy_validation_avg);

            // End of batch
        }
        // End of epoch
    }
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
