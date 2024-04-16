/**
 * @file main.c
 * @author Fern Lane
 * @brief Basic usage and tests for some functions
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
#include <string.h>

#include "activation.h"
#include "dropout.h"
#include "errors.h"
#include "flower.h"
#include "loss.h"
#include "metrics.h"
#include "optimizers.h"
#include "petal.h"
#include "random.h"

// h for approximating derivative
#define PERTURB_H 0.001f

// Epsilon to prevent division by zero and other undefined states
#ifndef EPSILON
#define EPSILON 1e-15f
#endif

/**
 * @brief Prints 1D array as 1D or multidimensional array
 *
 * @param array pointer to array
 * @param rows number of rows (height)
 * @param cols number of columns (width)
 * @param depth number of channel (depth)
 */
void print_array(float *array, uint32_t rows, uint32_t cols, uint32_t depth) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            if (depth > 1)
                printf("(");
            for (uint32_t channel = 0; channel < depth; ++channel) {
                printf("%.4f", array[(row * cols + col) * depth + channel]);

                if (channel < depth - 1)
                    printf(", ");
            }
            if (depth > 1)
                printf(")");
            printf("\t");
        }
        printf("\n");
    }
}

/**
 * @brief Checks if function and it's derivative are correct
 * by comparing actual derivative with approximated one
 *
 * @param derivative pointer to actual derivative
 * @param derivative_approx pointer to approximated derivative (using PERTURB_H)
 * @param array_length length of each array
 * @param delta maximum allowed deviation
 * @return true arrays are match
 * @return false not matched
 */
bool check_match(float *derivative, float *derivative_approx, uint32_t array_length, float delta) {
    for (uint32_t i = 0; i < array_length; ++i) {
        if (fabsf(derivative[i] - derivative_approx[i]) > delta) {
            printf("Failed\n");
            return false;
        }
    }
    printf("Passed\n");
    return true;
}

/**
 * @brief Tests activation functions independently using numerical derivative
 * approximation
 *
 * @param activation pointer to activation struct
 * @param test_data pointer to a 1D array of test data
 * @param test_data_length length of test array
 * @return true passed
 * @return false not passed
 */
bool check_activation(activation_s *activation, float *test_data, uint32_t test_data_length) {
    // Initialize temp arrays
    float *test_temp_forward;
    if (activation->type == ACTIVATION_SOFTMAX)
        test_temp_forward = (float *) malloc(test_data_length * test_data_length * sizeof(float));
    else
        test_temp_forward = (float *) malloc(test_data_length * sizeof(float));
    if (!test_temp_forward) {
        printf("Error allocating test_temp_forward!");
        return false;
    }
    float *test_temp = (float *) malloc(test_data_length * sizeof(float));
    if (!test_temp) {
        printf("Error allocating test_temp!");
        return false;
    }

    // Copy data
    memcpy(test_temp_forward, test_data, test_data_length * sizeof(float));
    for (uint32_t i = 0; i < test_data_length; ++i)
        test_temp[i] = test_temp_forward[i] + PERTURB_H;

    // Set some activation variables if needed and print initial message
    switch (activation->type) {
    case ACTIVATION_LINEAR:
        activation->linear_alpha = .5f;
        activation->linear_const = 1.f;
        printf("Linear activation with a=%.2f, c=%.2f:\t", activation->linear_alpha, activation->linear_const);
        break;

    case ACTIVATION_RELU:
        activation->relu_leak = 0.1f;
        printf("ReLU activation with leak=%.2f:\t\t", activation->relu_leak);
        break;

    case ACTIVATION_ELU:
        activation->elu_alpha = 0.1f;
        printf("ELU activation with alpha=%.2f:\t\t", activation->elu_alpha);
        break;

    case ACTIVATION_SOFTSIGN:
        printf("Softsign activation:\t\t\t");
        break;

    case ACTIVATION_SIGMOID:
        printf("Sigmoid activation:\t\t\t");
        break;

    case ACTIVATION_HARD_SIGMOID:
        printf("Hard-sigmoid activation:\t\t");
        break;

    case ACTIVATION_SWISH:
        activation->swish_beta = 2.f;
        printf("E-Swish activation with beta=%.2f:\t", activation->swish_beta);
        break;

    case ACTIVATION_SOFTMAX:
        printf("Softmax activation:\t\t\t");
        break;

    case ACTIVATION_TANH:
        printf("tanh activation:\t\t\t");
        break;

    default:
        printf("Wrong activation type: %u\n", activation->type);
        free(test_temp);
        free(test_temp_forward);
        return false;
        break;
    }

    // Forward
    activation_forward(activation, test_temp_forward, test_data_length, NULL);
    print_array(test_temp_forward, 1U, test_data_length, 1U);

    // Forward + epsilon
    activation_forward(activation, test_temp, test_data_length, NULL);

    // Calculate derivative approximation
    for (uint32_t i = 0; i < test_data_length; ++i)
        test_temp[i] = (test_temp[i] - test_temp_forward[i]) / PERTURB_H;

    bool match;
    // Softmax
    if (activation->type == ACTIVATION_SOFTMAX) {
        printf("Derivative:\n");
        activation_backward(activation, test_temp_forward, test_data_length, NULL);
        print_array(test_temp_forward, test_data_length, test_data_length, 1U);
        float softmax_check[] = {
            0.011520363521412946f,  -0.00036932676448486745f, -0.0010039341868832707f, -0.0027289760764688253f,
            -0.007418126333504915f, -0.00036932676448486745f, 0.03068098600488156f,    -0.0027289758436381817f,
            -0.007418125867843628f, -0.020164556801319122f,   -0.0010039341868832707f, -0.0027289758436381817f,
            0.07871041493490338f,   -0.020164556801319122f,   -0.054812945425510406f,  -0.0027289760764688253f,
            -0.007418125867843628f, -0.020164556801319122f,   0.1793087050318718f,     -0.14899703860282898f,
            -0.007418126333504915f, -0.020164556801319122f,   -0.054812945425510406f,  -0.14899703860282898f,
            0.23139268159866333f};
        match = check_match(test_temp_forward, softmax_check, test_data_length, 0.01f);
    }

    // Other functions
    else {
        // Backward
        printf("Derivative:\t\t\t\t");
        activation_backward(activation, test_temp_forward, test_data_length, NULL);
        print_array(test_temp_forward, 1U, test_data_length, 1U);

        // Check
        printf("Derivative approximation:\t\t");
        print_array(test_temp, 1U, test_data_length, 1U);
        match = check_match(test_temp_forward, test_temp, test_data_length, 0.01f);
    }

    // Clean and exit
    free(test_temp);
    free(test_temp_forward);
    return match;
}

/**
 * @brief Tests loss functions independently using numerical derivative
 * approximation
 *
 * @param loss pointer to loss struct
 * @param test_predicted pointer to 1D array of test predicted data
 * @param test_expected pointer to 1D array of test true data
 * @param test_length length of each array
 * @return true passed
 * @return false not passed
 */
bool check_loss(loss_s *loss, float *test_predicted, float *test_expected, uint32_t test_length) {
    // Set some activation variables if needed and print initial message
    switch (loss->type) {
    case LOSS_MEAN_SQUARED_ERROR:
        printf("Mean squared loss:\t\t\t\t");
        break;

    case LOSS_MEAN_SQUARED_LOG_ERROR:
        printf("Mean squared logarithmic loss:\t\t\t");
        break;

    case LOSS_ROOT_MEAN_SQUARED_LOG_ERROR:
        printf("Root mean squared logarithmic loss:\t\t");
        break;

    case LOSS_MEAN_ABS_ERROR:
        printf("Mean absolute loss:\t\t\t\t");
        break;

    case LOSS_BINARY_CROSSENTROPY:
        printf("Binary cross-entropy loss:\t\t\t");
        break;

    case LOSS_CATEGORICAL_CROSSENTROPY:
        printf("Categorical cross-entropy loss:\t\t\t");
        break;

    default:
        printf("Wrong loss type: %u\n", loss->type);
        return false;
        break;
    }

    // Allocate temp arrays
    float *perturbed_predicted = malloc(test_length * sizeof(float));
    if (!perturbed_predicted) {
        printf("Error allocating perturbed_predicted!");
        return false;
    }
    float *derivative_approximated = malloc(test_length * sizeof(float));
    if (!derivative_approximated) {
        printf("Error allocating derivative_approximated!");
        return false;
    }
    float *perturbations = malloc(test_length * sizeof(float));
    if (!perturbations) {
        printf("Error allocating perturbations!");
        return false;
    }

    // Perturb each input and calculate the numerical derivative approximation
    for (uint32_t i = 0; i < test_length; ++i) {
        perturbations[i] = test_predicted[i] * PERTURB_H;
        // Perturb the input
        for (uint32_t j = 0; j < test_length; ++j)
            perturbed_predicted[j] = test_predicted[j];
        perturbed_predicted[i] += perturbations[i];

        // Calculate the perturbed loss
        loss_forward(loss, perturbed_predicted, test_expected, test_length);

        derivative_approximated[i] = loss->loss[0];
    }

    // Forward
    loss_forward(loss, test_predicted, test_expected, test_length);
    printf("%.4f\n", loss->loss[0]);

    // Calculate approximations
    for (uint32_t i = 0; i < test_length; ++i)
        derivative_approximated[i] = (derivative_approximated[i] - loss->loss[0]) / (perturbations[i] + EPSILON);

    // Backward
    printf("Derivative:\t\t\t\t\t");
    loss_backward(loss, test_length);
    print_array(loss->loss, 1U, test_length, 1U);

    // Check
    printf("Derivative approximation:\t\t\t");
    print_array(derivative_approximated, 1U, test_length, 1U);
    bool match = check_match(loss->loss, derivative_approximated, test_length, 0.01f);

    // Clean and exit
    free(perturbations);
    free(derivative_approximated);
    free(perturbed_predicted);
    return match;
}

/**
 * @brief Tests activation functions and their derivatives
 *
 * @return uint8_t number of fails
 */
uint8_t test_activation_full() {
    // Fails counter
    uint8_t fails = 0U;

    // Initialize test data
    uint32_t test_length = 5U;
    float test_data[] = {-2.f, -1.f, 0.f, 1.f, 2.f};

    // Print test data
    printf("\nTesting activation functions on data:\t");
    print_array(test_data, 1U, test_length, 1U);
    printf("\n");

    // Initialize activation_s struct
    activation_s *activation = calloc(1U, sizeof(activation_s));

    // Test each activation function and it's derivative
    for (uint8_t type = 0; type <= ACTIVATION_MAX; ++type) {
        activation->type = type;
        if (!check_activation(activation, test_data, test_length))
            fails++;
        printf("\n");
    }

    // Clean and exit
    activation_destroy(activation);
    return fails;
}

/**
 * @brief Tests loss functions and their derivatives
 *
 * @return uint8_t number of fails
 */
uint8_t test_loss_full() {
    // Fails counter
    uint8_t fails = 0U;

    // Initialize test data
    uint32_t test_length = 6U;
    float test_predicted[] = {0.0f, 0.5f, 0.1f, 0.9f, 0.4f, 0.9f};
    float test_expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.f, 0.f};

    // Print test data
    printf("\nTesting loss functions on predicted data:\t");
    print_array(test_predicted, 1U, test_length, 1U);
    printf("Testing loss functions on expected data:\t");
    print_array(test_expected, 1U, test_length, 1U);
    printf("\n");

    // Initialize loss_s struct
    loss_s *loss = calloc(1U, sizeof(loss_s));

    // Test each loss function and it's derivative
    for (uint8_t type = 0; type <= LOSS_MAX; ++type) {
        loss->type = type;
        if (!check_loss(loss, test_predicted, test_expected, test_length))
            fails++;
        printf("\n");
    }

    // Clean and exit
    loss_destroy(loss);
    return fails;
}

/**
 * @brief Tests dropout bit array generation
 *
 * @return uint8_t number of fails (0 or 1)
 */
uint8_t test_dropout() {
    // Test data
    uint32_t bit_size = 50U;
    float target_ratio = 0.2;
    printf("\nTesting dropout on array with size %u and ratio: %.2f\n", bit_size, target_ratio);

    // Initialize array of bits
    bit_array_s *bit_array = bit_array_init(bit_size);

    // Calculate dropout
    dropout_generate_indices(bit_array, target_ratio);

    // Calculate ones and print array
    printf("Array of bits: ");
    uint32_t ones_counter = 0U;
    for (uint32_t i = 0; i < bit_size; ++i) {
        if (bit_array_get_bit(bit_array, i)) {
            ones_counter++;
            printf("1");
        } else
            printf("0");
    }
    printf("\n");

    // Calculated ratio
    float ones_ratio = (float) ones_counter / (float) bit_size;
    printf("Bits set: %u (%.4f%%)\n", ones_counter, ones_ratio * 100.f);

    // Clear
    bit_array_destroy(bit_array);

    // Check
    if (fabsf(target_ratio - ones_ratio) < 0.001f) {
        printf("Passed\n");
        return 0;
    }
    printf("Failed\n");
    return 1;
}

/**
 * @brief Tests all normalization petal types
 *
 * @return uint8_t number of fails
 */
uint8_t test_normalization() {
    // Fails counter
    uint8_t fails = 0U;

    // Test data
    float inputs[] = {2.f, 0.f, 10.f, -1.f, 1.f, 8.f, 2.f, 1.5f, 0.5f, -0.4f, -0.1f, 0.1f};
    petal_shape_s input_shape = (petal_shape_s){1U, 12U, 1U, 0UL};
    petal_shape_s output_shape = (petal_shape_s){1U, 12U, 1U, 0UL};
    printf("\nTesting normalization petals\n");

    // 1D
    petal_s *petal =
        petal_init(PETAL_TYPE_NORMALIZE_ALL, false, &input_shape, &output_shape, NULL, NULL, NULL, 0.f, 0.f, 1.f);
    printf("1D (PETAL_TYPE_NORMALIZE_ALL) Input data:\n");
    print_array(inputs, input_shape.rows, input_shape.cols, input_shape.depth);
    printf("Normalized:\n");
    petal_forward(petal, inputs, false);
    print_array(petal->output, output_shape.rows, output_shape.cols, output_shape.depth);
    float min = INFINITY;
    float max = -INFINITY;
    for (uint32_t i = 0; i < 12U; ++i) {
        if (petal->output[i] > max)
            max = petal->output[i];
        if (petal->output[i] < min)
            min = petal->output[i];
    }
    printf("Output range: %.4f to %.4f\n", min, max);
    if (min != -1.f || max != 1.f) {
        printf("Failed\n");
        fails++;
    } else
        printf("Passed\n");
    petal_destroy(petal, true, true, true);
    printf("\n");

    // 2D
    input_shape.rows = 3U;
    input_shape.cols = 4U;
    output_shape.rows = 3U;
    output_shape.cols = 4U;
    petal =
        petal_init(PETAL_TYPE_NORMALIZE_IN_ROWS, false, &input_shape, &output_shape, NULL, NULL, NULL, 0.f, 0.f, 1.f);
    printf("2D (PETAL_TYPE_NORMALIZE_IN_ROWS) Input data:\n");
    print_array(inputs, input_shape.rows, input_shape.cols, input_shape.depth);
    printf("Normalized:\n");
    petal_forward(petal, inputs, false);
    print_array(petal->output, output_shape.rows, output_shape.cols, output_shape.depth);
    min = INFINITY;
    max = -INFINITY;
    for (uint32_t i = 0; i < 12U; ++i) {
        if (petal->output[i] > max)
            max = petal->output[i];
        if (petal->output[i] < min)
            min = petal->output[i];
    }
    printf("Output range: %.4f to %.4f\n", min, max);
    if (min != -1.f || max != 1.f) {
        printf("Failed\n");
        fails++;
    } else
        printf("Passed\n");
    petal_destroy(petal, true, true, true);
    printf("\n");

    // 3D
    input_shape.rows = 3U;
    input_shape.cols = 2U;
    input_shape.depth = 2U;
    output_shape.rows = 3U;
    output_shape.cols = 2U;
    output_shape.depth = 2U;
    petal = petal_init(PETAL_TYPE_NORMALIZE_IN_CHANNELS, false, &input_shape, &output_shape, NULL, NULL, NULL, 0.f, 0.f,
                       1.f);
    printf("3D (PETAL_TYPE_NORMALIZE_IN_CHANNELS) Input data:\n");
    print_array(inputs, input_shape.rows, input_shape.cols, input_shape.depth);
    printf("Normalized:\n");
    petal_forward(petal, inputs, false);
    print_array(petal->output, output_shape.rows, output_shape.cols, output_shape.depth);
    min = INFINITY;
    max = -INFINITY;
    for (uint32_t i = 0; i < 12U; ++i) {
        if (petal->output[i] > max)
            max = petal->output[i];
        if (petal->output[i] < min)
            min = petal->output[i];
    }
    printf("Output range: %.4f to %.4f\n", min, max);
    if (min != -1.f || max != 1.f) {
        printf("Failed\n");
        fails++;
    } else
        printf("Passed\n");
    petal_destroy(petal, true, true, true);

    return fails;
}

/**
 * @brief Generates 2D array of random floats from -10.0 to 10.0
 *
 * @param rows number of rows (outer array length)
 * @param cols number of elements in each internal array
 * @return float** 2D array of random floats
 */
float **dense_generate_input_data(uint32_t rows, uint32_t cols) {
    // Allocate memory for the outer array (rows)
    float **array = (float **) malloc(rows * sizeof(float *));
    if (!array) {
        printf("Error allocating array for dense_generate_input_data!\n");
        return NULL;
    }

    for (uint32_t row = 0; row < rows; ++row) {
        // Allocate memory for each internal array (columns)
        array[row] = (float *) malloc(cols * sizeof(float));
        if (!array[row]) {
            printf("Error allocating array[row] for dense_generate_input_data!\n");
            return NULL;
        }

        // Populate the internal array with random float values in (-10, 10] interval
        for (uint32_t col = 0; col < cols; ++col) {
            array[row][col] = rk_float_() * 20.f - 10.f;
        }
    }
    return array;
}

/**
 * @brief Generates 2D array of expected outputs by comparing 1st and 2nd
 * elements of input_data array
 *
 * @param input_data 2D array of random floats
 * @param rows number of rows (outer array length)
 * @return float** true outputs (cols = 2)
 */
float **dense_generate_output_data(float **input_data, uint32_t rows) {
    // Allocate memory for the outer array (rows)
    float **array = (float **) malloc(rows * sizeof(float *));
    if (!array) {
        printf("Error allocating array for dense_generate_output_data!\n");
        return NULL;
    }

    for (uint32_t row = 0; row < rows; ++row) {
        // Allocate memory for each internal array (columns)
        array[row] = (float *) calloc(2U, sizeof(float));
        if (!array[row]) {
            printf("Error allocating array[row] for "
                   "dense_generate_output_data!\n");
            return NULL;
        }

        // 1 > 2
        if (input_data[row][0] > input_data[row][1])
            array[row][0] = 1.f;

        // 1 <= 2
        else
            array[row][1] = 1.f;
    }
    return array;
}

/**
 * @brief Performs test of dense layers by training simple classifier
 *
 * @return uint8_t number of fails
 */
uint8_t test_dense() {
    // Print about message
    printf("\nTesting simple classifier using 3 dense layers\n");

    // Fails counter
    uint8_t fails = 0U;

    // 1000 numbers from -10 to 10: 80% train, 20% validation
    uint32_t train_dataset_length = 800;
    uint32_t validation_dataset_length = 200;

    // Generate validation datasets
    float **train_dataset_inputs = dense_generate_input_data(train_dataset_length, 2U);
    float **validation_dataset_inputs = dense_generate_input_data(validation_dataset_length, 2U);
    if (!train_dataset_inputs || !validation_dataset_inputs) {
        printf("train_dataset_inputs or validation_dataset_inputs allocation failed\n");
        return 1U;
    }

    // Generate outputs
    float **train_dataset_outputs = dense_generate_output_data(train_dataset_inputs, train_dataset_length);
    float **validation_dataset_outputs =
        dense_generate_output_data(validation_dataset_inputs, validation_dataset_length);
    if (!train_dataset_outputs || !validation_dataset_outputs) {
        printf("train_dataset_outputs or est_dataset_outputs allocation failed\n");
        return 1U;
    }

    // Initialize petals
    petal_s *petal_hidden1 =
        petal_init(PETAL_TYPE_DENSE_1D, true, &(petal_shape_s){1U, 2U, 1U, 0UL}, &(petal_shape_s){1U, 2U, 1U, 0UL},
                   &(weights_s){true, WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN, 4U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(weights_s){true, WEIGHTS_INIT_CONSTANT, 2U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(activation_s){ACTIVATION_RELU, 1.f, 0.f, 0.0f, 0.00f, 1.f, NULL}, 0.0f, 0.f, 1.f);
    petal_s *petal_hidden2 =
        petal_init(PETAL_TYPE_DENSE_1D, false, &(petal_shape_s){1U, 2U, 1U, 0UL}, &(petal_shape_s){1U, 2U, 1U, 0UL},
                   &(weights_s){true, WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN, 4U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(weights_s){true, WEIGHTS_INIT_CONSTANT, 2U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(activation_s){ACTIVATION_RELU, 1.f, 0.f, 0.0f, 0.00f, 1.f, NULL}, 0.0f, 0.f, 1.f);
    petal_s *petal_output =
        petal_init(PETAL_TYPE_DENSE_1D, false, &(petal_shape_s){1U, 2U, 1U, 0UL}, &(petal_shape_s){1U, 2U, 1U, 0UL},
                   &(weights_s){true, WEIGHTS_INIT_XAVIER_GLOROT_GAUSSIAN, 6U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(weights_s){true, WEIGHTS_INIT_CONSTANT, 2U, NULL, NULL, 0.f, 1.f, NULL, NULL, 0U},
                   &(activation_s){ACTIVATION_SOFTMAX, 1.f, 0.f, 0.0f, 0.01f, 1.f, NULL}, 0.0f, 0.f, 1.f);

    // Print weights
    printf("In -> hidden 1 weights:\n");
    print_array(petal_hidden1->weights->weights, 2U, 2U, 1U);
    printf("In -> hidden 1 bias weights:\n");
    print_array(petal_hidden1->bias_weights->weights, 1U, 2U, 1U);

    printf("hidden 1 -> hidden 2 weights:\n");
    print_array(petal_hidden2->weights->weights, 2U, 2U, 1U);
    printf("hidden 1 -> hidden 2 bias weights:\n");
    print_array(petal_hidden2->bias_weights->weights, 1U, 2U, 1U);

    printf("hidden 2 -> out weights:\n");
    print_array(petal_output->weights->weights, 2U, 2U, 1U);
    printf("hidden 2 -> out bias weights:\n");
    print_array(petal_output->bias_weights->weights, 1U, 2U, 1U);

    // Initialize flower
    petal_s *petals[] = {petal_hidden1, petal_hidden2, petal_output};
    flower_s *flower = flower_init(petals, 3U);

    // Show prediction before training
    printf("Before training [1.0, 2.0] -> [1 > 2, 1 <= 2]:\t\t");
    print_array(flower_predict(flower, (float[]){1.f, 2.f}), 1U, 2U, 1U);

    // Initialize optimizer (Type, learning rate, momentum, beta 1, beta 2)
    optimizer_s optimizer = (optimizer_s){OPTIMIZER_ADAM, .01f, 0.f, .89f, .99f};

    // Initialize metrics
    metrics_s *metrics = metrics_init(1);
    metrics_add(metrics, METRICS_TIME_ELAPSED);
    metrics_add(metrics, METRICS_LOSS_TRAIN);
    metrics_add(metrics, METRICS_ACCURACY_TRAIN);
    metrics_add(metrics, METRICS_LOSS_VALIDATION);
    metrics_add(metrics, METRICS_ACCURACY_VALIDATION);

    // Train
    uint32_t epochs = 10;
    uint32_t batch_size = 40;
    flower_train(flower, LOSS_CATEGORICAL_CROSSENTROPY, &optimizer, metrics, train_dataset_inputs,
                 train_dataset_outputs, NULL, train_dataset_length, validation_dataset_inputs,
                 validation_dataset_outputs, NULL, validation_dataset_length, batch_size, epochs);

    // Test training result on a new data
    float *result;
    printf("After training [1.0, 10.0] -> [1 > 2, 1 <= 2]:\t\t");
    result = flower_predict(flower, (float[]){1.f, 10.f});
    print_array(result, 1U, 2U, 1U);
    if (result[0] >= result[1]) {
        printf("\t\t\t\t\t\t\t\tWRONG!\n");
        fails++;
    }

    printf("After training [20.0, 10.0] -> [1 > 2, 1 <= 2]:\t\t");
    result = flower_predict(flower, (float[]){20.f, 10.f});
    print_array(result, 1U, 2U, 1U);
    if (result[0] <= result[1]) {
        printf("\t\t\t\t\t\t\t\tWRONG!\n");
        fails++;
    }

    printf("After training [-1.0, 10.0] -> [1 > 2, 1 <= 2]:\t\t");
    result = flower_predict(flower, (float[]){-1.f, 10.f});
    print_array(result, 1U, 2U, 1U);
    if (result[0] >= result[1]) {
        printf("\t\t\t\t\t\t\t\tWRONG!\n");
        fails++;
    }

    // Print flower weight
    printf("Min flower size: %lu bytes\n", flower_estimate_min_size(flower));

    // Destroy internal array of weights
    weights_destroy(petal_hidden1->weights, false, true);
    weights_destroy(petal_hidden1->bias_weights, false, true);
    weights_destroy(petal_hidden2->weights, false, true);
    weights_destroy(petal_hidden2->bias_weights, false, true);
    weights_destroy(petal_output->weights, false, true);
    weights_destroy(petal_output->bias_weights, false, true);

    // Destroy flower without destroying petals
    flower_destroy(flower, false, false, false);

    // Destroy metrics
    metrics_destroy(metrics);

    // Destroy datasets
    for (uint16_t i = 0; i < train_dataset_length; ++i) {
        free(train_dataset_inputs[i]);
        free(train_dataset_outputs[i]);
    }
    for (uint16_t i = 0; i < validation_dataset_length; ++i) {
        free(validation_dataset_inputs[i]);
        free(validation_dataset_outputs[i]);
    }
    free(train_dataset_inputs);
    free(train_dataset_outputs);
    free(validation_dataset_inputs);
    free(validation_dataset_outputs);

    return fails;
}

/**
 * @brief Performs test of pseudo random number generator by validating rk_random_() and rk_float_() 5 times each
 * NOTE: call rk_seed_(0) for that to work
 *
 * @return uint8_t number of fails (mismatches)
 */
uint8_t test_random() {
    // Print about message
    printf("\nChecking whether the PRNG works correctly\n");

    // Fails counter
    uint8_t fails = 0U;

    // Test first 5 integers
    if (rk_random_() != 2357136044U)
        fails++;
    if (rk_random_() != 2546248239U)
        fails++;
    if (rk_random_() != 3071714933U)
        fails++;
    if (rk_random_() != 3626093760U)
        fails++;
    if (rk_random_() != 2588848963U)
        fails++;

    // Test next 5 floats
    if (rk_float_() != .85794562101364135742f)
        fails++;
    if (rk_float_() != .84725171327590942383f)
        fails++;
    if (rk_float_() != .62356370687484741211f)
        fails++;
    if (rk_float_() != .38438171148300170898f)
        fails++;
    if (rk_float_() != .29753458499908447266f)
        fails++;

    // Check
    if (fails == 0)
        printf("PRNG works correctly\n");
    else
        printf("PRNG DOES NOT WORK CORRECTLY!\n");

    return fails;
}

/**
 * @brief Automatically performs tests of most function
 *
 * @return int 0 in all test passed, -1 otherwise
 */
int main() {
    // Fails counter
    uint8_t fails = 0U;

    // Set random seed (seed must be 0 for test_random to work and also for results to be consistant)
    // rk_seed_(time(NULL) & 0xFFFFFFFFUL);
    rk_seed_(0);

    printf("\n--------------------------------- BEGIN TESTS ----------------------------------\n");

    // Validate random
    fails += test_random();
    printf("\n--------------------------------------------------------------------------------\n");

    // Test activation
    fails += test_activation_full();
    printf("\n--------------------------------------------------------------------------------\n");

    // Test loss
    fails += test_loss_full();
    printf("\n--------------------------------------------------------------------------------\n");

    // Test dropout
    fails += test_dropout();
    printf("\n--------------------------------------------------------------------------------\n");

    // Test normalization
    fails += test_normalization();
    printf("\n--------------------------------------------------------------------------------\n");

    // Test flower with dense petals
    fails += test_dense();
    printf("\n---------------------------------- END TESTS -----------------------------------\n");

    // Print number of fails during tests
    printf("\nFails: %u\n", fails);

    if (fails == 0U) {
        printf("All tests passed successfully!\n");
        return 0;
    }
    return -1;
}
