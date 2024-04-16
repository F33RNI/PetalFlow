/**
 * @file random.c
 * @author Jean Luc PONS, Fern Lane
 * @brief Knuth's PRNG as used in the Mersenne Twister reference implementation (original code from VanitySearch)
 *
 * @copyright Copyright (c) 2019 Jean Luc PONS, Copyright (c) 2023-2024 Fern Lane
 *
 * This file is part of the PetalFlow distribution <https://github.com/F33RNI/PetalFlow>.
 * Original of this file is part of the VanitySearch distribution <https://github.com/JeanLucPons/VanitySearch>.
 * This version contains some changes to make behavior consistent across different compilers and platforms
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

#include "random.h"

rk_state_s rk_state_global;

/**
 * @brief Initializes rk_state_global by using seed
 * NOTE: Please make sure rk_seed_() or rk_seed() called at least ones
 *
 * @param seed any value in range [0 - 4294967295] (including both ends)
 */
void rk_seed_(uint32_t seed) { rk_seed(seed, &rk_state_global); }

/**
 * @brief Slightly optimized reference implementation of the Mersenne Twister
 *
 * @return uniformly distributed value in the [0, 4294967295] interval (including both ends)
 */
inline uint32_t rk_random_() { return rk_random(&rk_state_global); }

/**
 * @brief rk_random_() but for double
 *
 * @return double a uniformly distributed value in the (0, 1) interval (excluding both ends)
 */
inline double rk_double_() { return rk_double(&rk_state_global); }

/**
 * @brief rk_random_() but for float
 *
 * @return float a uniformly distributed value in the (0, 1] interval (excluding 0, but sometimes including 1)
 * (see rk_float() docs for more info)
 */
inline float rk_float_() { return rk_float(&rk_state_global); }

/**
 * @brief Initializes rk_state by using seed
 * NOTE: Please make sure rk_seed_() or rk_seed() called at least ones
 *
 * @param seed any value in range [0 - 4294967295] (including both ends)
 * @param state pointer to rk_state_s struct
 */
void rk_seed(uint32_t seed, rk_state_s *state) {
    // Knuth's PRNG as used in the Mersenne Twister reference implementation
    for (uint32_t i = 0; i < RK_STATE_LEN; ++i) {
        state->key[i] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30UL)) + i + 1UL) & 0xffffffffUL;
    }

    state->pos = RK_STATE_LEN;
}

/**
 * @brief Slightly optimized reference implementation of the Mersenne Twister
 *
 * @param state pointer to rk_state_s struct
 * @return uint32_t uniformly distributed value in the [0, 4294967295] interval (including both ends)
 */
inline uint32_t rk_random(rk_state_s *state) {
    uint32_t y;

    if (state->pos == RK_STATE_LEN) {
        uint32_t i;

        for (i = 0; i < N - M; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i + 1] & LOWER_MASK);
            state->key[i] = state->key[i + M] ^ (y >> 1U) ^ (-(y & 1U) & MATRIX_A);
        }
        for (i; i < N - 1; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i + 1] & LOWER_MASK);
            state->key[i] = state->key[i + (M - N)] ^ (y >> 1U) ^ (-(y & 1U) & MATRIX_A);
        }
        y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
        state->key[N - 1] = state->key[M - 1U] ^ (y >> 1U) ^ (-(y & 1U) & MATRIX_A);

        state->pos = 0;
    }

    y = state->key[state->pos++];

    // Tempering
    y ^= (y >> 11U);
    y ^= (y << 7U) & 0x9d2c5680UL;
    y ^= (y << 15U) & 0xefc60000UL;
    y ^= (y >> 18U);

    return y;
}

/**
 * @brief rk_random() but for double
 *
 * @param state pointer to rk_state_s struct
 * @return double a uniformly distributed value in the (0, 1) interval (excluding both ends)
 */
inline double rk_double(rk_state_s *state) {
    // Shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000
    int32_t a = rk_random(state) >> 5U;
    int32_t b = rk_random(state) >> 6U;
    return ((double) a * 67108864. + (double) b) / 9007199254740992.;
}

/**
 * @brief rk_random() but for float
 *
 * @param state pointer to rk_state_s struct
 * @return float a uniformly distributed value in the (0, 1] interval (excluding 0, but sometimes including 1)
 * (for (0, 1) interval comment current return statement and uncomment bottom one)
 */
inline float rk_float(rk_state_s *state) {
    // Shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000
    int32_t a = rk_random(state) >> 5U;
    int32_t b = rk_random(state) >> 6U;
    return ((float) a * 67108864.f + (float) b) / 9007199254740992.f;

    // Ensure that the result is strictly between 0.0 and 1.0 (exclusive)
    // float result = ((float) a * 67108864.f + (float) b) / 9007199254740992.f;
    // return result > .000001f ? (result < .999999f ? result : .999999f) : .000001f;
}
