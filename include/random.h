/**
 * @file random.h
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
#ifndef RANDOM_H__
#define RANDOM_H__

#include <stdint.h>

#define RK_STATE_LEN 624U

// Magic Mersenne Twister constants
#define N          624U
#define M          397U
#define MATRIX_A   0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

#ifdef WIN32
// Disable "unary minus operator applied to unsigned type, result still unsigned" warning.
#pragma warning(disable : 4146)
#endif

typedef struct {
    uint32_t key[RK_STATE_LEN];
    uint32_t pos;
} rk_state_s;

extern rk_state_s rk_state_global;

void rk_seed_(uint32_t seed);
void rk_seed(uint32_t seed, rk_state_s *state);

extern uint32_t rk_random_();
extern uint32_t rk_random(rk_state_s *state);

extern double rk_double_();
extern double rk_double(rk_state_s *state);

extern float rk_float_();
extern float rk_float(rk_state_s *state);

#endif
