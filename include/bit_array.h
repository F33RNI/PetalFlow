/**
 * @file bit_array.h
 * @author Fern Lane
 * @brief Defines and stores bit array data
 * @version 1.0.0
 * @date 2023-11-17
 *
 * @copyright Copyright (c) 2023-2024 Fern Lane
 *
 * Copyright (C) 2023 Fern Lane, PetalFlow library
 * Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *       https://www.gnu.org/licenses/agpl-3.0.en.html
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef BIT_ARRAY_H__
#define BIT_ARRAY_H__

#include <stdbool.h>
#include <stdint.h>

/**
 * @struct bit_array_s
 * Stores "array of bits"
 *
 * @param data pointer to array of uint32_t numbers
 * @param length length of array in words (in uint32_t)
 * @param error_code initialization or runtime error code
 */
typedef struct {
    uint32_t *data;
    uint32_t length;
    uint8_t error_code;
} bit_array_s;

bit_array_s *bit_array_init(uint32_t size_bits);

void bit_array_set_bit(bit_array_s *bit_array, uint32_t index);

void bit_array_clear_bit(bit_array_s *bit_array, uint32_t index);

bool bit_array_get_bit(bit_array_s *bit_array, uint32_t index);

void bit_array_clear(bit_array_s *bit_array);

void bit_array_destroy(bit_array_s *bit_array);

#endif
