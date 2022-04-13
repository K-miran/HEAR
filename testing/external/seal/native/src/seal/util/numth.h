// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/util/defines.h"
#include "seal/smallmodulus.h"
#include "seal/util/common.h"
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>

namespace seal
{
    namespace util
    {
        SEAL_NODISCARD inline std::vector<int> naf(int value)
        {
          std::vector<int> res;

          // Record the sign of the original value and compute abs
          bool sign = value < 0;
          value = std::abs(value);

          // Transform to non-adjacent form (NAF)
          for (int i = 0; value; i++)
          {
            int zi = (value % 2) ? 2 - (value % 4) : 0;
            value = (value - zi) / 2;
            if (zi)
            {
              res.push_back((sign ? -zi : zi) * (1 << i));
            }
          }

          return res;
        }

        SEAL_NODISCARD inline std::uint64_t gcd(
            std::uint64_t x, std::uint64_t y)
        {
#ifdef SEAL_DEBUG
            if (x == 0)
            {
                throw std::invalid_argument("x cannot be zero");
            }
            if (y == 0)
            {
                throw std::invalid_argument("y cannot be zero");
            }
#endif
            if (x < y)
            {
                return gcd(y, x);
            }
            else if (y == 0)
            {
                return x;
            }
            else
            {
                std::uint64_t f = x % y;
                if (f == 0)
                {
                    return y;
                }
                else
                {
                    return gcd(y, f);
                }
            }
        }

        SEAL_NODISCARD inline auto xgcd(std::uint64_t x, std::uint64_t y)
            -> std::tuple<std::uint64_t, std::int64_t, std::int64_t>
        {
            /* Extended GCD:
            Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
            The numbers x, y are such that gcd = ax + by.
            */
#ifdef SEAL_DEBUG
            if (x == 0)
            {
                throw std::invalid_argument("x cannot be zero");
            }
            if (y == 0)
            {
                throw std::invalid_argument("y cannot be zero");
            }
#endif
            std::int64_t prev_a = 1;
            std::int64_t a = 0;
            std::int64_t prev_b = 0;
            std::int64_t b = 1;

            while (y != 0)
            {
                std::int64_t q = util::safe_cast<std::int64_t>(x / y);
                std::int64_t temp = util::safe_cast<std::int64_t>(x % y);
                x = y;
                y = util::safe_cast<std::uint64_t>(temp);

                temp = a;
                a = util::sub_safe(prev_a, mul_safe(q, a));
                prev_a = temp;

                temp = b;
                b = util::sub_safe(prev_b, mul_safe(q, b));
                prev_b = temp;
            }
            return std::make_tuple(x, prev_a, prev_b);
        }

        inline bool try_mod_inverse(std::uint64_t value,
            std::uint64_t modulus, std::uint64_t &result)
        {
#ifdef SEAL_DEBUG
            if (modulus <= 1)
            {
                throw std::invalid_argument("modulus must be at least 2");
            }
#endif
            if (value == 0)
            {
                return false;
            }
            auto gcd_tuple = xgcd(value, modulus);
            if (std::get<0>(gcd_tuple) != 1)
            {
                return false;
            }
            else if (std::get<1>(gcd_tuple) < 0)
            {
                result = static_cast<std::uint64_t>(std::get<1>(gcd_tuple)) + modulus;
                return true;
            }
            else
            {
                result = static_cast<std::uint64_t>(std::get<1>(gcd_tuple));
                return true;
            }
        }

        SEAL_NODISCARD std::vector<std::uint64_t> multiplicative_orders(
            std::vector<std::uint64_t> conjugate_classes,
            std::uint64_t modulus);

        SEAL_NODISCARD std::vector<std::uint64_t> conjugate_classes(
            std::uint64_t modulus, std::uint64_t subgroup_generator);

        void babystep_giantstep(std::uint64_t modulus,
            std::vector<std::uint64_t> &baby_steps,
            std::vector<std::uint64_t> &giant_steps);

        SEAL_NODISCARD auto decompose_babystep_giantstep(
            std::uint64_t modulus,
            std::uint64_t input,
            const std::vector<std::uint64_t> &baby_steps,
            const std::vector<std::uint64_t> &giant_steps)
            -> std::pair<std::size_t, std::size_t>;

        SEAL_NODISCARD bool is_prime(
            const SmallModulus &modulus, std::size_t num_rounds = 40);

        SEAL_NODISCARD std::vector<SmallModulus> get_primes(
            std::size_t ntt_size, int bit_size, std::size_t count);

        SEAL_NODISCARD inline SmallModulus get_prime(
            std::size_t ntt_size, int bit_size)
        {
            return get_primes(ntt_size, bit_size, 1)[0];
        }
    }
}
