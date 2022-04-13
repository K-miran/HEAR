// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <functional>
#include "seal/evaluator.h"
#include "seal/util/common.h"
#include "seal/util/uintarith.h"
#include "seal/util/polycore.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/numth.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        template<typename T, typename S>
        inline bool are_same_scale(const T &value1, const S &value2) noexcept
        {
            return util::are_close<double>(value1.scale(), value2.scale());
        }
    }

    Evaluator::Evaluator(shared_ptr<SEALContext> context) : context_(move(context))
    {
        // Verify parameters
        if (!context_)
        {
            throw invalid_argument("invalid context");
        }
        if (!context_->parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        // Calculate map from Zmstar to generator representation
        populate_Zmstar_to_generator();
    }

    void Evaluator::populate_Zmstar_to_generator()
    {
        uint64_t n = static_cast<uint64_t>(
            context_->first_context_data()->parms().poly_modulus_degree());
        uint64_t m = n << 1;

        for (uint64_t i = 0; i < n / 2; i++)
        {
            uint64_t galois_elt = exponentiate_uint64(3, i) & (m - 1);
            pair<uint64_t, uint64_t> temp_pair1{ i, 0 };
            Zmstar_to_generator_.emplace(galois_elt, temp_pair1);
            galois_elt = (exponentiate_uint64(3, i) * (m - 1)) & (m - 1);
            pair<uint64_t, uint64_t> temp_pair2{ i, 1 };
            Zmstar_to_generator_.emplace(galois_elt, temp_pair2);
        }
    }

    void Evaluator::negate_inplace(Ciphertext &encrypted)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        // Negate each poly in the array
        for (size_t j = 0; j < encrypted_size; j++)
        {
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                negate_poly_coeffmod(encrypted.data(j) + (i * coeff_count),
                    coeff_count, coeff_modulus[i], encrypted.data(j) + (i * coeff_count));
            }
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::add_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        // Size check
        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare destination
        encrypted1.resize(context_, context_data.parms_id(), max_count);

        // Add ciphertexts
        for (size_t j = 0; j < min_count; j++)
        {
            uint64_t *encrypted1_ptr = encrypted1.data(j);
            const uint64_t *encrypted2_ptr = encrypted2.data(j);
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                add_poly_poly_coeffmod(encrypted1_ptr + (i * coeff_count),
                    encrypted2_ptr + (i * coeff_count), coeff_count, coeff_modulus[i],
                    encrypted1_ptr + (i * coeff_count));
            }
        }

        // Copy the remainding polys of the array with larger count into encrypted1
        if (encrypted1_size < encrypted2_size)
        {
            set_poly_poly(encrypted2.data(min_count),
                coeff_count * (encrypted2_size - encrypted1_size),
                coeff_mod_count, encrypted1.data(encrypted1_size));
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::add_many(const vector<Ciphertext> &encrypteds, Ciphertext &destination)
    {
        if (encrypteds.empty())
        {
            throw invalid_argument("encrypteds cannot be empty");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }
        destination = encrypteds[0];
        for (size_t i = 1; i < encrypteds.size(); i++)
        {
            add_inplace(destination, encrypteds[i]);
        }
    }

    void Evaluator::sub_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        // Size check
        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare destination
        encrypted1.resize(context_, context_data.parms_id(), max_count);

        // Subtract polynomials.
        for (size_t j = 0; j < min_count; j++)
        {
            uint64_t *encrypted1_ptr = encrypted1.data(j);
            const uint64_t *encrypted2_ptr = encrypted2.data(j);
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                sub_poly_poly_coeffmod(encrypted1_ptr + (i * coeff_count),
                    encrypted2_ptr + (i * coeff_count), coeff_count, coeff_modulus[i],
                    encrypted1_ptr + (i * coeff_count));
            }
        }

        // If encrypted2 has larger count, negate remaining entries
        if (encrypted1_size < encrypted2_size)
        {
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                negate_poly_coeffmod(encrypted2.data(encrypted1_size) + (i * coeff_count),
                    coeff_count * (encrypted2_size - encrypted1_size), coeff_modulus[i],
                    encrypted1.data(encrypted1_size) + (i * coeff_count));
            }
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_inplace(Ciphertext &encrypted1,
        const Ciphertext &encrypted2, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }

        auto context_data_ptr = context_->first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::BFV:
            bfv_multiply(encrypted1, encrypted2, pool);
            break;

        case scheme_type::CKKS:
            ckks_multiply(encrypted1, encrypted2, pool);
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::bfv_multiply(Ciphertext &encrypted1,
        const Ciphertext &encrypted2, MemoryPoolHandle pool)
    {
        if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form())
        {
            throw invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        uint64_t plain_modulus = parms.plain_modulus().value();
        auto &base_converter = context_data.base_converter();
        auto &bsk_modulus = base_converter->get_bsk_mod_array();
        size_t bsk_base_mod_count = base_converter->bsk_base_mod_count();
        size_t bsk_mtilde_count = add_safe(bsk_base_mod_count, size_t(1));
        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();
        auto &bsk_small_ntt_tables = base_converter->get_bsk_small_ntt_tables();

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_count = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_count, coeff_count, bsk_mtilde_count))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare destination
        encrypted1.resize(context_, context_data.parms_id(), dest_count);

        size_t encrypted_ptr_increment = coeff_count * coeff_mod_count;
        size_t encrypted_bsk_mtilde_ptr_increment = coeff_count * bsk_mtilde_count;
        size_t encrypted_bsk_ptr_increment = coeff_count * bsk_base_mod_count;

        // Make temp polys for FastBConverter result from q ---> Bsk U {m_tilde}
        auto tmp_encrypted1_bsk_mtilde(allocate_poly(
            coeff_count * encrypted1_size, bsk_mtilde_count, pool));
        auto tmp_encrypted2_bsk_mtilde(allocate_poly(
            coeff_count * encrypted2_size, bsk_mtilde_count, pool));

        // Make temp polys for FastBConverter result from Bsk U {m_tilde} -----> Bsk
        auto tmp_encrypted1_bsk(allocate_poly(
            coeff_count * encrypted1_size, bsk_base_mod_count, pool));
        auto tmp_encrypted2_bsk(allocate_poly(
            coeff_count * encrypted2_size, bsk_base_mod_count, pool));

        // Step 0: fast base convert from q to Bsk U {m_tilde}
        // Step 1: reduce q-overflows in Bsk
        // Iterate over all the ciphertexts inside encrypted1
        for (size_t i = 0; i < encrypted1_size; i++)
        {
            base_converter->fastbconv_mtilde(
                encrypted1.data(i),
                tmp_encrypted1_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment),
                pool);
            base_converter->mont_rq(
                tmp_encrypted1_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment),
                tmp_encrypted1_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }

        // Iterate over all the ciphertexts inside encrypted2
        for (size_t i = 0; i < encrypted2_size; i++)
        {
            base_converter->fastbconv_mtilde(
                encrypted2.data(i),
                tmp_encrypted2_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), pool);
            base_converter->mont_rq(
                tmp_encrypted2_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment),
                tmp_encrypted2_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }

        // Step 2: compute product and multiply plain modulus to the result
        // We need to multiply both in q and Bsk. Values in encrypted_safe are in
        // base q and values in tmp_encrypted_bsk are in base Bsk. We iterate over
        // destination poly array and generate each poly based on the indices of
        // inputs (arbitrary sizes for ciphertexts). First allocate two temp polys:
        // one for results in base q and the other for the result in base Bsk. These
        // need to be zero for the arbitrary size multiplication; not for 2x2 though
        auto tmp_des_coeff_base(allocate_zero_poly(
            coeff_count * dest_count, coeff_mod_count, pool));
        auto tmp_des_bsk_base(allocate_zero_poly(
            coeff_count * dest_count, bsk_base_mod_count, pool));

        // Allocate two tmp polys: one for NTT multiplication results in base q and
        // one for result in base Bsk
        auto tmp1_poly_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));
        auto tmp1_poly_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count, pool));
        auto tmp2_poly_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));
        auto tmp2_poly_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count, pool));

        size_t current_encrypted1_limit = 0;

        // First convert all the inputs into NTT form
        auto copy_encrypted1_ntt_coeff_mod(allocate_poly(
            coeff_count * encrypted1_size, coeff_mod_count, pool));
        set_poly_poly(encrypted1.data(), coeff_count * encrypted1_size,
            coeff_mod_count, copy_encrypted1_ntt_coeff_mod.get());

        auto copy_encrypted1_ntt_bsk_base_mod(allocate_poly(
            coeff_count * encrypted1_size, bsk_base_mod_count, pool));
        set_poly_poly(tmp_encrypted1_bsk.get(), coeff_count * encrypted1_size,
            bsk_base_mod_count, copy_encrypted1_ntt_bsk_base_mod.get());

        auto copy_encrypted2_ntt_coeff_mod(allocate_poly(
            coeff_count * encrypted2_size, coeff_mod_count, pool));
        set_poly_poly(encrypted2.data(), coeff_count * encrypted2_size,
            coeff_mod_count, copy_encrypted2_ntt_coeff_mod.get());

        auto copy_encrypted2_ntt_bsk_base_mod(allocate_poly(
            coeff_count * encrypted2_size, bsk_base_mod_count, pool));
        set_poly_poly(tmp_encrypted2_bsk.get(), coeff_count * encrypted2_size,
            bsk_base_mod_count, copy_encrypted2_ntt_bsk_base_mod.get());

        for (size_t i = 0; i < encrypted1_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted1_ntt_coeff_mod.get() +
                    (j * coeff_count) + (i * encrypted_ptr_increment), coeff_small_ntt_tables[j]);
            }
            for (size_t j = 0; j < bsk_base_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted1_ntt_bsk_base_mod.get() +
                    (j * coeff_count) + (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables[j]);
            }
        }

        for (size_t i = 0; i < encrypted2_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted2_ntt_coeff_mod.get() +
                    (j * coeff_count) + (i * encrypted_ptr_increment), coeff_small_ntt_tables[j]);
            }
            for (size_t j = 0; j < bsk_base_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted2_ntt_bsk_base_mod.get() +
                    (j * coeff_count) + (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables[j]);
            }
        }

        // Perform multiplication on arbitrary size ciphertexts
        for (size_t secret_power_index = 0;
            secret_power_index < dest_count; secret_power_index++)
        {
            // Loop over encrypted1 components [i], seeing if a match exists with an encrypted2
            // component [j] such that [i+j]=[secret_power_index]
            // Only need to check encrypted1 components up to and including [secret_power_index],
            // and strictly less than [encrypted_array.size()]
            current_encrypted1_limit = min(encrypted1_size, secret_power_index + 1);

            for (size_t encrypted1_index = 0;
                encrypted1_index < current_encrypted1_limit; encrypted1_index++)
            {
                // check if a corresponding component in encrypted2 exists
                if (encrypted2_size > secret_power_index - encrypted1_index)
                {
                    size_t encrypted2_index = secret_power_index - encrypted1_index;

                    // NTT Multiplication and addition for results in q
                    for (size_t i = 0; i < coeff_mod_count; i++)
                    {
                        dyadic_product_coeffmod(
                            copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count) +
                            (encrypted_ptr_increment * encrypted1_index),
                            copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count) +
                            (encrypted_ptr_increment * encrypted2_index),
                            coeff_count, coeff_modulus[i],
                            tmp1_poly_coeff_base.get() + (i * coeff_count));
                        add_poly_poly_coeffmod(
                            tmp1_poly_coeff_base.get() + (i * coeff_count),
                            tmp_des_coeff_base.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * coeff_mod_count),
                            coeff_count, coeff_modulus[i],
                            tmp_des_coeff_base.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * coeff_mod_count));
                    }

                    // NTT Multiplication and addition for results in Bsk
                    for (size_t i = 0; i < bsk_base_mod_count; i++)
                    {
                        dyadic_product_coeffmod(
                            copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count) +
                            (encrypted_bsk_ptr_increment * encrypted1_index),
                            copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count) +
                            (encrypted_bsk_ptr_increment * encrypted2_index),
                            coeff_count, bsk_modulus[i],
                            tmp1_poly_bsk_base.get() + (i * coeff_count));
                        add_poly_poly_coeffmod(
                            tmp1_poly_bsk_base.get() + (i * coeff_count),
                            tmp_des_bsk_base.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * bsk_base_mod_count),
                            coeff_count, bsk_modulus[i],
                            tmp_des_bsk_base.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * bsk_base_mod_count));
                    }
                }
            }
        }

        // Convert back outputs from NTT form
        for (size_t i = 0; i < dest_count; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey(
                    tmp_des_coeff_base.get() + (i * (encrypted_ptr_increment)) +
                    (j * coeff_count), coeff_small_ntt_tables[j]);
            }
            for (size_t j = 0; j < bsk_base_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey(
                    tmp_des_bsk_base.get() + (i * (encrypted_bsk_ptr_increment)) +
                    (j * coeff_count), bsk_small_ntt_tables[j]);
            }
        }

        // Now we multiply plain modulus to both results in base q and Bsk and
        // allocate them together in one container as
        // (te0)q(te'0)Bsk | ... |te count)q (te' count)Bsk to make it ready for
        // fast_floor
        auto tmp_coeff_bsk_together(allocate_poly(
            coeff_count, dest_count * (coeff_mod_count + bsk_base_mod_count), pool));
        uint64_t *tmp_coeff_bsk_together_ptr = tmp_coeff_bsk_together.get();

        // Base q
        for (size_t i = 0; i < dest_count; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                multiply_poly_scalar_coeffmod(
                    tmp_des_coeff_base.get() + (j * coeff_count) + (i * encrypted_ptr_increment),
                    coeff_count, plain_modulus, coeff_modulus[j],
                    tmp_coeff_bsk_together_ptr + (j * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_ptr_increment;

            for (size_t k = 0; k < bsk_base_mod_count; k++)
            {
                multiply_poly_scalar_coeffmod(
                    tmp_des_bsk_base.get() + (k * coeff_count) + (i * encrypted_bsk_ptr_increment),
                    coeff_count, plain_modulus, bsk_modulus[k],
                    tmp_coeff_bsk_together_ptr + (k * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_bsk_ptr_increment;
        }

        // Allocate a new poly for fast floor result in Bsk
        auto tmp_result_bsk(allocate_poly(
            coeff_count, dest_count * bsk_base_mod_count, pool));
        for (size_t i = 0; i < dest_count; i++)
        {
            // Step 3: fast floor from q U {Bsk} to Bsk
            base_converter->fast_floor(
                tmp_coeff_bsk_together.get() +
                (i * (encrypted_ptr_increment + encrypted_bsk_ptr_increment)),
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), pool);

            // Step 4: fast base convert from Bsk to q
            base_converter->fastbconv_sk(
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment),
                encrypted1.data(i), pool);
        }
    }

    void Evaluator::ckks_multiply(Ciphertext &encrypted1,
        const Ciphertext &encrypted2, MemoryPoolHandle pool)
    {
        if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
        {
            throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        double new_scale = encrypted1.scale() * encrypted2.scale();

        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
            context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_count = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_count, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare destination
        encrypted1.resize(context_, context_data.parms_id(), dest_count);

        //pointer increment to switch to a next polynomial
        size_t encrypted_ptr_increment = coeff_count * coeff_mod_count;

        //Step 1: naive multiplication modulo the coefficient modulus
        //First allocate two temp polys :
        //one for results in base q. This need to be zero
        //for the arbitrary size multiplication; not for 2x2 though
        auto tmp_des(allocate_zero_poly(
            coeff_count * dest_count, coeff_mod_count, pool));

        //Allocate tmp polys for NTT multiplication results in base q
        auto tmp1_poly(allocate_poly(coeff_count, coeff_mod_count, pool));
        auto tmp2_poly(allocate_poly(coeff_count, coeff_mod_count, pool));

        // First convert all the inputs into NTT form
        auto copy_encrypted1_ntt(allocate_poly(
            coeff_count * encrypted1_size, coeff_mod_count, pool));
        set_poly_poly(encrypted1.data(), coeff_count * encrypted1_size,
            coeff_mod_count, copy_encrypted1_ntt.get());

        auto copy_encrypted2_ntt(allocate_poly(
            coeff_count * encrypted2_size, coeff_mod_count, pool));
        set_poly_poly(encrypted2.data(), coeff_count * encrypted2_size,
            coeff_mod_count, copy_encrypted2_ntt.get());

        // Perform multiplication on arbitrary size ciphertexts

        // Loop over encrypted1 components [i], seeing if a match exists with an encrypted2
        // component [j] such that [i+j]=[secret_power_index]
        // Only need to check encrypted1 components up to and including [secret_power_index],
        // and strictly less than [encrypted_array.size()]

        // Number of encrypted1 components to check
        size_t current_encrypted1_limit = 0;

        for (size_t secret_power_index = 0;
            secret_power_index < dest_count; secret_power_index++)
        {
            current_encrypted1_limit = min(encrypted1_size, secret_power_index + 1);

            for (size_t encrypted1_index = 0;
                encrypted1_index < current_encrypted1_limit; encrypted1_index++)
            {
                // check if a corresponding component in encrypted2 exists
                if (encrypted2_size > secret_power_index - encrypted1_index)
                {
                    size_t encrypted2_index = secret_power_index - encrypted1_index;

                    // NTT Multiplication and addition for results in q
                    for (size_t i = 0; i < coeff_mod_count; i++)
                    {
                        // ci * dj
                        dyadic_product_coeffmod(
                            copy_encrypted1_ntt.get() + (i * coeff_count) +
                            (encrypted_ptr_increment * encrypted1_index),
                            copy_encrypted2_ntt.get() + (i * coeff_count) +
                            (encrypted_ptr_increment * encrypted2_index),
                            coeff_count, coeff_modulus[i],
                            tmp1_poly.get() + (i * coeff_count));
                        // Dest[i+j]
                        add_poly_poly_coeffmod(
                            tmp1_poly.get() + (i * coeff_count),
                            tmp_des.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * coeff_mod_count),
                            coeff_count, coeff_modulus[i],
                            tmp_des.get() + (i * coeff_count) +
                            (secret_power_index * coeff_count * coeff_mod_count));
                    }
                }
            }
        }

        // Set the final result
        set_poly_poly(tmp_des.get(), coeff_count * dest_count,
            coeff_mod_count, encrypted1.data());

        // Set the scale
        encrypted1.scale() = new_scale;
    }

    void Evaluator::square_inplace(Ciphertext &encrypted, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::BFV:
            bfv_square(encrypted, move(pool));
            break;

        case scheme_type::CKKS:
            ckks_square(encrypted, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::bfv_square(Ciphertext &encrypted, MemoryPoolHandle pool)
    {
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        uint64_t plain_modulus = parms.plain_modulus().value();
        auto &base_converter = context_data.base_converter();
        auto &bsk_modulus = base_converter->get_bsk_mod_array();
        size_t bsk_base_mod_count = base_converter->bsk_base_mod_count();
        size_t bsk_mtilde_count = add_safe(bsk_base_mod_count, size_t(1));
        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();
        auto &bsk_small_ntt_tables = base_converter->get_bsk_small_ntt_tables();

        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_size != 2)
        {
            bfv_multiply(encrypted, encrypted, move(pool));
            return;
        }

        // Determine destination_array.size()
        size_t dest_count = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_count, coeff_count, bsk_mtilde_count))
        {
            throw logic_error("invalid parameters");
        }

        size_t encrypted_ptr_increment = coeff_count * coeff_mod_count;
        size_t encrypted_bsk_mtilde_ptr_increment = coeff_count * bsk_mtilde_count;
        size_t encrypted_bsk_ptr_increment = coeff_count * bsk_base_mod_count;

        // Prepare destination
        encrypted.resize(context_, context_data.parms_id(), dest_count);

        // Make temp poly for FastBConverter result from q ---> Bsk U {m_tilde}
        auto tmp_encrypted_bsk_mtilde(allocate_poly(
            coeff_count * encrypted_size, bsk_mtilde_count, pool));

        // Make temp poly for FastBConverter result from Bsk U {m_tilde} -----> Bsk
        auto tmp_encrypted_bsk(allocate_poly(
            coeff_count * encrypted_size, bsk_base_mod_count, pool));

        // Step 0: fast base convert from q to Bsk U {m_tilde}
        // Step 1: reduce q-overflows in Bsk
        // Iterate over all the ciphertexts inside encrypted1
        for (size_t i = 0; i < encrypted_size; i++)
        {
            base_converter->fastbconv_mtilde(
                encrypted.data(i),
                tmp_encrypted_bsk_mtilde.get() +
                (i * encrypted_bsk_mtilde_ptr_increment), pool);
            base_converter->mont_rq(
                tmp_encrypted_bsk_mtilde.get() +
                (i * encrypted_bsk_mtilde_ptr_increment),
                tmp_encrypted_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }

        // Step 2: compute product and multiply plain modulus to the result.
        // We need to multiply both in q and Bsk. Values in encrypted_safe are
        // in base q and values in tmp_encrypted_bsk are in base Bsk. We iterate
        // over destination poly array and generate each poly based on the indices
        // of inputs (arbitrary sizes for ciphertexts). First allocate two temp polys:
        // one for results in base q and the other for the result in base Bsk.
        auto tmp_des_coeff_base(allocate_poly(
            coeff_count * dest_count, coeff_mod_count, pool));
        auto tmp_des_bsk_base(allocate_poly(
            coeff_count * dest_count, bsk_base_mod_count, pool));

        // First convert all the inputs into NTT form
        auto copy_encrypted_ntt_coeff_mod(allocate_poly(
            coeff_count * encrypted_size, coeff_mod_count, pool));
        set_poly_poly(encrypted.data(), coeff_count * encrypted_size,
            coeff_mod_count, copy_encrypted_ntt_coeff_mod.get());

        auto copy_encrypted_ntt_bsk_base_mod(allocate_poly(
            coeff_count * encrypted_size, bsk_base_mod_count, pool));
        set_poly_poly(tmp_encrypted_bsk.get(), coeff_count * encrypted_size,
            bsk_base_mod_count, copy_encrypted_ntt_bsk_base_mod.get());

        for (size_t i = 0; i < encrypted_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                ntt_negacyclic_harvey_lazy(
                    copy_encrypted_ntt_coeff_mod.get() + (j * coeff_count) +
                    (i * encrypted_ptr_increment), coeff_small_ntt_tables[j]);
            }
            for (size_t j = 0; j < bsk_base_mod_count; j++)
            {
                ntt_negacyclic_harvey_lazy(
                    copy_encrypted_ntt_bsk_base_mod.get() + (j * coeff_count) +
                    (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables[j]);
            }
        }

        // Perform fast squaring
        // Compute c0^2 in base q
        for (size_t i = 0; i < coeff_mod_count; i++)
        {
            // Des[0] in q
            dyadic_product_coeffmod(
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count),
                coeff_count, coeff_modulus[i],
                tmp_des_coeff_base.get() + (i * coeff_count));

            // Des[2] in q
            dyadic_product_coeffmod(
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment,
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment,
                coeff_count, coeff_modulus[i],
                tmp_des_coeff_base.get() + (i * coeff_count) + (2 * encrypted_ptr_increment));
        }

        // Compute c0^2 in base bsk
        for (size_t i = 0; i < bsk_base_mod_count; i++)
        {
            // Des[0] in bsk
            dyadic_product_coeffmod(
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count),
                coeff_count, bsk_modulus[i],
                tmp_des_bsk_base.get() + (i * coeff_count));

            // Des[2] in bsk
            dyadic_product_coeffmod(
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment,
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment,
                coeff_count, bsk_modulus[i],
                tmp_des_bsk_base.get() + (i * coeff_count) + (2 * encrypted_bsk_ptr_increment));
        }

        auto tmp_second_mul_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));

        // Compute 2*c0*c1 in base q
        for (size_t i = 0; i < coeff_mod_count; i++)
        {
            dyadic_product_coeffmod(
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment,
                coeff_count, coeff_modulus[i],
                tmp_second_mul_coeff_base.get() + (i * coeff_count));
            add_poly_poly_coeffmod(
                tmp_second_mul_coeff_base.get() + (i * coeff_count),
                tmp_second_mul_coeff_base.get() + (i * coeff_count),
                coeff_count, coeff_modulus[i],
                tmp_des_coeff_base.get() + (i * coeff_count) + encrypted_ptr_increment);
        }

        auto tmp_second_mul_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count, pool));

        // Compute 2*c0*c1 in base bsk
        for (size_t i = 0; i < bsk_base_mod_count; i++)
        {
            dyadic_product_coeffmod(
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment,
                coeff_count, bsk_modulus[i],
                tmp_second_mul_bsk_base.get() + (i * coeff_count));
            add_poly_poly_coeffmod(
                tmp_second_mul_bsk_base.get() + (i * coeff_count),
                tmp_second_mul_bsk_base.get() + (i * coeff_count),
                coeff_count, bsk_modulus[i],
                tmp_des_bsk_base.get() + (i * coeff_count) + encrypted_bsk_ptr_increment);
        }

        // Convert back outputs from NTT form
        for (size_t i = 0; i < dest_count; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey_lazy(
                    tmp_des_coeff_base.get() + (i * (encrypted_ptr_increment)) + (j * coeff_count),
                    coeff_small_ntt_tables[j]);
            }
            for (size_t j = 0; j < bsk_base_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey_lazy(
                    tmp_des_bsk_base.get() + (i * (encrypted_bsk_ptr_increment)) +
                    (j * coeff_count), bsk_small_ntt_tables[j]);
            }
        }

        // Now we multiply plain modulus to both results in base q and Bsk and
        // allocate them together in one container as (te0)q(te'0)Bsk | ... |te count)q (te' count)Bsk
        // to make it ready for fast_floor
        auto tmp_coeff_bsk_together(allocate_poly(
            coeff_count, dest_count * (coeff_mod_count + bsk_base_mod_count), pool));
        uint64_t *tmp_coeff_bsk_together_ptr = tmp_coeff_bsk_together.get();

        // Base q
        for (size_t i = 0; i < dest_count; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                multiply_poly_scalar_coeffmod(
                    tmp_des_coeff_base.get() + (j * coeff_count) + (i * encrypted_ptr_increment),
                    coeff_count, plain_modulus, coeff_modulus[j],
                    tmp_coeff_bsk_together_ptr + (j * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_ptr_increment;

            for (size_t k = 0; k < bsk_base_mod_count; k++)
            {
                multiply_poly_scalar_coeffmod(
                    tmp_des_bsk_base.get() + (k * coeff_count) + (i * encrypted_bsk_ptr_increment),
                    coeff_count, plain_modulus, bsk_modulus[k],
                    tmp_coeff_bsk_together_ptr + (k * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_bsk_ptr_increment;
        }

        // Allocate a new poly for fast floor result in Bsk
        auto tmp_result_bsk(allocate_poly(coeff_count, dest_count * bsk_base_mod_count, pool));
        for (size_t i = 0; i < dest_count; i++)
        {
            // Step 3: fast floor from q U {Bsk} to Bsk
            base_converter->fast_floor(
                tmp_coeff_bsk_together.get() + (i * (encrypted_ptr_increment + encrypted_bsk_ptr_increment)),
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), pool);

            // Step 4: fast base convert from Bsk to q
            base_converter->fastbconv_sk(
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), encrypted.data(i), pool);
        }
    }

    void Evaluator::ckks_square(Ciphertext &encrypted, MemoryPoolHandle pool)
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        double new_scale = encrypted.scale() * encrypted.scale();

        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
            context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_count = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_count, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare destination
        encrypted.resize(context_, context_data.parms_id(), dest_count);

        //pointer increment to switch to a next polynomial
        size_t encrypted_ptr_increment = coeff_count * coeff_mod_count;

        //Step 1: naive multiplication modulo the coefficient modulus
        //First allocate two temp polys :
        //one for results in base q. This need to be zero
        //for the arbitrary size multiplication; not for 2x2 though
        auto tmp_des(allocate_zero_poly(
            coeff_count * dest_count, coeff_mod_count, pool));

        //Allocate tmp polys for NTT multiplication results in base q
        auto tmp1_poly(allocate_poly(coeff_count, coeff_mod_count, pool));
        auto tmp2_poly(allocate_poly(coeff_count, coeff_mod_count, pool));

        // First convert all the inputs into NTT form
        auto copy_encrypted_ntt(allocate_poly(
            coeff_count * encrypted_size, coeff_mod_count, pool));
        set_poly_poly(encrypted.data(), coeff_count * encrypted_size,
            coeff_mod_count, copy_encrypted_ntt.get());

        // The simplest case when the ciphertext dimension is 2
        if (encrypted_size == 2)
        {
            //Compute c0^2, 2*c0 + c1 and c1^2 modulo q
            //tmp poly to keep 2 * c0 * c1
            auto tmp_second_mul(allocate_poly(coeff_count, coeff_mod_count, pool));

            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                //Des[0] = c0^2 in NTT
                dyadic_product_coeffmod(
                    copy_encrypted_ntt.get() + (i * coeff_count),
                    copy_encrypted_ntt.get() + (i * coeff_count),
                    coeff_count, coeff_modulus[i],
                    tmp_des.get() + (i * coeff_count));

                //Des[1] = 2 * c0 * c1
                dyadic_product_coeffmod(
                    copy_encrypted_ntt.get() + (i * coeff_count),
                    copy_encrypted_ntt.get() + (i * coeff_count) + encrypted_ptr_increment,
                    coeff_count, coeff_modulus[i],
                    tmp_second_mul.get() + (i * coeff_count));
                add_poly_poly_coeffmod(
                    tmp_second_mul.get() + (i * coeff_count),
                    tmp_second_mul.get() + (i * coeff_count),
                    coeff_count, coeff_modulus[i],
                    tmp_des.get() + (i * coeff_count) + encrypted_ptr_increment);

                //Des[2] = c1^2 in NTT
                dyadic_product_coeffmod(
                    copy_encrypted_ntt.get() + (i * coeff_count) + encrypted_ptr_increment,
                    copy_encrypted_ntt.get() + (i * coeff_count) + encrypted_ptr_increment,
                    coeff_count, coeff_modulus[i],
                    tmp_des.get() + (i * coeff_count) + (2 * encrypted_ptr_increment));
            }
        }
        else
        {
            // Perform multiplication on arbitrary size ciphertexts

            // Loop over encrypted1 components [i], seeing if a match exists with an encrypted2
            // component [j] such that [i+j]=[secret_power_index]
            // Only need to check encrypted1 components up to and including [secret_power_index],
            // and strictly less than [encrypted_array.size()]

            // Number of encrypted1 components to check
            size_t current_encrypted_limit = 0;

            for (size_t secret_power_index = 0; secret_power_index < dest_count; secret_power_index++)
            {
                current_encrypted_limit = min(encrypted_size, secret_power_index + 1);

                for (size_t encrypted1_index = 0; encrypted1_index < current_encrypted_limit;
                    encrypted1_index++)
                {
                    // check if a corresponding component in encrypted2 exists
                    if (encrypted_size > secret_power_index - encrypted1_index)
                    {
                        size_t encrypted2_index = secret_power_index - encrypted1_index;

                        // NTT Multiplication and addition for results in q
                        for (size_t i = 0; i < coeff_mod_count; i++)
                        {
                            // ci * dj
                            dyadic_product_coeffmod(
                                copy_encrypted_ntt.get() + (i * coeff_count) +
                                (encrypted_ptr_increment * encrypted1_index),
                                copy_encrypted_ntt.get() + (i * coeff_count) +
                                (encrypted_ptr_increment * encrypted2_index),
                                coeff_count, coeff_modulus[i],
                                tmp1_poly.get() + (i * coeff_count));

                            // Dest[i+j]
                            add_poly_poly_coeffmod(
                                tmp1_poly.get() + (i * coeff_count),
                                tmp_des.get() + (i * coeff_count) +
                                (secret_power_index * coeff_count * coeff_mod_count),
                                coeff_count, coeff_modulus[i],
                                tmp_des.get() + (i * coeff_count) +
                                (secret_power_index * coeff_count * coeff_mod_count));
                        }
                    }
                }
            }
        }

        // Set the final result
        set_poly_poly(tmp_des.get(), coeff_count * dest_count, coeff_mod_count, encrypted.data());

        // Set the scale
        encrypted.scale() = new_scale;
    }

    void Evaluator::relinearize_internal(Ciphertext &encrypted,
        const RelinKeys &relin_keys, size_t destination_size,
        MemoryPoolHandle pool)
    {
        // Verify parameters.
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (relin_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }

        size_t encrypted_size = encrypted.size();

        // Verify parameters.
        if (destination_size < 2 || destination_size > encrypted_size)
        {
            throw invalid_argument("destination_size must be at least 2 and less than or equal to current count");
        }
        if (relin_keys.size() < sub_safe(encrypted_size, size_t(2)))
        {
            throw invalid_argument("not enough relinearization keys");
        }

        // If encrypted is already at the desired level, return
        if (destination_size == encrypted_size)
        {
            return;
        }

        // Calculate number of relinearize_one_step calls needed
        size_t relins_needed = encrypted_size - destination_size;
        for (size_t i = 0; i < relins_needed; i++)
        {
            switch_key_inplace(
                encrypted,
                encrypted.data(encrypted_size - 1),
                static_cast<const KSwitchKeys &>(relin_keys),
                RelinKeys::get_index(encrypted_size - 1),
                pool);
            encrypted_size--;
        }

        // Put the output of final relinearization into destination.
        // Prepare destination only at this point because we are resizing down
        encrypted.resize(context_, context_data_ptr->parms_id(), destination_size);
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_switch_scale_to_next(const Ciphertext &encrypted,
        Ciphertext &destination, MemoryPoolHandle pool)
    {
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (context_data_ptr->parms().scheme() == scheme_type::BFV &&
            encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (context_data_ptr->parms().scheme() == scheme_type::CKKS &&
            !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &next_context_data = *context_data.next_context_data();
        auto &next_parms = next_context_data.parms();

        // q_1,...,q_{k-1}
        auto &next_coeff_modulus = next_parms.coeff_modulus();
        size_t next_coeff_mod_count = next_coeff_modulus.size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.size();
        auto &inv_last_coeff_mod_array =
            context_data.base_converter()->get_inv_last_coeff_mod_array();

        // Size test
        if (!product_fits_in(coeff_count, encrypted_size, next_coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // In CKKS need to transform away from NTT form
        Ciphertext encrypted_copy(pool);
        encrypted_copy = encrypted;
        if (next_parms.scheme() == scheme_type::CKKS)
        {
            transform_from_ntt_inplace(encrypted_copy);
        }

        auto temp1(allocate_uint(coeff_count, pool));

        // Allocate enough room for the result
        auto temp2(allocate_poly(coeff_count * encrypted_size, next_coeff_mod_count, pool));
        auto temp2_ptr = temp2.get();

        for (size_t poly_index = 0; poly_index < encrypted_size; poly_index++)
        {
            // Set temp1 to ct mod qk
            set_uint_uint(
                encrypted_copy.data(poly_index) + next_coeff_mod_count * coeff_count,
                coeff_count, temp1.get());
            // Add (p-1)/2 to change from flooring to rounding.
            auto last_modulus = context_data.parms().coeff_modulus().back();
            uint64_t half = last_modulus.value() >> 1;
            for (size_t j = 0; j < coeff_count; j++)
            {
                temp1.get()[j] = barrett_reduce_63(temp1.get()[j] + half, last_modulus);
            }
            for (size_t mod_index = 0; mod_index < next_coeff_mod_count; mod_index++,
                temp2_ptr += coeff_count)
            {
                // (ct mod qk) mod qi
                modulo_poly_coeffs_63(temp1.get(), coeff_count,
                    next_coeff_modulus[mod_index], temp2_ptr);
                uint64_t half_mod = barrett_reduce_63(half, next_coeff_modulus[mod_index]);
                for (size_t j = 0; j < coeff_count; j++)
                {
                   temp2_ptr[j] = sub_uint_uint_mod(temp2_ptr[j], half_mod, next_coeff_modulus[mod_index]);
                }
                // ((ct mod qi) - (ct mod qk)) mod qi
                sub_poly_poly_coeffmod(
                    encrypted_copy.data(poly_index) + mod_index * coeff_count, temp2_ptr,
                    coeff_count, next_coeff_modulus[mod_index], temp2_ptr);
                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(temp2_ptr, coeff_count,
                    inv_last_coeff_mod_array[mod_index],
                    next_coeff_modulus[mod_index], temp2_ptr);
            }
        }

        // Resize destination
        destination.resize(context_, next_context_data.parms_id(), encrypted_size);
        destination.is_ntt_form() = false;

        set_poly_poly(temp2.get(), coeff_count * encrypted_size, next_coeff_mod_count,
            destination.data());

        // In CKKS need to transform back to NTT form
        if (next_parms.scheme() == scheme_type::CKKS)
        {
            transform_to_ntt_inplace(destination);

            // Also change the scale
            destination.scale() = encrypted.scale() /
                static_cast<double>(context_data.parms().coeff_modulus().back().value());
        }
    }

    void Evaluator::mod_switch_drop_to_next(const Ciphertext &encrypted,
        Ciphertext &destination, MemoryPoolHandle pool)
    {
        // Assuming at this point encrypted is already validated.
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (context_data_ptr->parms().scheme() == scheme_type::CKKS &&
            !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = next_context_data.parms();

        // Check that scale is positive and not too large
        if (encrypted.scale() <= 0 || (static_cast<int>(log2(encrypted.scale())) >=
            next_context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // q_1,...,q_{k-1}
        size_t next_coeff_mod_count = next_parms.coeff_modulus().size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.size();

        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, next_coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        size_t rns_poly_total_count = next_coeff_mod_count * coeff_count;

        if (&encrypted == &destination)
        {
            // Switching in-place so need temporary space
            auto temp(allocate_uint(rns_poly_total_count * encrypted_size, pool));

            // Copy data over to temp
            for (size_t i = 0; i < encrypted_size; i++)
            {
                const uint64_t *encrypted_ptr = encrypted.data(i);
                for (size_t j = 0; j < next_coeff_mod_count; j++)
                {
                    set_uint_uint(encrypted_ptr + (j * coeff_count), coeff_count,
                        temp.get() + (i * rns_poly_total_count) + (j * coeff_count));
                }
            }

            // Resize destination before writing
            destination.resize(context_, next_context_data.parms_id(), encrypted_size);
            destination.is_ntt_form() = true;
            destination.scale() = encrypted.scale();

            // Copy data to destination
            set_uint_uint(temp.get(), rns_poly_total_count * encrypted_size,
                destination.data());
        }
        else
        {
            // Resize destination before writing
            destination.resize(context_, next_context_data.parms_id(), encrypted_size);
            destination.is_ntt_form() = true;
            destination.scale() = encrypted.scale();

            // Copy data directly to new destination
            for (size_t i = 0; i < encrypted_size; i++)
            {
                for (size_t j = 0; j < next_coeff_mod_count; j++)
                {
                    const uint64_t *encrypted_ptr = encrypted.data(i);
                    set_uint_uint(encrypted_ptr + (j * coeff_count), coeff_count,
                        destination.data() + (i * rns_poly_total_count) + (j * coeff_count));
                }
            }
        }
    }

    void Evaluator::mod_switch_drop_to_next(Plaintext &plain)
    {
        // Assuming at this point plain is already validated.
        auto context_data_ptr = context_->get_context_data(plain.parms_id());
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (!context_data_ptr->next_context_data())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }

        // Extract encryption parameters.
        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = context_data_ptr->next_context_data()->parms();

        // Check that scale is positive and not too large
        if (plain.scale() <= 0 || (static_cast<int>(log2(plain.scale())) >=
            next_context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // q_1,...,q_{k-1}
        auto &next_coeff_modulus = next_parms.coeff_modulus();
        size_t next_coeff_mod_count = next_coeff_modulus.size();
        size_t coeff_count = next_parms.poly_modulus_degree();

        // Compute destination size first for exception safety
        auto dest_size = mul_safe(next_coeff_mod_count, coeff_count);

        plain.parms_id() = parms_id_zero;
        plain.resize(dest_size);
        plain.parms_id() = next_context_data.parms_id();
    }

    void Evaluator::mod_switch_to_next(const Ciphertext &encrypted,
        Ciphertext &destination, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (context_->last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_->first_context_data()->parms().scheme())
        {
        case scheme_type::BFV:
            // Modulus switching with scaling
            mod_switch_scale_to_next(encrypted, destination, move(pool));
            break;

        case scheme_type::CKKS:
            // Modulus switching without scaling
            mod_switch_drop_to_next(encrypted, destination, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (destination.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_switch_to_inplace(Ciphertext &encrypted,
        parms_id_type parms_id, MemoryPoolHandle pool)
    {
        // Verify parameters.
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_->get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(encrypted, pool);
        }
    }

    void Evaluator::mod_switch_to_inplace(Plaintext &plain, parms_id_type parms_id)
    {
        // Verify parameters.
        auto context_data_ptr = context_->get_context_data(plain.parms_id());
        auto target_context_data_ptr = context_->get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!context_->get_context_data(parms_id))
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(plain);
        }
    }

    void Evaluator::rescale_to_next(const Ciphertext &encrypted, Ciphertext &destination,
        MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (context_->last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_->first_context_data()->parms().scheme())
        {
        case scheme_type::BFV:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::CKKS:
            // Modulus switching with scaling
            mod_switch_scale_to_next(encrypted, destination, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (destination.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::rescale_to_inplace(Ciphertext &encrypted, parms_id_type parms_id,
        MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_->get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::BFV:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::CKKS:
            while (encrypted.parms_id() != parms_id)
            {
                // Modulus switching with scaling
                mod_switch_scale_to_next(encrypted, encrypted, pool);
            }
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_many(const vector<Ciphertext> &encrypteds,
        const RelinKeys &relin_keys, Ciphertext &destination,
        MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (encrypteds.size() == 0)
        {
            throw invalid_argument("encrypteds vector must not be empty");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }

        // There is at least one ciphertext
        auto context_data_ptr = context_->get_context_data(encrypteds[0].parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypteds is not valid for encryption parameters");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();

        if (parms.scheme() != scheme_type::BFV)
        {
            throw logic_error("unsupported scheme");
        }

        // If there is only one ciphertext, return it.
        if (encrypteds.size() == 1)
        {
            destination = encrypteds[0];
            return;
        }

        // Do first level of multiplications
        vector<Ciphertext> product_vec;
        for (size_t i = 0; i < encrypteds.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            if (encrypteds[i].data() == encrypteds[i + 1].data())
            {
                square(encrypteds[i], temp);
            }
            else
            {
                multiply(encrypteds[i], encrypteds[i + 1], temp);
            }
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(move(temp));
        }
        if (encrypteds.size() & 1)
        {
            product_vec.emplace_back(encrypteds.back());
        }

        // Repeatedly multiply and add to the back of the vector until the end is reached
        for (size_t i = 0; i < product_vec.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            multiply(product_vec[i], product_vec[i + 1], temp);
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(move(temp));
        }

        destination = product_vec.back();
    }

    void Evaluator::exponentiate_inplace(Ciphertext &encrypted, uint64_t exponent,
        const RelinKeys &relin_keys, MemoryPoolHandle pool)
    {
        // Verify parameters.
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_->get_context_data(relin_keys.parms_id()))
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be 0");
        }

        // Fast case
        if (exponent == 1)
        {
            return;
        }

        // Create a vector of copies of encrypted
        vector<Ciphertext> exp_vector(static_cast<size_t>(exponent), encrypted);
        multiply_many(exp_vector, relin_keys, encrypted, move(pool));
    }

    void Evaluator::add_plain_inplace(Ciphertext &encrypted, const Plaintext &plain)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::BFV && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (parms.scheme() == scheme_type::CKKS && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (plain.is_ntt_form() != encrypted.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (encrypted.is_ntt_form() &&
            (encrypted.parms_id() != plain.parms_id()))
        {
            throw invalid_argument("encrypted and plain parameter mismatch");
        }
        if (!are_same_scale(encrypted, plain))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::BFV:
        {
            multiply_add_plain_with_scaling_variant(
                plain, context_data, encrypted.data());
            break;
        }

        case scheme_type::CKKS:
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                add_poly_poly_coeffmod(encrypted.data() + (j * coeff_count),
                    plain.data() + (j*coeff_count), coeff_count,
                    coeff_modulus[j], encrypted.data() + (j * coeff_count));
                
                
                if(j == 0){
                    for (int k = 0; k < coeff_count; k++)
                    {
                        std::cout << k << ": " << plain.data()[k] << std::endl;
                    }
                }
            }
            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    
    void Evaluator::sub_plain_inplace(Ciphertext &encrypted, const Plaintext &plain)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::BFV && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (parms.scheme() == scheme_type::CKKS && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (plain.is_ntt_form() != encrypted.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (encrypted.is_ntt_form() &&
            (encrypted.parms_id() != plain.parms_id()))
        {
            throw invalid_argument("encrypted and plain parameter mismatch");
        }
        if (!are_same_scale(encrypted, plain))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::BFV:
        {
            multiply_sub_plain_with_scaling_variant(
                plain, context_data, encrypted.data());
            break;
        }

        case scheme_type::CKKS:
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                sub_poly_poly_coeffmod(encrypted.data() + (j * coeff_count),
                    plain.data() + (j * coeff_count), coeff_count,
                    coeff_modulus[j], encrypted.data() + (j * coeff_count));
            }
            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_plain_inplace(Ciphertext &encrypted,
        const Plaintext &plain, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form() != plain.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (encrypted.is_ntt_form())
        {
            multiply_plain_ntt(encrypted, plain);
        }
        else
        {
            multiply_plain_normal(encrypted, plain, move(pool));
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_plain_normal(Ciphertext &encrypted,
        const Plaintext &plain, MemoryPool &pool)
    {
        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();

        auto plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();
        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();

        size_t encrypted_size = encrypted.size();
        size_t plain_coeff_count = plain.coeff_count();
        size_t plain_nonzero_coeff_count = plain.nonzero_coeff_count();

        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        double new_scale = encrypted.scale() * plain.scale();

        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
            context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // Set the scale
        encrypted.scale() = new_scale;

        /*
        Optimizations for constant / monomial multiplication can lead to the
        presence of a timing side-channel in use-cases where the plaintext
        data should also be kept private.
        */
        if (plain_nonzero_coeff_count == 1)
        {
            // Multiplying by a monomial?
            size_t mono_exponent = plain.significant_coeff_count() - 1;

            if (plain[mono_exponent] >= plain_upper_half_threshold)
            {
                if (!context_data.qualifiers().using_fast_plain_lift)
                {
                    auto adjusted_coeff(allocate_uint(coeff_mod_count, pool));
                    auto decomposed_coeff(allocate_uint(coeff_mod_count, pool));
                    add_uint_uint64(plain_upper_half_increment, plain[mono_exponent],
                        coeff_mod_count, adjusted_coeff.get());
                    decompose_single_coeff(context_data, adjusted_coeff.get(),
                        decomposed_coeff.get(), pool);

                    for (size_t i = 0; i < encrypted_size; i++)
                    {
                        for (size_t j = 0; j < coeff_mod_count; j++)
                        {
                            negacyclic_multiply_poly_mono_coeffmod(
                                encrypted.data(i) + (j * coeff_count), coeff_count,
                                decomposed_coeff[j], mono_exponent, coeff_modulus[j],
                                encrypted.data(i) + (j * coeff_count), pool);
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < encrypted_size; i++)
                    {
                        for (size_t j = 0; j < coeff_mod_count; j++)
                        {
                            negacyclic_multiply_poly_mono_coeffmod(
                                encrypted.data(i) + (j * coeff_count), coeff_count,
                                plain[mono_exponent] + plain_upper_half_increment[j],
                                mono_exponent, coeff_modulus[j],
                                encrypted.data(i) + (j * coeff_count), pool);
                        }
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < encrypted_size; i++)
                {
                    for (size_t j = 0; j < coeff_mod_count; j++)
                    {
                        negacyclic_multiply_poly_mono_coeffmod(
                            encrypted.data(i) + (j * coeff_count), coeff_count,
                            plain[mono_exponent], mono_exponent, coeff_modulus[j],
                            encrypted.data(i) + (j * coeff_count), pool);
                    }
                }
            }

            return;
        }

        // Generic plain case
        auto adjusted_poly(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
        auto decomposed_poly(allocate_uint(coeff_count * coeff_mod_count, pool));
        uint64_t *poly_to_transform = nullptr;
        if (!context_data.qualifiers().using_fast_plain_lift)
        {
            // Reposition coefficients.
            const uint64_t *plain_ptr = plain.data();
            uint64_t *adjusted_poly_ptr = adjusted_poly.get();
            for (size_t i = 0; i < plain_coeff_count; i++, plain_ptr++,
                adjusted_poly_ptr += coeff_mod_count)
            {
                if (*plain_ptr >= plain_upper_half_threshold)
                {
                    add_uint_uint64(plain_upper_half_increment,
                        *plain_ptr, coeff_mod_count, adjusted_poly_ptr);
                }
                else
                {
                    *adjusted_poly_ptr = *plain_ptr;
                }
            }
            decompose(context_data, adjusted_poly.get(), decomposed_poly.get(), pool);
            poly_to_transform = decomposed_poly.get();
        }
        else
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                const uint64_t *plain_ptr = plain.data();
                uint64_t *adjusted_poly_ptr = adjusted_poly.get() + (j * coeff_count);
                uint64_t current_plain_upper_half_increment = plain_upper_half_increment[j];
                for (size_t i = 0; i < plain_coeff_count; i++, plain_ptr++, adjusted_poly_ptr++)
                {
                    // Need to lift the coefficient in each qi
                    if (*plain_ptr >= plain_upper_half_threshold)
                    {
                        *adjusted_poly_ptr = *plain_ptr + current_plain_upper_half_increment;
                    }
                    // No need for lifting
                    else
                    {
                        *adjusted_poly_ptr = *plain_ptr;
                    }
                }
            }
            poly_to_transform = adjusted_poly.get();
        }

        // Need to multiply each component in encrypted with decomposed_poly (plain poly)
        // Transform plain poly only once
        for (size_t i = 0; i < coeff_mod_count; i++)
        {
            ntt_negacyclic_harvey(
                poly_to_transform + (i * coeff_count), coeff_small_ntt_tables[i]);
        }

        for (size_t i = 0; i < encrypted_size; i++)
        {
            uint64_t *encrypted_ptr = encrypted.data(i);
            for (size_t j = 0; j < coeff_mod_count; j++, encrypted_ptr += coeff_count)
            {
                // Explicit inline to avoid unnecessary copy
                //ntt_multiply_poly_nttpoly(encrypted.data(i) + (j * coeff_count),
                //poly_to_transform + (j * coeff_count),
                //    coeff_small_ntt_tables_[j], encrypted.data(i) + (j * coeff_count), pool);

                // Lazy reduction
                ntt_negacyclic_harvey_lazy(encrypted_ptr, coeff_small_ntt_tables[j]);
                dyadic_product_coeffmod(encrypted_ptr, poly_to_transform + (j * coeff_count),
                    coeff_count, coeff_modulus[j], encrypted_ptr);
                inverse_ntt_negacyclic_harvey(encrypted_ptr, coeff_small_ntt_tables[j]);
            }
        }
    }

    void Evaluator::multiply_plain_ntt(Ciphertext &encrypted_ntt,
        const Plaintext &plain_ntt)
    {
        // Verify parameters.
        if (!plain_ntt.is_ntt_form())
        {
            throw invalid_argument("plain_ntt is not in NTT form");
        }
        if (encrypted_ntt.parms_id() != plain_ntt.parms_id())
        {
            throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted_ntt.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_ntt_size = encrypted_ntt.size();

        // Size check
        if (!product_fits_in(encrypted_ntt_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        double new_scale = encrypted_ntt.scale() * plain_ntt.scale();

        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
            context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        for (size_t i = 0; i < encrypted_ntt_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                dyadic_product_coeffmod(
                    encrypted_ntt.data(i) + (j * coeff_count),
                    plain_ntt.data() + (j * coeff_count),
                    coeff_count, coeff_modulus[j],
                    encrypted_ntt.data(i) + (j * coeff_count));
            }
        }

        // Set the scale
        encrypted_ntt.scale() = new_scale;
    }


    void Evaluator::transform_to_ntt_inplace(Plaintext &plain,
        parms_id_type parms_id, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for the current context");
        }
        if (plain.is_ntt_form())
        {
            throw invalid_argument("plain is already in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t plain_coeff_count = plain.coeff_count();

        auto plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();

        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Resize to fit the entire NTT transformed (ciphertext size) polynomial
        // Note that the new coefficients are automatically set to 0
        plain.resize(coeff_count * coeff_mod_count);

        // Verify if plain lift is needed
        if (!context_data.qualifiers().using_fast_plain_lift)
        {
            auto adjusted_poly(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
            for (size_t i = 0; i < plain_coeff_count; i++)
            {
                if (plain[i] >= plain_upper_half_threshold)
                {
                    add_uint_uint64(plain_upper_half_increment, plain[i],
                        coeff_mod_count, adjusted_poly.get() + (i * coeff_mod_count));
                }
                else
                {
                    adjusted_poly[i * coeff_mod_count] = plain[i];
                }
            }
            decompose(context_data, adjusted_poly.get(), plain.data(), pool);
        }
        // No need for composed plain lift and decomposition
        else
        {
            for (size_t j = coeff_mod_count; j--; )
            {
                const uint64_t *plain_ptr = plain.data();
                uint64_t *adjusted_poly_ptr = plain.data() + (j * coeff_count);
                uint64_t current_plain_upper_half_increment = plain_upper_half_increment[j];
                for (size_t i = 0; i < plain_coeff_count; i++, plain_ptr++, adjusted_poly_ptr++)
                {
                    // Need to lift the coefficient in each qi
                    if (*plain_ptr >= plain_upper_half_threshold)
                    {
                        *adjusted_poly_ptr = *plain_ptr + current_plain_upper_half_increment;
                    }
                    // No need for lifting
                    else
                    {
                        *adjusted_poly_ptr = *plain_ptr;
                    }
                }
            }
        }

        // Transform to NTT domain
        for (size_t i = 0; i < coeff_mod_count; i++)
        {
            ntt_negacyclic_harvey(
                plain.data() + (i * coeff_count), coeff_small_ntt_tables[i]);
        }

        plain.parms_id() = parms_id;
    }

    void Evaluator::transform_to_ntt_inplace(Ciphertext &encrypted)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted is already in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Transform each polynomial to NTT domain
        for (size_t i = 0; i < encrypted_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                ntt_negacyclic_harvey(
                    encrypted.data(i) + (j * coeff_count), coeff_small_ntt_tables[j]);
            }
        }

        // Finally change the is_ntt_transformed flag
        encrypted.is_ntt_form() = true;
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::transform_from_ntt_inplace(Ciphertext &encrypted_ntt)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted_ntt, context_) ||
            !is_buffer_valid(encrypted_ntt))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_->get_context_data(encrypted_ntt.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }
        if (!encrypted_ntt.is_ntt_form())
        {
            throw invalid_argument("encrypted_ntt is not in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = parms.coeff_modulus().size();
        size_t encrypted_ntt_size = encrypted_ntt.size();

        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Transform each polynomial from NTT domain
        for (size_t i = 0; i < encrypted_ntt_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey(
                    encrypted_ntt.data(i) + (j * coeff_count), coeff_small_ntt_tables[j]);
            }
        }

        // Finally change the is_ntt_transformed flag
        encrypted_ntt.is_ntt_form() = false;
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted_ntt.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::apply_galois_inplace(Ciphertext &encrypted, uint64_t galois_elt,
        const GaloisKeys &galois_keys, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Don't validate all of galois_keys but just check the parms_id.
        if (galois_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // Check if Galois key is generated or not.
        if (!galois_keys.has_key(galois_elt))
        {
            throw invalid_argument("Galois key not present");
        }

        uint64_t m = mul_safe(static_cast<uint64_t>(coeff_count), uint64_t(2));
        int n_power_of_two = get_power_of_two(static_cast<uint64_t>(coeff_count));

        // Verify parameters
        if (!(galois_elt & 1) || unsigned_geq(galois_elt, m))
        {
            throw invalid_argument("Galois element is not valid");
        }
        if (encrypted_size > 2)
        {
            throw invalid_argument("encrypted size must be 2");
        }

        auto temp(allocate_poly(coeff_count, coeff_mod_count, pool));

        // DO NOT CHANGE EXECUTION ORDER OF FOLLOWING SECTION
        // BEGIN: Apply Galois for each ciphertext
        // Execution order is sensitive, since apply_galois is not inplace!
        if (parms.scheme() == scheme_type::BFV)
        {
            // !!! DO NOT CHANGE EXECUTION ORDER!!!
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                util::apply_galois(
                    encrypted.data(0) + i * coeff_count,
                    n_power_of_two,
                    galois_elt,
                    coeff_modulus[i],
                    temp.get() + i * coeff_count);
            }
            // copy result to encrypted.data(0)
            set_poly_poly(temp.get(), coeff_count, coeff_mod_count,
                encrypted.data(0));
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                util::apply_galois(
                    encrypted.data(1) + i * coeff_count,
                    n_power_of_two,
                    galois_elt,
                    coeff_modulus[i],
                    temp.get() + i * coeff_count);
            }
        }
        else if (parms.scheme() == scheme_type::CKKS)
        {
            // !!! DO NOT CHANGE EXECUTION ORDER!!!
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                util::apply_galois_ntt(
                    encrypted.data(0) + i * coeff_count,
                    n_power_of_two,
                    galois_elt,
                    temp.get() + i * coeff_count);
            }
            // copy result to encrypted.data(0)
            set_poly_poly(temp.get(), coeff_count, coeff_mod_count,
                encrypted.data(0));
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                util::apply_galois_ntt(
                    encrypted.data(1) + i * coeff_count,
                    n_power_of_two,
                    galois_elt,
                    temp.get() + i * coeff_count);
            }
        }
        else
        {
            throw logic_error("scheme not implemented");
        }
        // wipe encrypted.data(1)
        set_zero_poly(coeff_count, coeff_mod_count, encrypted.data(1));
        // END: Apply Galois for each ciphertext
        // REORDERING IS SAFE NOW

        // Calculate (temp * galois_key[0], temp * galois_key[1]) + (ct[0], 0)
        switch_key_inplace(
            encrypted,
            temp.get(),
            static_cast<const KSwitchKeys &>(galois_keys),
            GaloisKeys::get_index(galois_elt),
            pool);
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::rotate_internal(Ciphertext &encrypted, int steps,
        const GaloisKeys &galois_keys, MemoryPoolHandle pool)
    {
        auto context_data_ptr = context_->get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_data_ptr->qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }
        if (galois_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        // Is there anything to do?
        if (steps == 0)
        {
            return;
        }

        size_t coeff_count = context_data_ptr->parms().poly_modulus_degree();

        // Check if Galois key is generated or not.
        if (galois_keys.has_key(galois_elt_from_step(steps, coeff_count)))
        {
            // Perform rotation and key switching
            apply_galois_inplace(encrypted,
                                 galois_elt_from_step(steps, coeff_count),
                                 galois_keys, move(pool));
        }
        else
        {
            // Convert the steps to NAF: guarantees using smallest HW
            vector<int> naf_steps = naf(steps);

            // If naf_steps contains only one element, then this is a power-of-two
            // rotation and we would have expected not to get to this part of the
            // if-statement.
            if (naf_steps.size() == 1)
            {
                throw invalid_argument("Galois key not present");
            }

            for (size_t i = 0; i < naf_steps.size(); i++)
            {
                // We might have a NAF-term of size coeff_count / 2; this corresponds
                // to no rotation so we skip it.
                if (safe_cast<size_t>(abs(naf_steps[i])) == (coeff_count >> 1))
                {
                    continue;
                }

                // Apply rotation for this step
                rotate_internal(encrypted, naf_steps[i], galois_keys, pool);
            }
        }
    }

    void Evaluator::switch_key_inplace(
        Ciphertext &encrypted,
        const uint64_t *target,
        const KSwitchKeys &kswitch_keys,
        size_t kswitch_keys_index,
        MemoryPoolHandle pool)
    {
        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_->get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &key_context_data = *context_->key_context_data();
        auto &key_parms = key_context_data.parms();
        auto scheme = parms.scheme();

        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target)
        {
            throw invalid_argument("target");
        }
        if (!context_->using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        // Don't validate all of kswitch_keys but just check the parms_id.
        if (kswitch_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }

        if (kswitch_keys_index >= kswitch_keys.data().size())
        {
            throw out_of_range("kswitch_keys_index");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::BFV && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::CKKS && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_mod_count = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_mod_count = key_modulus.size();
        size_t rns_mod_count = decomp_mod_count + 1;
        auto &small_ntt_tables = key_context_data.small_ntt_tables();
        auto &modswitch_factors = key_context_data.base_converter()->
            get_inv_last_coeff_mod_array();

        // Size check
        if (!product_fits_in(coeff_count, rns_mod_count, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare input
        auto &key_vector = kswitch_keys.data()[kswitch_keys_index];

        // Check only the used component in KSwitchKeys.
        for (auto &each_key : key_vector)
        {
            if (!is_metadata_valid_for(each_key, context_) ||
                !is_buffer_valid(each_key))
            {
                throw invalid_argument(
                    "kswitch_keys is not valid for encryption parameters");
            }
        }

        // Temporary results
        Pointer<uint64_t> temp_poly[2] {
            allocate_zero_poly(2 * coeff_count, rns_mod_count, pool),
            allocate_zero_poly(2 * coeff_count, rns_mod_count, pool)
        };

        // RNS decomposition index = key index
        for (size_t i = 0; i < decomp_mod_count; i++)
        {
            // For each RNS decomposition, multiply with key data and sum up.
            auto local_small_poly_0(allocate_uint(coeff_count, pool));
            auto local_small_poly_1(allocate_uint(coeff_count, pool));
            auto local_small_poly_2(allocate_uint(coeff_count, pool));

            const uint64_t *local_encrypted_ptr = nullptr;
            set_uint_uint(
                target + i * coeff_count,
                coeff_count,
                local_small_poly_0.get());
            if (scheme == scheme_type::CKKS)
            {
                inverse_ntt_negacyclic_harvey(
                    local_small_poly_0.get(),
                    small_ntt_tables[i]);
            }
            // Key RNS representation
            for (size_t j = 0; j < rns_mod_count; j++)
            {
                size_t index = (j == decomp_mod_count ? key_mod_count - 1 : j);
                if (scheme == scheme_type::CKKS && i == j)
                {
                    local_encrypted_ptr = target + j * coeff_count;
                }
                else
                {
                    // Reduce modulus only if needed
                    if (key_modulus[i].value() <= key_modulus[index].value())
                    {
                        set_uint_uint(
                            local_small_poly_0.get(),
                            coeff_count,
                            local_small_poly_1.get());
                    }
                    else
                    {
                        modulo_poly_coeffs_63(
                            local_small_poly_0.get(),
                            coeff_count,
                            key_modulus[index],
                            local_small_poly_1.get());
                    }

                    // Lazy reduction, output in [0, 4q).
                    ntt_negacyclic_harvey_lazy(
                        local_small_poly_1.get(),
                        small_ntt_tables[index]);
                    local_encrypted_ptr = local_small_poly_1.get();
                }
                // Two components in key
                for (size_t k = 0; k < 2; k++)
                {
                    // dyadic_product_coeffmod(
                    //     local_encrypted_ptr,
                    //     key_vector[i].data(k) + index * coeff_count,
                    //     coeff_count,
                    //     key_modulus[index],
                    //     local_small_poly_2.get());
                    // add_poly_poly_coeffmod(
                    //     local_small_poly_2.get(),
                    //     temp_poly[k].get() + j * coeff_count,
                    //     coeff_count,
                    //     key_modulus[index],
                    //     temp_poly[k].get() + j * coeff_count);
                    const uint64_t *key_ptr = key_vector[i].data().data(k);
                    for (size_t l = 0; l < coeff_count; l++)
                    {
                        unsigned long long local_wide_product[2];
                        unsigned long long local_low_word;
                        unsigned char local_carry;

                        multiply_uint64(
                            local_encrypted_ptr[l],
                            key_ptr[(index * coeff_count) + l],
                            local_wide_product);
                        local_carry = add_uint64(
                            temp_poly[k].get()[(j * coeff_count + l) * 2],
                            local_wide_product[0],
                            &local_low_word);
                        temp_poly[k].get()[(j * coeff_count + l) * 2] =
                            local_low_word;
                        temp_poly[k].get()[(j * coeff_count + l) * 2 + 1] +=
                            local_wide_product[1] + local_carry;
                    }
                }
            }
        }

        // Results are now stored in temp_poly[k]
        // Modulus switching should be performed
        auto local_small_poly(allocate_uint(coeff_count, pool));
        for (size_t k = 0; k < 2; k++)
        {
            // Reduce (ct mod 4qk) mod qk
            uint64_t *temp_poly_ptr = temp_poly[k].get() +
                decomp_mod_count * coeff_count * 2;
            for (size_t l = 0; l < coeff_count; l++)
            {
                temp_poly_ptr[l] = barrett_reduce_128(
                    temp_poly_ptr + l * 2,
                    key_modulus[key_mod_count - 1]);
            }
            // Lazy reduction, they are then reduced mod qi
            uint64_t *temp_last_poly_ptr = temp_poly[k].get() + decomp_mod_count * coeff_count * 2;
            inverse_ntt_negacyclic_harvey_lazy(
                temp_last_poly_ptr,
                small_ntt_tables[key_mod_count - 1]);

            // Add (p-1)/2 to change from flooring to rounding.
            uint64_t half = key_modulus[key_mod_count - 1].value() >> 1;
            for (size_t l = 0; l < coeff_count; l++)
            {
                temp_last_poly_ptr[l] = barrett_reduce_63(temp_last_poly_ptr[l] + half,
                    key_modulus[key_mod_count - 1]);
            }

            uint64_t *encrypted_ptr = encrypted.data(k);
            for (size_t j = 0; j < decomp_mod_count; j++)
            {
                temp_poly_ptr = temp_poly[k].get() + j * coeff_count * 2;
                // (ct mod 4qi) mod qi
                for (size_t l = 0; l < coeff_count; l++)
                {
                    temp_poly_ptr[l] = barrett_reduce_128(
                        temp_poly_ptr + l * 2,
                        key_modulus[j]);
                }
                // (ct mod 4qk) mod qi
                modulo_poly_coeffs_63(
                    temp_last_poly_ptr,
                    coeff_count,
                    key_modulus[j],
                    local_small_poly.get());

                uint64_t half_mod = barrett_reduce_63(half, key_modulus[j]);
                for (size_t l = 0; l < coeff_count; l++)
                {
                    local_small_poly.get()[l] = sub_uint_uint_mod(local_small_poly.get()[l],
                        half_mod,
                        key_modulus[j]);
                }

                if (scheme == scheme_type::CKKS)
                {
                    ntt_negacyclic_harvey(
                        local_small_poly.get(),
                        small_ntt_tables[j]);
                }
                else if (scheme == scheme_type::BFV)
                {
                    inverse_ntt_negacyclic_harvey(
                        temp_poly_ptr,
                        small_ntt_tables[j]);
                }
                // ((ct mod qi) - (ct mod qk)) mod qi
                sub_poly_poly_coeffmod(
                    temp_poly_ptr,
                    local_small_poly.get(),
                    coeff_count,
                    key_modulus[j],
                    temp_poly_ptr);
                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(
                    temp_poly_ptr,
                    coeff_count,
                    modswitch_factors[j],
                    key_modulus[j],
                    temp_poly_ptr);
                add_poly_poly_coeffmod(
                    temp_poly_ptr,
                    encrypted_ptr + j * coeff_count,
                    coeff_count,
                    key_modulus[j],
                    encrypted_ptr + j * coeff_count);
            }
        }
    }


    // miran
    // c0 <- pt + c0
    void Evaluator::add_plain(
        const Plaintext &plain,
        const SEALContext::ContextData &context_data,
        uint64_t *destination) const
    {
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_mod_count = coeff_modulus.size();  // 1
        size_t plain_coeff_count = plain.coeff_count(); // 4
        
        for (size_t i = 0; i < plain_coeff_count; i++, destination++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                uint64_t plain_coeff = plain.data()[i];
                destination[j * coeff_count] = util::add_uint_uint_mod(
                    destination[j * coeff_count], plain_coeff, coeff_modulus[j]);
            }
        }
    }

 

    // miran
    // add by a plaintext polynomial which has a message in its coefficient
    void Evaluator::add_plain_inplace_ckks(Ciphertext &encrypted, const Plaintext &plain)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::BFV && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (parms.scheme() == scheme_type::CKKS && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (plain.is_ntt_form() != encrypted.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (encrypted.is_ntt_form() &&
            (encrypted.parms_id() != plain.parms_id()))
        {
            throw invalid_argument("encrypted and plain parameter mismatch");
        }
        if (!are_same_scale(encrypted, plain))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
            case scheme_type::BFV:
            {
                add_plain(plain, context_data, encrypted.data());
                break;
            }

            default:
                throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    
    void Evaluator::add_const_inplace(Ciphertext &encrypted, const vector<uint64_t> &plain)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::BFV)
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (parms.scheme() == scheme_type::CKKS && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
 
        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t N = parms.poly_modulus_degree();//N
        size_t coeff_mod_count = coeff_modulus.size(); // l
        
        // Size check
        if (!product_fits_in(N, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
        
        for (size_t j = 0; j < coeff_mod_count; j++)
        {
            // defined in polyarithsmallmod.h
            add_poly_const_coeffmod(encrypted.data() + (j * N),
                                   plain[j], N,
                                   coeff_modulus[j], encrypted.data() + (j * N));
        }
        
        
    #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
    #endif
    }


    
    // miran
    // multiply by a plaintext polynomial which has a message in its coefficient
    void Evaluator::multiply_plain_inplace_ckks(Ciphertext &encrypted,
                                            const Plaintext &plain, MemoryPoolHandle pool)
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form() != plain.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
                
        if (encrypted.is_ntt_form())
        {
            multiply_plain_ntt(encrypted, plain);
        }
        else
        {
            multiply_plain_normal_ckks(encrypted, plain, move(pool));
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        // Transparent ciphertext output is not allowed.
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    // miran
    void Evaluator::multiply_plain_normal_ckks(Ciphertext &encrypted,
                                            const Plaintext &plain, MemoryPool &pool)
    {
        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
            
        //auto plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        //auto plain_upper_half_increment = context_data.plain_upper_half_increment();
        auto &coeff_small_ntt_tables = context_data.small_ntt_tables();
            
        size_t encrypted_size = encrypted.size();
        size_t plain_coeff_count = plain.coeff_count();
        size_t plain_nonzero_coeff_count = plain.nonzero_coeff_count();
            
        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
            
        double new_scale = encrypted.scale() * plain.scale();
            
        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
                                context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }
            
        // Set the scale
        encrypted.scale() = new_scale;
            
        /*
            Optimizations for constant / monomial multiplication can lead to the
            presence of a timing side-channel in use-cases where the plaintext
            data should also be kept private.
          */
        if (plain_nonzero_coeff_count == 1)
        {
            // Multiplying by a monomial?
            size_t mono_exponent = plain.significant_coeff_count() - 1;
            for (size_t i = 0; i < encrypted_size; i++)
            {
                for (size_t j = 0; j < coeff_mod_count; j++)
                {
                    negacyclic_multiply_poly_mono_coeffmod(
                        encrypted.data(i) + (j * coeff_count), coeff_count,
                        plain[mono_exponent], mono_exponent, coeff_modulus[j],
                        encrypted.data(i) + (j * coeff_count), pool);
                }
            }
            return;
        }
    }
    
    //-----------------------------
    // Homomorphic operations by a plainxt polynomial
    // which is encoded at a specific lower level (than the highest level)
    //-----------------------------

    // miran
    // add by a plaintext polynomial encoded at a specific level
        void Evaluator::add_plain_leveled_inplace(Ciphertext &encrypted, const Plaintext &plain)
        {
            // Verify parameters.
            if (!is_metadata_valid_for(encrypted, context_))
            {
                throw invalid_argument("encrypted is not valid for encryption parameters");
            }
            if (!is_valid_for(plain, context_))
            {
                throw invalid_argument("plain is not valid for encryption parameters");
            }
            
            auto &context_data = *context_->get_context_data(encrypted.parms_id());
            auto &parms = context_data.parms();
            if (parms.scheme() == scheme_type::BFV)
            {
                throw invalid_argument("BFV encrypted cannot be in NTT form");
            }
            if (parms.scheme() == scheme_type::CKKS && !encrypted.is_ntt_form())
            {
                throw invalid_argument("CKKS encrypted must be in NTT form");
            }
            if (plain.is_ntt_form() != encrypted.is_ntt_form())
            {
                throw invalid_argument("NTT form mismatch");
            }
            //            if (encrypted.is_ntt_form() &&
            //                (encrypted.parms_id() != plain.parms_id()))
            //            {
            //                throw invalid_argument("encrypted and plain parameter mismatch");
            //            }
            if (!are_same_scale(encrypted, plain))
            {
                throw invalid_argument("scale mismatch");
            }
            
            // Extract encryption parameters.
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_count = parms.poly_modulus_degree();
            size_t coeff_mod_count = coeff_modulus.size();
            
            // Size check
            if (!product_fits_in(coeff_count, coeff_mod_count))
            {
                throw logic_error("invalid parameters");
            }
            
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                add_poly_poly_coeffmod(encrypted.data() + (j * coeff_count),
                                       plain.data() + (j*coeff_count), coeff_count,
                                       coeff_modulus[j], encrypted.data() + (j * coeff_count));
            }
            
    #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
            // Transparent ciphertext output is not allowed.
            if (encrypted.is_transparent())
            {
                throw logic_error("result ciphertext is transparent");
            }
    #endif
        }

    // miran
    // multiply by a plaintext polynomial encoded at a specific level
    void Evaluator::multiply_plain_leveled_inplace(Ciphertext &encrypted,
                                                   const Plaintext &plain, MemoryPoolHandle pool)
    {
        if (!is_metadata_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!context_->get_context_data(encrypted.parms_id()))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form() != plain.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        
        if (encrypted.is_ntt_form())
        {
            multiply_plain_ntt_leveled(encrypted, plain);
        }
        else
        {
            throw logic_error("result ciphertext is transparent");
        }
    }

    // miran
    // encrypted_ntt.level <= plain_ntt.level <= L
    void Evaluator::multiply_plain_ntt_leveled(Ciphertext &encrypted_ntt,
                                               const Plaintext &plain_ntt)
    {
        // Verify parameters.
        if (!plain_ntt.is_ntt_form())
        {
            throw invalid_argument("plain_ntt is not in NTT form");
        }
        
        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted_ntt.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_ntt_size = encrypted_ntt.size();
        
        // Size check
        if (!product_fits_in(encrypted_ntt_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
        
        double new_scale = encrypted_ntt.scale() * plain_ntt.scale();
        
        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
                               context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }
        
        for (size_t i = 0; i < encrypted_ntt_size; i++)
        {
            for (size_t j = 0; j < coeff_mod_count; j++)
            {
                dyadic_product_coeffmod(
                                        encrypted_ntt.data(i) + (j * coeff_count),
                                        plain_ntt.data() + (j * coeff_count),
                                        coeff_count, coeff_modulus[j],
                                        encrypted_ntt.data(i) + (j * coeff_count));
            }
        }
        
        // Set the scale
        encrypted_ntt.scale() = new_scale;
    }

    // miran
    // multiply by a plaintext polynomial encoded at a specific level
    void Evaluator::multiply_plain_leveled_inplace_fast(Ciphertext &encrypted,
                                                   const Plaintext &plain, MemoryPoolHandle pool)
    {
        if (!is_metadata_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!context_->get_context_data(encrypted.parms_id()))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form() != plain.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        
        if (encrypted.is_ntt_form())
        {
            multiply_plain_ntt_leveled_fast(encrypted, plain);
        }
        else
        {
            throw logic_error("result ciphertext is transparent");
        }
    }

    // miran
    // encrypted_ntt.level <= plain_ntt.level <= L
    void Evaluator::multiply_plain_ntt_leveled_fast(Ciphertext &encrypted_ntt,
                                               const Plaintext &plain_ntt)
    {
        // Verify parameters.
        if (!plain_ntt.is_ntt_form())
        {
            throw invalid_argument("plain_ntt is not in NTT form");
        }
        
        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted_ntt.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_ntt_size = encrypted_ntt.size();   // length of ciphertext
        
        // Size check
        if (!product_fits_in(encrypted_ntt_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
        
        double new_scale = encrypted_ntt.scale() * plain_ntt.scale();
        
        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
                               context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }
        
        for (size_t i = 0; i < encrypted_ntt_size; i++)
        {
            // q0 is usually larger than 31-bit
            dyadic_product_coeffmod(encrypted_ntt.data(i) + (0 * coeff_count),
                                    plain_ntt.data() + (0 * coeff_count),
                                    coeff_count, coeff_modulus[0],
                                    encrypted_ntt.data(i) + (0 * coeff_count));
            
            // qi's (i>0) is approximately 31 bits
            for (size_t j = 1; j < coeff_mod_count; j++)
            {
                dyadic_product_coeffmod_fast(
                                             encrypted_ntt.data(i) + (j * coeff_count),
                                             plain_ntt.data() + (j * coeff_count),
                                             coeff_count, coeff_modulus[j],
                                             encrypted_ntt.data(i) + (j * coeff_count));
            }
        }
        
        // Set the scale
        encrypted_ntt.scale() = new_scale;
    }
    
    // miran (from ngraph-HE2) 
    void multiply_poly_scalar_coeffmod64(const uint64_t* poly, size_t coeff_count,
                                         uint64_t scalar,
                                         const seal::SmallModulus& modulus, std::uint64_t* result) {
      const uint64_t modulus_value = modulus.value();
      const uint64_t const_ratio_1 = modulus.const_ratio()[1];

      // NOLINTNEXTLINE
      for (; coeff_count--; poly++, result++) {
        // Multiplication
        auto z = *poly * scalar;

        // Barrett base 2^64 reduction
        unsigned long long carry;
        // Carry will store the result modulo 2^64
        seal::util::multiply_uint64_hw64(z, const_ratio_1, &carry);
        // Barrett subtraction
        carry = z - carry * modulus_value;
        // Possible correction term
        *result = carry -
                    (modulus_value & static_cast<uint64_t>(-static_cast<int64_t>(carry >= modulus_value)));
      }
    }

    // miran 
    void Evaluator::multiply_const_inplace(Ciphertext &encrypted,
                                        const vector<uint64_t> &plain, MemoryPoolHandle pool)
    {
        // Extract encryption parameters.
        auto &context_data = *context_->get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();
        
        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
        
        double scale = encrypted.scale();
        double new_scale = scale * scale;
        
        // Check that scale is positive and not too large
        if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >= context_data.total_coeff_modulus_bit_count()))
        {
          std::cout << "new_scale " << new_scale << " (" << static_cast<int>(log2(new_scale)) << " bits) out of bounds";
            std::cout << "Coeff mod bit count " << context_data.total_coeff_modulus_bit_count();
        }
        for (size_t i = 0; i < encrypted_size; i++) {
          for (size_t j = 0; j < coeff_mod_count; j++) {
            // Multiply by scalar instead of doing dyadic product
            if (coeff_modulus[j].value() < (1UL << 31U)) {
              multiply_poly_scalar_coeffmod64(encrypted.data(i) + (j * coeff_count),
                                              coeff_count, plain[j],
                                              coeff_modulus[j],
                                              encrypted.data(i) + (j * coeff_count));
            } else {
              seal::util::multiply_poly_scalar_coeffmod(
                  encrypted.data(i) + (j * coeff_count), coeff_count,
                  plain[j], coeff_modulus[j],
                  encrypted.data(i) + (j * coeff_count));
            }
          }
        }
        // Set the scale
        encrypted.scale() = new_scale;
    }

    //-----------------------------
    // Rotations of a ciphertext
    // we apply the hoisting technique of HS18 for efficiency
    //-----------------------------
    // miran
    /*
     rotations of a ciphertext
    */
    void Evaluator::rotate_internal_many(vector<Ciphertext> &encrypted, vector<int> steps,
                                         const GaloisKeys &galois_keys, MemoryPoolHandle pool)
    {
        auto context_data_ptr = context_->get_context_data(encrypted[0].parms_id());
        
#ifdef SEAL_DEBUG
        cout << "debug" << endl;
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_data_ptr->qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }
        if (galois_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }
#endif
        size_t coeff_count = context_data_ptr->parms().poly_modulus_degree();
        
        // Generate galois elements
        vector<uint64_t> galois_elt (steps.size(), 0);
        for(size_t i = 0; i < steps.size(); i++)
        {
            // Is there anything to do?
            // Or compute the galois element
            if (steps[i] == 0)
            {
                throw logic_error("trasparent rotation");
                return;
            }
            else{
                galois_elt[i] = galois_elt_from_step(steps[i], coeff_count); // steps_to_galois_elt (ver 3.3) -> galois_elt_from_step (ver 3.4)
            }
            encrypted[i + 1] = encrypted[0];
        }
        
        // Perform rotation and key switching
        apply_galois_many_fast(encrypted,
                               galois_elt,
                               galois_keys, move(pool));
    }
    
    // miran
    /*
     This function precomputes some numbers for applying galois_ntt
     */
    void Evaluator::apply_galois_many_fast(vector<Ciphertext> &encrypted, vector<uint64_t> galois_elt,
                                           const GaloisKeys &galois_keys, MemoryPoolHandle pool)
    {
        size_t steps_size = galois_elt.size();
        
        auto &context_data = *context_->get_context_data(encrypted[0].parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_mod_count = coeff_modulus.size();
        size_t encrypted_size = encrypted[0].size();
        
        //KSwitchKeys kswitch_keys = static_cast<const KSwitchKeys>(galois_keys); // 0.5 sec
        
        auto parms_id = encrypted[0].parms_id();
        auto &key_context_data = *context_->key_context_data();
        auto &key_parms = key_context_data.parms();
        auto scheme = parms.scheme();
        
        int n_power_of_two = get_power_of_two(static_cast<uint64_t>(coeff_count));
        size_t decomp_mod_count = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_mod_count = key_modulus.size();
        size_t rns_mod_count = decomp_mod_count + 1;
        auto &small_ntt_tables = key_context_data.small_ntt_tables();
        auto &modswitch_factors = key_context_data.base_converter()-> get_inv_last_coeff_mod_array();
        
        // Prepare input
        uint64_t *target = encrypted[0].data(1);
        
        vector<size_t> kswitch_keys_index (steps_size, 0);
        for(size_t r = 0; r < steps_size; r++)
        {
            kswitch_keys_index[r] = GaloisKeys::get_index(galois_elt[r]);
        }
        
        for(size_t r = 0; r < steps_size; ++r)
        {
            set_zero_poly(coeff_count, coeff_mod_count, encrypted[r + 1].data(1));
        }
        
#ifdef SEAL_DEBUG
        for(size_t r = 0; r < steps_size; r++)
        {
            if (kswitch_keys_index[r] >= galois_keys.data().size())
            {
                throw out_of_range("kswitch_keys_index");
            }
        }
        if(steps_size != (encrypted.size() - 1))
        {
            throw invalid_argument("encrypted is not initialized correctly");
        }
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted[0], context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        
        // Don't validate all of galois_keys but just check the parms_id.
        if (galois_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }
        auto scheme = parms.scheme();
        // Size check
        if (!product_fits_in(coeff_count, coeff_mod_count))
        {
            throw logic_error("invalid parameters");
        }
        uint64_t m = mul_safe(static_cast<uint64_t>(coeff_count), uint64_t(2));
        for(size_t r = 0; r < steps_size; r++)
        {
            if (!(galois_elt[r] & 1) || unsigned_geq(galois_elt[r], m))
            {
                throw invalid_argument("Galois element is not valid");
            }
            
            // Check if Galois key is generated or not.
            // If not, attempt a bit decomposition; maybe we have log(n) many keys
            if (!galois_keys.has_key(galois_elt[r]))
            {
                throw logic_error("Galois key is not supported by the context");
            }
        }
        if (encrypted_size > 2)
        {
            throw invalid_argument("encrypted size must be 2");
        }
        
        // Verify parameters.
        if (!target)
        {
            throw invalid_argument("target");
        }
        if (!context_->using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }
        
        // Don't validate all of kswitch_keys but just check the parms_id.
        if (galois_keys.parms_id() != context_->key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }
        
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::BFV && encrypted[0].is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::CKKS && !encrypted[0].is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        
        // Size check
        if (!product_fits_in(coeff_count, rns_mod_count, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
#endif
        
        // Temporary results (allocating : 141 ms)
        auto temp(allocate_poly(coeff_count, coeff_mod_count, pool));
        Pointer<uint64_t> temp_poly[2 * steps_size];
        
        for(size_t i = 0; i < 2 * steps_size; ++i)
        {
            temp_poly[i] = allocate_zero_poly(2 * coeff_count, rns_mod_count, pool);
        }
        
        // compute index for gal in NTT
        uint64_t m_minus_one = 2 * coeff_count - 1;
        
        uint64_t **gal_ntt_index = new uint64_t*[steps_size];
        for(size_t r = 0; r < steps_size; r++)
        {
            gal_ntt_index[r] = new uint64_t[coeff_count];
        }
        
        for (size_t l = 0; l < coeff_count; l++)
        {
            uint64_t reversed = reverse_bits(l, n_power_of_two);
            for(size_t r = 0; r < steps_size; r++)
            {
                uint64_t index_raw = galois_elt[r] * (2 * reversed + 1);
                index_raw &= m_minus_one;
                gal_ntt_index[r][l] = reverse_bits((index_raw - 1) >> 1, n_power_of_two);
            }
        }
        
        for (size_t i = 0; i < decomp_mod_count; i++)
        {
            // RNS decomposition of ct[0][1]
            auto local_small_poly_0(allocate_uint(coeff_count, pool));
            auto local_small_poly_1(allocate_uint(coeff_count, pool));
            
            set_uint_uint(
                          target + i * coeff_count,
                          coeff_count,
                          local_small_poly_0.get());
            
            if (scheme == scheme_type::CKKS)
            {
                inverse_ntt_negacyclic_harvey(
                                              local_small_poly_0.get(),
                                              small_ntt_tables[i]);
            }
            
            // Key RNS representation
            for (size_t j = 0; j < rns_mod_count; j++)
            {
                const uint64_t *local_encrypted_ptr = nullptr;
                size_t index = (j == decomp_mod_count ? key_mod_count - 1 : j);
                // Reduce modulus only if needed
                if (key_modulus[i].value() <= key_modulus[index].value())
                {
                    set_uint_uint(
                                  local_small_poly_0.get(),
                                  coeff_count,
                                  local_small_poly_1.get());
                }
                else
                {
                    modulo_poly_coeffs_63(
                                          local_small_poly_0.get(),
                                          coeff_count,
                                          key_modulus[index],
                                          local_small_poly_1.get());
                }
                
                // Lazy reduction, output in [0, 4q).
                ntt_negacyclic_harvey_lazy(
                                           local_small_poly_1.get(),
                                           small_ntt_tables[index]);
                local_encrypted_ptr = local_small_poly_1.get();
                
                // Apply Galois to ct[r][1] in NTT form
                for(size_t r = 0; r < steps_size; ++r)
                {
                    const uint64_t *local_encrypted_ptr_step = nullptr;
                    
                    util::apply_galois_ntt_fast(local_encrypted_ptr,
                                                gal_ntt_index[r], coeff_count,
                                                temp.get());
                    
                    local_encrypted_ptr_step = temp.get();
                    
                    // Two components in key
                    // Calculate (temp * galois_key[0], temp * galois_key[1]) + (ct[0], 0)
                    // For each RNS decomposition, multiply with key data and sum up.
                    auto &key_vector = galois_keys.data()[kswitch_keys_index[r]];
                    
                    for (size_t k = 0; k < 2; k++)
                    {
                        const uint64_t *key_ptr = key_vector[i].data().data(k);
                        for (size_t l = 0; l < coeff_count; l++)
                        {
                            unsigned long long local_wide_product[2];
                            unsigned long long local_low_word;
                            unsigned char local_carry;
                            
                            multiply_uint64(
                                            local_encrypted_ptr_step[l],
                                            key_ptr[(index * coeff_count) + l],
                                            local_wide_product);
                            local_carry = add_uint64(
                                                     temp_poly[k + 2 * r].get()[(j * coeff_count + l) * 2],
                                                     local_wide_product[0],
                                                     &local_low_word);
                            temp_poly[k + 2 * r].get()[(j * coeff_count + l) * 2] = local_low_word;
                            temp_poly[k + 2 * r].get()[(j * coeff_count + l) * 2 + 1] += local_wide_product[1] + local_carry;
                        }
                    }
                }
            }
        }
        
        
        // Apply Galois to ct[r][0] in NTT representation
        for(size_t r = 0; r < steps_size; r++)
        {
            for (size_t i = 0; i < coeff_mod_count; i++)
            {
                if (scheme == scheme_type::CKKS)
                {
                    //                    util::apply_galois_ntt(
                    //                                           encrypted[0].data(0) + i * coeff_count,
                    //                                           n_power_of_two,
                    //                                           galois_elt[r],
                    //                                           temp.get() + i * coeff_count);
                    
                    util::apply_galois_ntt_fast(encrypted[0].data(0) + i * coeff_count,
                                                gal_ntt_index[r], coeff_count,
                                                temp.get() + i * coeff_count);
                }
                else if (scheme == scheme_type::BFV)
                {
                    util::apply_galois(
                                       encrypted[0].data(0) + i * coeff_count,
                                       n_power_of_two,
                                       galois_elt[r],
                                       coeff_modulus[i],
                                       temp.get() + i * coeff_count);
                }
                
            }
            // copy result to encrypted.data(0)
            set_poly_poly(temp.get(), coeff_count, coeff_mod_count, encrypted[r + 1].data(0));
        }
        
        // Results are now stored in temp_poly[k]
        // Modulus switching should be performed
        auto local_small_poly(allocate_uint(coeff_count, pool));
        for(size_t r = 0; r < steps_size; r++)
        {
            for (size_t k = 0; k < 2; k++)
            {
                // Reduce (ct mod 4qk) mod qk
                uint64_t *temp_poly_ptr = temp_poly[k + 2 * r].get() + decomp_mod_count * coeff_count * 2;
                for (size_t l = 0; l < coeff_count; l++)
                {
                    temp_poly_ptr[l] = barrett_reduce_128(
                                                          temp_poly_ptr + l * 2,
                                                          key_modulus[key_mod_count - 1]);
                }
                // Lazy reduction, they are then reduced mod qi
                uint64_t *temp_last_poly_ptr = temp_poly[k + 2 * r].get() + decomp_mod_count * coeff_count * 2;
                inverse_ntt_negacyclic_harvey_lazy(
                                                   temp_last_poly_ptr,
                                                   small_ntt_tables[key_mod_count - 1]);
                
                // Add (p-1)/2 to change from flooring to rounding.
                uint64_t half = key_modulus[key_mod_count - 1].value() >> 1;
                for (size_t l = 0; l < coeff_count; l++)
                {
                    temp_last_poly_ptr[l] = barrett_reduce_63(temp_last_poly_ptr[l] + half,
                                                              key_modulus[key_mod_count - 1]);
                }
                
                // store the results
                uint64_t *encrypted_ptr = encrypted[r + 1].data(k);
                for (size_t j = 0; j < decomp_mod_count; j++)
                {
                    temp_poly_ptr = temp_poly[k + 2 * r].get() + j * coeff_count * 2;
                    // (ct mod 4qi) mod qi
                    for (size_t l = 0; l < coeff_count; l++)
                    {
                        temp_poly_ptr[l] = barrett_reduce_128(
                                                              temp_poly_ptr + l * 2,
                                                              key_modulus[j]);
                    }
                    // (ct mod 4qk) mod qi
                    modulo_poly_coeffs_63(
                                          temp_last_poly_ptr,
                                          coeff_count,
                                          key_modulus[j],
                                          local_small_poly.get());
                    
                    uint64_t half_mod = barrett_reduce_63(half, key_modulus[j]);
                    for (size_t l = 0; l < coeff_count; l++)
                    {
                        local_small_poly.get()[l] = sub_uint_uint_mod(local_small_poly.get()[l],
                                                                      half_mod,
                                                                      key_modulus[j]);
                    }
                    
                    if (scheme == scheme_type::CKKS)
                    {
                        ntt_negacyclic_harvey(local_small_poly.get(),
                                              small_ntt_tables[j]);
                    }
                    else if (scheme == scheme_type::BFV)
                    {
                        inverse_ntt_negacyclic_harvey(temp_poly_ptr,
                                                      small_ntt_tables[j]);
                    }
                    
                    // ((ct mod qi) - (ct mod qk)) mod qi
                    sub_poly_poly_coeffmod(
                                           temp_poly_ptr,
                                           local_small_poly.get(),
                                           coeff_count,
                                           key_modulus[j],
                                           temp_poly_ptr);
                    // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                    multiply_poly_scalar_coeffmod(
                                                  temp_poly_ptr,
                                                  coeff_count,
                                                  modswitch_factors[j],
                                                  key_modulus[j],
                                                  temp_poly_ptr);
                    // add with (ct[0], 0)
                    add_poly_poly_coeffmod(
                                           temp_poly_ptr,
                                           encrypted_ptr + j * coeff_count,
                                           coeff_count,
                                           key_modulus[j],
                                           encrypted_ptr + j * coeff_count);
                }
            }
        }
    }
}
