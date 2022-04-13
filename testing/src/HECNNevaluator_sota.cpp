/*
 * We modified the open-source project of nGraph-HE2: https://github.com/IntelAI/he-transformer/src/seal/seal_util.cpp
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "math.h"
#include "time.h"
#include <chrono>
#include <sys/resource.h>

#include "utils.h"
#include "thread.h"
#include "param.h"
#include "HECNNevaluator_sota.h"
#include <omp.h>

#define DEBUG true
using namespace std;
using namespace seal::util;

namespace seal
{
    HEnGraphCNN::HEnGraphCNN(shared_ptr<SEALContext> context):
        context_(move(context))
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

        auto &context_data = *context_->first_context_data();
        if (context_data.parms().scheme() != scheme_type::CKKS)
        {
            throw invalid_argument("unsupported scheme");
        }
        if (!context_data.qualifiers().using_batching)
        {
            throw invalid_argument("encryption parameters are not valid for batching");
        }
    }

    // It is taken from seal/seal_util.cpp: encode function (Line 311)
    void HEnGraphCNN::encode(vector<uint64_t>& destination, double input, double scale, int coeff_mod_count, MemoryPoolHandle pool)
    {
        auto &context_data = *context_->first_context_data();
        auto& parms = context_data.parms();
        auto& coeff_modulus = parms.coeff_modulus();
        double two_pow_64 = pow(2.0, 64);
        
        // Check that scale is positive and not too large
        if (scale <= 0 || (static_cast<int>(log2(scale)) >=
                           context_data.total_coeff_modulus_bit_count())){
          cout << "scale " << scale;
          cout << "context_data.total_coeff_modulus_bit_count "
                     << context_data.total_coeff_modulus_bit_count();
          throw invalid_argument("scale out of bounds");
        }
        
        double value = input * scale; // Compute the scaled value
    
        int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
        double coeffd = std::round(value);
        bool is_negative = std::signbit(coeffd);
        coeffd = fabs(coeffd);
        destination.resize (coeff_mod_count);
        
        // mod operation
        if (coeff_bit_count <= 64) {
            auto coeffu = static_cast<uint64_t>(fabs(coeffd));
            if (is_negative) {
#pragma omp parallel for
                 for (int l = 0; l < coeff_mod_count; l++){
                    destination[l] = seal::util::negate_uint_mod(coeffu % coeff_modulus[l].value(), coeff_modulus[l]);
                }
            } else {
                for (int l = 0; l < coeff_mod_count; l++){
                    destination[l] = coeffu % coeff_modulus[l].value();
                }
            }
        } else if (coeff_bit_count <= 128){
            uint64_t coeffu[2]{static_cast<uint64_t>(fmod(coeffd, two_pow_64)), static_cast<uint64_t>(coeffd / two_pow_64)};
            if (is_negative) {
                for (int l = 0; l < coeff_mod_count; l++){
                    destination[l] = seal::util::negate_uint_mod(seal::util::barrett_reduce_128(coeffu, coeff_modulus[l]), coeff_modulus[l]);
                }
            } else {
                for (int l = 0; l < coeff_mod_count; l++){
                    destination[l] = seal::util::barrett_reduce_128(coeffu, coeff_modulus[l]);
                }
            }
        } else {
            throw invalid_argument("Error: coeff_bit_cnt > 128");
        }
        
        // Slow case
        auto coeffu(seal::util::allocate_uint(coeff_mod_count, pool));
        auto decomp_coeffu(seal::util::allocate_uint(coeff_mod_count, pool));

        // We are at this point guaranteed to fit in the allocated space
        seal::util::set_zero_uint(coeff_mod_count, coeffu.get());
        auto coeffu_ptr = coeffu.get();
        while (coeffd >= 1){
            *coeffu_ptr++ = static_cast<uint64_t>(fmod(coeffd, two_pow_64));
            coeffd /= two_pow_64;
        }

        // Next decompose this coefficient (from evaluator.h)
        // decompose_single_coeff(context_data, coeffu.get(), decomp_coeffu.get(), pool);
        if (coeff_mod_count == 1){
            seal::util::set_uint_uint(coeffu.get(), coeff_mod_count, decomp_coeffu.get());
        }
        auto value_copy(util::allocate_uint(coeff_mod_count, pool));
        for (std::size_t l = 0; l < coeff_mod_count; l++){
            seal::util::set_uint_uint(coeffu.get(), coeff_mod_count, value_copy.get());

            // Starting from the top, reduce always 128-bit blocks
            for (std::size_t k = coeff_mod_count - 1; k--;){
                value_copy[k] = seal::util::barrett_reduce_128(value_copy.get() + k, coeff_modulus[l]);
            }
            decomp_coeffu.get()[l] = value_copy[0];
        }
        
        // Finally replace the sign if necessary
        if (is_negative) {
            for (int l = 0; l < coeff_mod_count; l++){
                destination[l] = seal::util::negate_uint_mod(decomp_coeffu[l], coeff_modulus[l]);
            }
        } else {
            for (int l = 0; l < coeff_mod_count; l++){
                destination[l] = decomp_coeffu[l];
            }
        }
    }

    /*
        @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_size])
        @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
        @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
        @param[in] dense_bias, the bias of the dense layer, [10]
        @param[in] NB_CHANNELS, the number of channels
        @param[out] ker_plain, the encoded polynomials of convolution kernels
        @param[out] act_plain, the encoded polynomials of "real_poly"
        @param[out] ker2, the encoded polynomials of kernels at B2
        @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
        @param[out] bias_poly, the encoded polynomials of the dense bias
     */
    void HEnGraphCNN::prepare_network1d(vector<vector<vector<vector<vector<uint64_t>>>>> &ker_plain,
                                    vector<vector<vector<vector<uint64_t>>>> &act_plain,
                                    vector<vector<vector<uint64_t>>> &dense_ker_plain, vector<vector<uint64_t>> &dense_bias_plain,
                                    vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, vector<int> NB_CHANNELS)
    {
        ker_plain.resize(3);
        for(int i = 0; i < 3; i++){
            int batch_size = NB_CHANNELS[i + 1];
            int nch = NB_CHANNELS[i];
            int height = 3;
            int lvl = Param_HEAR::ker_poly_lvl[i];
            int coeff_mod_count = lvl + 1;
            
            ker_plain[i].resize(batch_size,
                        vector<vector<vector<uint64_t>>> (nch, vector<vector<uint64_t>>(height, vector<uint64_t> (coeff_mod_count))));

            for(int ch = 0; ch < nch; ch++){
                for(int ht = 0; ht < height; ht++){
#pragma omp parallel for
                    for(int n = 0; n < batch_size; n++){
                        vector<uint64_t> destination;
                        encode(destination, ker[i][n][ch][ht], Param_HEAR::qcscale, coeff_mod_count);
                        ker_plain[i][n][ch][ht] = destination;
                    }
                }
            }
        }
        
        for(int i = 0; i < 3; i++){
            int const_lvl = Param_HEAR::ker_poly_lvl[i] - 3;    // level = Param_conv2d::ker_poly_lvl[0]
            int nonconst_lvl = Param_HEAR::ker_poly_lvl[i] - 2;
            int batch_size = NB_CHANNELS[i + 1];
            vector<vector<vector<uint64_t>>> temp (batch_size); // [batch_size][3][lvl+1]
            
#pragma omp parallel for
            for(int j = 0; j < batch_size; j++){
                vector<vector<uint64_t>> tempj; // [3][lvl+1]
                
                vector<uint64_t> destination;
                encode(destination, real_poly[i][j][0], Param_HEAR::qscale, const_lvl + 1);     // act_plain[i][j][0]
                tempj.push_back(destination);
                
                destination.clear();
                encode(destination, real_poly[i][j][1], Param_HEAR::qcscale, nonconst_lvl + 1); // act_plain[i][j][1]
                tempj.push_back(destination);
                
                destination.clear();
                encode(destination, real_poly[i][j][2], Param_HEAR::qcscale, nonconst_lvl + 1); // act_plain[i][j][1]
                tempj.push_back(destination);
            
                temp[j] = tempj;
            }
            act_plain.push_back(temp);
        }
         
        // dense_ker_plain[10][4*nch][lvl+1] = [10][NB[3]][lvl+1]
        int nin = dense_ker[0].size();
        int nout = dense_ker.size();
        
        for(int i = 0; i < nout; ++i){
            vector<vector<uint64_t>> tempj (nin);
            
#pragma omp parallel for
            for(int j = 0; j < nin; ++j){
                vector<uint64_t> destination;
                double value = dense_ker[i][j]/10.0;
                encode(destination, value, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[3] + 1);
                tempj[j] = destination;
            }
            dense_ker_plain.push_back(tempj);
            
            vector<uint64_t> destination;
            double value = dense_bias[i]/10.0;
            encode(destination, value, Param_HEAR::qscale, 1);
            dense_bias_plain.push_back(destination);
        }
    }

    /*
        @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_height][filter_width])
        @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
        @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
        @param[in] dense_bias, the bias of the dense layer, [10]
        @param[in] NB_CHANNELS, the number of channels
        @param[out] ker_plain, the encoded polynomials of convolution kernels
        @param[out] act_plain, the encoded polynomials of "real_poly"
        @param[out] ker2, the encoded polynomials of kernels at B2
        @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
        @param[out] bias_poly, the encoded polynomials of the dense bias
     */
    void HEnGraphCNN::prepare_network2d(vector<vector<vector<vector<vector<vector<uint64_t>>>>>> &ker_plain,
                                    vector<vector<vector<vector<uint64_t>>>> &act_plain,
                                    vector<vector<vector<uint64_t>>> &dense_ker_plain, vector<vector<uint64_t>> &dense_bias_plain,
                                    vector<ften> ker, vector<dmat> real_poly,
                                    dmat dense_ker, dvec dense_bias, vector<int> NB_CHANNELS)
    {
        ker_plain.resize(3);

        for(int i = 0; i < 3; i++){
            int batch_size = NB_CHANNELS[i + 1]; // nout
            int nch = NB_CHANNELS[i]; // nin
            int height = 3;
            int width = 3;
            int lvl = Param_HEAR::ker_poly_lvl[i];
            int coeff_mod_count = lvl + 1;
            
            ker_plain[i].resize(batch_size,
                        vector<vector<vector<vector<uint64_t>>>> (nch,
                                                                  vector<vector<vector<uint64_t>>>(height, vector<vector<uint64_t>>(width, vector<uint64_t> (coeff_mod_count)))));

            for(int ch = 0; ch < nch; ch++){
                for(int ht = 0; ht < height; ht++){
                    for(int wt = 0; wt < width; wt++){
#pragma omp parallel for
                        for(int n = 0; n < batch_size; n++){
                            vector<uint64_t> destination;
                            encode(destination, ker[i][n][ch][ht][wt], Param_HEAR::qcscale, coeff_mod_count);
                            ker_plain[i][n][ch][ht][wt] = destination;
                        }
                    }
                }
            }
        }
        
        for(int i = 0; i < 3; i++){
            int const_lvl = Param_HEAR::ker_poly_lvl[i] - 3;    // level = Param_conv2d::ker_poly_lvl[0]
            int nonconst_lvl = Param_HEAR::ker_poly_lvl[i] - 2;
            int batch_size = NB_CHANNELS[i + 1];
            vector<vector<vector<uint64_t>>> temp (batch_size); // [batch_size][3][lvl+1]
            
#pragma omp parallel for
            for(int j = 0; j < batch_size; j++){
                vector<vector<uint64_t>> tempj; // [3][lvl+1]
                
                vector<uint64_t> destination;
                encode(destination, real_poly[i][j][0], Param_HEAR::qscale, const_lvl + 1);     // act_plain[i][j][0]
                tempj.push_back(destination);
                
                destination.clear();
                encode(destination, real_poly[i][j][1], Param_HEAR::qcscale, nonconst_lvl + 1); // act_plain[i][j][1]
                tempj.push_back(destination);
                
                destination.clear();
                encode(destination, real_poly[i][j][2], Param_HEAR::qcscale, nonconst_lvl + 1); // act_plain[i][j][1]
                tempj.push_back(destination);
            
                temp[j] = tempj;
            }
            act_plain.push_back(temp);
        }
         
        // dense_ker_plain[10][4*nch][lvl+1] = [10][NB[3]][lvl+1]
        int nin = dense_ker[0].size();
        int nout = dense_ker.size();
        
        for(int i = 0; i < nout; ++i){
            vector<vector<uint64_t>> tempj (nin);
            
#pragma omp parallel for
            for(int j = 0; j < nin; ++j){
                vector<uint64_t> destination;
                double value = dense_ker[i][j]/10.0;
                encode(destination, value, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[3] + 1);
                tempj[j] = destination;
            }
            
            dense_ker_plain.push_back(tempj);
            
            vector<uint64_t> destination;
            double value = dense_bias[i]/10.0;
            encode(destination, value, Param_HEAR::qscale, 1);
            dense_bias_plain.push_back(destination);
        }
    }
}
   
/*
 @param[in] ct, input ciphertexts ([nch][ht] where nch is the input channel size)
 @param[in] plain, [bsize][nch][3][lvl+1] where bsize is the output channel size (=nch=NB_CHANNELS[1])
 @param[out] res, the ciphertext ciphertexts (size [bsize][ht])
 This is for the 1d homomorphic convolution at the 1st layer.
*/
void HEnGraphEval::Eval_Conv1d(vector<vector<Ciphertext>> &res,
                      vector<vector<Ciphertext>> ct, vector<vector<vector<vector<uint64_t>>>> plain)
{
    int batch_size = plain.size();
    int nch = ct.size();
    int ht = ct[0].size();

    res.resize(batch_size, vector<Ciphertext> (ht));
    
#pragma omp parallel for
    for(int bsize = 0; bsize < batch_size; bsize++){
        for(int i = 0; i < ht; i++){
            // 1*3 convolution
            // Step 1.1. 0-th ch: the middle point
            res[bsize][i] = ct[0][i];
            evaluator.multiply_const_inplace(res[bsize][i], plain[bsize][0][1]);
            
            // Step 1.2. 0-th ch: outside the middle point
            for(int k = - 1; k <= 1; k++){
                int i1 = i + k;
                if ((k != 0) && (i1 >= 0) && (i1 < ht)){
                    Ciphertext temp = ct[0][i1];
                    evaluator.multiply_const_inplace(temp, plain[bsize][0][k+1]);
                    evaluator.add_inplace(res[bsize][i], temp);
                }
            }
            
            // Step 2. the remaining channels
            for(int ch = 1; ch < nch; ch++){
                for(int k = - 1; k <= 1; k++){
                    int i1 = i + k;
                    if ((i1 >= 0) && (i1 < ht)){
                        Ciphertext temp = ct[ch][i1];
                        evaluator.multiply_const_inplace(temp, plain[bsize][ch][k+1]);
                        evaluator.add_inplace(res[bsize][i], temp);
                    }
                }
            }
            evaluator.rescale_to_next_inplace(res[bsize][i]);
        }
    }
}

/*
 @param[in] ct, puctured input ciphertexts ([nch][2*ht] where nch is the input channel size)
 @param[in] plain, [bsize][nch][3][lvl+1] where bsize is the output channel size (NB_CHANNELS[2] or NB_CHANNELS[3])
 @param[out] res, the ciphertext ciphertexts ([bsize][ht])
 This is for the 1d homomorphic convolution at the 2nd or the 3rd layer.
*/
void HEnGraphEval::Eval_Puctured_Conv1d(vector<vector<Ciphertext>> &res,
                      vector<vector<Ciphertext>> ct, vector<vector<vector<vector<uint64_t>>>> plain)
{
    int batch_size = plain.size();
    int nch = ct.size();
    int ht = ct[0].size()/2;

    res.resize(batch_size, vector<Ciphertext> (ht));
    
#pragma omp parallel for
    for(int bsize = 0; bsize < batch_size; bsize++){
        for(int i = 0; i < ht; i++){
            // 1*3 convolution
            // Step 1.1. 0-th ch: the middle point
            res[bsize][i] = ct[0][2*i];
            evaluator.multiply_const_inplace(res[bsize][i], plain[bsize][0][1]);
            
            // Step 1.2. 0-th ch: outside the middle point
            for(int k = - 1; k <= 1; k++){
                int i1 = i + k;
                if ((k != 0) && (i1 >= 0) && (i1 < ht)){
                    Ciphertext temp = ct[0][2*i1];
                    evaluator.multiply_const_inplace(temp, plain[bsize][0][k+1]);
                    evaluator.add_inplace(res[bsize][i], temp);
                }
            }
            
            // Step 2.The remaining channels
            for(int ch = 1; ch < nch; ch++){
                for(int k = - 1; k <= 1; k++){
                    int i1 = i + k;
                    if ((i1 >= 0) && (i1 < ht)){
                        Ciphertext temp = ct[ch][2*i1];
                        evaluator.multiply_const_inplace(temp, plain[bsize][ch][k+1]);
                        evaluator.add_inplace(res[bsize][i], temp);
                    }
                }
            }
            evaluator.rescale_to_next_inplace(res[bsize][i]);
        }
    }
}

/*
 @param[in] ct, input ciphertexts ([nch][ht][wt] where nch is the input channel size)
 @param[in] plain, [bsize][nch][3][3][lvl+1] where bsize is the output channel size (NB_CHANNELS[1])
 @param[out] res, the ciphertext ciphertexts ([bsize][ht][wt])
 Complexity: bsize * ht * wt * (3*3*nch)
 This is for the 2d homomorphic convolution at the 1st layer.
*/
void HEnGraphEval::Eval_Conv2d(vector<vector<vector<Ciphertext>>> &res,
                      vector<vector<vector<Ciphertext>>> ct,
                      vector<vector<vector<vector<vector<uint64_t>>>>> plain)
{
    int batch_size = plain.size();  // = NB_CHANNELS[1]
    int nch = ct.size();            // number of input channels
    int ht = ct[0].size();
    int wt = ct[0][0].size();

    res.resize(batch_size, vector<vector<Ciphertext>> (ht, vector<Ciphertext>(wt)));
    
#pragma omp parallel for
    for(int bsize = 0; bsize < batch_size; bsize++){
        for(int i = 0; i < ht; i++){
            for(int j = 0; j < wt; j++){
                // 3*3 convolution
                // Step 1.1. 0-th ch: the middle point
                res[bsize][i][j] = ct[0][i][j];
                evaluator.multiply_const_inplace(res[bsize][i][j], plain[bsize][0][1][1]);
               
                // Step 1.2. 0-th ch: outside the middle point
                for(int k = - 1; k <= 1; k++){
                    for(int l = - 1; l <= 1; l++){
                        int i1 = i + k;
                        int j1 = j + l;
                        if (((k != 0) || (l != 0)) && (i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wt)){
                            Ciphertext temp = ct[0][i1][j1];
                            evaluator.multiply_const_inplace(temp, plain[bsize][0][k+1][l+1]);
                            evaluator.add_inplace(res[bsize][i][j], temp);
                        }
                    }
                }
                
                // Step 2. the remaining channels
                for(int ch = 1; ch < nch; ch++){
                    for(int k = - 1; k <= 1; k++){
                        for(int l = - 1; l <= 1; l++){
                            int i1 = i + k;
                            int j1 = j + l;
                            if ((i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wt)){
                                Ciphertext temp = ct[ch][i1][j1];
                                evaluator.multiply_const_inplace(temp, plain[bsize][ch][k+1][l+1]);
                                evaluator.add_inplace(res[bsize][i][j], temp);
                            }
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[bsize][i][j]);
            }
        }
    }
}

/*
 @param[in] ct, input ciphertexts ([nch][2*ht][2*wt] where nch is the input channel size)
 @param[in] plain, [bsize][nch][3][3][lvl+1] where bsize is the output channel size (NB_CHANNELS[2] or NB_CHANNELS[3])
 @param[out] res, the ciphertext ciphertexts ([bsize][ht][wt])
 Complexity: bsize * ht * wt * (3*3*nch)
 This is for the 2d homomorphic convolution at the 2nd or the 3rd layer.
*/
void HEnGraphEval::Eval_Puctured_Conv2d(vector<vector<vector<Ciphertext>>> &res,
                      vector<vector<vector<Ciphertext>>> ct,
                      vector<vector<vector<vector<vector<uint64_t>>>>> plain)
{
    int batch_size = plain.size(); // = NB_CHANNELS[2] or NB_CHANNELS[3]
    int nch = ct.size();
    int ht = ct[0].size()/2;
    int wt = ct[0][0].size()/2;

    res.resize(batch_size, vector<vector<Ciphertext>> (ht, vector<Ciphertext>(wt)));
    
#pragma omp parallel for
    for(int bsize = 0; bsize < batch_size; bsize++){
        for(int i = 0; i < ht; i++){
            for(int j = 0; j < wt; j++){
                // 3*3 convolution
                // Step 1.1. 0-th ch: the middle point
                res[bsize][i][j] = ct[0][2*i][2*j];
                evaluator.multiply_const_inplace(res[bsize][i][j], plain[bsize][0][1][1]);
               
                // Step 1.2. 0-th ch: outside the middle point
                for(int k = - 1; k <= 1; k++){
                    for(int l = - 1; l <= 1; l++){
                        int i1 = i + k;
                        int j1 = j + l;
                        if (((k != 0) || (l != 0)) && (i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wt)){
                            Ciphertext temp = ct[0][2*i1][2*j1];
                            evaluator.multiply_const_inplace(temp, plain[bsize][0][k+1][l+1]);
                            evaluator.add_inplace(res[bsize][i][j], temp);
                        }
                    }
                }
                
                // Step 2. the remaining channels
                for(int ch = 1; ch < nch; ch++){
                    for(int k = - 1; k <= 1; k++){
                        for(int l = - 1; l <= 1; l++){
                            int i1 = i + k;
                            int j1 = j + l;
                            if ((i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wt)){
                                Ciphertext temp = ct[ch][2*i1][2*j1];
                                evaluator.multiply_const_inplace(temp, plain[bsize][ch][k+1][l+1]);
                                evaluator.add_inplace(res[bsize][i][j], temp);
                            }
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[bsize][i][j]);
            }
        }
    }
}

/*
 @param[in] ct, input ciphertexts
 @param[in] plain, plaintext polynomials for activation
 if B1: res[128][32*15], act_plain[128][3]
 if B2: res[256][16*7], act_plain[256][3]
 if B3: res[512][8*3], act_plain[512][3]
 Complexity: ch * ht * (1HM + 2SM)
*/
void HEnGraphEval::Eval_BN_ActPoly1d(vector<vector<Ciphertext>> &ct, vector<vector<vector<uint64_t>>> plain)
{
    int ch = ct.size();
    int ht = ct[0].size();

#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){
            Ciphertext ct2;
            evaluator.square(ct[i][j], ct2);
            evaluator.relinearize_inplace(ct2, relin_keys);
            evaluator.rescale_to_next_inplace(ct2);   // rescale by q
            
            evaluator.mod_switch_to_inplace(ct[i][j], ct2.parms_id());  // first take the same level before mult
            evaluator.multiply_const_inplace(ct2, plain[i][2]);         // (q * m2) * (qc * poly)
            evaluator.multiply_const_inplace(ct[i][j], plain[i][1]);    // (q * m) * (qc * poly)
            
            ct2.scale() = ct[i][j].scale();
            evaluator.add_inplace(ct[i][j], ct2);
            evaluator.rescale_to_next_inplace(ct[i][j]);                // rescale by qc, (q * msg)

            evaluator.add_const_inplace(ct[i][j], plain[i][0]);
        }
    }
}

/*
 @param[in] ct, input ciphertexts
 @param[in] plain, plaintext polynomials for activation
 if B1: res[128][32][15], act_plain[128][3]
 if B2: res[256][16][7], act_plain[256][3]
 if B3: res[512][8][3], act_plain[512][3]
 Complexity: ch * ht * wt * (1HM + 2SM)
*/
void HEnGraphEval::Eval_BN_ActPoly2d(vector<vector<vector<Ciphertext>>> &ct, vector<vector<vector<uint64_t>>> plain)
{
    int ch = ct.size();
    int ht = ct[0].size();
    int wt = ct[0][0].size();
    
#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){
            for(int k = 0; k < wt; k++){
                Ciphertext ct2;
                evaluator.square(ct[i][j][k], ct2);
                evaluator.relinearize_inplace(ct2, relin_keys);
                evaluator.rescale_to_next_inplace(ct2);   // rescale by q
                
                evaluator.mod_switch_to_inplace(ct[i][j][k], ct2.parms_id());   // first take the same level before mult
                evaluator.multiply_const_inplace(ct2, plain[i][2]);             // (q * m2) * (qc * poly)
                evaluator.multiply_const_inplace(ct[i][j][k], plain[i][1]);     // (q * m2) * (qc * poly)
                
                ct2.scale() = ct[i][j][k].scale();
                evaluator.add_inplace(ct[i][j][k], ct2);
                evaluator.rescale_to_next_inplace(ct[i][j][k]);                 // rescale by qc, (q * msg)
 
                evaluator.add_const_inplace(ct[i][j][k], plain[i][0]);
            }
        }
    }
}

/*
 @param[in] ct, input ciphertexts
 Update the input ciphertexts by using the 1d-aggregation, so it yields puctured ciphertexts
*/
void HEnGraphEval::Eval_Average1d(vector<vector<Ciphertext>> &ct)
{
    int ch = ct.size();
    int ht = ct[0].size()/ 2;

#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){
            evaluator.add_inplace(ct[i][2*j], ct[i][2*j+1]);
        }
    }
}

/*
 @param[in] ct, input ciphertexts
 Update the input ciphertexts by using the 2d-aggregation, so it yields puctured ciphertexts
*/
void HEnGraphEval::Eval_Average2d(vector<vector<vector<Ciphertext>>> &ct)
{
    int ch = ct.size();
    int ht = ct[0].size()/ 2;
    int wt = ct[0][0].size()/ 2;

#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){
            for(int k = 0; k < wt; k++){
                evaluator.add_inplace(ct[i][2*j][2*k], ct[i][2*j][2*k+1]);
                evaluator.add_inplace(ct[i][2*j][2*k], ct[i][2*j+1][2*k]);
                evaluator.add_inplace(ct[i][2*j][2*k], ct[i][2*j+1][2*k+1]);
            }
        }
    }
}
 
/*
 @param[in] ct, input ciphertexts  (size [4*nch][120])
 Update the input ciphertexts by using the 1d-global aggregation
*/
void HEnGraphEval::Eval_Global_Average1d(vector<vector<Ciphertext>> &ct)
{
    int ch = ct.size();
    int ht = ct[0].size();
    
#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int j = 1; j < ht; j++){
            evaluator.add_inplace(ct[i][0], ct[i][j]);
        }
    }
}

/*
 @param[in] ct, input ciphertexts  (size [4*nch][8][3])
 Update the input ciphertexts by using the 2d-global aggregation
*/
void HEnGraphEval::Eval_Global_Average2d(vector<vector<vector<Ciphertext>>> &ct)
{
    int ch = ct.size();
    int ht = ct[0].size();
    int wt = ct[0][0].size();
    
#pragma omp parallel for
    for(int i = 0; i < ch; i++){
        for(int k = 1; k < wt; k++){
            evaluator.add_inplace(ct[i][0][0], ct[i][0][k]);
        }
        for(int j = 1; j < ht; j++){
            for(int k = 0; k < wt; k++){
                evaluator.add_inplace(ct[i][0][0], ct[i][j][k]);
            }
        }
    }
}


/*
 @param[in] ct, input ciphertexts  (size [4*nch][-])
 @param[in] dense_ker_plain, plaintext polynomials for the dense kernel ([10][4*nch][lvl+1])
 @param[in] dense_bias_plain, plaintext polynomials for the dense bias ([10][1])
 @param[out] res, output ciphertexts
*/
void HEnGraphEval::Eval_Dense1d(vector<vector<Ciphertext>> &res, vector<vector<Ciphertext>> ct,
                                vector<vector<vector<uint64_t>>> dense_ker_plain, vector<vector<uint64_t>> dense_bias_plain)
{
    int nclass = dense_bias_plain.size();   // 10
    int nch = ct.size();                    // 512
    
    res.clear();
    res.resize(nclass, vector<Ciphertext> (nch)); // [10][512][1]
    
    for(int i = 0; i < nclass; i++){
#pragma omp parallel for
        for(int j = 0; j < nch; j++){
            res[i][j] = ct[j][0];
            evaluator.multiply_const_inplace(res[i][j], dense_ker_plain[i][j]);
        }
        
        for(int j = 1; j < nch; j++){
            evaluator.add_inplace(res[i][0], res[i][j]);
        }
        evaluator.rescale_to_next_inplace(res[i][0]);
        evaluator.add_const_inplace(res[i][0], dense_bias_plain[i]);
    }
}
 
/*
 @param[in] ct, input ciphertexts  (size [4*nch][-][-])
 @param[in] dense_ker_plain, plaintext polynomials for the dense kernel ([10][4*nch][lvl+1])
 @param[in] dense_bias_plain, plaintext polynomials for the dense bias ([10][1])
 @param[out] res, output ciphertexts
*/
void HEnGraphEval::Eval_Dense2d(vector<vector<vector<Ciphertext>>> &res, vector<vector<vector<Ciphertext>>> ct,
                       vector<vector<vector<uint64_t>>> dense_ker_plain, vector<vector<uint64_t>> dense_bias_plain)
{
    int nclass = dense_bias_plain.size();   // 10
    int nch = ct.size();                    // 512
    
    res.clear();
    res.resize(nclass, vector<vector<Ciphertext>> (nch, vector<Ciphertext> (1))); //[10][512][1]
    
    for(int i = 0; i < nclass; i++){
#pragma omp parallel for
        for(int j = 0; j < nch; j++){
            res[i][j][0] = ct[j][0][0];
            evaluator.multiply_const_inplace(res[i][j][0], dense_ker_plain[i][j]);
        }
        
        for(int j = 1; j < nch; j++){
            evaluator.add_inplace(res[i][0][0], res[i][j][0]);
        }
        evaluator.rescale_to_next_inplace(res[i][0][0]);
        evaluator.add_const_inplace(res[i][0][0], dense_bias_plain[i]);
    }
}

