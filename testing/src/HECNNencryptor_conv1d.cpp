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
#include "HECNNencryptor_conv1d.h"

using namespace std;

/*
 @param[in] data, the input tensor [2][32][15] = c * h * w
 @param[in] scale, the scaling factor of data
 @param[out] res, the ciphertexts,
 res[i] encrypts a plaintext vector of data[i][*][*] (of length 32*15) for i=0,1
 */
void HECNNenc1d::encryptdata(vector<Ciphertext> &res, vector<vector<vector<double>>> data, double scale)
{
    res.resize(2);

    MT_EXEC_RANGE(2, first, last);
    for(int c = first; c < last; ++c){
        // encrypt data[c][32][15] as an one ciphertext
        vector<double> msg_one_block;   // msg_one_block = [32 * 16], first generate the message vector filled with zeros at the right
        for(int i = 0; i < 32; ++i){
            for(int j = 0; j < 15; ++j){
                msg_one_block.push_back(data[c][i][j]);
            }
        }
        for(int i = 0; i < 32; ++i){
            msg_one_block.push_back(0.0);
        }
        
        // msg_one_block: [32 * 16] -> msg_full: [32 * 16 * 16] = fully packed slots
        vector<double> msg_full;
        for(int l = 0; l < 16; ++l){
            msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
        }
        
        Plaintext plain;
        encoder.encode(msg_full, scale, plain);
        encryptor.encrypt(plain, res[c]);
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] data, the input tensor [2][32][15] = c * h * w
 @param[in] scale, the scaling factor of data
 @param[out] res, the ciphertext
 Note that res[0] = Enc(I0|I1|...|I0|I1) where I0=data[0][*][*], I1=data[1][*][*]
 */
void HECNNenc1d::encryptdata_packed(Ciphertext &res, vector<vector<vector<double>>> data, double scale)
{
    // First, generate the plaintext vector of I0=data[0][*][*]
    vector<double> msg_one_block; // msg_one_block = [32 * 16 * 2],
    for(int i = 0; i < 32; ++i){
        for(int j = 0; j < 15; ++j){
            msg_one_block.push_back(data[0][i][j]);
        }
    }
    for(int i = 0; i < 32; ++i){
        msg_one_block.push_back(0.0);
    }
    
    // Second, pad the subvector of I1=data[1][*][*]
    for(int i = 0; i < 32; ++i){
        for(int j = 0; j < 15; ++j){
            msg_one_block.push_back(data[1][i][j]);
        }
    }
    for(int i = 0; i < 32; ++i){
        msg_one_block.push_back(0.0);
    }

    // Third, generate the fully-packed plaintext vector (of size 32 * 16 * 16) while interlacing I0 and I1
    vector<double> msg_full;
    for(int l = 0; l < (16/2); ++l){
        msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
    }
    
    if(msg_full.size() != encoder.slot_count()){
        throw invalid_argument("Error: encypt size mismatch");
    }
    
    Plaintext plain;
    encoder.encode(msg_full, scale, plain);
    encryptor.encrypt(plain, res);
}

/*
@param[in] ct, a ciphertext
@param[out] dmsg, the double-type vector which is a decrypted result of ct
*/
void HECNNenc1d::decrypt_vector(vector<double> &dmsg, Ciphertext ct){
    Plaintext pmsg;
    decryptor.decrypt(ct, pmsg);
    encoder.decode(pmsg, dmsg);
}

/* -----------------------------------
 * Encoding weights of the first conv layer
 * -----------------------------------
 */
 
/*
 @param[in] kernel, the kernel of the first convoluational layer
                    [out_channels][in_channels][filter_size]; [64][2][3] when nch=64 and [128][2][3] when nch=128
 @param[in] NB_CHANNELS, the number of channels
 @param[in] scale, the scaling factor of the kernel
 @param[in] level, the encoding level of the kernel
 @param[out] ker_poly, the encoded polynomials of kernels as a native plaintext format (size = [NB_CHANNELS[1]/16][2][3])
 The ouput polynomials are scaled by the factor of "scale" at "level"
 */
void HECNNenc1d::encode_conv1(vector<vector<vector<Plaintext>>> &ker_poly, dten kernel, vector<int> NB_CHANNELS, double scale, int level)
{
    int ncols = NB_CHANNELS[1] / 16;
    int nrows = NB_CHANNELS[0];
    
    ker_poly.resize(ncols, vector<vector<Plaintext>>(nrows, vector<Plaintext> (3)));    // [4][2][3] or [8][2][3]
    size_t slot_count = encoder.slot_count();
    int num = ncols * Param_conv1d::DIM_FILTERS;
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int j = (int) floor((double) n / (double) Param_conv1d::DIM_FILTERS);  // 0 <= j < ncols
        int j_start = (j << 4);
        int option = (n % Param_conv1d::DIM_FILTERS);   // 0,1,2

        dmat temp_full_slots;       // [2][8192]
        temp_full_slots.resize(nrows);

        for(int i = 0; i < nrows; ++i){
            for(int k = 0; k < 16; ++k){
                dvec temp_short;    // size = (32 * 16)
                ker1_to_vector_conv1d(temp_short, kernel[j_start + k][i], option);
                temp_full_slots[i].insert(temp_full_slots[i].end(), temp_short.begin(), temp_short.end());
            }
            encoder.encode(temp_full_slots[i], scale, level, ker_poly[j][i][option]);
        }
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] kernel, the kernel of the first convoluational layer
                    [out_channels][in_channels][filter_size]; [64][2][3] when nch=64 and [128][2][3] when nch=128
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[in] NB_CHANNELS, the number of channels
 @param[in] scale, the scaling factor of the kernel
 @param[in] level, the encoding level of the kernel
 @param[out] ker_poly, the encoded polynomials of kernels as a native plaintext format (size = [NB_CHANNELS[1]/16][2][3])
 The ouput polynomials are scaled by the factor of "scale" at "level"
*/
void HECNNenc1d::encode_conv1(vector<vector<vector<Plaintext>>> &ker_poly, dten kernel, string mode,
                              vector<int> NB_CHANNELS, double scale, int level)
{
    int ncols = NB_CHANNELS[1] / 16;
    int nrows = NB_CHANNELS[0];
    
    ker_poly.resize(ncols, vector<vector<Plaintext>>(nrows, vector<Plaintext> (3)));
    size_t slot_count = encoder.slot_count();
    int num = ncols * Param_conv1d::DIM_FILTERS;
    
    if(mode == "fully"){
        MT_EXEC_RANGE(num, first, last);
        for(int n = first; n < last; ++n){
            int j = (int) floor((double) n / (double) Param_conv1d::DIM_FILTERS);  // 0 <= j < ncols
            int j_start = (j << 4);
            int option = (n % Param_conv1d::DIM_FILTERS); // 0 <= option < 3
            
            dmat temp_full_slots;   // [2][8192]
            temp_full_slots.resize(nrows);
            
            // For I0
            for(int k = 0; k < 16; ++k){
                dvec temp_short;    // size = (32 * 16)
                bool i1 = (k % 2);
                ker1_to_vector_conv1d(temp_short, kernel[j_start + k][i1], option);
                temp_full_slots[0].insert(temp_full_slots[0].end(), temp_short.begin(), temp_short.end());
            }
            encoder.encode(temp_full_slots[0], scale, level, ker_poly[j][0][option]);
            
            // For I1
            for(int k = 0; k < 16; ++k){
                dvec temp_short;    // size = (32 * 16)
                bool i1 = ((k + 1) % 2);
                ker1_to_vector_conv1d(temp_short, kernel[j_start + k][i1], option);
                temp_full_slots[1].insert(temp_full_slots[1].end(), temp_short.begin(), temp_short.end());
            }
            encoder.encode(temp_full_slots[1], scale, level, ker_poly[j][1][option]);
        }
        MT_EXEC_RANGE_END
    }
    else if(mode == "baby"){
        MT_EXEC_RANGE(num, first, last);
        for(int n = first; n < last; ++n){
            int j = (int) floor((double) n / (double) Param_conv1d::DIM_FILTERS);  // 0 <= j < ncols
            int j_start = (j << 4);
            int option = (n % Param_conv1d::DIM_FILTERS); // 0 <= option < 9
            
            dmat temp_full_slots;   // [2][8192]
            temp_full_slots.resize(nrows);
            
            // For I0
            for(int k = 0; k < 16; ++k){
                dvec temp_short;    // size = (32 * 16)
                bool i1 = (k % 2);
                ker1_to_vector_conv1d(temp_short, kernel[j_start + k][i1], option);
                temp_full_slots[0].insert(temp_full_slots[0].end(), temp_short.begin(), temp_short.end());
            }
            encoder.encode(temp_full_slots[0], scale, level, ker_poly[j][0][option]);
            
            // For I1
            for(int k = 0; k < 16; ++k){
                dvec temp_short;    // size = (32 * 16)
                bool i1 = ((k + 1) % 2);
                ker1_to_vector_conv1d(temp_short, kernel[j_start + k][i1], option);
                temp_full_slots[1].insert(temp_full_slots[1].end(), temp_short.begin(), temp_short.end());
            }
            msgrightRotate_inplace(temp_full_slots[1], Param_conv1d::shift); // it is changed for the baby-strategy method
            encoder.encode(temp_full_slots[1], scale, level, ker_poly[j][1][option]);
        }
        MT_EXEC_RANGE_END
    }
}

/* -----------------------------------------------
 * Encoding weights of the second/third conv layers
 * -----------------------------------------------
 */

/*
 @param[in] kernel, the kernel of the "conv_block"-th convoluational layer
                 [out_channels][in_channels][filter_size];
 @params[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[in] NB_CHANNELS, the number of channels
 @param[in] scale, the scaling factor of the kernel
 @param[in] level, the encoding level of the kernel
 @param[out] ker_poly, the encoded polynomials of kernels as a native plaintext format (size = [NB_CHANNELS[1]/16][2][3])
 The ouput polynomials are scaled by the factor of "scale" at "level"
*/
void HECNNenc1d::encode_conv(vector<vector<vector<Plaintext>>> &ker_poly, dten kernel, int conv_block, string mode,
                             vector<int> NB_CHANNELS, double scale, int level)
{
    size_t slot_count = encoder.slot_count();
    int conv_block1 = conv_block - 1;
    int num_out = (NB_CHANNELS[conv_block] / 16);    // (out_channels)
    int num_in = NB_CHANNELS[conv_block1];          //(in_channels)
    int dist = (conv_block == 2 ? 2 : 4);
    
    ker_poly.resize(num_out, vector<vector<Plaintext>>(num_in, vector<Plaintext> (3)));
    
    if(mode == "fully"){
        for(int i = 0; i < num_out; ++i){ // poly[i][j][-]
            MT_EXEC_RANGE(num_in, first, last);
            for(int j = first; j < last; ++j){
                for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                    dvec temp_slots;    // (32 * 16) * 16 = 8192
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (j1 - j)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (16 - j + j1)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
            MT_EXEC_RANGE_END
        }
    }
    else if(mode == "baby"){
        for(int i = 0; i < num_out; ++i){
            MT_EXEC_RANGE(num_in, first, last);
            for(int j = first; j < last; ++j){
                for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                    dvec temp_slots;    // (32 * 16) * 16 = 8192
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (j1 - j)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (16 - j + j1)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                
                    if((j % 16) != 0) {
                        msgrightRotate_inplace(temp_slots, (j % 16) * Param_conv1d::shift); // It is changed for the baby-strategy method
                    }
                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
            MT_EXEC_RANGE_END
        }
    }
    else if(mode == "giant"){
        for(int i = 0; i < num_out; ++i){
            MT_EXEC_RANGE(num_in, first, last);
            for(int j = first; j < last; ++j){
                for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                    dvec temp_slots;    // (32 * 16) * 16 = 8192
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (j1 - j)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        ker2_to_vector_conv1d(temp, kernel[i * 16 + (16 - j + j1)][j1], option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    // additional rotations for giant hoisting
                    // e.g., k0 * rho(ct[1], -1) = rho((rho(k0, 1) * ct[1]), -1);
                    // k2 * rho(ct[2], 1) = rho((rho(k2, -1) * ct[2]), 1)
                    if(option == 0){
                        msgleftRotate_inplace(temp_slots, Param_conv1d::steps_conv[conv_block1][1]);
                    } else if(option == 2){
                        msgrightRotate_inplace(temp_slots, Param_conv1d::steps_conv[conv_block1][1]);
                    }
                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
            MT_EXEC_RANGE_END
        }
    }
}

/*
 @param[in] kernel, the kernel of the "conv_block"-th convoluational layer
                 [out_channels][in_channels][filter_size];
 @params[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[in] NB_CHANNELS, the number of channels
 @param[in] scale, the scaling factor of the kernel
 @param[in] level, the encoding level of the kernel
 @param[out] ker_poly, the encoded polynomials of kernels as a native plaintext format (size = [NB_CHANNELS[1]/16][2][3])
 The ouput polynomials are scaled by the factor of "scale" at "level"
 */
void HECNNenc1d::encode_conv_interlaced(vector<vector<vector<Plaintext>>> &ker_poly,
                                      dten kernel, int conv_block, string mode, vector<int> NB_CHANNELS, double scale, int level)
{
    size_t slot_count = encoder.slot_count();
    int conv_block1 = conv_block - 1;
    int num_out = (NB_CHANNELS[conv_block] / 16);
    int num_in = NB_CHANNELS[conv_block1];
    int dist = (conv_block == 2 ? 2 : 4);
    int batch_in = (conv_block == 2 ? 2 : 4);        // the number of interlacing entries
    int num_in1 = (int) num_in/ batch_in;            // the number of actual outputs; (128/4) = 32 and (256/16)= 16 when nch=128
    
    ker_poly.resize(num_out, vector<vector<Plaintext>>(num_in1, vector<Plaintext> (3)));    // [16][128][3]
    
    if(mode == "fully"){
        MT_EXEC_RANGE(num_out, first, last);
        for(int i = first; i < last; ++i){
            for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                for(int j = 0; j < num_in1; ++j){
                    dvec temp_slots;
                    
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    int jstart_batch = jstart * batch_in;  // 0, 16*4=64
                    
                    // upper diagonal entries
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (j1 - j)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist); // len = (32 * 16)
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    // lower diagonal entries
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (16 - j + j1)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
        }
        MT_EXEC_RANGE_END
    }
    else if(mode == "baby"){
        MT_EXEC_RANGE(num_out, first, last);
        for(int i = first; i < last; ++i){
            for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                for(int j = 0; j < num_in1; ++j){
                    dvec temp_slots;
                    
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    int jstart_batch = jstart * batch_in;
                    
                    // upper diagonal entries
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (j1 - j)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    // lower diagonal entries
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (16 - j + j1)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    if((j % 16) != 0) {
                        msgrightRotate_inplace(temp_slots, (j % 16) * Param_conv1d::shift);
                    }
                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
        }
        MT_EXEC_RANGE_END
    }
    else if(mode == "giant"){
        MT_EXEC_RANGE(num_out, first, last);
        for(int i = first; i < last; ++i){
            for(int option = 0; option < Param_conv1d::DIM_FILTERS; ++option){
                for(int j = 0; j < num_in1; ++j){
                    dvec temp_slots;
                    
                    int jend = 16 * ceil((double)(j + 1) / 16.0);
                    int jstart = jend - 16;
                    int jstart_batch = jstart * batch_in;
                    
                    // upper diagonal entries
                    for(int j1 = j; j1 < jend; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (j1 - j)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    // lower diagonal entries
                    for(int j1 = jstart; j1 < j; ++j1){
                        dvec temp;
                        dmat kernels;
                        for(int k = 0; k < batch_in; k++){
                            kernels.push_back(kernel[i * 16 + (16 - j + j1)][jstart_batch + (j1 - jstart) + k * 16]);
                        }
                        ker2_to_interlaced_vector_conv1d(temp, kernels, option, dist);
                        temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                    }
                    
                    // additional rotations for giant hoisting
                    // e.g., k0 * rho(ct[1], -1) = rho((rho(k0, 1) * ct[1]), -1);
                    // k2 * rho(ct[2], 1) = rho((rho(k2, -1) * ct[2]), 1)
                    if(option == 0){
                        msgleftRotate_inplace(temp_slots, Param_conv1d::steps_conv[conv_block1][1]);
                    } else if(option == 2){
                        msgrightRotate_inplace(temp_slots, Param_conv1d::steps_conv[conv_block1][1]);
                    }

                    encoder.encode(temp_slots, scale, level, ker_poly[i][j][option]);
                }
            }
        }
        MT_EXEC_RANGE_END
    }
}


/*
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT, [128][3]
 @param[in] NB_CHANNELS, the number of channels at the current activation layer (=nch)
 @param[in] qscale, the scaling factor of the parameters which are handled with ciphertexts
 @param[in] qcscale, the scaling factor of the parameters which are handled with plaintexts
 @param[in] level, the encoding level of the input ciphertext
 @param[out] act_poly, the encoded polynomials of "real_poly", [nch/16][3]
 We encode 16 information into one plaintext polynomial
 The contant term is scaled by factor of "qscale" (lvl = Param_conv1d::ker_poly_lvl[1]),
 while the other terms are scaled by "qcscale" (lvl = Param_conv1d::ker_poly_lvl[1] + 1)
*/
void HECNNenc1d::encode_act1(vector<vector<Plaintext>> &act_poly, dmat real_poly, int NB_CHANNELS,
                           double qscale, double qcscale, int level)
{
    size_t slot_count = encoder.slot_count();
    int len = 32 * 16;
    int len_actual = 32 * 15;
    
    int num = (NB_CHANNELS / 16);
    act_poly.resize(num, vector<Plaintext>(3));
    
    int const_lvl = level - 3;  // = Param_conv1d::ker_poly_lvl[0]
    int nonconst_lvl = level - 2;
    
    int num3 = num * 3;

    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        int i1 = (i << 4);
        
        // A fully-packed plaintext vector of size (32*16*16)
        for(int j = 0; j < 16; ++j){
            dvec temp_short;    // size = (32 * 16)
            val1_to_vector_conv1d(temp_short, real_poly[i1 + j][l], len, len_actual); 
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, qscale, const_lvl, act_poly[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, qcscale, nonconst_lvl, act_poly[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, qcscale, nonconst_lvl, act_poly[i][2]);
        }
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT, [2*nch][3] in B2 or [4*nch][3] in B3
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] NB_CHANNELS, the number of channels at the current activation layer
 @param[in] qscale, the scaling factor of the parameters which are handled with ciphertexts
 @param[in] qcscale, the scaling factor of the parameters which are handled with plaintexts
 @param[in] level, the encoding level of the input ciphertext
 @param[out] act_poly, the encoded polynomials of "real_poly", [2*nch/16][3] or [4*nch/16][3]
 We encode 16 information into one plaintext polynomial
 The contant term is scaled by factor of "qscale" (lvl = Param_conv1d::ker_poly_lvl[conv_block]),
 while the other terms are scaled by "qcscale" (lvl = Param_conv1d::ker_poly_lvl[conv_block] + 1)
 */
void HECNNenc1d::encode_act(vector<vector<Plaintext>> &act_poly, dmat real_poly, int conv_block, int NB_CHANNELS,
                          double qscale, double qcscale, int level)
{
    size_t slot_count = encoder.slot_count();
    int len = 32 * 16;
    int len_actual = 32 * 15;
    
    int dist = (1 << (conv_block -1));           // 2 for B2, 4 for B3
    int num = (NB_CHANNELS / 16);                // 2*nch for B2; 4*nch for B3
    act_poly.resize(num, vector<Plaintext>(3)); // [2*nch/16][3] for B2; [4*nch/16][3] for B3
    
    int const_lvl = level - 3;                  // Param_conv1d::ker_poly_lvl[conv_block - 1]
    int nonconst_lvl = level - 2;
    
    int num3 = num * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        dvec temp_full_slots;  // (32 * 16) * 16 = 8192
        int i1 = (i << 4);
        
        for(int j = 0; j < 16; ++j){
            dvec temp_short;    // size = (32 * 16)
            val_to_vector_conv1d(temp_short, real_poly[i1 + j][l], len, len_actual, dist);
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
       
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, qscale, const_lvl, act_poly[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, qcscale, nonconst_lvl, act_poly[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, qcscale, nonconst_lvl, act_poly[i][2]);
        }
    }
    MT_EXEC_RANGE_END
}

/* -----------------------------------------------
 * Encoding weights of the fully-connected layer
 * -----------------------------------------------
 */

/*
 @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
 @param[in] NB_CHANNELS, the number of channels at the current layer (= 4*nch)
 @param[in] scale, the scaling factor of the parameters
 @param[in] level, the encoding level of the input ciphertext
 @param[out] dense_ker_poly, the encoded polynomials of the dense kernel, size [4*nch/16][16]
 The parameters are scaled by "scale" at lvl = Param_conv1d::ker_poly_lvl[3])
 */
void HECNNenc1d::encode_dense_kerpoly(vector<vector<Plaintext>> &dense_ker_poly, dmat dense_ker, int NB_CHANNELS, double scale, int level)
{
    size_t slot_count = encoder.slot_count();
    size_t numi = NB_CHANNELS/16;   // 4*nch/16 = 32
    dense_ker_poly.resize(numi, vector<Plaintext> (16));
    
    MT_EXEC_RANGE(numi, first, last);
    for(int i = first; i < last; ++i){
        int i1 = (i << 4);
        int diff = i1; // initial diff
        
        // the first seven diagonal vectors
        for(int j = 0; j < 7; ++j){
            dvec temp_slots(slot_count, 0.0);
            for(int l = 0; l < 10; ++l){
                int l1 = ((j + l) << Param_conv1d::logshift);
                temp_slots[l1] = (dense_ker[l][diff + l]);
            }
            encoder.encode(temp_slots, scale, level, dense_ker_poly[i][j]);
            diff++;
        }
        
        // the second nine diagonal vectors
        for(int j = 7; j < 16; ++j){
            dvec temp_slots(slot_count, 0.0);
            
            for(int l = 0; l < j - 6; ++l){
                int l1 = (l << Param_conv1d::logshift);
                int xcord = (i1 + l);
                temp_slots[l1] = (dense_ker[l - j + 16][xcord]);
            }
            
            for(int l = j; l < 16; ++l){
                int l1 = (l << Param_conv1d::logshift);
                int ycord = l - j;
                temp_slots[l1] = (dense_ker[ycord][diff + ycord]);
            }
            encoder.encode(temp_slots, scale, level, dense_ker_poly[i][j]);
            diff++;
        }
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] scale, the scaling factor of the bias parameters
 @param[out] bias_poly, the encoded polynomials of the dense bias
 The input vector is converted into a plaintext vector (bias[0], 0,,,0| bias[1], 0,,,0| ..., |bias[9], 0,,,0).
 And then it is scaled by "scale" at lvl = 0)
 */
void HECNNenc1d::encode_dense_biaspoly(Plaintext &bias_poly, dvec dense_bias, double scale)
{
    size_t slot_count = encoder.slot_count();
    int lvl = 0;
    dvec temp_slots(slot_count, 0ULL);
    
    for(int i = 0; i < (int) dense_bias.size(); ++i){
        int i1 = Param_conv1d::shift * i;
        temp_slots[i1] = dense_bias[i];
    }
    encoder.encode(temp_slots, scale, lvl, bias_poly);
}

/*
 @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_size])
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
 @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation strategy of homomorphic convolution at B1
 @param[in] mode2, the evaluation strategy of homomorphic convolution at B2
 @param[in] mode3, the evaluation strategy of homomorphic convolution at B3
 @param[in] NB_CHANNELS, the number of channels
 @param[out] ker1, the encoded polynomials of kernels at B1
 @param[out] act1, the encoded polynomials of "real_poly" at B1
 @param[out] ker2, the encoded polynomials of kernels at B2
 @param[out] act2, the encoded polynomials of "real_poly" at B2
 @param[out] ker3, the encoded polynomials of kernels at B3
 @param[out] act3, the encoded polynomials of "real_poly" at B3
 @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
 @param[out] bias_poly, the encoded polynomials of the dense bias
 We generate plaintext polynomials that are compatible with HEAR
*/
void HECNNenc1d::prepare_network(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                    vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                    vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                    vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly,
                    vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                    string mode1, string mode2, string mode3, vector<int> NB_CHANNELS)
{
    // B1
    if (mode1 == "nonpacked"){
        encode_conv1(ker1, ker[0], NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    } else{
        encode_conv1(ker1, ker[0], mode1, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    }
    encode_act1(act1, real_poly[0], NB_CHANNELS[1], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    
    // B2
    encode_conv(ker2, ker[1], 2, mode2, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[1]);
    encode_act(act2, real_poly[1], 2, NB_CHANNELS[2], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[1]);

    // B3
    encode_conv(ker3, ker[2], 3, mode3, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[2]);
    encode_act(act3, real_poly[2], 3, NB_CHANNELS[3], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[2]);
 
    // Dense
    dmat scaled_dense_ker (dense_ker.size(), vector<double> (dense_ker[0].size()));
    for(int i = 0; i < dense_ker.size(); ++i){
        for(int j = 0; j < dense_ker[0].size(); ++j){
            scaled_dense_ker[i][j] = dense_ker[i][j]/10.0;
        }
    }
    
    dvec scaled_dense_bias (dense_bias.size());
    for(int i = 0; i < dense_bias.size(); ++i){
        scaled_dense_bias[i] = dense_bias[i]/10.0;
    }
    
    encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[3]);
    encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_HEAR::qscale);
}

/*
 @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_size])
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
 @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation strategy of homomorphic convolution at B1
 @param[in] mode2, the evaluation strategy of homomorphic convolution at B2
 @param[in] mode3, the evaluation strategy of homomorphic convolution at B3
 @param[in] NB_CHANNELS, the number of channels
 @param[in] modulus_bits, the bit lengths of ciphertext modulus
 @param[out] ker1, the encoded polynomials of kernels at B1
 @param[out] act1, the encoded polynomials of "real_poly" at B1
 @param[out] ker2, the encoded polynomials of kernels at B2
 @param[out] act2, the encoded polynomials of "real_poly" at B2
 @param[out] ker3, the encoded polynomials of kernels at B3
 @param[out] act3, the encoded polynomials of "real_poly" at B3
 @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
 @param[out] bias_poly, the encoded polynomials of the dense bias
 @param[out] zero_one_poly, the plaintext polynomial with 0-1 values, (used for a pre-processing step)
 We generate plaintext polynomials that are compatible with Fast-HEAR
*/
void HECNNenc1d::prepare_network_interlaced(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                                         vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                                         vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                                         vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly, vector<Plaintext> &zero_one_poly,
                                         vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                                         string mode1, string mode2, string mode3, vector<int> NB_CHANNELS, vector<int> modulus_bits)
{
    // Generate plaintext polynomials for B1
    if (mode1 == "nonpacked"){
        encode_conv1(ker1, ker[0], NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    } else{
        encode_conv1(ker1, ker[0], mode1, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    }
    encode_act1(act1, real_poly[0], NB_CHANNELS[1], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    
    // Generate plaintext polynomials for B2
    encode_conv_interlaced(ker2, ker[1], 2, mode2, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[1]);
    encode_act(act2, real_poly[1], 2, NB_CHANNELS[2], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[1]);
   
    // Generate plaintext polynomials for B3
    encode_conv_interlaced(ker3, ker[2], 3, mode3, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[2]);
    encode_act(act3, real_poly[2], 3, NB_CHANNELS[3], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[2]);

    // Generate plaintext polynomials forDense
    dmat scaled_dense_ker (dense_ker.size(), vector<double> (dense_ker[0].size()));
    for(int i = 0; i < dense_ker.size(); ++i){
        for(int j = 0; j < dense_ker[0].size(); ++j){
            scaled_dense_ker[i][j] = dense_ker[i][j]/10.0;
        }
    }
    
    dvec scaled_dense_bias (dense_bias.size());
    for(int i = 0; i < dense_bias.size(); ++i){
        scaled_dense_bias[i] = dense_bias[i]/10.0;
    }
    
    double q1 = pow(2.0, modulus_bits[1]);
    encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], q1, Param_FHEAR::ker_poly_lvl[3]);
    encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_FHEAR::qscale);
    
    // Plaintext polynomials for pre-processing step
    // zero_one_poly[0]: for conv2, zero_one_poly[1]: for conv3
    zero_one_poly.resize(2);
    for(int k = 0; k < 2; ++k){
        vector<double> msg_one_block (32 * 16, 0ULL);   // msg_one_block = [32 * 16]
        int lvl;
        
        if(k == 0){ // (1,0) (1,0) ...
            lvl = Param_FHEAR::ker_poly_lvl[0] - 3;
            for(int i = 0; i < 32 * 15; i+=2){
                msg_one_block[i] = 1.0;
            }
        }
        else{ // (1,0,0,0) (1,0,0,0) ...
            lvl = Param_FHEAR::ker_poly_lvl[1] - 3;
            for(int i = 0; i < 32 * 15; i+=4){
                msg_one_block[i] = 1.0;
            }
        }
        
        vector<double> msg_full;        // [32 * 16 * 16] = fully packed slots
        for(int l = 0; l < 16; ++l){
            msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
        }
        encoder.encode(msg_full, Param_FHEAR::qcscale_small, lvl, zero_one_poly[k]);
    }
}

/*
 @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_size])
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
 @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation strategy of homomorphic convolution at B1
 @param[in] mode2, the evaluation strategy of homomorphic convolution at B2
 @param[in] mode3, the evaluation strategy of homomorphic convolution at B3
 @param[in] NB_CHANNELS, the number of channels
 @param[in] modulus_bits, the bit lengths of ciphertext modulus
 @param[out] ker1, the encoded polynomials of kernels at B1
 @param[out] act1, the encoded polynomials of "real_poly" at B1
 @param[out] ker2, the encoded polynomials of kernels at B2
 @param[out] act2, the encoded polynomials of "real_poly" at B2
 @param[out] ker3, the encoded polynomials of kernels at B3
 @param[out] act3, the encoded polynomials of "real_poly" at B3
 @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
 @param[out] bias_poly, the encoded polynomials of the dense bias
 We generate plaintext polynomials that are compatible with HEAR.
 In particular, all the parameters are encoded at the highest encoding level (without the level-aware encoding method)
*/
void HECNNenc1d::prepare_network_naive(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                    vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                    vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                    vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly,
                    vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                    string mode1, string mode2, string mode3, vector<int> NB_CHANNELS)
{
    // B1
    if (mode1 == "nonpacked"){
        encode_conv1(ker1, ker[0], NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    } else{
        encode_conv1(ker1, ker[0], mode1, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    }
    encode_act1(act1, real_poly[0], NB_CHANNELS[1], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    
    // B2
    encode_conv(ker2, ker[1], 2, mode2, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    encode_act(act2, real_poly[1], 2, NB_CHANNELS[2], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);

    // B3
    encode_conv(ker3, ker[2], 3, mode3, NB_CHANNELS, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    encode_act(act3, real_poly[2], 3, NB_CHANNELS[3], Param_HEAR::qscale, Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
 
    // Dense
    dmat scaled_dense_ker (dense_ker.size(), vector<double> (dense_ker[0].size()));
    for(int i = 0; i < dense_ker.size(); ++i){
        for(int j = 0; j < dense_ker[0].size(); ++j){
            scaled_dense_ker[i][j] = dense_ker[i][j]/10.0;
        }
    }
    
    dvec scaled_dense_bias (dense_bias.size());
    for(int i = 0; i < dense_bias.size(); ++i){
        scaled_dense_bias[i] = dense_bias[i]/10.0;
    }
    
    encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], Param_HEAR::qcscale, Param_HEAR::ker_poly_lvl[0]);
    encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_HEAR::qscale);
}


/*
 @param[in] ker, the kernel of the convoluational layers ([3][out_channels][in_channels][filter_size])
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT
 @param[in] dense_ker, the kernel matrix of the dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation strategy of homomorphic convolution at B1
 @param[in] mode2, the evaluation strategy of homomorphic convolution at B2
 @param[in] mode3, the evaluation strategy of homomorphic convolution at B3
 @param[in] NB_CHANNELS, the number of channels
 @param[in] modulus_bits, the bit lengths of ciphertext modulus
 @param[out] ker1, the encoded polynomials of kernels at B1
 @param[out] act1, the encoded polynomials of "real_poly" at B1
 @param[out] ker2, the encoded polynomials of kernels at B2
 @param[out] act2, the encoded polynomials of "real_poly" at B2
 @param[out] ker3, the encoded polynomials of kernels at B3
 @param[out] act3, the encoded polynomials of "real_poly" at B3
 @param[out] dense_ker_poly, the encoded polynomials of the dense kernel
 @param[out] bias_poly, the encoded polynomials of the dense bias
 @param[out] zero_one_poly, the plaintext polynomial with 0-1 values, (used for a pre-processing step)
 We generate plaintext polynomials that are compatible with Fast-HEAR
 In particular, all the parameters are encoded at the highest encoding level (without the level-aware encoding method).
*/
void HECNNenc1d::prepare_network_interlaced_naive(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                                         vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                                         vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                                         vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly, vector<Plaintext> &zero_one_poly,
                                         vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                                         string mode1, string mode2, string mode3, vector<int> NB_CHANNELS, vector<int> modulus_bits)
{
    // Generate plaintext polynomials for B1
    if (mode1 == "nonpacked"){
        encode_conv1(ker1, ker[0], NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    } else{
        encode_conv1(ker1, ker[0], mode1, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    }
    encode_act1(act1, real_poly[0], NB_CHANNELS[1], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    
    // Generate plaintext polynomials for B2
    encode_conv_interlaced(ker2, ker[1], 2, mode2, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    encode_act(act2, real_poly[1], 2, NB_CHANNELS[2], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
   
    // Generate plaintext polynomials for B3
    encode_conv_interlaced(ker3, ker[2], 3, mode3, NB_CHANNELS, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);
    encode_act(act3, real_poly[2], 3, NB_CHANNELS[3], Param_FHEAR::qscale, Param_FHEAR::qcscale, Param_FHEAR::ker_poly_lvl[0]);

    // Generate plaintext polynomials forDense
    dmat scaled_dense_ker (dense_ker.size(), vector<double> (dense_ker[0].size()));
    for(int i = 0; i < dense_ker.size(); ++i){
        for(int j = 0; j < dense_ker[0].size(); ++j){
            scaled_dense_ker[i][j] = dense_ker[i][j]/10.0;
        }
    }
    
    dvec scaled_dense_bias (dense_bias.size());
    for(int i = 0; i < dense_bias.size(); ++i){
        scaled_dense_bias[i] = dense_bias[i]/10.0;
    }
    
    double q1 = pow(2.0, modulus_bits[1]);
    encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], q1, Param_FHEAR::ker_poly_lvl[0]);
    encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_FHEAR::qscale);
    
    zero_one_poly.resize(2);
    for(int k = 0; k < 2; ++k){
        vector<double> msg_one_block (32 * 16, 0ULL);
        int lvl;
        
        if(k == 0){ // (1,0) (1,0) ...
            lvl = Param_FHEAR::ker_poly_lvl[0] - 3;
            for(int i = 0; i < 32 * 15; i+=2){
                msg_one_block[i] = 1.0;
            }
        }
        else{ // (1,0,0,0) (1,0,0,0) ...
            lvl = Param_FHEAR::ker_poly_lvl[1] - 3;
            for(int i = 0; i < 32 * 15; i+=4){
                msg_one_block[i] = 1.0;
            }
        }
        
        vector<double> msg_full;        // [32 * 16 * 16] = fully packed slots
        for(int l = 0; l < 16; ++l){
            msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
        }
        encoder.encode(msg_full, Param_FHEAR::qcscale_small, lvl, zero_one_poly[k]);
    }
}
