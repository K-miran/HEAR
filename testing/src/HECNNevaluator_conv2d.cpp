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
#include "HECNNevaluator_conv2d.h"

#define DEBUG true
using namespace std;

/* -----------------------------------------------
 * Generate the rotated ciphertexts
 * -----------------------------------------------
 */

/*
@param[in] ct, input ciphertexts (size 2)
@param[out] res, rotated ciphertexts (size [2][9])
 res[i][*] is the rotated ciphertext of the ciphertext ct[i]
*/
void HECNNeval::generate_rotations_conv1(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct)
{
    res.resize(2, vector<Ciphertext> (9));
    
    for(int i = 0; i < res.size(); ++i){
        res[i][0] = ct[i];
    }
  
    int NUM_THREADS = Thread::availableThreads();
    int NUM_THREADS2 = (NUM_THREADS/2);
    int total_nrots = res[0].size() - 1;     // the total number of needed rotations
    int nrows = res.size();
    
    if(total_nrots <= NUM_THREADS){
        MT_EXEC_RANGE(total_nrots, first, last);
        for(int j = first; j < last; ++j){
            evaluator.rotate_vector(ct[0], Param_conv2d::steps_conv[0][j], gal_keys, res[0][j + 1]);
            evaluator.rotate_vector(ct[1], Param_conv2d::steps_conv[0][j], gal_keys, res[1][j + 1]);
        }
        MT_EXEC_RANGE_END
    }
    else{
        // For convenience, we break subtasks up into smaller, almost-equal-sized groups of subtasks
        int block_len = (int)ceil((double)total_nrots / NUM_THREADS);
        int final_block_len = (total_nrots % block_len);
        if (final_block_len == 0) final_block_len = block_len;
        
        MT_EXEC_RANGE(NUM_THREADS, first, last);
        for(int k = first; k < last; ++k){
            int nstart = k * block_len;
            int sub_len = (k == (NUM_THREADS - 1) ? final_block_len : block_len);
            
            vector<int> new_steps;
            for (size_t j = nstart; j < nstart + sub_len; ++j) {
                new_steps.push_back(Param_conv2d::steps_conv[0][j]);
            }
            
            vector<Ciphertext> ct_temp0(sub_len + 1);
            ct_temp0[0] = ct[0];
            evaluator.rotate_vector_many(res[0][0], new_steps, gal_keys, ct_temp0);
            
            vector<Ciphertext> ct_temp1(sub_len + 1);
            ct_temp1[0] = ct[1];
            evaluator.rotate_vector_many(res[1][0], new_steps, gal_keys, ct_temp1);
            
            for (size_t i = 1; i < sub_len + 1; ++i) {
                res[0][i + nstart] = ct_temp0[i];
                res[1][i + nstart] = ct_temp1[i];
            }
        }
        MT_EXEC_RANGE_END
    }
}

/*
@param[in] ct, an input ciphertext
@param[in] mode, the evaluation strategy of homomorphic convolution
@param[out] res, rotated ciphertexts by using the hoisting technique (size 9)
 if mode = baby, we compute rho(ct; +-1), rho(ct; +-15), rho(ct; +-16), rho(ct; +-17) by using the "divide-and-conquer" method
 if mode = fully, we generate the following ciphertexts:
 rho(ct; +-1), rho(ct; +-15), rho(ct; +-16), rho(ct; +-17)
 rho(ct; r), rho(ct; r+-1), rho(ct; r+-15), rho(ct; r+-16), rho(ct; r+-17)
*/
void HECNNeval::generate_rotations_conv1(vector<Ciphertext> &res, Ciphertext ct, string mode)
{
    int NUM_THREADS = Thread::availableThreads();
    
    if(mode == "fully"){
        res.resize(9 * 2);
        res[0] = ct;
        evaluator.rotate_vector(ct, Param_conv2d::shift, gal_keys, res[9]);
        
        int total_nrots = res.size()/2 - 1; // the total number of needed rotations
        
        if(total_nrots < NUM_THREADS){
            MT_EXEC_RANGE(NUM_THREADS, first, last);
            for(int j = first; j < last; ++j){
                if(j < 8){
                    evaluator.rotate_vector(ct, Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1]);
                } else{
                    evaluator.rotate_vector(res[9], Param_conv2d::steps_conv[0][j - 8], gal_keys, res[j + 2]);
                }
            }
            MT_EXEC_RANGE_END
        }
        else if(total_nrots == NUM_THREADS){
            MT_EXEC_RANGE(total_nrots, first, last);
            for(int j = first; j < last; ++j){
                evaluator.rotate_vector(ct, Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1]);
                evaluator.rotate_vector(res[9], Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1 + 9]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            int block_len = (int)ceil((double)total_nrots / NUM_THREADS);
            int final_block_len = (total_nrots % block_len);
            if (final_block_len == 0) final_block_len = block_len;
            
            MT_EXEC_RANGE(NUM_THREADS, first, last);
            for(int k = first; k < last; ++k){
                int nstart = k * block_len;
                int sub_len = (k == (NUM_THREADS - 1) ? final_block_len : block_len);
                
                vector<int> new_steps;
                for (size_t j = nstart; j < nstart + sub_len; ++j) {
                    new_steps.push_back(Param_conv2d::steps_conv[0][j]);
                }
                
                vector<Ciphertext> ct_temp(sub_len + 1);
                ct_temp[0] = ct;
                evaluator.rotate_vector_many(res[0], new_steps, gal_keys, ct_temp);
                
                vector<Ciphertext> ct_temp1(sub_len + 1);
                ct_temp1[0] = res[9];
                evaluator.rotate_vector_many(res[9], new_steps, gal_keys, ct_temp1);
                
                for (size_t i = 1; i < sub_len + 1; ++i) {
                    res[i + nstart] = ct_temp[i];
                    res[i + nstart + 9] = ct_temp1[i];
                }
            }
            MT_EXEC_RANGE_END
        }
    }
    else if(mode == "baby"){
        res.resize(9);
        int total_nrots = res.size() - 1;     // the total number of needed rotations
        res[0] = ct;
        
        if(total_nrots <= NUM_THREADS){
            MT_EXEC_RANGE(total_nrots, first, last);
            for(int j = first; j < last; ++j){
                evaluator.rotate_vector(ct, Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            int block_len = (int)ceil((double)total_nrots / NUM_THREADS);
            int final_block_len = (total_nrots % block_len);
            if (final_block_len == 0) final_block_len = block_len;
            
            MT_EXEC_RANGE(NUM_THREADS, first, last);
            for(int k = first; k < last; ++k){
                int nstart = k * block_len;
                int sub_len = (k == (NUM_THREADS - 1) ? final_block_len : block_len);
                
                vector<int> new_steps;
                for (size_t j = nstart; j < nstart + sub_len; ++j) {
                    new_steps.push_back(Param_conv2d::steps_conv[0][j]);
                }
                
                vector<Ciphertext> ct_temp(sub_len + 1);
                ct_temp[0] = ct;
                evaluator.rotate_vector_many(res[0], new_steps, gal_keys, ct_temp);
                
                for (size_t i = 1; i < sub_len + 1; ++i) {
                    res[i + nstart] = ct_temp[i];
                }
            }
            MT_EXEC_RANGE_END
        }
    }
}

/*
@param[in] ct, an input ciphertext
@param[in] mode, the evaluation strategy of homomorphic convolution
@param[out] res, rotated ciphertexts without the hoisting technique (size 9)
 if mode = baby, we compute rho(ct; +-1), rho(ct; +-15), rho(ct; +-16), rho(ct; +-17) by using the "divide-and-conquer" method
 if mode = fully, we generate the following ciphertexts:
 rho(ct; +-1), rho(ct; +-15), rho(ct; +-16), rho(ct; +-17)
 rho(ct; r), rho(ct; r+-1), rho(ct; r+-15), rho(ct; r+-16), rho(ct; r+-17)
*/
void HECNNeval::generate_rotations_conv1_wo_hoisting(vector<Ciphertext> &res, Ciphertext ct, string mode)
{
    int NUM_THREADS = Thread::availableThreads();
    
    if(mode == "fully"){
        res.resize(9 * 2);
        res[0] = ct;
        evaluator.rotate_vector(ct, Param_conv2d::shift, gal_keys, res[9]);
        
        int total_nrots = res.size()/2 - 1;
        
        if(total_nrots <= NUM_THREADS){
            MT_EXEC_RANGE(total_nrots, first, last);
            for(int j = first; j < last; ++j){
                evaluator.rotate_vector(ct, Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1]);
                evaluator.rotate_vector(res[9], Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1 + 9]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            int block_len = (int)ceil((double)total_nrots / NUM_THREADS);
            int final_block_len = (total_nrots % block_len);
            if (final_block_len == 0) final_block_len = block_len;
            
            MT_EXEC_RANGE(NUM_THREADS, first, last);
            for(int k = first; k < last; ++k){
                int nstart = k * block_len;
                int sub_len = (k == (NUM_THREADS - 1) ? final_block_len : block_len);
                
                vector<int> new_steps;
                for (size_t j = nstart; j < nstart + sub_len; ++j) {
                    new_steps.push_back(Param_conv2d::steps_conv[0][j]);
                }
                
                vector<Ciphertext> ct_temp(sub_len + 1);
                ct_temp[0] = ct;
                for(int i = 0; i < sub_len; ++i){
                    evaluator.rotate_vector(res[0], new_steps[i], gal_keys, ct_temp[i + 1]);
                }
                
                vector<Ciphertext> ct_temp1(sub_len + 1);
                ct_temp1[0] = res[9];
                for(int i = 0; i < sub_len; ++i){
                    evaluator.rotate_vector(res[9], new_steps[i], gal_keys, ct_temp1[i + 1]);
                }
                
                for (size_t i = 1; i < sub_len + 1; ++i) {
                    res[i + nstart] = ct_temp[i];
                    res[i + nstart + 9] = ct_temp1[i];
                }
            }
            MT_EXEC_RANGE_END
        }
    }
    else if(mode == "baby"){
        res.resize(9);
        int total_nrots = res.size() - 1;
        res[0] = ct;
        
        if(total_nrots <= NUM_THREADS){
            MT_EXEC_RANGE(total_nrots, first, last);
            for(int j = first; j < last; ++j){
                evaluator.rotate_vector(ct, Param_conv2d::steps_conv[0][j], gal_keys, res[j + 1]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            int block_len = (int)ceil((double)total_nrots / NUM_THREADS);
            int final_block_len = (total_nrots % block_len);
            if (final_block_len == 0) final_block_len = block_len;
            
            MT_EXEC_RANGE(NUM_THREADS, first, last);
            for(int k = first; k < last; ++k){
                int nstart = k * block_len;
                int sub_len = (k == (NUM_THREADS - 1) ? final_block_len : block_len);
                
                vector<int> new_steps;
                for (size_t j = nstart; j < nstart + sub_len; ++j) {
                    new_steps.push_back(Param_conv2d::steps_conv[0][j]);
                }
                
                vector<Ciphertext> ct_temp(sub_len + 1);
                ct_temp[0] = ct;
                
                for(int i = 0; i < sub_len; ++i){
                    evaluator.rotate_vector(res[0], new_steps[i], gal_keys, ct_temp[i+ 1]);
                }
                
                for (size_t i = 1; i < sub_len + 1; ++i) {
                    res[i + nstart] = ct_temp[i];
                }
            }
            MT_EXEC_RANGE_END
        }
    }
}

/*
@param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
@param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
@param[out] res, rotated ciphertexts (size [2*nch][9*16] in B2; [4*nch][9*16] in B3)
 We generate the giant ciphertexts by using the native rotation method.
 And then generate the baby ciphertexts by using the hoisting method.
*/
void HECNNeval::generate_rotations_conv_fully(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block)
{
    int conv_block1 = conv_block - 1;
    int nin_ctxts = ct.size();
    int steps_size1 = Param_conv2d::steps_size + 1;
    int num = nin_ctxts * (Param_conv2d::steps_giant_size + 1);
    res.resize(nin_ctxts, vector<Ciphertext> (16 * 9));
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int i = (n % nin_ctxts);
        int j = (int) floor((double) n / (double) nin_ctxts);
         
        vector<Ciphertext> temp(steps_size1);
        if(j == 0){
            temp[0] = ct[i];
        } else{
            evaluator.rotate_vector(ct[i], j * Param_conv2d::shift, gal_keys, temp[0]);
        }
        
        evaluator.rotate_vector_many(temp[0], Param_conv2d::steps_conv[conv_block1], gal_keys, temp);

        for(int l = 0; l < steps_size1; ++l){
            res[i][j * (steps_size1) + l] = temp[l];
        }
    }
    MT_EXEC_RANGE_END
}

/*
@param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
@param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
@param[out] res, rotated ciphertexts (size [2*nch][9*16] in B2; [4*nch][9*16] in B3)
 We generate the giant ciphertexts by using the native rotation method.
 And then generate the baby ciphertexts by using the hoisting method.
 We optimize the memory usage by not generating temporary ciphertexts.
*/
void HECNNeval::generate_rotations_conv_fully_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block)
{
    // compute the parameters
    int conv_block1 = conv_block - 1;
    int nin_ctxts = ct.size();                              // (2*nch/16 or 4*nch/16) = (8 or 16), number of ciphertexts
    int steps_size1 = Param_conv2d::steps_size + 1;         // 8+1=9
    int num = nin_ctxts * (Param_conv2d::steps_giant_size + 1);
    res.resize(nin_ctxts, vector<vector<Ciphertext>> (16, vector<Ciphertext> (9)));
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int i = (n % nin_ctxts);                             // 0 <= i < nin_ctxts
        int j = (int) floor((double) n / (double) nin_ctxts);   // 0 <= j < (Param_conv2d::steps_giant_size + 1) = 16
         
        if(j == 0){
            res[i][j][0] = ct[i];
        } else{
            evaluator.rotate_vector(ct[i], j * Param_conv2d::shift, gal_keys, res[i][j][0]);  // the ciphertexts in the first column (giant ciphertexts)
        }
        
        evaluator.rotate_vector_many(res[i][j][0], Param_conv2d::steps_conv[conv_block1], gal_keys, res[i][j]);  // baby ciphertexts
    }
    MT_EXEC_RANGE_END
}

/*
@param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
@param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
@param[out] res, rotated ciphertexts (size [2*nch][9*16] in B2; [4*nch][9*16] in B3)
 We generate the giant ciphertexts by using the native rotation method.
 And then generate the baby ciphertexts without the hoisting method.
*/
void HECNNeval::generate_rotations_conv_fully_wo_hoisting(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block)
{
    int conv_block1 = conv_block - 1;
    int nin_ctxts = ct.size();
    
    res.resize(nin_ctxts, vector<Ciphertext> (9 * 16));
    int steps_size1 = Param_conv2d::steps_size + 1;
    
    int num = nin_ctxts * (Param_conv2d::steps_giant_size + 1);
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int i = (n % nin_ctxts);
        int j = (int) floor((double) n / (double) nin_ctxts);
        
        vector<Ciphertext> temp(steps_size1);
        if(j == 0){
            temp[0] = ct[i];
        } else{
            evaluator.rotate_vector(ct[i], j * Param_conv2d::shift, gal_keys, temp[0]);
        }
        
        for(int i = 0; i < Param_conv2d::steps_size; ++i){
            evaluator.rotate_vector(temp[0], Param_conv2d::steps_conv[conv_block1][i], gal_keys, temp[i + 1]);
        }
        
        for(int l = 0; l < steps_size1; ++l){
            res[i][j * (steps_size1) + l] = temp[l];
        }
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[out] res, rotated ciphertexts (size [2*nch][3*16] in B2; [4*nch][3*16] in B3)
 We generate the baby ciphertexts for each blocks (requiring 8 * 8 rotations).
 Here, the rotations are specified by "Param_conv3d::steps_conv[1]".
 If mode = baby, we pre-compute the rotated baby ciphertexts. (8 rotation for each input ctxt)
 If mode = giant, we pre-compute the rotated giant ciphertexts. (15 rotation for each input ctxt)
*/
void HECNNeval::generate_rotations_conv_babygiant(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block, string mode)
{
    int conv_block1 = (conv_block - 1);
    int nin_ctxts = ct.size();
    
    vector<int> steps;
    int nrot;
    if(mode == "baby"){
        nrot = Param_conv2d::steps_size;
        steps.insert(steps.end(), Param_conv2d::steps_conv[conv_block1].begin(), Param_conv2d::steps_conv[conv_block1].end());
    } else if(mode == "giant"){
        nrot = Param_conv2d::steps_giant_size;
        steps.insert(steps.end(), Param_conv2d::steps_giant.begin(), Param_conv2d::steps_giant.end());
    }
    
    int nrot_total = nin_ctxts * nrot;
    res.resize(nin_ctxts, vector<Ciphertext> (nrot + 1));
    for(int i = 0; i < nin_ctxts; ++i){
        res[i][0] = ct[i];
    }
    
    int NUM_THREADS = Thread::availableThreads();
    
    if(nrot_total <= NUM_THREADS)
    {
        MT_EXEC_RANGE(nrot_total, first, last);
        for(int k = first; k < last; k++){
            int i = k / nrot;
            int j = k % nrot;
            evaluator.rotate_vector(ct[i], steps[j], gal_keys, res[i][j + 1]);
        }
        MT_EXEC_RANGE_END
    }
    else if(NUM_THREADS <= nin_ctxts){
        MT_EXEC_RANGE(nin_ctxts, first, last);
        for(int i = first; i < last; i++){
            evaluator.rotate_vector_many(res[i][0], steps, gal_keys, res[i]);
        }
        MT_EXEC_RANGE_END
    }
    else{
        int nsubcols = (NUM_THREADS / nin_ctxts);
        int block_len = (int)ceil((double) nrot / nsubcols);
        int final_block_len = (nrot % block_len);
        if (final_block_len == 0) final_block_len = block_len;
        
        MT_EXEC_RANGE(NUM_THREADS, first, last);
        for(int n = first; n < last; ++n){
            int i = n % nin_ctxts;
            int l = (int) floor((double) n / (double) nin_ctxts);
            
            int nstart = l * block_len;
            int sub_len = (l == (nsubcols - 1) ? final_block_len : block_len);
            
            vector<int> new_steps;
            for (size_t j = nstart; j < nstart + sub_len; ++j) {
                new_steps.push_back(steps[j]);
            }
            
            vector<Ciphertext> ct_temp(sub_len + 1);
            ct_temp[0] = ct[i];
            evaluator.rotate_vector_many(res[i][0], new_steps, gal_keys, ct_temp);
            
            for (l = 1; l < sub_len + 1; ++l) {
                res[i][l + nstart] = ct_temp[l];
            }
        }
        MT_EXEC_RANGE_END
    }
}

/*
 @param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[out] res, rotated ciphertexts (size [2*nch][3*16] in B2; [4*nch][3*16] in B3)
 We generate the baby ciphertexts for each blocks (requiring 8 * 8 rotations).
 Here, the rotations are specified by "Param_conv3d::steps_conv[1]".
 If mode = baby, we pre-compute the rotated baby ciphertexts. (8 rotation for each input ctxt)
 If mode = giant, we pre-compute the rotated giant ciphertexts. (15 rotation for each input ctxt)
*/
void HECNNeval::generate_rotations_conv_babygiant_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block, string mode)
{
    int conv_block1 = (conv_block - 1);
    int nin_ctxts = ct.size();                  // = Param_conv2d::NUM_CHANNELS[conv_block1] / 16;
    
    // Compute the required rotation amounts
    vector<int> steps;
    int nrot;
    if(mode == "baby"){
        nrot = Param_conv2d::steps_size;        // 9 - 1 = 8
        steps.insert(steps.end(), Param_conv2d::steps_conv[conv_block1].begin(), Param_conv2d::steps_conv[conv_block1].end());
    } else if(mode == "giant"){
        nrot = Param_conv2d::steps_giant_size;  // 16 - 1 = 15
        steps.insert(steps.end(), Param_conv2d::steps_giant.begin(), Param_conv2d::steps_giant.end());
    }
    
    // Initialize
    // when nch=128, the size of [8][9] is reduced to [2][9] in B2; or the size of [16][9] is reduced to [1][9] in B3
    int nrot_total = nin_ctxts * nrot;    // actual number of required rotations
    res.resize(1, vector<vector<Ciphertext>>(nin_ctxts, vector<Ciphertext> (nrot + 1)));
    for(int i = 0; i < nin_ctxts; ++i){
        res[0][i][0] = ct[i];
    }
    
    int NUM_THREADS = Thread::availableThreads();
    
    if(nrot_total <= NUM_THREADS)   // Perform rotation each by each
    {
        MT_EXEC_RANGE(nrot_total, first, last);
        for(int k = first; k < last; k++){
            int i = k / nrot;    // 0 <= i < nin_ctxts
            int j = k % nrot;    // 0 <= j < nrot_unit = {0,1, ..., 7} or {0,1,...14}
            evaluator.rotate_vector(ct[i], steps[j], gal_keys, res[0][i][j + 1]);
        }
        MT_EXEC_RANGE_END
    }
    else if(NUM_THREADS <= nin_ctxts){
        MT_EXEC_RANGE(nin_ctxts, first, last);
        for(int i = first; i < last; i++){
            evaluator.rotate_vector_many(res[0][i][0], steps, gal_keys, res[0][i]);
        }
        MT_EXEC_RANGE_END
    }
    else{
        int nsubcols = (NUM_THREADS / nin_ctxts);  // number of subblocks in the column
        int block_len = (int)ceil((double) nrot / nsubcols);
        int final_block_len = (nrot % block_len);
        if (final_block_len == 0) final_block_len = block_len;
        
        // We divide the task into small many sub-tasks (nthreds/in),
        // and perform the computation on each sub-tasks in parallel
        MT_EXEC_RANGE(NUM_THREADS, first, last);
        for(int n = first; n < last; ++n){
            int i = n % nin_ctxts;    // 0 <= i < num_in
            int l = (int) floor((double) n / (double) nin_ctxts);
            
            int nstart = l * block_len;
            int sub_len = (l == (nsubcols - 1) ? final_block_len : block_len);
            
            vector<int> new_steps;
            for (size_t j = nstart; j < nstart + sub_len; ++j) {
                new_steps.push_back(steps[j]);
            }
            
            vector<Ciphertext> ct_temp(sub_len + 1);
            ct_temp[0] = ct[i];
            evaluator.rotate_vector_many(res[0][i][0], new_steps, gal_keys, ct_temp);
            
            for (l = 1; l < sub_len + 1; ++l) {
                res[0][i][l + nstart] = ct_temp[l];
            }
        }
        MT_EXEC_RANGE_END
    }
}

/*
 @param[in] ct, input ciphertexts (size [2*nch/16] in B2; [4*nch/16] in B3)
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[in] mode, the evaluation strategy of homomorphic convolution
 @param[out] res, rotated ciphertexts (size [2*nch][3*16] in B2; [4*nch][3*16] in B3)
 We generate the baby ciphertexts for each blocks (requiring 8 * 8 rotations).
 Here, the rotations are specified by "Param_conv3d::steps_conv[1]".
 If mode = baby, we pre-compute the rotated baby ciphertexts. (8 rotation for each input ctxt)
 If mode = giant, we pre-compute the rotated giant ciphertexts. (15 rotation for each input ctxt)
 This implementation does not use the hoisting technique.
*/
void HECNNeval::generate_rotations_conv_babygiant_wo_hoisting(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block, string mode)
{
    int conv_block1 = (conv_block - 1);
    int nin_ctxts = ct.size();
    
    vector<int> steps;
    int nrot;
    if(mode == "baby"){
        nrot = Param_conv2d::steps_size;
        steps.insert(steps.end(), Param_conv2d::steps_conv[conv_block1].begin(), Param_conv2d::steps_conv[conv_block1].end());
    } else if(mode == "giant"){
        nrot = Param_conv2d::steps_giant_size;
        steps.insert(steps.end(), Param_conv2d::steps_giant.begin(), Param_conv2d::steps_giant.end());
    }
    
    res.resize(nin_ctxts, vector<Ciphertext> (nrot + 1));
    for(int i = 0; i < nin_ctxts; ++i){
        res[i][0] = ct[i];
    }
    
    int NUM_THREADS = Thread::availableThreads();
    
    if(NUM_THREADS <= nin_ctxts){
        MT_EXEC_RANGE(nin_ctxts, first, last);
        for(int i = first; i < last; i++){
            for(int j = 0; j < steps.size(); ++j){
                evaluator.rotate_vector(res[i][0], steps[j], gal_keys, res[i][j + 1]);
            }
        }
        MT_EXEC_RANGE_END
    }
    else{
        int nsubcols = (NUM_THREADS / nin_ctxts);
        if(nsubcols > nrot){
            NUM_THREADS = nrot;
            nsubcols = nrot;
        }
        int block_len = (int)ceil((double) nrot / nsubcols);
        int final_block_len = (nrot % block_len);
        if (final_block_len == 0) final_block_len = block_len;
        
        MT_EXEC_RANGE(NUM_THREADS, first, last);
        for(int n = first; n < last; ++n){
            int i = n % nin_ctxts;
            int l = (int) floor((double) n / (double) nin_ctxts);
            
            int nstart = l * block_len;
            int sub_len = (l == (nsubcols - 1) ? final_block_len : block_len);
            
            vector<int> new_steps;
            for (size_t j = nstart; j < nstart + sub_len; ++j) {
                new_steps.push_back(steps[j]);
            }
            
            vector<Ciphertext> ct_temp(sub_len + 1);
            ct_temp[0] = ct[i];
            
            for(int j = 0; j < new_steps.size(); ++j){
                evaluator.rotate_vector(res[i][0], new_steps[j], gal_keys, ct_temp[j + 1]);
            }
            
            for (l = 1; l < sub_len + 1; ++l) {
                res[i][l + nstart] = ct_temp[l];
            }
        }
        MT_EXEC_RANGE_END
    }
}

/* -----------------------------------------------
 * Evaluation of BN-Act, pool or fully-connected layer
 * -----------------------------------------------
 */

/*
 @param[in] ct, intput ciphertexts that are scaled by the factor of q
 @param[in] poly, the plaintext polynomials of activation coefficients
 Update the "ct" by evaluating the coefficients of "poly".
 This implementation uses the native plaintext-ciphertext multiplication (multiply_plain_leveled)
*/
void HECNNeval::Eval_BN_ActPoly_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly){
    MT_EXEC_RANGE(ct.size(), first, last);
    for(int i = first; i < last; ++i){
        // ct^2 * poly[0]
        Ciphertext ct2;
        evaluator.square(ct[i], ct2);
        evaluator.relinearize_inplace(ct2, relin_keys);
        evaluator.rescale_to_next_inplace(ct2);   // rescale by q
        
        evaluator.mod_switch_to_inplace(poly[i][2], ct2.parms_id());
        evaluator.multiply_plain_leveled(ct2, poly[i][2], ct2);         // (q * m2) * (qc * poly[2])
        
        // ct^2 * poly[0] + ct * poly[1]
        evaluator.mod_switch_to_inplace(ct[i], ct2.parms_id());
        evaluator.mod_switch_to_inplace(poly[i][1], ct[i].parms_id());
        evaluator.multiply_plain_leveled(ct[i], poly[i][1], ct[i]);     // (q * m) * (qc * poly[1])
        
        ct[i].scale() = ct2.scale();
        evaluator.add_inplace(ct2, ct[i]);
        evaluator.rescale_to_next_inplace(ct2); // rescale by qc, then the resulting ciphertext is scaled by q
        
        // ct^2 * poly[0] + ct * poly[1] + poly[0]
        poly[i][0].scale() = ct2.scale();
        evaluator.mod_switch_to_inplace(poly[i][0], ct2.parms_id());
        evaluator.add_plain_leveled_inplace(ct2, poly[i][0]);
        ct[i] = ct2;
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] ct, intput ciphertexts that are scaled by the factor of q
 @param[in] poly, the plaintext polynomials of activation coefficients
 Update the "ct" by evaluating the coefficients of "poly".
 This implementation uses the fast plaintext-ciphertext multiplication (multiply_plain_leveled_fast)
*/
void HECNNeval::Eval_BN_ActPoly_Fast_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly){
    MT_EXEC_RANGE(ct.size(), first, last);
    for(int i = first; i < last; ++i){
        // ct^2 * poly[0]
        Ciphertext ct2;
        evaluator.square(ct[i], ct2);
        evaluator.relinearize_inplace(ct2, relin_keys);
        evaluator.rescale_to_next_inplace(ct2);   // rescale by q
        
        evaluator.mod_switch_to_inplace(poly[i][2], ct2.parms_id());
        evaluator.multiply_plain_leveled_fast(ct2, poly[i][2], ct2);      // (q * m2) * (qc * poly[2])
        
        // ct^2 * poly[0] + ct * poly[1]
        evaluator.mod_switch_to_inplace(ct[i], ct2.parms_id());
        evaluator.mod_switch_to_inplace(poly[i][1], ct[i].parms_id());
        evaluator.multiply_plain_leveled_fast(ct[i], poly[i][1], ct[i]);  // (q * m) * (qc * poly[1])
        
        ct[i].scale() = ct2.scale();
        evaluator.add_inplace(ct2, ct[i]);
        evaluator.rescale_to_next_inplace(ct2);  // rescale by qc, then the resulting ciphertext is scaled by q
        
        // ct^2 * poly[0] + ct * poly[1] + poly[0]
        poly[i][0].scale() = ct2.scale();
        evaluator.mod_switch_to_inplace(poly[i][0], ct2.parms_id());
        evaluator.add_plain_leveled_inplace(ct2, poly[i][0]);
        ct[i] = ct2;
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] ct, intput ciphertexts that are scaled by the factor of q
 @param[in] poly, the plaintext polynomials of activation coefficients
 Update the "ct" by evaluating the coefficients of "poly".
 This implementation does not use the lazy rescaling. 
*/
void HECNNeval::Eval_BN_ActPoly_Fast_inplace_wo_lazyres(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly){
    MT_EXEC_RANGE(ct.size(), first, last);
    for(int i = first; i < last; ++i){
        // ct^2 * poly[0]
        Ciphertext ct2;
        evaluator.square(ct[i], ct2);
        evaluator.relinearize_inplace(ct2, relin_keys);
        evaluator.rescale_to_next_inplace(ct2);   // rescale by q
        
        evaluator.mod_switch_to_inplace(poly[i][2], ct2.parms_id());
        evaluator.multiply_plain_leveled_fast(ct2, poly[i][2], ct2);      // (q * m2) * (qc * poly[2])
        evaluator.mod_switch_to_inplace(ct[i], ct2.parms_id());
        evaluator.rescale_to_next_inplace(ct2);                           // rescale by qc, (q * m2)
        
        // ct^2 * poly[0] + ct * poly[1]
        evaluator.mod_switch_to_inplace(poly[i][1], ct[i].parms_id());
        evaluator.multiply_plain_leveled_fast(ct[i], poly[i][1], ct[i]);  // (q * m) * (qc * poly[1])
        evaluator.rescale_to_next_inplace(ct[i]);                         // rescale by qc, (q * m)
        
        ct[i].scale() = ct2.scale();
        evaluator.add_inplace(ct2, ct[i]);
        
        // ct^2 * poly[0] + ct * poly[1] + poly[0]
        poly[i][0].scale() = ct2.scale();
        evaluator.mod_switch_to_inplace(poly[i][0], ct2.parms_id());
        evaluator.add_plain_leveled_inplace(ct2, poly[i][0]);
        ct[i] = ct2;
    }
    MT_EXEC_RANGE_END
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] conv_block, the indicator of the current block
 Update the "ct" by aggregating over slots.
*/
void HECNNeval::Eval_Avg_inplace(vector<Ciphertext> &ct, int conv_block)
{
    if(conv_block == 1){
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(ct[i], 1, gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], 16, gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
        }
        MT_EXEC_RANGE_END
    }
    else if(conv_block == 2){
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(ct[i], 2, gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], 32, gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
        }
        MT_EXEC_RANGE_END
    }
    else if(conv_block == 3){
        int row_shift_amount[] = {16 * 4, 16 * 4 * 2, 16 * 4 * 4};
        int column_shift_amount[] = {4, 8};
        
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(ct[i], row_shift_amount[0], gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], row_shift_amount[1], gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], row_shift_amount[2], gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], column_shift_amount[0], gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
            
            evaluator.rotate_vector(ct[i], column_shift_amount[1], gal_keys, ctemp);
            evaluator.add_inplace(ct[i], ctemp);
        }
        MT_EXEC_RANGE_END
    }
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] ker_poly, the encoded polynomials of the dense kernel, size [4*nch/16][16]
 @param[in] bias_poly, the encoded polynomials of the dense bias
 @param[in] nch, the number of channels at the current layer
 @param[out] res, output ciphertext
 We aggregate in a vertical direction followed by the single rescaling & rotation.
 Then we aggregate in a horizontal direction to get the final result.
 This implementation uses the native plaintext-ciphertext multiplication (multiply_plain_leveled)
 */
void HECNNeval::Eval_Dense(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch){
    int numi = nch/16;
    int num = 16 * numi;
    vector<vector<Ciphertext>> ctemp(numi, vector<Ciphertext> (16));
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int j = (n % 16);
        int i = (int) floor(n / 16.0);
        evaluator.multiply_plain_leveled(ct[i], ker_poly[i][j], ctemp[i][j]);
    }
    MT_EXEC_RANGE_END
    
    MT_EXEC_RANGE(16, first, last);
    for(int j = first; j < last; ++j){
        for(int i = 1; i < numi; ++i){
            evaluator.add_inplace(ctemp[0][j], ctemp[i][j]);    // Aggregate in a vertical direction
        }
        if(j != 0){
            evaluator.rotate_vector_inplace(ctemp[0][j], Param_conv2d::steps_giant[j - 1], gal_keys);
        }
    }
    MT_EXEC_RANGE_END
    
    res = ctemp[0][0];
    for(int j = 1; j < 16; ++j){
        evaluator.add_inplace(res, ctemp[0][j]);    // Aggregate in a horizontal direction
    }
    evaluator.rescale_to_next_inplace(res);
    
    // Add the bias
    bias_poly.scale() = res.scale();
    evaluator.add_plain_leveled_inplace(res, bias_poly);
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] ker_poly, the encoded polynomials of the dense kernel, size [4*nch/16][16]
 @param[in] bias_poly, the encoded polynomials of the dense bias
 @param[in] nch, the number of channels at the current layer
 @param[out] res, output ciphertext
 We aggregate in a vertical direction followed by the single rescaling & rotation.
 Then we aggregate in a horizontal direction to get the final result.
 This implementation uses the fast plaintext-ciphertext multiplication (multiply_plain_leveled_fast)
 */
void HECNNeval::Eval_Dense_Fast(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch){
    int numi = nch/16;
    int num = 16 * numi;
    vector<vector<Ciphertext>> ctemp(numi, vector<Ciphertext> (16));
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int j = (n % 16);
        int i = (int) floor(n / 16.0);
        evaluator.multiply_plain_leveled_fast(ct[i], ker_poly[i][j], ctemp[i][j]);
    }
    MT_EXEC_RANGE_END
    
    MT_EXEC_RANGE(16, first, last);
    for(int j = first; j < last; ++j){
        for(int i = 1; i < numi; ++i){
            evaluator.add_inplace(ctemp[0][j], ctemp[i][j]);    // Aggregate in a vertical direction
        }
        if(j != 0){
            evaluator.rotate_vector_inplace(ctemp[0][j], Param_conv2d::steps_giant[j - 1], gal_keys);
        }
    }
    MT_EXEC_RANGE_END
    
    res = ctemp[0][0];
    for(int j = 1; j < 16; ++j){
        evaluator.add_inplace(res, ctemp[0][j]);    // Aggregate in a horizontal direction
    }
    evaluator.rescale_to_next_inplace(res);
    
    // Add the bias
    bias_poly.scale() = res.scale();
    evaluator.add_plain_leveled_inplace(res, bias_poly);
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] ker_poly, the encoded polynomials of the dense kernel, size [4*nch/16][16]
 @param[in] bias_poly, the encoded polynomials of the dense bias
 @param[in] nch, the number of channels at the current layer
 @param[out] res, output ciphertext
 We aggregate in a vertical direction followed by the single rescaling & rotation.
 Then we aggregate in a horizontal direction to get the final result.
 This implementation uses the fast plaintext-ciphertext multiplication (multiply_plain_leveled_fast)
 We slightly changed "Eval_Dense_Fast" to make the dimension of intermediate ciphertexts from 2d to 1d,
 so we can reduce the memory usage during computation.
 */
void HECNNeval::Eval_Dense_Fast_Light(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch){
    int numi = nch/16;
    vector<Ciphertext> ctemp(16);
    
    MT_EXEC_RANGE(16, first, last);
    for(int j = first; j < last; ++j){
        evaluator.multiply_plain_leveled_fast(ct[0], ker_poly[0][j], ctemp[j]);
        
        Ciphertext temp;
        for(int i = 1; i < numi; ++i){
            evaluator.multiply_plain_leveled_fast(ct[i], ker_poly[i][j], temp);
            evaluator.add_inplace(ctemp[j], temp);    // Aggregate in a vertical direction
        }
       
        if(j != 0){
            evaluator.rotate_vector_inplace(ctemp[j], Param_conv2d::steps_giant[j - 1], gal_keys);
        }
    }
    MT_EXEC_RANGE_END
    
    res = ctemp[0];
    for(int j = 1; j < 16; ++j){
        evaluator.add_inplace(res, ctemp[j]);        // Aggregate in a horizontal direction
    }
    evaluator.rescale_to_next_inplace(res);
    
    // Add the bias
    bias_poly.scale() = res.scale();
    evaluator.add_plain_leveled_inplace(res, bias_poly);
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] ker_poly, the encoded polynomials of the dense kernel, size [4*nch/16][16]
 @param[in] bias_poly, the encoded polynomials of the dense bias
 @param[in] nch, the number of channels at the current layer
 @param[out] res, output ciphertext
 We aggregate in a vertical direction followed by the single rescaling & rotation.
 Then we aggregate in a horizontal direction to get the final result.
 This implementation does not use the lazy rescaling method.
 */
void HECNNeval::Eval_Dense_Fast_wo_lazyres(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch){
    int numi = nch/16;
    int num = 16 * numi;
    vector<vector<Ciphertext>> ctemp(numi, vector<Ciphertext> (16));
    
    MT_EXEC_RANGE(num, first, last);
    for(int n = first; n < last; ++n){
        int j = (n % 16);
        int i = (int) floor(n / 16.0);
        evaluator.multiply_plain_leveled_fast(ct[i], ker_poly[i][j], ctemp[i][j]);
        evaluator.rescale_to_next_inplace(ctemp[i][j]);
    }
    MT_EXEC_RANGE_END
    
    MT_EXEC_RANGE(16, first, last);
    for(int j = first; j < last; ++j){
        for(int i = 1; i < numi; ++i){
            evaluator.add_inplace(ctemp[0][j], ctemp[i][j]);
        }
        if(j != 0){
            evaluator.rotate_vector_inplace(ctemp[0][j], Param_conv2d::steps_giant[j - 1], gal_keys);
        }
    }
    MT_EXEC_RANGE_END
    
    res = ctemp[0][0];
    for(int j = 1; j < 16; ++j){
        evaluator.add_inplace(res, ctemp[0][j]);
    }
   
    bias_poly.scale() = res.scale();
    evaluator.add_plain_leveled_inplace(res, bias_poly);
}

/*
 @param[in] ct, intput ciphertexts
 @param[in] zero_one_poly, the plaintext polynomial with 0-1 values, (used for a pre-processing step)
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[out] res, output ciphertexts
 */
void HECNNeval::interlace_ctxts(vector<Ciphertext> &res, vector<Ciphertext> &ct, Plaintext zero_one_poly, int conv_block)
{
    if(conv_block == 2){
        // Input: ct[0], ..., ct[7]
        // Zero-out the garbage values to combine
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            evaluator.multiply_plain_leveled_fast(ct[i], zero_one_poly, ct[i]);
            //evaluator.rescale_to_next_inplace(ct[i]);
        }
        MT_EXEC_RANGE_END
       
        // Case1 (nch=128): Input ct = {c[0], ..., c[7]} => combine (c[0], ..., c[3]), (c[4], ..., c[7])
        // Case2 (nch=64): Input ct = {c[0], ..., c[3]} => combine (c[0], ..., c[3])
        if(ct.size() == 8){
            MT_EXEC_RANGE(6, first, last);
            for(int i = first; i < last; ++i){
                if(i == 0){
                    evaluator.rotate_vector_inplace(ct[1], -1, gal_keys);
                } else if(i == 1){
                    evaluator.rotate_vector_inplace(ct[2], -16, gal_keys);
                } else if(i == 2){
                    evaluator.rotate_vector_inplace(ct[3], -17, gal_keys);
                } else if(i == 3){
                    evaluator.rotate_vector_inplace(ct[5], -1, gal_keys);
                } else if(i == 4){
                    evaluator.rotate_vector_inplace(ct[6], -16, gal_keys);
                } else if(i == 5){
                    evaluator.rotate_vector_inplace(ct[7], -17, gal_keys);
                }
            }
            MT_EXEC_RANGE_END
            
            res.resize(2);
            MT_EXEC_RANGE(2, first, last);
            for(int i = first; i < last; ++i){
                res[i] = ct[4 * i];
                for(int j = 1; j < 4; ++j){
                    evaluator.add_inplace(res[i], ct[4 * i + j]);
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else if(ct.size() == 4){
            MT_EXEC_RANGE(3, first, last);
            for(int i = first; i < last; ++i){
                if(i == 0){
                    evaluator.rotate_vector_inplace(ct[1], -1, gal_keys);
                } else if(i == 1){
                    evaluator.rotate_vector_inplace(ct[2], -16, gal_keys);
                } else if(i == 2){
                    evaluator.rotate_vector_inplace(ct[3], -17, gal_keys);
                }
            }
            MT_EXEC_RANGE_END
            
            res.resize(1);
            res[0] = ct[0];
            for(int j = 1; j < 4; ++j){
                evaluator.add_inplace(res[0], ct[j]);
            }
            evaluator.rescale_to_next_inplace(res[0]);
        }
        else if(ct.size() == 2){
            evaluator.rotate_vector_inplace(ct[1], -1, gal_keys);
            res.resize(1);
            res[0] = ct[0];
            evaluator.add_inplace(res[0], ct[1]);
            evaluator.rescale_to_next_inplace(res[0]);
        }
    }
    else if(conv_block == 3){
        // Input: ct[0], ..., ct[15]
        // Zero-out the garbage values to combine
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            evaluator.multiply_plain_leveled_fast(ct[i], zero_one_poly, ct[i]);
        }
        MT_EXEC_RANGE_END
        
        MT_EXEC_RANGE(ct.size() - 1, first, last);
        for(int n = first; n < last; ++n){
            int n1 = n + 1;
            int rot_amount = - ((n1 % 4) + 16 * (int) floor((double) n1 / (double) 4));
            evaluator.rotate_vector_inplace(ct[n + 1], rot_amount, gal_keys);   // ct[1],...,ct[15]
        }
        MT_EXEC_RANGE_END
    
        res.resize(1);
        res[0] = ct[0];
        for(int j = 1; j < ct.size(); ++j){
            evaluator.add_inplace(res[0], ct[j]);
        }
        evaluator.rescale_to_next_inplace(res[0]);
    }
}


/*
 @param[in] ct, intput ciphertexts
 @param[in] zero_one_poly, the plaintext polynomial with 0-1 values, (used for a pre-processing step)
 @param[in] conv_block, the indicator of the current block, (e.g. 2 for B2; 3 for B3)
 @param[out] res, output ciphertexts
 This implementation does not use the lazy rescaling method.
 */
void HECNNeval::interlace_ctxts_wo_lazyres(vector<Ciphertext> &res, vector<Ciphertext> &ct, Plaintext zero_one_poly, int conv_block)
{
    if(conv_block == 2){
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            evaluator.multiply_plain_leveled_fast(ct[i], zero_one_poly, ct[i]);
            evaluator.rescale_to_next_inplace(ct[i]);
        }
        MT_EXEC_RANGE_END
        
        MT_EXEC_RANGE(6, first, last);
        for(int i = first; i < last; ++i){
            if(i == 0){
                evaluator.rotate_vector_inplace(ct[1], -1, gal_keys);
            } else if(i == 1){
                evaluator.rotate_vector_inplace(ct[2], -16, gal_keys);
            } else if(i == 2){
                evaluator.rotate_vector_inplace(ct[3], -17, gal_keys);
            } else if(i == 3){
                evaluator.rotate_vector_inplace(ct[5], -1, gal_keys);
            } else if(i == 4){
                evaluator.rotate_vector_inplace(ct[6], -16, gal_keys);
            } else if(i == 5){
                evaluator.rotate_vector_inplace(ct[7], -17, gal_keys);
            }
        }
        MT_EXEC_RANGE_END
        
        res.resize(2);
        MT_EXEC_RANGE(2, first, last);
        for(int i = first; i < last; ++i){
            res[i] = ct[4 * i];
            for(int j = 1; j < 4; ++j){
                evaluator.add_inplace(res[i], ct[4 * i + j]);
            }
        }
        MT_EXEC_RANGE_END
    }
    else if(conv_block == 3){
        MT_EXEC_RANGE(ct.size(), first, last);
        for(int i = first; i < last; ++i){
            evaluator.multiply_plain_leveled_fast(ct[i], zero_one_poly, ct[i]);
            evaluator.rescale_to_next_inplace(ct[i]);
        }
        MT_EXEC_RANGE_END
        
        MT_EXEC_RANGE(15, first, last);
        for(int n = first; n < last; ++n){
            int n1 = n + 1;
            int rot_amount = - ((n1 % 4) + 16 * (int) floor((double) n1 / (double) 4));
            evaluator.rotate_vector_inplace(ct[n + 1], rot_amount, gal_keys);   // ct[1]~ct[15]
        }
        MT_EXEC_RANGE_END
        
        res.resize(1);
        res[0] = ct[0];
        for(int j = 1; j < 16; ++j){
            evaluator.add_inplace(res[0], ct[j]);
        }
    }
}

/* -----------------------------------------------
 * Functions for Debugging
 -----------------------------------------------
 */

void HECNNeval::Debug_encrypt(Ciphertext res)
{
    vector<double> dmsg;
    hecnnenc.decrypt_vector(dmsg, res); // [32][16]
    
    for(int i = 0; i < (32 * 1); i++){
        for(int j = 0; j < 16; j ++){  // one row
            cout << dmsg[i * 16 + j]  << " ";
        }
        if((((i + 1) % 32) == 0)){
            cout << endl;
            cout << "=================================================" << endl;
        } else{
            cout << endl;
        }
    }
}

void HECNNeval::Debug_Conv1(vector<Ciphertext> res1, string param_dir)
{
    int bias_index = 0;
    dvec conv_bias1;    // [128]
    read_onecol(conv_bias1, param_dir + "conv1.bias_torch.Size([128]).csv");
    
    for(int l = 0; l < 1; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res1[l]); // [32][16]
        for(int i = 0; i < (32 * 2); i++){
            for(int j = 0; j < 16; j ++){  // one row
                cout << dmsg[i * 16 + j] + conv_bias1[bias_index] << " ";
            }
            if((((i + 1) % 32) == 0)){
                bias_index++;
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_Act1(vector<Ciphertext> res1)
{
    for(int l = 0; l < 1; l++){ // num_out; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res1[l]); // [32][16]
        for(int i = 0; i < (32 * 2); i++){
            for(int j = 0; j < 16; j ++){  // one row
                cout << dmsg[i * 16 + j] << " ";
            }
            if((((i + 1) % 32) == 0)){
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_B1(vector<Ciphertext> res1, int NB_CHANNELS)
{
    cout << "B1 = " << endl;
    int num_out = NB_CHANNELS / 16;   // 8
    
    for(int l = 0; l < num_out; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res1[l]); // the first 16 many 2D-(16*7)
        //output_with_file(dmsg, Param_conv2d::dir + "res/HE_conv1_output.txt", 16);
        
        for(int i = 0; i < (32 * 1); i+=2){    // 32 * 16
            for(int j = 0; j < 14; j+=2){  // one row
                cout << dmsg[i * 16 + j]  << " ";
            }
            if((((i + 2) % 32) == 0)){
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_Conv2(vector<Ciphertext> res2, string param_dir)
{
    int bias_index = 0;
    dvec conv_bias2;    // [128]
    read_onecol(conv_bias2, param_dir + "conv2.bias_torch.Size([256]).csv");
    
    for(int l = 0; l < 1; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res2[l]); // the first 16 many 2D-(16*7)
        
        for(int i = 0; i < (32); i+=2){    // * 16
            for(int j = 0; j < 14; j+=2){  // one row
                cout << dmsg[i * 16 + j] + conv_bias2[bias_index] << " ";
            }
            
            if((((i + 2) % 32) == 0)){
                bias_index++;
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_B2(vector<Ciphertext> res2)
{
    cout << "B2 = "  << endl;
    for(int l = 0; l < 1; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res2[l]); // the first 16 many 2D-(8*3)
        
        for(int i = 0; i < (32 * 1); i+=4){ // 32*16: whole
            for(int j = 0; j < 4 * 3; j+=4){  // one row: 0, 4, 8
                cout << dmsg[i * 16 + j]  << " ";
            }
            
            if((((i + 4) % 32) == 0)){
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_Conv3(vector<Ciphertext> res3, string param_dir)
{
    int bias_index = 0;
    dvec conv_bias3;    // [128]
    read_onecol(conv_bias3, param_dir + "conv3.bias_torch.Size([512]).csv");
    
    for(int l = 0; l < 1; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res3[l]); // the first 16 many 2D-(8*3)
        
        for(int i = 0; i < (32 * 1); i+=4){ // 32 * 16
            for(int j = 0; j < 4 * 3; j+=4){  // one row: 0, 4, 8
                cout << dmsg[i * 16 + j] + conv_bias3[bias_index] << " ";
            }
            
            if((((i + 4) % 32) == 0)){
                bias_index++;
                cout << endl;
                cout << "=================================================" << endl;
            } else{
                cout << endl;
            }
        }
    }
}

void HECNNeval::Debug_B3(vector<Ciphertext> res3)
{
    cout << "B3 = "  << endl;
    for(int l = 0; l < 1; l++){
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res3[l]); // the first 16 many 2D-(8*3)
        
        for(int i = 0; i < (32 * 16 * 16); i+=(32 * 16)){
            cout << dmsg[i] << endl;
        }
    }
}
