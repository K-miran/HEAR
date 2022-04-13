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
#include <random>
#include <iomanip>
#include <unistd.h>

#include "utils.h"
#include "thread.h"
#include "HECNNevaluator_conv1d.h"
#include "HECNNevaluator_conv2d.h"
#include "Test_HEAR.h"

#define DEBUG false
#define unit 200

using namespace std;

/*
 @param[in] test_input, [608][3*32*15]
 @param[in] id_st, the starting id index for the testing sample
 @param[in] id_end, the ending id index for the testing sample
 @param[in] ker[3][out_nchannels][in_nchannels][3]
 in_nchannels is the number of in-channles of filters/kernels tensors
 out_nchannels is the number of out-channles of filters/kernels tensors
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT, [3][*][*]
 @param[in] dense_ker, the kernel of dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation option for the 1st conv layer
 @param[in] mode2, the evaluation option for the 2nd conv layer
 @param[in] mode3, the evaluation option for the 3rd conv layer
 @param[in] nch, the number of channels at the 1st conv layer
 @param[in] method, the underlying method for homomorphic evaluation of the convolution
 @param[out] output, the predicted result, [id_end-id_st+1][10]
 Note: the inference times are recorded as "time_conv1d_nch.txt".
       the detailed memory usages are recorded as "memory_conv1d_nch.txt".
       the detailed timing for convolutions are recorded as "convtime_conv1d_nch.txt".
 */

void TestHEAR1d::hecnn(dmat &output, dmat test_input, int id_st, int id_end,
                       vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                       string mode1, string mode2, string mode3, int nch, string method)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::high_resolution_clock::time_point time_start_detail, time_end_detail;
 
    string filename_time;
    string filename_memory;
    string filename_convtime;
    
    filename_time = "result/time_conv1d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
    filename_memory = "result/memory_conv1d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
    filename_convtime = "result/convtime_conv1d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
    
    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfc;
    outfc.open(filename_convtime.c_str(), fstream::in | fstream::out | fstream::app);
 
    vector<vector<double>> eval_times;
    vector<vector<double>> conv_times; // (conv2-pre,conv2-conv,conv2-post,conv3-pre,conv3-conv,conv3-post)
    vector<vector<double>> memory;

    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    vector<int> bit_sizes_vec;
    if(method == "hear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_HEAR::logq0, Param_HEAR::logq, Param_HEAR::logqc,
                          Param_HEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);    // n = degree
        parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
    } else if(method == "fhear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_FHEAR::logq0, Param_FHEAR::logq, Param_FHEAR::logqc,
                          Param_FHEAR::logqc_small, Param_FHEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_FHEAR::poly_modulus_degree);    // n = degree
        parms.set_coeff_modulus(CoeffModulus::Create(Param_FHEAR::poly_modulus_degree, bit_sizes_vec));
    }
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    vector<int> steps_all;  // the required rotations
    steps_all.insert(steps_all.end(), Param_conv1d::steps_giant.begin(), Param_conv1d::steps_giant.end());
    steps_all.insert(steps_all.end(), Param_conv1d::steps_conv[0].begin(), Param_conv1d::steps_conv[0].end());
    steps_all.insert(steps_all.end(), Param_conv1d::steps_conv[1].begin(), Param_conv1d::steps_conv[1].end());
    steps_all.insert(steps_all.end(), Param_conv1d::steps_conv[2].begin(), Param_conv1d::steps_conv[2].end());
    steps_all.insert(steps_all.end(), Param_conv1d::steps_pool.begin(), Param_conv1d::steps_pool.end());
    
    if(method == "fhear"){
        steps_all.insert(steps_all.end(), Param_conv1d::steps_interlacing.begin(), Param_conv1d::steps_interlacing.end());
    }
    GaloisKeys gal_keys = keygen.galois_keys(steps_all);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
//    std::cout << "| --->  poly_modulus_degree (n): " << context_data -> parms().poly_modulus_degree() << std::endl;
//    std::cout << "| --->  coeff_modulus size (logQ): ";
//    std::cout << context_data->total_coeff_modulus_bit_count() << endl;
//    print_modulus_chain(bit_sizes_vec);
   
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Prepare Network: " ;
//    cout << "+------------------------------------+" << endl;

    time_start = chrono::high_resolution_clock::now();
    
    CKKSEncoder encoder(context);
    HECNNenc1d hecnnenc(encryptor, decryptor, encoder);

    vector<vector<vector<Plaintext>>> ker_poly_1;   // [nch/16][2][3]
    vector<vector<Plaintext>> act_poly_1;           // [nch/16][3], here 3 is degree of the approximate activation

    vector<vector<vector<Plaintext>>> ker_poly_2;   // [2*nch/16][128/4][3]
    vector<vector<Plaintext>> act_poly_2;           // [2*nch/16][3]

    vector<vector<vector<Plaintext>>> ker_poly_3;   // [4*nch/16][256/16][3]
    vector<vector<Plaintext>> act_poly_3;           // [4*nch/16][3]

    vector<vector<Plaintext>> dense_ker_poly;
    Plaintext dense_bias_poly;
    vector<Plaintext> zero_one_poly;

    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    if(method == "hear"){
        hecnnenc.prepare_network(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                dense_ker_poly, dense_bias_poly,
                                ker, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, NB_CHANNELS);
    } else if(method == "fhear") {
        hecnnenc.prepare_network_interlaced(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                            dense_ker_poly, dense_bias_poly, zero_one_poly,
                                            ker, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, NB_CHANNELS, bit_sizes_vec);
    }

    cout << "(done) " ;
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    int ntest = id_end + 1 - id_st;
    vector<Ciphertext> xCipher(ntest);
    
    MT_EXEC_RANGE(ntest, first, last);
    for(int i = first; i < last; ++i){
        int id = i + id_st;
        dten test_sample; // the input tensor [2][32][15] = c * h * w
        reshape(test_sample, test_input[id], 3, 32, 15);
        
        if(method == "hear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_HEAR::qscale);
        } else if(method == "fhear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_FHEAR::qscale);
        }
    }
    MT_EXEC_RANGE_END
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << ",";
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
//    cout << "|             Evaluation             |" << endl;
//    cout << "+------------------------------------+" << endl;
        
    HECNNeval1d hecnneval(evaluator, relin_keys, gal_keys, hecnnenc);
    output.resize(ntest, vector<double> (10));    // clarify the size of output
    
    vector<Ciphertext> res;                     // ciphertexts for intermediate results; [8] -> [16] -> [32] during inference for nch=128
    vector<vector<vector<Ciphertext>>> rot;
    Ciphertext *block = new Ciphertext [NB_CHANNELS[0] * NB_CHANNELS[1]/16];
    
    for(int t = 0; t < ntest; t++){
//        cout << "+------------------------------+" << endl;
//        cout << "|        Evaluation (B1)       |" << endl;
//        cout << "+------------------------------+" << endl;
        
        vector<double> detailed_eval_times;
        vector<double> detailed_memory;

        chrono::microseconds time_total_eval(0);
        
        cout << "=========(" << t + id_st << ")=========" << endl;
        cout << "> Evaluation (B1): " ;
        
        time_start = chrono::high_resolution_clock::now();
        
        // compute the hyper-parameters
        int nin_ctxts = NB_CHANNELS[0];          // 2 = nrows of blocks (in),
        int nout_ctxts = NB_CHANNELS[1] / 16;    // 8 = ncols of blocks (total number of independent ciphertexts), (out)
        int nblocks = nin_ctxts * nout_ctxts;   // 2 * 8 = 16
        int num = (nin_ctxts * nout_ctxts * Param_conv1d::DIM_FILTERS);   // 2 * 8 * 9 = 144
        res.resize(nout_ctxts);
        
        rot.resize(1, vector<vector<Ciphertext>> (1, vector<Ciphertext> (1)));
        hecnneval.generate_rotations_conv1(rot[0][0], xCipher[t], mode1);
        
        if(mode1 == "fully"){
            // perform scalar multiplication on each blocks
            MT_EXEC_RANGE(num, first, last);
            for(int n = first; n < last; ++n){  // 0 <= k < 3, 0 <= l < nin * nout, 0 <= i < 2, 0 <= j < 8
                int k = (n % Param_conv1d::DIM_FILTERS);
                int l = (int) floor(n / Param_conv1d::DIM_FILTERS);
                int i = (l % nin_ctxts);
                int j = (int) floor((double)l / (double)nin_ctxts);
                int i1 = (i * 3);
                
                Ciphertext ctemp;
                if(k == 1){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1], ker_poly_1[j][i][1], ctemp);
                } else if(k == 0){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + 1], ker_poly_1[j][i][0], ctemp);
                } else if(k == 2){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + 2], ker_poly_1[j][i][2], ctemp);
                }
                
                // aggregate in a horizontal direction: temp[l][0] += temp[l][1] + ... + temp[l][8]
                if (i == 0) {
                    if(k == 0) {
                        res[j] = ctemp;
                    } else{
                        evaluator.add_inplace(res[j], ctemp);
                    }
                } else{
                    if(k == 0) {
                        block[j] = ctemp;
                    } else{
                        evaluator.add_inplace(block[j], ctemp);
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        else if(mode1 == "baby"){
            MT_EXEC_RANGE(nblocks, first, last);    // nblocks = 2*8 = 16
            for(int l = first; l < last; ++l){
                int i = (l % nin_ctxts);            // 0 <= i < 2
                int j = (int) floor((double)l / (double)nin_ctxts); // 0 <= j < 8
               
                for(int k = 0; k < Param_conv1d::DIM_FILTERS; ++k){
                    Ciphertext ctemp;
                    if(k == 1){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][0], ker_poly_1[j][i][1], ctemp);
                    } else if(k == 0){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][1], ker_poly_1[j][i][0], ctemp);
                    } else if(k == 2){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][2], ker_poly_1[j][i][2], ctemp);
                    }
                   
                    if (i == 0) {
                        if(k == 0) {
                            res[j] = ctemp;
                        } else{
                            evaluator.add_inplace(res[j], ctemp);
                        }
                    } else {
                        if(k == 0) {
                            block[j] = ctemp;
                        } else{
                            evaluator.add_inplace(block[j], ctemp);
                        }
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        
        // aggregate the temporary results to res
        MT_EXEC_RANGE(nout_ctxts, first, last);
        for(int j = first; j < last; ++j){
            if(mode1 == "baby") {
                evaluator.rotate_vector_inplace(block[j], Param_conv1d::shift, gal_keys, MemoryPoolHandle().New(false));
            }
            evaluator.add_inplace(res[j], block[j]);
            evaluator.rescale_to_next_inplace(res[j]);     // rescale by qc
        }
        MT_EXEC_RANGE_END
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;

        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B2): ";
//    cout << "+------------------------------------+" << endl;
        
        time_start = chrono::high_resolution_clock::now();
        time_start_detail = chrono::high_resolution_clock::now();
        vector<double> detailed_conv_times;
        
        // Step 2.1. Pre-processing step
        // res_packed = res[0] + rho(res[1], -1) + rho(res[2], -16) + rho(res[3], -17)
        vector<Ciphertext> res_packed;          // 128/16 = 8 -> 8/4 = 2 when nch=128
        
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[1] / 16;
        } else if(method == "fhear"){
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[0], 2);
            nin_ctxts = res_packed.size();
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        nout_ctxts = NB_CHANNELS[2] / 16;
        rot.clear();
        
        // Step 2.2. Ordinary homomorphic convolutions
        // Fist, generate the rotated ciphertexts of input
        // Second, multiply the rotated ciphertext by the kernel plaintexts.
        time_start_detail = chrono::high_resolution_clock::now();
        if(mode2 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 2);
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 2);
            }
            res.resize(nout_ctxts);
        
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 3 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][1], res[i]);
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        evaluator.multiply_plain_leveled_fast(rot[k][j][1], ker_poly_2[i][j0][0], ctemp);
                        evaluator.add_inplace(res[i], ctemp);
                        evaluator.multiply_plain_leveled_fast(rot[k][j][2], ker_poly_2[i][j0][2], ctemp);
                        evaluator.add_inplace(res[i], ctemp);
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 2, mode2); // [8][3]
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 2, mode2); // [2][3]
            }
            res.resize(nout_ctxts);
        
            if(mode2 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][1], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            evaluator.multiply_plain_leveled_fast(rot[0][k][1], ker_poly_2[i][j0][0], ctemp);
                            evaluator.add_inplace(ctempj, ctemp);
                            evaluator.multiply_plain_leveled_fast(rot[0][k][2], ker_poly_2[i][j0][2], ctemp);
                            evaluator.add_inplace(ctempj, ctemp);
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv1d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false)); // rho(j*shift)
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            } else if(mode2 == "giant"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv1d::steps_size + 1; l++){ // 0 <= l < 9
                        int ker_index;
                        switch(l){
                            case 0:
                                ker_index = 1;
                                break;
                            case 1:
                                ker_index = 0;
                                break;
                            case 2:
                                ker_index = 2;
                                break;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctempl); // ct0 * k4
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctemp);  // ct0 * k4
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv1d::steps_conv[1][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        // Step 2.3. Post-processing
        // Aggregate across the slots (using rot-and-sum): ct + rho(ct; 1)
        time_start_detail = chrono::high_resolution_clock::now();
        if(method == "fhear"){
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                evaluator.add_inplace(res[i], ctemp);
            }
            MT_EXEC_RANGE_END
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B3): ";
 //    cout << "+------------------------------------+" << endl;
        time_start = chrono::high_resolution_clock::now();
        time_start_detail = chrono::high_resolution_clock::now();
        
        nout_ctxts = NB_CHANNELS[3] / 16;   // 512/16 = 32 (out) for nch=128
        
        // Step 3.1: Pre-processing
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[2] / 16;
        } else if(method == "fhear"){
            // first combine ciphertexts: res_packed = res[0] + rho(res[1], -1) + rho(res[2], -16) + rho(res[3], -17)
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[1], 3);
            nin_ctxts = res_packed.size();
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_start_detail = chrono::high_resolution_clock::now();
        rot.clear();
        
        // Step 3.2. Ordinary homomorphic convolutions
        // Fist, generate the rotated ciphertexts of input
        // Multiply the rotated ciphertext by the kernel plaintexts.
        if(mode3 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 3);
            }
            res.resize(nout_ctxts);
            
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 9 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][1], res[i]);
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        
                        evaluator.multiply_plain_leveled_fast(rot[k][j][1], ker_poly_3[i][j0][0], ctemp);
                        evaluator.add_inplace(res[i], ctemp);
                        evaluator.multiply_plain_leveled_fast(rot[k][j][2], ker_poly_3[i][j0][2], ctemp);
                        evaluator.add_inplace(res[i], ctemp);
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        } else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 3, mode3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 3, mode3);
            }
            res.resize(nout_ctxts);
            
            if(mode3 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][1], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            evaluator.multiply_plain_leveled_fast(rot[0][k][1], ker_poly_3[i][j0][0], ctemp);
                            evaluator.add_inplace(ctempj, ctemp);
                            evaluator.multiply_plain_leveled_fast(rot[0][k][2], ker_poly_3[i][j0][2], ctemp);
                            evaluator.add_inplace(ctempj, ctemp);
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv1d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false)); // rho(j*shift)
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
            else if(mode3 == "giant"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv1d::steps_size + 1; l++){ // 0 <= l < 3
                        int ker_index;
                        switch(l){
                            case 0:
                                ker_index = 1;
                                break;
                            case 1:
                                ker_index = 0;
                                break;
                            case 2:
                                ker_index = 2;
                                break;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctempl);
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctemp);
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv1d::steps_conv[2][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
            
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_start_detail = chrono::high_resolution_clock::now();
        
        // Step 3.3: post-processing
        if(method == "fhear"){
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                evaluator.add_inplace(res[i], ctemp);

                evaluator.rotate_vector(res[i], 2, gal_keys, ctemp, MemoryPoolHandle().New(false));
                evaluator.add_inplace(res[i], ctemp);
            }
            MT_EXEC_RANGE_END
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;

        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_3);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 3);  // res3[0], ,,, res3[31]
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Dense... " ;
//    cout << "+------------------------------------+" << endl;
        
        time_start = chrono::high_resolution_clock::now();
        
        Ciphertext res_prediction;
        hecnneval.Eval_Dense_Fast_Light(res_prediction, res, dense_ker_poly, dense_bias_poly, NB_CHANNELS[3]);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        time_total_eval += time_diff;
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        cout << ">> Total Eval Time = [" << time_total_eval.count()/1000000.0 << " s] " << endl;
        detailed_eval_times.push_back(time_total_eval.count()/1000000.0) ;

//    cout << "+------------------------------------+" << endl;
        cout << ">  Decryption... " ;
//    cout << "+------------------------------------+" << endl;
        time_start = chrono::high_resolution_clock::now();
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res_prediction);
        
        int j = 0;
        for(int i = 0; i < (32 * 16 * 10); i+=(32 * 16)){
            output[t][j] = dmsg[i];
            j++;
        }
                
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv1d::memoryscale));
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        
        // output probabilities
        for(int i = 0; i < 9; i++){
            cout << output[t][i] * 10.0 << ",";
        }
        cout << output[t][9] * 10.0 << endl;
                
        eval_times.push_back(detailed_eval_times);
        memory.push_back(detailed_memory);
        conv_times.push_back(detailed_conv_times);
        
        res.clear();
        res.shrink_to_fit();
        
        rot.clear();
        rot.shrink_to_fit();
    }

    // Write time and memory
    for(int t = 0; t < ntest; t++){
        for(int i = 0; i < eval_times[0].size() - 1; i++){
            outf << eval_times[t][i] << ",";
            outfm << memory[t][i] << ",";
        }

        outf << eval_times[t][eval_times[0].size() - 1] << endl;
        outfm << memory[t][eval_times[0].size() - 1] << endl;

        for(int i = 0; i < conv_times[0].size() - 1; i++){
            outfc << conv_times[t][i] << ",";
        }
        outfc << conv_times[t][conv_times[0].size()  - 1] << endl;
    }

    outf.close();
    outfm.close();
    outfc.close();
}

/*
 @param[in] test_input, [608][3*32*15]
 @param[in] id_st, the starting id index for the testing sample
 @param[in] id_end, the ending id index for the testing sample
 @param[in] ker[3][out_nchannels][in_nchannels][3][3]
 in_nchannels is the number of in-channles of filters/kernels tensors
 out_nchannels is the number of out-channles of filters/kernels tensors
 @param[in] real_poly, the coefficients of polynomials of conv_bias/BN/ACT, [3][*][*]
 @param[in] dense_ker, the kernel of dense layer, [10][4*nch]
 @param[in] dense_bias, the bias of the dense layer, [10]
 @param[in] mode1, the evaluation option for the 1st conv layer
 @param[in] mode2, the evaluation option for the 2nd conv layer
 @param[in] mode3, the evaluation option for the 3rd conv layer
 @param[in] nch, the number of channels at the 1st conv layer
 @param[in] method, the underlying method for homomorphic evaluation of the convolution
 @param[out] output, the predicted result, [id_end-id_st+1][10]
 Note: the inference times are recorded as "time_conv2d_nch_method_mode2.txt".
       the detailed memory usages are recorded as "memory_conv2d_nch_method_mode2.txt".
       the detailed timing for convolutions are recorded as "convtime_conv2d_nch_method_mode2.txt".
 */

void TestHEAR2d::hecnn(dmat &output, dmat test_input, int id_st, int id_end,
                       vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                       string mode1, string mode2, string mode3, int nch, string method)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::high_resolution_clock::time_point time_start_detail, time_end_detail;
    
//#if 0  // (for keygen, encoding)
//    string time = "result/fig4/time_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt"; // (keygen,encoding)
//    fstream outf;
//    outf.open(time.c_str(), fstream::in | fstream::out | fstream::app);
//
//    string peak_memory = "result/fig4/memory_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt"; // (keygen,encoding,encrypt,evaluation,decrypt)
//    fstream outfm;
//    outfm.open(peak_memory.c_str(), fstream::in | fstream::out | fstream::app);
//#endif
//
    string filename_time;
    string filename_memory;
    string filename_convtime;

    if((method == "hear") && (nch == 128) && (mode2 == "fully")){
        int resid = id_st/unit;
        filename_time = "result/time_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + "_" + to_string(resid) + ".txt";
        filename_memory = "result/memory_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + "_" + to_string(resid) + ".txt";
        filename_convtime = "result/convtime_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + "_" + to_string(resid) + ".txt";
    } else{
        filename_time = "result/time_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
        filename_memory = "result/memory_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
        filename_convtime = "result/convtime_conv2d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
    }

    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);

    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);

    fstream outfc;
    outfc.open(filename_convtime.c_str(), fstream::in | fstream::out | fstream::app);

    vector<vector<double>> eval_times;
    vector<vector<double>> conv_times; // (conv2-pre,conv2-conv,conv2-post, conv3-pre,conv3-conv,conv3-post)
    vector<vector<double>> memory;

    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    vector<int> bit_sizes_vec;
    if(method == "hear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_HEAR::logq0, Param_HEAR::logq, Param_HEAR::logqc,
                          Param_HEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
    } else if(method == "fhear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_FHEAR::logq0, Param_FHEAR::logq, Param_FHEAR::logqc,
                          Param_FHEAR::logqc_small, Param_FHEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_FHEAR::poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(Param_FHEAR::poly_modulus_degree, bit_sizes_vec));
    }
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    vector<int> steps_all;  // the required rotations
    steps_all.insert(steps_all.end(), Param_conv2d::steps_giant.begin(), Param_conv2d::steps_giant.end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[0].begin(), Param_conv2d::steps_conv[0].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[1].begin(), Param_conv2d::steps_conv[1].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[2].begin(), Param_conv2d::steps_conv[2].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_pool.begin(), Param_conv2d::steps_pool.end());

    if(method == "fhear"){
        steps_all.insert(steps_all.end(), Param_conv2d::steps_interlacing.begin(), Param_conv2d::steps_interlacing.end());
    }
    GaloisKeys gal_keys = keygen.galois_keys(steps_all);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
//    std::cout << "| --->  poly_modulus_degree (n): " << context_data -> parms().poly_modulus_degree() << std::endl;
//    std::cout << "| --->  coeff_modulus size (logQ): ";
//    std::cout << context_data->total_coeff_modulus_bit_count() << endl;
//    print_modulus_chain(bit_sizes_vec);
   
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Prepare Network: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    CKKSEncoder encoder(context);
    HECNNenc hecnnenc(encryptor, decryptor, encoder);

    vector<vector<vector<Plaintext>>> ker_poly_1;   // [128/16][2][9]
    vector<vector<Plaintext>> act_poly_1;           // [128/16][3], here 3 is degree of the approximate activation

    vector<vector<vector<Plaintext>>> ker_poly_2;   // [256/16][128/4][9] = [16][32][9]
    vector<vector<Plaintext>> act_poly_2;           // [256/16][3] =[16][3]

    vector<vector<vector<Plaintext>>> ker_poly_3;   // [512/16][256/16][9] = [32][16][9]
    vector<vector<Plaintext>> act_poly_3;           // [512/16][3] = [32][3]

    vector<vector<Plaintext>> dense_ker_poly;
    Plaintext dense_bias_poly;
    vector<Plaintext> zero_one_poly;

    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    if(method == "hear"){
        hecnnenc.prepare_network(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                           dense_ker_poly, dense_bias_poly,
                                           ker, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, NB_CHANNELS);
    } else if(method == "fhear") {
        hecnnenc.prepare_network_interlaced(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                           dense_ker_poly, dense_bias_poly, zero_one_poly,
                                           ker, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, NB_CHANNELS, bit_sizes_vec);
    }

    cout << "(done) " ;
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    int ntest = id_end + 1 - id_st;
    vector<Ciphertext> xCipher(ntest);
    
    MT_EXEC_RANGE(ntest, first, last);
    for(int i = first; i < last; ++i){
        int id = i + id_st;
        dten test_sample;   // the input tensor [2][32][15] = c * h * w
        reshape(test_sample, test_input[id], 3, 32, 15);
        
        if(method == "hear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_HEAR::qscale);
        } else if(method == "fhear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_FHEAR::qscale);
        }
    }
    MT_EXEC_RANGE_END
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
//    cout << "|             Evaluation             |" << endl;
//    cout << "+------------------------------------+" << endl;
    
    HECNNeval hecnneval(evaluator, relin_keys, gal_keys, hecnnenc);
    output.resize(ntest, vector<double> (10));  // clarify the size of output

    vector<Ciphertext> res;         // ciphertexts for intermediate results, [8] -> [16] -> [32] during inference when nch=128
    vector<vector<vector<Ciphertext>>> rot;
    Ciphertext *block = new Ciphertext [NB_CHANNELS[0] * NB_CHANNELS[1]/16];

    for(int t = 0; t < ntest; t++){
//        cout << "+------------------------------+" << endl;
//        cout << "|        Evaluation (B1)       |" << endl;
//        cout << "+------------------------------+" << endl;
       
        vector<double> detailed_eval_times;
        vector<double> detailed_memory;
        
        chrono::microseconds time_total_eval(0);
        
        cout << "=========(" << t + id_st << ")=========" << endl;
        cout << "> Evaluation (B1): " ;
        
        time_start = chrono::high_resolution_clock::now();
        
        // compute the hyper-parameters
        int nin_ctxts = NB_CHANNELS[0];         // 2 = nrows of blocks (in),
        int nout_ctxts = NB_CHANNELS[1] / 16;   // 8 = ncols of blocks (total number of independent ciphertexts), (out)
        int nblocks = nin_ctxts * nout_ctxts;   // 2 * 8 = 16
        int num = (nin_ctxts * nout_ctxts * Param_conv2d::DIM2_FILTERS);   // 2 * 8 * 9 = 144
        res.resize(nout_ctxts);
       
        if(mode1 == "fully"){
            rot.resize(1, vector<vector<Ciphertext>> (1, vector<Ciphertext> (1)));
            hecnneval.generate_rotations_conv1(rot[0][0], xCipher[t], mode1);
            
            // perform scalar multiplication on each blocks
            MT_EXEC_RANGE(num, first, last);
            for(int n = first; n < last; ++n){ // 0 <= k < 9, 0 <= l < 16, 0 <= i < 2, 0 <= j < 8
                int k = (n % Param_conv2d::DIM2_FILTERS);
                int l = (int) floor(n / Param_conv2d::DIM2_FILTERS);
                
                int i = (l % nin_ctxts);
                int j = (int) floor((double)l / (double)nin_ctxts);
                int i1 = (i * 9);
                
                Ciphertext ctemp;
                if(k == 4){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1], ker_poly_1[j][i][4], ctemp);   // ct0 * k4, [8][3][9]
                } else if(k < (Param_conv2d::steps_halfsize)){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + k + 1], ker_poly_1[j][i][k], ctemp);
                } else{
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + k], ker_poly_1[j][i][k], ctemp);
                }
                
                // aggregate in a horizontal direction: temp[l][0] += temp[l][1] + ... + temp[l][8]
                if (i == 0) {  // l=even
                    if(k == 0) {
                        res[j] = ctemp;
                    } else{
                        evaluator.add_inplace(res[j], ctemp);
                    }
                } else{
                    if(k == 0) {
                        block[j] = ctemp;
                    } else{
                        evaluator.add_inplace(block[j], ctemp);
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        else if(mode1 == "baby"){
            rot.resize(1, vector<vector<Ciphertext>> (1, vector<Ciphertext> (1)));
            hecnneval.generate_rotations_conv1(rot[0][0], xCipher[t], mode1);

            MT_EXEC_RANGE(nblocks, first, last);    // nblocks = 2*8 = 16
            for(int l = first; l < last; ++l){
                int i = (l % nin_ctxts);            // 0 <= i < 2
                int j = (int) floor((double)l / (double)nin_ctxts); // 0 <= j < 8
               
                for(int k = 0; k < Param_conv2d::DIM2_FILTERS; ++k){
                    Ciphertext ctemp;
                    if(k == 4){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][0], ker_poly_1[j][i][4], ctemp);   // ct0 * k4, [8][3][9]
                    } else if(k < (Param_conv2d::steps_halfsize)){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][k + 1], ker_poly_1[j][i][k], ctemp);
                    } else{
                        evaluator.multiply_plain_leveled_fast(rot[0][0][k], ker_poly_1[j][i][k], ctemp);
                    }
                   
                    if (i == 0) {
                        if(k == 0) {
                            res[j] = ctemp;
                        } else{
                            evaluator.add_inplace(res[j], ctemp);
                        }
                    } else{
                        if(k == 0) {
                            block[j] = ctemp;
                        } else{
                            evaluator.add_inplace(block[j], ctemp);
                        }
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        
        // aggregate the temporary results to res
        MT_EXEC_RANGE(nout_ctxts, first, last);
        for(int j = first; j < last; ++j){
            if(mode1 == "baby") {
                evaluator.rotate_vector_inplace(block[j], Param_conv2d::shift, gal_keys, MemoryPoolHandle().New(false));
            }
            evaluator.add_inplace(res[j], block[j]);
            evaluator.rescale_to_next_inplace(res[j]);     // rescale by qc
        }
        MT_EXEC_RANGE_END
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B2): ";
//    cout << "+------------------------------------+" << endl;
//
        time_start = chrono::high_resolution_clock::now();
        time_start_detail = chrono::high_resolution_clock::now();
        vector<double> detailed_conv_times;
        
        // Step 2.1. Pre-processing step
        // res_packed = res[0] + rho(res[1], -1) + rho(res[2], -16) + rho(res[3], -17)
        vector<Ciphertext> res_packed;    // 128/16 = 8 -> 8/4 = 2
        
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[1] / 16;     // 128/16 = 8 (in)
        } else if(method == "fhear"){
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[0], 2);
            nin_ctxts = res_packed.size();
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        nout_ctxts = NB_CHANNELS[2] / 16;   // 256/16 = 16 (out)
        rot.clear();

        // Step 2.2. Ordinary homomorphic convolutions
        // Fist, generate the rotated ciphertexts of input
        // Second, multiply the rotated ciphertext by the kernel plaintexts.
        time_start_detail = chrono::high_resolution_clock::now();
        if(mode2 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 2);
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 2);
            }
            res.resize(nout_ctxts);
            
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 9 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][4], res[i]);
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][4], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        
                        for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_2[i][j0][l], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_2[i][j0][l + 1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 2, mode2);          // [8][9]
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 2, mode2);   // [2][9]
            }
            res.resize(nout_ctxts);
            
            if(mode2 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][4], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][4], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_2[i][j0][l], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_2[i][j0][l + 1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv2d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false)); // rho(j*shift)
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
            else if(mode2 == "giant") {
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv2d::steps_size + 1; l++){ // 0 <= l < 9
                        int ker_index;
                        if(l == 0){
                            ker_index = 4;
                        } else if(l < (Param_conv2d::steps_halfsize + 1)){
                            ker_index = l - 1;
                        } else{
                            ker_index = l;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctempl);
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctemp);
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv2d::steps_conv[1][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);

        // Step 2.3. Post-processing
        time_start_detail = chrono::high_resolution_clock::now();
        if(method == "fhear"){
            if(NB_CHANNELS[1] == 32){    // packed: (ct0, ct1)
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
            else{
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B3): ";
//    cout << "+------------------------------------+" << endl;
        
        time_start = chrono::high_resolution_clock::now();
        time_start_detail = chrono::high_resolution_clock::now();
        
        nout_ctxts = NB_CHANNELS[3] / 16;   // 512/16 = 32 (out) when nch=128
        
        // Step 3.1: pre-processing
        // res_packed = res[0] + rho(res[1], -1) + rho(res[2], -16) + rho(res[3], -17)
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[2] / 16;
        } else if(method == "fhear"){
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[1], 3);
            nin_ctxts = res_packed.size();
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_start_detail = chrono::high_resolution_clock::now();
        rot.clear();
        
        // Step 3.2. Ordinary homomorphic convolutions
        // Fist, generate the rotated ciphertexts of input
        // Second, multiply the rotated ciphertext by the kernel plaintexts.
        if(mode3 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 3);
            }
            res.resize(nout_ctxts);
            
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 9 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][4], res[i]);
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][4], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        
                        for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_3[i][j0][l], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_3[i][j0][l + 1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 3, mode3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 3, mode3);
            }
            res.resize(nout_ctxts);
            
            if(mode3 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][4], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][4], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_3[i][j0][l], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_3[i][j0][l + 1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv2d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false)); // rho(j*shift)
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
            else if(mode3 == "giant"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv2d::steps_size + 1; l++){ // 0 <= l < 9
                        int ker_index;
                        if(l == 0){
                            ker_index = 4;
                        } else if(l < (Param_conv2d::steps_halfsize + 1)){
                            ker_index = l - 1;
                        } else{
                            ker_index = l;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctempl); // ct0 * k4
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctemp); // ct0 * k4
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv2d::steps_conv[2][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);   // rescale by qc
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_start_detail = chrono::high_resolution_clock::now();
        
        // Step 3.3: post-processing
        if(method == "fhear"){
            if(NB_CHANNELS[1] <= 64){    // packed: (ct0, ct1, ..., ct7): 8 * 16 = 128 = NB_CHANNELS[3]
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 2, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
            else{
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 2, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 32, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end_detail = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end_detail - time_start_detail);
        detailed_conv_times.push_back(time_diff.count()/1000000.0);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "conv... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_3);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 3);  // res3[0], ,,, res3[31]
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Dense... " ;
//    cout << "+------------------------------------+" << endl;
        
        time_start = chrono::high_resolution_clock::now();
        
        Ciphertext res_prediction;
        hecnneval.Eval_Dense_Fast_Light(res_prediction, res, dense_ker_poly, dense_bias_poly, NB_CHANNELS[3]);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        cout << ">> Total Eval Time = [" << time_total_eval.count()/1000000.0 << " s] " << endl;
        detailed_eval_times.push_back(time_total_eval.count()/1000000.0) ;
     
//    cout << "+------------------------------------+" << endl;
        cout << ">  Decryption... " ;
//    cout << "+------------------------------------+" << endl;
        time_start = chrono::high_resolution_clock::now();
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res_prediction);
        
        int j = 0;
        for(int i = 0; i < (32 * 16 * 10); i+=(32 * 16)){
            output[t][j] = dmsg[i];
            j++;
        }
                
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
         
        // output probabilities
        for(int i = 0; i < 9; i++){
            cout << output[t][i] * 10.0 << ",";
        }
        cout << output[t][9] * 10.0 << endl;
                
        eval_times.push_back(detailed_eval_times);
        memory.push_back(detailed_memory);
        conv_times.push_back(detailed_conv_times);
        
        res.clear();
        res.shrink_to_fit();
        
        rot.clear();
        rot.shrink_to_fit();
    }

    // Write time and memory
    for(int t = 0; t < ntest; t++){
        for(int i = 0; i < eval_times[0].size() - 1; i++){
            outf << eval_times[t][i] << ",";
            outfm << memory[t][i] << ",";
        }

        outf << eval_times[t][eval_times[0].size() - 1] << endl;
        outfm << memory[t][eval_times[0].size() - 1] << endl;

        for(int i = 0; i < conv_times[0].size() - 1; i++){
            outfc << conv_times[t][i] << ",";
        }
        outfc << conv_times[t][conv_times[0].size()  - 1] << endl;
    }
    
    outf.close();
    outfm.close();
    outfc.close();
}


/*
 Note: This is the same as hecnn. The difference is that it outputs only the evaluation time.
 */

void TestHEAR2d::hecnn_threading(dmat &output, dmat test_input, int id_st, int id_end,
                       vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                       string mode1, string mode2, string mode3, int nch, string method, int NUM_THREADS)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
   
    string filename_evaltime;
   
    if((method == "hear") && (nch == 128) && (mode2 == "fully")){
        int resid = id_st/unit;
        filename_evaltime = "result/multi-threading_" + method + "_" + mode2 + "_" + to_string(NUM_THREADS) + "_" + to_string(resid) + ".txt";
        
    } else{
        filename_evaltime = "result/multi-threading_" + method + "_" + mode2 + "_" + to_string(NUM_THREADS) + ".txt";
    }
 
    fstream outf;
    outf.open(filename_evaltime.c_str(), fstream::in | fstream::out | fstream::app);
    
    vector<double> eval_times;

    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    vector<int> bit_sizes_vec;
    if(method == "hear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_HEAR::logq0, Param_HEAR::logq, Param_HEAR::logqc,
                          Param_HEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);    // n = degree
        parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
    } else if(method == "fhear") {
        get_modulus_chain(bit_sizes_vec,
                          Param_FHEAR::logq0, Param_FHEAR::logq, Param_FHEAR::logqc,
                          Param_FHEAR::logqc_small, Param_FHEAR::logp0);
        
        parms.set_poly_modulus_degree(Param_FHEAR::poly_modulus_degree);    // n = degree
        parms.set_coeff_modulus(CoeffModulus::Create(Param_FHEAR::poly_modulus_degree, bit_sizes_vec));
    }
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    vector<int> steps_all;  // the required rotations
    steps_all.insert(steps_all.end(), Param_conv2d::steps_giant.begin(), Param_conv2d::steps_giant.end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[0].begin(), Param_conv2d::steps_conv[0].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[1].begin(), Param_conv2d::steps_conv[1].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_conv[2].begin(), Param_conv2d::steps_conv[2].end());
    steps_all.insert(steps_all.end(), Param_conv2d::steps_pool.begin(), Param_conv2d::steps_pool.end());

    if(method == "fhear"){
        steps_all.insert(steps_all.end(), Param_conv2d::steps_interlacing.begin(), Param_conv2d::steps_interlacing.end());
    }
    GaloisKeys gal_keys = keygen.galois_keys(steps_all);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
   
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Prepare Network: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    CKKSEncoder encoder(context);
    HECNNenc hecnnenc(encryptor, decryptor, encoder);

    vector<vector<vector<Plaintext>>> ker_poly_1;   // [128/16][2][9]
    vector<vector<Plaintext>> act_poly_1;           // [128/16][3], here 3 is degree of the approximate activation

    vector<vector<vector<Plaintext>>> ker_poly_2;   // [256/16][128/4][9] = [16][32][9]
    vector<vector<Plaintext>> act_poly_2;           // [256/16][3] =[16][3]

    vector<vector<vector<Plaintext>>> ker_poly_3;   // [512/16][256/16][9] = [32][16][9]
    vector<vector<Plaintext>> act_poly_3;           // [512/16][3] = [32][3]

    vector<vector<Plaintext>> dense_ker_poly;
    Plaintext dense_bias_poly;
    vector<Plaintext> zero_one_poly;

    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    if(method == "hear"){
        hecnnenc.prepare_network(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                           dense_ker_poly, dense_bias_poly,
                                           ker, real_poly, dense_ker, dense_bias,
                                           mode1, mode2, mode3, NB_CHANNELS);
    } else if(method == "fhear") {
        hecnnenc.prepare_network_interlaced(ker_poly_1, act_poly_1, ker_poly_2, act_poly_2, ker_poly_3, act_poly_3,
                                           dense_ker_poly, dense_bias_poly, zero_one_poly,
                                           ker, real_poly, dense_ker, dense_bias,
                                           mode1, mode2, mode3, NB_CHANNELS, bit_sizes_vec);
    }

    cout << "(done) " ;
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;


//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    int ntest = id_end + 1 - id_st;
    vector<Ciphertext> xCipher(ntest);
    
    MT_EXEC_RANGE(ntest, first, last);
    for(int i = first; i < last; ++i){
        int id = i + id_st;
        dten test_sample; // the input tensor [2][32][15] = c * h * w
        reshape(test_sample, test_input[id], 3, 32, 15);
        
        if(method == "hear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_HEAR::qscale);
        } else if(method == "fhear"){
            hecnnenc.encryptdata_packed(xCipher[i], test_sample, Param_FHEAR::qscale);
        }
    }
    MT_EXEC_RANGE_END
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
//    cout << "|             Evaluation             |" << endl;
//    cout << "+------------------------------------+" << endl;
    
    HECNNeval hecnneval(evaluator, relin_keys, gal_keys, hecnnenc);
    output.resize(ntest, vector<double> (10));

    
    vector<Ciphertext> res;
    vector<vector<vector<Ciphertext>>> rot;
    Ciphertext *block = new Ciphertext [NB_CHANNELS[0] * NB_CHANNELS[1]/16];

    for(int t = 0; t < ntest; t++){
//        cout << "+------------------------------+" << endl;
//        cout << "|        Evaluation (B1)       |" << endl;
//        cout << "+------------------------------+" << endl;
       
        chrono::microseconds time_total_eval(0);
        
        cout << "=========(" << t + id_st << ")=========" << endl;
        cout << "> Evaluation (B1): " ;
        
        time_start = chrono::high_resolution_clock::now();
        
        // compute the hyper-parameters
        int nin_ctxts = NB_CHANNELS[0];         // 2 = nrows of blocks (in),
        int nout_ctxts = NB_CHANNELS[1] / 16;   // 8 = ncols of blocks (total number of independent ciphertexts), (out)
        int nblocks = nin_ctxts * nout_ctxts;   // 2 * 8 = 16
        int num = (nin_ctxts * nout_ctxts * Param_conv2d::DIM2_FILTERS);   // 2 * 8 * 9 = 144
        res.resize(nout_ctxts);
       
        if(mode1 == "fully"){
            rot.resize(1, vector<vector<Ciphertext>> (1, vector<Ciphertext> (1)));
            hecnneval.generate_rotations_conv1(rot[0][0], xCipher[t], mode1);
            
            MT_EXEC_RANGE(num, first, last);
            for(int n = first; n < last; ++n){
                int k = (n % Param_conv2d::DIM2_FILTERS);
                int l = (int) floor(n / Param_conv2d::DIM2_FILTERS);
                
                int i = (l % nin_ctxts);
                int j = (int) floor((double)l / (double)nin_ctxts);
                int i1 = (i * 9);
                
                Ciphertext ctemp;
                if(k == 4){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1], ker_poly_1[j][i][4], ctemp);
                } else if(k < (Param_conv2d::steps_halfsize)){
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + k + 1], ker_poly_1[j][i][k], ctemp);
                } else{
                    evaluator.multiply_plain_leveled_fast(rot[0][0][i1 + k], ker_poly_1[j][i][k], ctemp);
                }
                
                if (i == 0) {
                    if(k == 0) {
                        res[j] = ctemp;
                    } else{
                        evaluator.add_inplace(res[j], ctemp);
                    }
                } else{
                    if(k == 0) {
                        block[j] = ctemp;
                    } else{
                        evaluator.add_inplace(block[j], ctemp);
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        else if(mode1 == "baby"){
            rot.resize(1, vector<vector<Ciphertext>> (1, vector<Ciphertext> (1)));
            hecnneval.generate_rotations_conv1(rot[0][0], xCipher[t], mode1);

            MT_EXEC_RANGE(nblocks, first, last);
            for(int l = first; l < last; ++l){
                int i = (l % nin_ctxts);
                int j = (int) floor((double)l / (double)nin_ctxts);
               
                for(int k = 0; k < Param_conv2d::DIM2_FILTERS; ++k){
                    Ciphertext ctemp;
                    if(k == 4){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][0], ker_poly_1[j][i][4], ctemp);
                    } else if(k < (Param_conv2d::steps_halfsize)){
                        evaluator.multiply_plain_leveled_fast(rot[0][0][k + 1], ker_poly_1[j][i][k], ctemp);
                    } else{
                        evaluator.multiply_plain_leveled_fast(rot[0][0][k], ker_poly_1[j][i][k], ctemp);
                    }
                   
                    if (i == 0) {
                        if(k == 0) {
                            res[j] = ctemp;
                        } else{
                            evaluator.add_inplace(res[j], ctemp);
                        }
                    } else{
                        if(k == 0) {
                            block[j] = ctemp;
                        } else{
                            evaluator.add_inplace(block[j], ctemp);
                        }
                    }
                }
            }
            MT_EXEC_RANGE_END
        }
        
        MT_EXEC_RANGE(nout_ctxts, first, last);
        for(int j = first; j < last; ++j){
            if(mode1 == "baby") {
                evaluator.rotate_vector_inplace(block[j], Param_conv2d::shift, gal_keys, MemoryPoolHandle().New(false));
            }
            evaluator.add_inplace(res[j], block[j]);
            evaluator.rescale_to_next_inplace(res[j]);
        }
        MT_EXEC_RANGE_END
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "conv... " ;

        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 1);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B2): ";
//    cout << "+------------------------------------+" << endl;
//
        time_start = chrono::high_resolution_clock::now();
         
        vector<Ciphertext> res_packed;
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[1] / 16;
        } else if(method == "fhear"){
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[0], 2);
            nin_ctxts = res_packed.size();
        }
        nout_ctxts = NB_CHANNELS[2] / 16;
        rot.clear();
        
        if(mode2 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 2);
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 2);
            }
            res.resize(nout_ctxts);
            
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 9 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][4], res[i]); // ct0 * k4
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_2[i][j0][4], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        
                        for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_2[i][j0][l], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_2[i][j0][l + 1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 2, mode2); // [8][9]
            } else if (method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 2, mode2); // [2][9]
            }
            res.resize(nout_ctxts);
            
            if(mode2 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][4], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_2[i][j0][4], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_2[i][j0][l], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_2[i][j0][l + 1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv2d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);
                }
                MT_EXEC_RANGE_END
            }
            else if(mode2 == "giant") {
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv2d::steps_size + 1; l++){
                        int ker_index;
                        if(l == 0){
                            ker_index = 4;
                        } else if(l < (Param_conv2d::steps_halfsize + 1)){
                            ker_index = l - 1;
                        } else{
                            ker_index = l;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctempl);
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_2[i][j0][ker_index], ctemp);
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv2d::steps_conv[1][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);
                }
                MT_EXEC_RANGE_END
            }
        }
        
        if(method == "fhear"){
            if(NB_CHANNELS[1] == 32){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
            else{
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
        }
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "conv... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 2);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
        cout << "> Evaluation (B3): ";
//    cout << "+------------------------------------+" << endl;
        
        time_start = chrono::high_resolution_clock::now();
        
        nout_ctxts = NB_CHANNELS[3] / 16;
        
        if(method == "hear"){
            nin_ctxts = NB_CHANNELS[2] / 16;
        } else if(method == "fhear"){
            hecnneval.interlace_ctxts(res_packed, res, zero_one_poly[1], 3);
            nin_ctxts = res_packed.size();
        }
    
        rot.clear();
        
        if(mode3 == "fully"){
            if(method == "hear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res, 3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_fully_light(rot, res_packed, 3);
            }
            res.resize(nout_ctxts);
            
            MT_EXEC_RANGE(nout_ctxts, first, last);
            for(int i = first; i < last; ++i){
                Ciphertext ctemp;
                for(int k = 0; k < nin_ctxts; ++k){
                    for(int j = 0; j < 16; ++j){
                        int j1 = 9 * j;
                        int j0 = j + 16 * k;
                        if((j == 0) && (k == 0)){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][4], res[i]);
                        } else{
                            evaluator.multiply_plain_leveled_fast(rot[k][j][0], ker_poly_3[i][j0][4], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        
                        for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_3[i][j0][l], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                        for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                            evaluator.multiply_plain_leveled_fast(rot[k][j][l + 1], ker_poly_3[i][j0][l + 1], ctemp);
                            evaluator.add_inplace(res[i], ctemp);
                        }
                    }
                }
                evaluator.rescale_to_next_inplace(res[i]);
            }
            MT_EXEC_RANGE_END
        }
        else{
            if(method == "hear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res, 3, mode3);
            } else if(method == "fhear"){
                hecnneval.generate_rotations_conv_babygiant_light(rot, res_packed, 3, mode3);
            }
            res.resize(nout_ctxts);
            
            if(mode3 == "baby"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int j = 0; j < 16; ++j){
                        Ciphertext ctemp;
                        Ciphertext ctempj;
                        
                        for(int k = 0; k < nin_ctxts; ++k){
                            int j0 = j + 16 * k;
                            
                            if(k == 0) {
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][4], ctempj);
                            } else{
                                evaluator.multiply_plain_leveled_fast(rot[0][k][0], ker_poly_3[i][j0][4], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            
                            for(int l = 0; l < Param_conv2d::steps_halfsize; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_3[i][j0][l], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                            for(int l = Param_conv2d::steps_halfsize; l < Param_conv2d::steps_size; l++){
                                evaluator.multiply_plain_leveled_fast(rot[0][k][l + 1], ker_poly_3[i][j0][l + 1], ctemp);
                                evaluator.add_inplace(ctempj, ctemp);
                            }
                        }
                        
                        if(j == 0){
                            res[i] = ctempj;
                        } else{
                            evaluator.rotate_vector_inplace(ctempj, Param_conv2d::steps_giant[j - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempj);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);
                }
                MT_EXEC_RANGE_END
            }
            else if(mode3 == "giant"){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    for(int l = 0; l < Param_conv2d::steps_size + 1; l++){
                        int ker_index;
                        if(l == 0){
                            ker_index = 4;
                        } else if(l < (Param_conv2d::steps_halfsize + 1)){
                            ker_index = l - 1;
                        } else{
                            ker_index = l;
                        }
                        
                        Ciphertext ctemp;
                        Ciphertext ctempl;
                        for(int j = 0; j < 16; ++j){
                            for(int k = 0; k < nin_ctxts; ++k){
                                int j0 = j + 16 * k;
                                if(j0 == 0) {
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctempl);
                                } else{
                                    evaluator.multiply_plain_leveled_fast(rot[0][k][j], ker_poly_3[i][j0][ker_index], ctemp);
                                    evaluator.add_inplace(ctempl, ctemp);
                                }
                            }
                        }
                        
                        if (l == 0){
                            res[i] = ctempl;
                        } else{
                            evaluator.rotate_vector_inplace(ctempl, Param_conv2d::steps_conv[2][l - 1], gal_keys, MemoryPoolHandle().New(false));
                            evaluator.add_inplace(res[i], ctempl);
                        }
                    }
                    evaluator.rescale_to_next_inplace(res[i]);
                }
                MT_EXEC_RANGE_END
            }
        }
      
        if(method == "fhear"){
            if(NB_CHANNELS[1] <= 64){
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 2, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
            else{
                MT_EXEC_RANGE(nout_ctxts, first, last);
                for(int i = first; i < last; ++i){
                    Ciphertext ctemp;
                    evaluator.rotate_vector(res[i], 1, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 2, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 16, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);

                    evaluator.rotate_vector(res[i], 32, gal_keys, ctemp, MemoryPoolHandle().New(false));
                    evaluator.add_inplace(res[i], ctemp);
                }
                MT_EXEC_RANGE_END
            }
        }
          
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "conv... " ;

        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_BN_ActPoly_Fast_inplace(res, act_poly_3);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();
        hecnneval.Eval_Avg_inplace(res, 3);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "avg... [" << time_total_eval.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        
//    cout << "+------------------------------------+" << endl;
        cout << "> Dense... " ;
//    cout << "+------------------------------------+" << endl;
        time_start = chrono::high_resolution_clock::now();
        
        Ciphertext res_prediction;
        hecnneval.Eval_Dense_Fast_Light(res_prediction, res, dense_ker_poly, dense_bias_poly, NB_CHANNELS[3]);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        time_total_eval += time_diff;
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
        cout << ">> Total Eval Time = [" << time_total_eval.count()/1000000.0 << " s] " << endl;
        eval_times.push_back(time_total_eval.count()/1000000.0) ;
     
//    cout << "+------------------------------------+" << endl;
        cout << ">  Decryption... " ;
//    cout << "+------------------------------------+" << endl;
        time_start = chrono::high_resolution_clock::now();
        vector<double> dmsg;
        hecnnenc.decrypt_vector(dmsg, res_prediction); // the first 16 many 2D-(8*3)
        
        int j = 0;
        for(int i = 0; i < (32 * 16 * 10); i+=(32 * 16)){
            output[t][j] = dmsg[i];
            j++;
        }
                
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
          
        // output probabilities
        for(int i = 0; i < 9; i++){
            cout << output[t][i] * 10.0 << ",";
        }
        cout << output[t][9] * 10.0 << endl;
                
        res.clear();
        res.shrink_to_fit();
        
        rot.clear();
        rot.shrink_to_fit();
    }

    // time 
    for(int t = 0; t < ntest; t++){
        outf << eval_times[t] << endl;
    }
    outf.close();
}
