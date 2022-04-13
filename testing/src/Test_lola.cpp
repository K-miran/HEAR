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
#include "HECNNevaluator_sota.h"
#include "HECNNevaluator_conv2d.h"
#include "Test_lola.h"

#define DEBUG false
using namespace std;

void TestLoLA::hecnn1d(dmat &output, dmat test_input, int id_st, int id_end,
                      vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    
    int resid = id_st/50;
    string filename_time = "result/time_conv1d_" + to_string(nch) + "_lola_" + to_string(resid) + ".txt";
    string filename_memory = "result/memory_conv1d_" + to_string(nch) + "_lola_" + to_string(resid) + ".txt";
    
    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);
    
    vector<vector<double>> eval_times;
    vector<vector<double>> memory;

    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    vector<int> bit_sizes_vec = {
        Param_FHEAR::logq0,                     // The last modulus
        Param_FHEAR::logqc,                     // FC (1 -> 0)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act3 (3->1)
        Param_FHEAR::logqc_small,               // merge (4->3)
        Param_FHEAR::logqc,                     // conv3 (5->4)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act2 (7->5)
        Param_FHEAR::logqc_small,               // merge (8->7)
        Param_FHEAR::logqc,                     // conv2 (9->8)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act1 (11->9)
        Param_FHEAR::logqc,                     // conv1 (12->11)
        Param_FHEAR::logp0
    };
        
    parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);    // n = degree
    parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
    vector<int> ker_poly_lvl = {12, 9, 5, 1};
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    vector<int> steps_all = {
        1, // avg1
        32*16, 32*16*2, 32*16*4,  32*16*8,  //cov2
        2, -2, // conv2
        4, -4, // conv3
        8, 16, 32*1, 32*2,32*3,32*4,32*5,32*6,32*7,32*8,32*9,32*10,32*11,32*12,32*13,32*14 // pool3
    };
    
    vector<int> steps_merge = { // steps_giant (for merging as the post-processing after the conv)
        -32*16, -32*16*2, -32*16*3,
        -32*16*4, -32*16*5, -32*16*6, -32*16*7,
        -32*16*8, -32*16*9, -32*16*10, -32*16*11,
        -32*16*12, -32*16*13, -32*16*14, -32*16*15
    };
    steps_all.insert(steps_all.end(), steps_merge.begin(), steps_merge.end());
    GaloisKeys gal_keys = keygen.galois_keys(steps_all);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Prepare Network: B1" ;
//    cout << "+------------------------------------+" << endl;

    time_start = chrono::high_resolution_clock::now();

    CKKSEncoder encoder(context);
    HECNNenc hecnnenc(encryptor, decryptor, encoder); // in HECNNencryptor_conv2d.h
    
    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    int nout_ctxts1 = NB_CHANNELS[1]/16;
    int nout_ch2 = NB_CHANNELS[2];
    int nout_ctxts2 = NB_CHANNELS[2]/16;
    int nout_ch3 = NB_CHANNELS[3];
    int nout_ctxts3 = nout_ch3/16;
    
    // Input: ker[0] = size[nch][2][3] = [out_channels][in_channels][filter_height]
    vector<vector<Plaintext>> ker_poly_1;                   // [8][3*2]=[8][6]
    ker_poly_1.resize(nout_ctxts1, vector<Plaintext>(3*2));
    size_t slot_count = encoder.slot_count();
    vector<double> temp_zero32(32, 0.0);
    
    for(int ch = 0; ch < 2; ++ch){
        for(int ht = 0; ht < 3; ++ht){
            MT_EXEC_RANGE(nout_ctxts1, first, last);
            for(int n = first; n < last; n++){              // repeat 8 times (16 kernels per iteration)
                vector<double> ker_full;
                for(int l = n * 16; l < (n+1) * 16; l++){    // 16 kernels
                    vector<double> ker_one_block(32*15, ker[0][l][ch][ht]); // copy the entry
                    ker_one_block.insert(ker_one_block.end(), temp_zero32.begin(), temp_zero32.end());
                    ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                }
                
                if(ker_full.size() != slot_count){
                    throw invalid_argument("Error: encode_conv1_kernel");
                }
                encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[0], ker_poly_1[n][3 * ch + ht]);// lvl=12
            }
            MT_EXEC_RANGE_END
        }
    }
    
    vector<vector<Plaintext>> act_poly_1;   // [nch/16][3], here 3 is degree of the approximate activation
    int nrows = 32;
    int ncols = 16;

    act_poly_1.resize(nout_ctxts1, vector<Plaintext>(3));   // [nch/16][3]

    int const_lvl = Param_FHEAR::ker_poly_lvl[0] - 3;        // level of the input ciphertext
    int nonconst_lvl = Param_FHEAR::ker_poly_lvl[0] - 2;

    int len = 32 * 16;
    int len_actual = 32 * 15;
    
    int num3 = nout_ctxts1 * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        int i1 = (i << 4);

        for(int j = 0; j < 16; ++j){
            dvec temp_short;    // size = (32 * 16) = 512
            val1_to_vector_conv1d(temp_short, real_poly[0][i1 + j][l], len, len_actual);
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }

        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_1[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_1[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_1[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    
    getrusage(RUSAGE_SELF, &usage);
    cout << "(done:" << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << "GB), B2" ;
    
    //-------------
    // Block2
    //-------------
    // Input: ker[1]: size [2*nch][nch][3]
    // Output: ker_poly_2[2*nch][240][nch/16]
    int num_out = 240; // size of input
    vector<vector<vector<Plaintext>>> ker_poly_2;
    ker_poly_2.resize(nout_ch2, vector<vector<Plaintext>>(num_out, vector<Plaintext> (nout_ctxts1)));
    
    MT_EXEC_RANGE(nout_ch2, first, last);
    for(int n = first; n < last; ++n){
        for(int i = 0; i < 240; i++){
            for(int k = 0; k < nout_ctxts1; k++){   // 8
                vector<double> ker_full;
                for(int l = k * 16; l < (k+1) * 16; l++){   // encode 16 kernels together (k=0: 0~15; k=1: 16~31; ..., k=7: 112~127)
                    vector<double> ker_one_block(32*16, 0.0);
                    for(int ht = -1; ht < 2; ht++){
                        int i1 = i + ht;
                        if ((i1 >= 0) && (i1 < 240)) {
                            int id = (2 * i1);
                            ker_one_block[id] = ker[1][n][l][ht+1];
                        }
                    }
                    ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                }
                if(ker_full.size() != slot_count) throw invalid_argument("Error: encode_conv2_kernel");

                encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[1], ker_poly_2[n][i][k]); // lvl=9
            }
        }
    }
    MT_EXEC_RANGE_END
    
    vector<Plaintext> mask_poly2 (240);
    for(int i = 0; i < 240; i++){
        vector<double> ker_one_block(slot_count, 0.0);
        ker_one_block[2 * i] = 1.0; // (0, 2, 4, 6, 8, ..., 478)
        encoder.encode(ker_one_block, Param_FHEAR::qcscale_small, ker_poly_lvl[1]-1, mask_poly2[i]);  //lvl=8
    }
 
    vector<vector<Plaintext>> act_poly_2;                   // [2*nch/16][3], here 3 is degree of the approximate activation
    act_poly_2.resize(nout_ctxts2, vector<Plaintext>(3));
    
    const_lvl = ker_poly_lvl[1] - 3;
    nonconst_lvl = ker_poly_lvl[1] - 2;
    
    int dist = 2;
    num3 = nout_ctxts2 * 3; // 240 * (3 coff)
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        for(int j = 0; j < 16; ++j){
            dvec temp_short;
            val_to_vector_conv1d(temp_short, real_poly[1][i * 16 + j][l], len, len_actual, dist);
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        if(temp_full_slots.size() != slot_count) throw invalid_argument("Error: encode_act2_kernel");
        
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_2[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_2[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_2[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    
    getrusage(RUSAGE_SELF, &usage);
    cout << "(done:" << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << "GB), B3" ;
    
    //-------------
    // Block3
    //-------------
    // Input: ker[2]: size [4*nch][2*nch][3]
    // Output: ker_poly_3[4*nch][120][2*nch/16]
    num_out = 120; // size of input
    vector<vector<vector<Plaintext>>> ker_poly_3;
    ker_poly_3.resize(nout_ch3, vector<vector<Plaintext>>(num_out, vector<Plaintext> (nout_ctxts2)));

    MT_EXEC_RANGE(nout_ch3, first, last);
    for(int n = first; n < last; ++n){
        for(int i = 0; i < 120; i++){
            for(int k = 0; k < nout_ctxts2; k++){
                vector<double> ker_full;
                for(int l = k * 16; l < (k+1) * 16; l++){
                    vector<double> ker_one_block(32*16, 0.0);
                    for(int ht = -1; ht < 2; ht++){
                        int i1 = i + ht;
                        if ((i1 >= 0) && (i1 < 120)) {
                            int id = (4 * i1) ;
                            ker_one_block[id] = ker[2][n][l][ht+1];
                        }
                    }
                    ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                }
                if(ker_full.size() != slot_count) throw invalid_argument("Error: encode_conv3_kernel");

                encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[2], ker_poly_3[n][i][k]);
            }
        }
    }
    MT_EXEC_RANGE_END

//    vector<Plaintext> mask_poly3 (120);
//    for(int i = 0; i < 120; i++){
//        vector<double> ker_one_block(slot_count, 0.0);
//        ker_one_block[4 * i] = 1.0; // (0, 4, 8, ..., 476)
//        encoder.encode(ker_one_block, Param_FHEAR::qcscale_small, ker_poly_lvl[2]-1, mask_poly3[i]);  //lvl=8
//    }

    vector<vector<Plaintext>> act_poly_3;                   // [4*nch/16][3], here 3 is degree of the approximate activation
    act_poly_3.resize(nout_ctxts3, vector<Plaintext>(3));
    
    const_lvl = ker_poly_lvl[2] - 3;
    nonconst_lvl = ker_poly_lvl[2] - 2;
    
    dist = 4;
    num3 = nout_ctxts3 * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        for(int j = 0; j < 16; ++j){
            dvec temp_short;
            val1_to_vector_conv1d(temp_short, real_poly[2][i * 16 + j][l], len, len_actual);
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        if(temp_full_slots.size() != slot_count) throw invalid_argument("Error: encode_act3_kernel");
        
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_3[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_3[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_3[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    
    getrusage(RUSAGE_SELF, &usage);
    cout << "(done:" << (double) usage.ru_maxrss/(Param_conv1d::memoryscale) << "GB), Dense" ;
    
    vector<vector<Plaintext>> dense_ker_poly;
    Plaintext dense_bias_poly;
    
    // Generate plaintext polynomials for the dense layer
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
    
    double q1 = pow(2.0, bit_sizes_vec[1]);
    hecnnenc.encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], q1, ker_poly_lvl[3]);
    hecnnenc.encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_FHEAR::qscale);

    cout << "(done) " ;
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    //cout << "> Prepare Network Time  [" << time_diff.count()/1000.0 << " ms] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

#if 1
//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    int ntest = id_end + 1 - id_st;
    vector<vector<Ciphertext>> xCipher(ntest, vector<Ciphertext> (3*2));
    
    MT_EXEC_RANGE(ntest, first, last);
    for(int t = first; t < last; ++t){
        int id = t + id_st;
        dten test_sample;   // the input tensor of size [2][32][15] = c * h * w
        reshape(test_sample, test_input[id], 3, 32, 15);
        
        for(int ch = 0; ch < 2; ch++){
            vector<double> input_padded;        // input[ch]: 32*15=480 -> (1+480+1)=482
            input_padded.push_back(0.0);
            for(int i = 0; i < 32; i++){
                input_padded.insert(input_padded.end(), test_sample[ch][i].begin(), test_sample[ch][i].end());
            }
            input_padded.push_back(0.0);
            
            for(int i = -1; i <= 1; i++){
                vector<double> msg_one_block;   // msg_one_block = [32 * 16]
                for(int k = 0; k < 32 * 15; k++){
                    msg_one_block.push_back(input_padded[k+1+i]);
                }
                msg_one_block.insert(msg_one_block.end(), temp_zero32.begin(), temp_zero32.end());
                
                vector<double> msg_full;        // [32 * 16 * 16] = fully-packed slots
                for(int l = 0; l < 16; ++l){
                    msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
                }
                
                Plaintext plain;
                encoder.encode(msg_full, Param_FHEAR::qscale, plain);
                encryptor.encrypt(plain, xCipher[t][3 * ch + (i+1)]);
            }
        }
    }
    MT_EXEC_RANGE_END
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB), xCtxt=[" << xCipher.size() << "][" << xCipher[0].size() << "]" << endl;
    
//    cout << "+------------------------------------+" << endl;
//    cout << "|             Evaluation             |" << endl;
//    cout << "+------------------------------------+" << endl;
    
    HECNNeval LoLaEval(evaluator, relin_keys, gal_keys, hecnnenc);
    output.resize(ntest, vector<double> (10));
    
    for(int t = 0; t < ntest; t++){
        vector<double> detailed_eval_times;
        vector<double> detailed_memory;
        
        chrono::microseconds time_total_eval(0);
        
        cout << "=========(" << t + id_st << ")=========" << endl;
        cout << "> Evaluation (B1): " ;
    
        time_start = chrono::high_resolution_clock::now();
        
        // Input: xCipher[6], ker_poly_1[8][3*2]
        vector<Ciphertext> res1 (nout_ctxts1);  // = nch/16
        
        MT_EXEC_RANGE(nout_ctxts1, first, last);
        for(int n = first; n < last; ++n){
            evaluator.multiply_plain_leveled_fast(xCipher[t][0], ker_poly_1[n][0], res1[n]);
            for(int i = 1; i < 3 * 2; i++){
                Ciphertext ctemp;
                evaluator.multiply_plain_leveled_fast(xCipher[t][i], ker_poly_1[n][i], ctemp);
                evaluator.add_inplace(res1[n], ctemp);
            }
            evaluator.rescale_to_next_inplace(res1[n]);
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
        
        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res1, act_poly_1);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        MT_EXEC_RANGE(res1.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(res1[i], 1, gal_keys, ctemp);
            evaluator.add_inplace(res1[i], ctemp);
        }
        MT_EXEC_RANGE_END

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

        time_start = chrono::high_resolution_clock::now();

        // Input:  res1[8],  ker_poly_2 [2*nch][240][8]
        vector<Ciphertext> res_conv2 (nout_ch2);
        vector<Ciphertext> res_temp (nout_ch2);
        
        // n-th channel: res1 & ker_poly_2[n][240][8]
        MT_EXEC_RANGE(nout_ch2, first, last);
        for(int n = first; n < last; ++n){
            for(int i = 0; i < 240; i++){
                evaluator.multiply_plain_leveled_fast(res1[0], ker_poly_2[n][i][0], res_temp[n]);
                for(int k = 1; k < nout_ctxts1; k++){
                    Ciphertext ctemp;
                    evaluator.multiply_plain_leveled_fast(res1[k], ker_poly_2[n][i][k], ctemp);
                    evaluator.add_inplace(res_temp[n], ctemp);
                }
                evaluator.rescale_to_next_inplace(res_temp[n]);
        
                Ciphertext ctemp1;
                Ciphertext ctemp2;

                // Require additional log2(16*7)=8 rotations
                // Aggeregate over single ciphertext over distinct channels: 16 channels
                evaluator.rotate_vector(res_temp[n], 32*16, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp[n], ctemp1);
                
                evaluator.rotate_vector(res_temp[n], 32*16*2, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp[n], ctemp1);
                
                evaluator.rotate_vector(res_temp[n], 32*16*4, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp[n], ctemp1);
                
                evaluator.rotate_vector(res_temp[n], 32*16*8, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp[n], ctemp1);
                    
                // Perform rotations (over inside 240 blocks)
                if(i == 0){
                    evaluator.rotate_vector(res_temp[n], 2, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp1);
                } else if (i == 240){
                    evaluator.rotate_vector(res_temp[n], -2, gal_keys, ctemp2);
                    evaluator.add_inplace(res_temp[n], ctemp2);
                } else{
                    evaluator.rotate_vector(res_temp[n], 2, gal_keys, ctemp1);
                    evaluator.rotate_vector(res_temp[n], -2, gal_keys, ctemp2);
                    evaluator.add_inplace(res_temp[n], ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp2);
                }
                
                // Multiply an (1-0) plaintext vector
                evaluator.multiply_plain_leveled_fast(res_temp[n], mask_poly2[i], res_temp[n]);
                evaluator.rescale_to_next_inplace(res_temp[n]);
            
                // Merge a distinct ciphertext to make them a packed ciphertext
                if(i == 0){
                    res_conv2[n] = res_temp[n];
                } else{
                    evaluator.add_inplace(res_conv2[n], res_temp[n]);
                }
            }
        }
        MT_EXEC_RANGE_END
        
        // Merge compactly
        // output: res2[0]+=(res[1]+...+res[15]), res2[16], res2[32], ...,
        vector<Ciphertext> res2 (nout_ctxts2);
        MT_EXEC_RANGE(nout_ctxts2, first, last);
        for(int n = first; n < last; ++n){
            res2[n] = res_conv2[n * 16];
            for(int i = 1; i < 16; i++){
                evaluator.rotate_vector(res_conv2[n * 16 + i], steps_merge[i-1], gal_keys, res_conv2[n * 16 + i]);
                evaluator.add_inplace(res2[n], res_conv2[n * 16 + i]);
            }
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
        
        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res2, act_poly_2);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        MT_EXEC_RANGE(res2.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(res2[i], 2, gal_keys, ctemp);
            evaluator.add_inplace(res2[i], ctemp);
        }
        MT_EXEC_RANGE_END

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

        // Input: res2[16], ker_poly_3[4*nch][120][16]
        vector<Ciphertext> res_conv3 (nout_ch3);
        vector<Ciphertext> res_temp3 (nout_ch3);
        
        // n-th channel: res2 & ker_poly_3[n][120][16]
        MT_EXEC_RANGE(nout_ch3, first, last);
        for(int n = first; n < last; ++n){
            for(int i = 0; i < 120; i++){
                evaluator.multiply_plain_leveled_fast(res2[0], ker_poly_3[n][i][0], res_temp3[n]);
                for(int k = 1; k < nout_ctxts2; k++){ // 16 many ctxts
                    Ciphertext ctemp;
                    evaluator.multiply_plain_leveled_fast(res2[k], ker_poly_3[n][i][k], ctemp);
                    evaluator.add_inplace(res_temp3[n], ctemp);
                }
                evaluator.rescale_to_next_inplace(res_temp3[n]);

                Ciphertext ctemp1;
                Ciphertext ctemp2;
                    
                // Aggregate the intermediate computed convolution results over slots
                // Use additional 8 rotations
                evaluator.rotate_vector(res_temp3[n], 32*16, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp3[n], ctemp1);
                
                evaluator.rotate_vector(res_temp3[n], 32*16*2, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp3[n], ctemp1);
                
                evaluator.rotate_vector(res_temp3[n], 32*16*4, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp3[n], ctemp1);
                
                evaluator.rotate_vector(res_temp3[n], 32*16*8, gal_keys, ctemp1);
                evaluator.add_inplace(res_temp3[n], ctemp1);
                    
                // Perform rotations over inside 120 blocks
                if(i == 0){
                    evaluator.rotate_vector(res_temp3[n], 4, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
                } else if(i == 120){
                    evaluator.rotate_vector(res_temp3[n], -4, gal_keys, ctemp2);
                    evaluator.add_inplace(res_temp3[n], ctemp2);
                } else{
                    evaluator.rotate_vector(res_temp3[n], 4, gal_keys, ctemp1);
                    evaluator.rotate_vector(res_temp3[n], -4, gal_keys, ctemp2);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp2);
                }
                
                // Multiply an (1-0) plaintext vector
                Plaintext mask_poly = mask_poly2[2*i];
                evaluator.mod_switch_to_inplace(mask_poly, res_temp3[n].parms_id());
                evaluator.multiply_plain_leveled_fast(res_temp3[n], mask_poly, res_temp3[n]);
                evaluator.rescale_to_next_inplace(res_temp3[n]);
            
                // Merge distinct ciphertext to make them a packed ciphertext
                if(i == 0){
                    res_conv3[n] = res_temp3[n];
                } else{
                    evaluator.add_inplace(res_conv3[n], res_temp3[n]);
                }
            }
        }
        MT_EXEC_RANGE_END
         
        // output: res3[0], res3[16], res3[32], ...,
        vector<Ciphertext> res3 (nout_ctxts3);
        
        MT_EXEC_RANGE(nout_ctxts3, first, last);
        for(int n = first; n < last; ++n){
            res3[n] = res_conv3[n * 16];
            for(int i = 1; i < 16; i++){
                evaluator.rotate_vector(res_conv3[n * 16 + i], steps_merge[i-1], gal_keys, res_conv3[n * 16 + i]);
                evaluator.add_inplace(res3[n], res_conv3[n * 16 + i]);
            }
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

        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res3, act_poly_3);

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        // rotation: 4*(2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6) (#=log(120)=8)
        MT_EXEC_RANGE(res3.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp1;
            evaluator.rotate_vector(res3[i], 4 * 1, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
                
            evaluator.rotate_vector(res3[i], 4 * 2, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
            
            evaluator.rotate_vector(res3[i], 4 * 4, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
            
            // After that, we have (0~7, ...., 8~15, ..., ..., 112~119, *** )
            vector<Ciphertext> ctemp(14);
            for(int j = 1; j < 15; j++){
                evaluator.rotate_vector(res3[i], 4 * 8 * j, gal_keys, ctemp[j-1]);
            }
            
            for(int j = 1; j < 15; j++){
                evaluator.add_inplace(res3[i], ctemp[j-1]);
            }
        }
        MT_EXEC_RANGE_END

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
        
        Ciphertext HEres;
        LoLaEval.Eval_Dense_Fast(HEres, res3, dense_ker_poly, dense_bias_poly, NB_CHANNELS[3]);
        
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
        hecnnenc.decrypt_vector(dmsg, HEres);
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
    }
    
    // time and memory
    for(int t = 0; t < ntest; t++){
        for(int i = 0; i < eval_times[0].size() - 1; i++){
            outf << eval_times[t][i] << ",";
            outfm << memory[t][i] << ",";
        }
        
        outf << eval_times[t][eval_times[0].size() - 1] << endl;
        outfm << memory[t][eval_times[0].size() - 1] << endl;
    }
    
    outf.close();
    outfm.close();
    
    //cout << "> Decryption Time = [" << time_diff.count()/1000.0 << " ms] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB)" << endl;
#endif
}



void TestLoLA::hecnn2d(dmat &output, dmat test_input, int id_st, int id_end,
                      vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    
    int resid = id_st/50;
    string filename_time = "result/time_conv2d_" + to_string(nch) + "_lola_" + to_string(resid) + ".txt";
    string filename_memory = "result/memory_conv2d_" + to_string(nch) + "_lola_" + to_string(resid) + ".txt";
    
    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);
    
    vector<vector<double>> eval_times;
    vector<vector<double>> memory;

    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    vector<int> bit_sizes_vec = {
        Param_FHEAR::logq0,                     // The last modulus
        Param_FHEAR::logqc,                     // FC (1 -> 0)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act3 (3->1)
        Param_FHEAR::logqc_small,               // merge (4->3)
        Param_FHEAR::logqc,                     // conv3 (5->4)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act2 (7->5)
        Param_FHEAR::logqc_small,               // merge (8->7)
        Param_FHEAR::logqc,                     // conv2 (9->8)
        Param_FHEAR::logqc, Param_FHEAR::logq,  // act1 (11->9)
        Param_FHEAR::logqc,                     // conv1 (12->11)
        Param_FHEAR::logp0
    };
        
    parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);    // n = degree
    parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
    vector<int> ker_poly_lvl = {12, 9, 5, 1};
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    vector<int> steps_all = {
        1, 15, // avg1
        32*16, 32*16*2, 32*16*4, 32*16*8, //cov2
        2, -2, 30, -30,  // conv2
        4, -4, 60, -60,  // conv3
        8, 60*2, 60*4    // pool3 (4,8,15*4=60,60*2,60*4)
    };
    vector<int> steps_merge = { // steps_giant
        -32*16, -32*16*2, -32*16*3,
        -32*16*4, -32*16*5, -32*16*6, -32*16*7,
        -32*16*8, -32*16*9, -32*16*10, -32*16*11,
        -32*16*12, -32*16*13, -32*16*14, -32*16*15
    };
    steps_all.insert(steps_all.end(), steps_merge.begin(), steps_merge.end());
    GaloisKeys gal_keys = keygen.galois_keys(steps_all);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
    //std::cout << "| --->  poly_modulus_degree (n): " << context_data -> parms().poly_modulus_degree() << std::endl;
    //std::cout << "| --->  coeff_modulus size (logQ): ";
    //std::cout << context_data->total_coeff_modulus_bit_count() << endl;
    //print_modulus_chain(bit_sizes_vec);
    
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Prepare Network: B1" ;
//    cout << "+------------------------------------+" << endl;

    time_start = chrono::high_resolution_clock::now();

    CKKSEncoder encoder(context);
    HECNNenc hecnnenc(encryptor, decryptor, encoder); // in HECNNencryptor_conv2d.h
    
    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    int nout_ctxts1 = NB_CHANNELS[1]/16;
    int nout_ch2 = NB_CHANNELS[2];
    int nout_ctxts2 = NB_CHANNELS[2]/16;
    int nout_ch3 = NB_CHANNELS[3];
    int nout_ctxts3 = nout_ch3/16;
    
    // Input: ker[0] = size[nch][2][3][3] = [out_channels][in_channels][filter_height][filter_width])
    vector<vector<Plaintext>> ker_poly_1;   // [8][18]
    ker_poly_1.resize(nout_ctxts1, vector<Plaintext>(3*3*2));
    size_t slot_count = encoder.slot_count();
    vector<double> temp_zero32(32, 0.0);
    
    for(int ch = 0; ch < 2; ++ch){
        for(int ht = 0; ht < 3; ++ht){
            for(int wt = 0; wt < 3; ++wt){
                //-----------------
                for(int n = 0; n < nout_ctxts1; n++){   // repeat 8 times (16 kernels per iteration)
                    vector<double> ker_full;
                    for(int l = n * 16; l < (n+1) * 16; l++){   // 16 kernels
                        vector<double> ker_one_block(32*15, ker[0][l][ch][ht][wt]); // copy the entry
                        ker_one_block.insert(ker_one_block.end(), temp_zero32.begin(), temp_zero32.end());
                        ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                    }
                    
                    if(ker_full.size() != slot_count){
                        throw invalid_argument("Error: encode_conv1_kernel");
                    }
                    encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[0], ker_poly_1[n][9 * ch + 3 * ht + wt]);// lvl=12
                }
            }
        }
    }
     
    vector<vector<Plaintext>> act_poly_1;   // [nch/16][3], here 3 is degree of the approximate activation
    int nrows = 32;
    int ncols = 16;

    act_poly_1.resize(nout_ctxts1, vector<Plaintext>(3));   // [nch/16][3]

    int const_lvl = Param_FHEAR::ker_poly_lvl[0] - 3;       // level of the input ciphertext
    int nonconst_lvl = Param_FHEAR::ker_poly_lvl[0] - 2;

    int num3 = nout_ctxts1 * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        int i1 = (i << 4);

        for(int j = 0; j < 16; ++j){
            dvec temp_short(nrows * (ncols -1), real_poly[0][i1 + j][l]);    // size = (32 * 16) = 512
            temp_short.insert(temp_short.end(), temp_zero32.begin(), temp_zero32.end());
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }

        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_1[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_1[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_1[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    cout << "(done), B2" ;
    
    //-------------
    // Block2
    //-------------
    // Input: ker[1]: size [2*nch][nch][3][3]
    // Output: ker_poly_2[2*nch][16*7][nch/16]
    
    int num_out = 16 * 7;
    vector<vector<vector<Plaintext>>> ker_poly_2;
    ker_poly_2.resize(nout_ch2, vector<vector<Plaintext>>(num_out, vector<Plaintext> (nout_ctxts1)));
 
    MT_EXEC_RANGE(nout_ch2, first, last);
    for(int n = first; n < last; ++n){
        for(int i = 0; i < 16; i++){
            for(int j = 0; j < 7; j++){
                for(int k = 0; k < nout_ctxts1; k++){
                    vector<double> ker_full;
                    for(int l = k * 16; l < (k+1) * 16; l++){   // 16 kernels  (k=0: 0~15; k=1: 16~31; ..., k=7: 112~127)
                        vector<double> ker_one_block(32*16, 0.0);
                        for(int ht = -1; ht < 2; ht++){
                            for(int wt = -1; wt < 2; wt++){
                                // 2*(i+ht,j+wt) in 32*15 matrix -> 15 * 2 * (i+ht) + 2 * (j + wt)
                                int i1 = i + ht;
                                int j1 = j + wt;
                                if ((i1 >= 0) && (i1 < 16) && (j1 >= 0) && (j1 < 7)) {
                                    int id = 15 * (2 * i1) + (2 * j1);
                                    ker_one_block[id] = ker[1][n][l][ht+1][wt+1];
                                }
                            }
                        }
                        ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                    }
                    if(ker_full.size() != slot_count) throw invalid_argument("Error: encode_conv2_kernel");

                    encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[1], ker_poly_2[n][7 * i + j][k]); // lvl=9
                }
            }
        }
    }
    MT_EXEC_RANGE_END
    
    vector<vector<Plaintext>> mask_poly2 (16, vector<Plaintext>(7));
    for(int i = 0; i < 16; i++){
        for(int j = 0; j < 7; j++){
            vector<double> ker_one_block(slot_count, 0.0);
            int ker_nonzero_id = 15 * (2 * i) + (2 * j);    // 2 * (15*i+j):
            ker_one_block[ker_nonzero_id] = 1.0;
            encoder.encode(ker_one_block, Param_FHEAR::qcscale_small, ker_poly_lvl[1]-1, mask_poly2[i][j]);  //lvl=8
        }
    }
 
    vector<vector<Plaintext>> act_poly_2;                   // [2*nch/16][3], here 3 is degree of the approximate activation
    act_poly_2.resize(nout_ctxts2, vector<Plaintext>(3));
    
    const_lvl = ker_poly_lvl[1] - 3;
    nonconst_lvl = ker_poly_lvl[1] - 2;
    
    int dist = 2;
    num3 = nout_ctxts2 * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        for(int j = 0; j < 16; ++j){
            dvec temp_short;
            val_to_vector(temp_short, real_poly[1][i * 16 + j][l], 32, 15, dist);
            temp_short.insert(temp_short.end(), temp_zero32.begin(), temp_zero32.end());// size = (32 * 16)
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        if(temp_full_slots.size() != slot_count) throw invalid_argument("Error: encode_act2_kernel");
        
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_2[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_2[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_2[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    cout << "(done), B3";

    //-------------
    // Block3
    //-------------
    // Input: ker[2]: size [4*nch][2*nch][3]
    // Output: ker_poly_3[4*nch][8*3][2*nch/16]
    num_out = 8 * 3;
    vector<vector<vector<Plaintext>>> ker_poly_3;
    ker_poly_3.resize(nout_ch3, vector<vector<Plaintext>>(num_out, vector<Plaintext> (nout_ctxts2)));

    MT_EXEC_RANGE(nout_ch3, first, last);
    for(int n = first; n < last; ++n){
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < nout_ctxts2; k++){
                    vector<double> ker_full;
                    for(int l = k * 16; l < (k+1) * 16; l++){   // 16 kernels  (k=0: 0~15; k=1: 16~31; ..., k=15: ~255)
                        vector<double> ker_one_block(32*16, 0.0);
                        for(int ht = -1; ht < 2; ht++){
                            for(int wt = -1; wt < 2; wt++){
                                // 2*(i+ht,j+wt) in 32*15 matrix -> 15 * 2 * (i+ht) + 2 * (j + wt)
                                int i1 = i + ht;
                                int j1 = j + wt;
                                if ((i1 >= 0) && (i1 < 8) && (j1 >= 0) && (j1 < 3)) {
                                    int id = 15 * (4 * i1) + (4 * j1);
                                    ker_one_block[id] = ker[2][n][l][ht+1][wt+1];
                                }
                            }
                        }
                        ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                    }
                    if(ker_full.size() != slot_count) throw invalid_argument("Error: encode_conv3_kernel");

                    encoder.encode(ker_full, Param_FHEAR::qcscale, ker_poly_lvl[2], ker_poly_3[n][3 * i + j][k]);
                }
            }
        }
    }
    MT_EXEC_RANGE_END

    vector<vector<Plaintext>> mask_poly3 (8, vector<Plaintext>(3));
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 3; j++){
            vector<double> ker_one_block(slot_count, 0.0);
            int ker_nonzero_id = 15 * (4 * i) + (4 * j); // 4*(15*i+j)=2*(15*2i+2j)
            ker_one_block[ker_nonzero_id] = 1.0;
            encoder.encode(ker_one_block, Param_FHEAR::qcscale_small, ker_poly_lvl[2]-1, mask_poly3[i][j]);  //lvl=8
        }
    }

    vector<vector<Plaintext>> act_poly_3;                   // [4*nch/16][3], here 3 is degree of the approximate activation
    act_poly_3.resize(nout_ctxts3, vector<Plaintext>(3));
    
    const_lvl = ker_poly_lvl[2] - 3;
    nonconst_lvl = ker_poly_lvl[2] - 2;
    
    dist = 4;
    num3 = nout_ctxts3 * 3;
    MT_EXEC_RANGE(num3, first, last);
    for(int k = first; k < last; ++k){
        int i = (int) floor (k / 3.0);
        int l = (k % 3);
        
        dvec temp_full_slots;   // (32 * 16) * 16 = 8192
        for(int j = 0; j < 16; ++j){
            dvec temp_short;
            val_to_vector(temp_short, real_poly[2][i * 16 + j][l], 32, 15, dist);
            temp_short.insert(temp_short.end(), temp_zero32.begin(), temp_zero32.end());// size = (32 * 16)
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        if(temp_full_slots.size() != slot_count) throw invalid_argument("Error: encode_act3_kernel");
        
        // Encode each coefficient
        if(l == 0){
            encoder.encode(temp_full_slots, Param_FHEAR::qscale, const_lvl, act_poly_3[i][0]);
        } else if(l == 1){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_3[i][1]);
        } else if(l == 2){
            encoder.encode(temp_full_slots, Param_FHEAR::qcscale, nonconst_lvl, act_poly_3[i][2]);
        }
    }
    MT_EXEC_RANGE_END
    cout << "(done), Dense";
    
    vector<vector<Plaintext>> dense_ker_poly;
    Plaintext dense_bias_poly;
    
    // Generate plaintext polynomials for the dense layer
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
    
    double q1 = pow(2.0, bit_sizes_vec[1]);
    hecnnenc.encode_dense_kerpoly(dense_ker_poly, scaled_dense_ker, NB_CHANNELS[3], q1, ker_poly_lvl[3]);
    hecnnenc.encode_dense_biaspoly(dense_bias_poly, scaled_dense_bias, Param_FHEAR::qscale);

    cout << "(done) " ;
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

#if 1
//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    int ntest = id_end + 1 - id_st;
    vector<vector<Ciphertext>> xCipher(ntest, vector<Ciphertext> (18));
    
    MT_EXEC_RANGE(ntest, first, last);
    for(int t = first; t < last; ++t){
        int id = t + id_st;
        dten test_sample;   // the input tensor [2][32][15] = c * h * w
        reshape(test_sample, test_input[id], 3, 32, 15);
        
        for(int ch = 0; ch < 2; ch++){
            vector<vector<double>> input_padded; // input[ch]: 32*15 -> 34*17
            vector<double> zero_vec (17, 0.0);
            input_padded.push_back(zero_vec);
            for(int i = 0; i < 32; i++){
                vector<double> temp_vec;
                temp_vec.push_back(0.0);
                temp_vec.insert(temp_vec.end(), test_sample[ch][i].begin(), test_sample[ch][i].end());
                temp_vec.push_back(0.0);
                input_padded.push_back(temp_vec);
            }
            input_padded.push_back(zero_vec);
            
            for(int i = -1; i < 2; i++){
                for(int j = -1; j < 2; j++){
                    vector<double> msg_one_block;       // msg_one_block = [32 * 16]
                    for(int k = 0; k < 32; k++){
                        for(int l = 0; l < 15; l++){     //middle: input_padded[k+1][l+1]
                            msg_one_block.push_back(input_padded[k+1+i][l+1+j]);
                        }
                    }
                    msg_one_block.insert(msg_one_block.end(), temp_zero32.begin(), temp_zero32.end());
                    
                    vector<double> msg_full;            // [32 * 16 * 16] = fully packed slots
                    for(int l = 0; l < 16; ++l){
                        msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
                    }
                    
                    Plaintext plain;
                    encoder.encode(msg_full, Param_FHEAR::qscale, plain);
                    encryptor.encrypt(plain, xCipher[t][9 * ch + 3*(i+1)+(j+1)]);
                }
            }
        }
    }
    MT_EXEC_RANGE_END
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB), xCtxt=[" << xCipher.size() << "][" << xCipher[0].size() << "]" << endl;
    
//    cout << "+------------------------------------+" << endl;
//    cout << "|             Evaluation             |" << endl;
//    cout << "+------------------------------------+" << endl;
    
    HECNNeval LoLaEval(evaluator, relin_keys, gal_keys, hecnnenc);
    output.resize(ntest, vector<double> (10));
    
    for(int t = 0; t < ntest; t++){
        vector<double> detailed_eval_times;
        vector<double> detailed_memory;
        
        chrono::microseconds time_total_eval(0);
        
        cout << "=========(" << t + id_st << ")=========" << endl;
        cout << "> Evaluation (B1): " ;
    
        time_start = chrono::high_resolution_clock::now();
        
        // Input: xCipher[18], ker_poly_1[8][3*3*2]
        vector<Ciphertext> res1 (nout_ctxts1);
       
        MT_EXEC_RANGE(nout_ctxts1, first, last);
        for(int n = first; n < last; ++n){
            evaluator.multiply_plain_leveled_fast(xCipher[t][0], ker_poly_1[n][0], res1[n]);
            for(int i = 1; i < 18; i++){
                Ciphertext ctemp;
                evaluator.multiply_plain_leveled_fast(xCipher[t][i], ker_poly_1[n][i], ctemp);
                evaluator.add_inplace(res1[n], ctemp);
            }
            evaluator.rescale_to_next_inplace(res1[n]);
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
        
        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res1, act_poly_1);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        MT_EXEC_RANGE(res1.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(res1[i], 1, gal_keys, ctemp);
            evaluator.add_inplace(res1[i], ctemp);
            
            evaluator.rotate_vector(res1[i], 15, gal_keys, ctemp);
            evaluator.add_inplace(res1[i], ctemp);
        }
        MT_EXEC_RANGE_END

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

        time_start = chrono::high_resolution_clock::now();

        // Input:  res1[8],  ker_poly_2 [2*nch][16*7][8]
        vector<Ciphertext> res_conv2 (nout_ch2);
        vector<Ciphertext> res_temp (nout_ch2);
        
        // n-th channel: res1 & ker_poly_2[n][16*7][8]
        MT_EXEC_RANGE(nout_ch2, first, last);
        for(int n = first; n < last; ++n){
            for(int i = 0; i < 16; i++){
                for(int j = 0; j < 7; j++){
                    // step1: Multiply the ciphertext with the ker_poly (over distint ciphertexts)
                    int id = 7 * i + j;
                    evaluator.multiply_plain_leveled_fast(res1[0], ker_poly_2[n][id][0], res_temp[n]);
                    for(int k = 1; k < nout_ctxts1; k++){
                        Ciphertext ctemp;
                        evaluator.multiply_plain_leveled_fast(res1[k], ker_poly_2[n][id][k], ctemp);
                        evaluator.add_inplace(res_temp[n], ctemp);
                    }
                    evaluator.rescale_to_next_inplace(res_temp[n]);

                    Ciphertext ctemp1;
                    Ciphertext ctemp2;
                    
                    // Require additional log2(16*7)=8 rotations
                    // aggeregate over single ciphertext over distinct channels (rot-and-sum)
                    evaluator.rotate_vector(res_temp[n], 32*16, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp[n], 32*16*2, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp[n], 32*16*4, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp[n], 32*16*8, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp[n], ctemp1);
            
                    // Perform rotations over inside 16*7 blocks
                    if(j == 0){
                        evaluator.rotate_vector(res_temp[n], 2, gal_keys, ctemp1);
                        evaluator.add_inplace(res_temp[n], ctemp1);
                    } else if(j == 6){
                        evaluator.rotate_vector(res_temp[n], -2, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp[n], ctemp2);
                    } else{
                        evaluator.rotate_vector(res_temp[n], 2, gal_keys, ctemp1);
                        evaluator.rotate_vector(res_temp[n], -2, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp[n], ctemp1);
                        evaluator.add_inplace(res_temp[n], ctemp2);
                    }
                    
                    if(i == 0){
                        evaluator.rotate_vector(res_temp[n], 30, gal_keys, ctemp1);
                        evaluator.add_inplace(res_temp[n], ctemp1);
                    } else if (i == 15){
                        evaluator.rotate_vector(res_temp[n], -30, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp[n], ctemp2);
                    } else{
                        evaluator.rotate_vector(res_temp[n], 30, gal_keys, ctemp1);
                        evaluator.rotate_vector(res_temp[n], -30, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp[n], ctemp1);
                        evaluator.add_inplace(res_temp[n], ctemp2);
                    }
                    
                    // Multiply an (1-0) plaintext vector
                    evaluator.multiply_plain_leveled_fast(res_temp[n], mask_poly2[i][j], res_temp[n]);
                    evaluator.rescale_to_next_inplace(res_temp[n]);
                
                    // Merge a distinct ciphertext to make them a packed ciphertext
                    if((i == 0) && (j == 0)){
                        res_conv2[n] = res_temp[n];
                    } else{
                        evaluator.add_inplace(res_conv2[n], res_temp[n]);
                    }
                }
            }
        }
        MT_EXEC_RANGE_END
        
        
        // Merge compactly
        // Output: res2[0]+=(res[1]+...+res[15]), res2[16], res2[32], ...,
        vector<Ciphertext> res2 (nout_ctxts2);
        MT_EXEC_RANGE(nout_ctxts2, first, last);
        for(int n = first; n < last; ++n){
            res2[n] = res_conv2[n * 16];
            for(int i = 1; i < 16; i++){
                evaluator.rotate_vector(res_conv2[n * 16 + i], steps_merge[i-1], gal_keys, res_conv2[n * 16 + i]);
                evaluator.add_inplace(res2[n], res_conv2[n * 16 + i]);
            }
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
        
        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res2, act_poly_2);
        
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        MT_EXEC_RANGE(res2.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp;
            evaluator.rotate_vector(res2[i], 2, gal_keys, ctemp);
            evaluator.add_inplace(res2[i], ctemp);

            evaluator.rotate_vector(res2[i], 30, gal_keys, ctemp);
            evaluator.add_inplace(res2[i], ctemp);
        }
        MT_EXEC_RANGE_END

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

        // Input: res2[16], ker_poly_3[4*nch][8*3][16]
        vector<Ciphertext> res_conv3 (nout_ch3);
        vector<Ciphertext> res_temp3 (nout_ch3);
        
        // n-th channel: res2 & ker_poly_3[n][8*3][16]
        MT_EXEC_RANGE(nout_ch3, first, last);
        for(int n = first; n < last; ++n){
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 3; j++){
                    // step1: Multiply the ciphertext with the ker_poly (over distint ciphertexts)
                    int id = 3 * i + j;
                    evaluator.multiply_plain_leveled_fast(res2[0], ker_poly_3[n][id][0], res_temp3[n]);
                    for(int k = 1; k < nout_ctxts2; k++){ // 16 many ctxts
                        Ciphertext ctemp;
                        evaluator.multiply_plain_leveled_fast(res2[k], ker_poly_3[n][id][k], ctemp);
                        evaluator.add_inplace(res_temp3[n], ctemp);
                    }
                    evaluator.rescale_to_next_inplace(res_temp3[n]);

                    Ciphertext ctemp1;
                    Ciphertext ctemp2;
                    
                    // Aggregate the intermediate computed convolution results over slots
                    // Use additional 8 rotations
                    evaluator.rotate_vector(res_temp3[n], 32*16, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp3[n], 32*16*2, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp3[n], 32*16*4, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
                    
                    evaluator.rotate_vector(res_temp3[n], 32*16*8, gal_keys, ctemp1);
                    evaluator.add_inplace(res_temp3[n], ctemp1);
            
                    // Perform rotations over inside 8*3 blocks
                    if(j == 0){
                        evaluator.rotate_vector(res_temp3[n], 4, gal_keys, ctemp1);
                        evaluator.add_inplace(res_temp3[n], ctemp1);
                    } else if(j == 2){
                        evaluator.rotate_vector(res_temp3[n], -4, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp3[n], ctemp2);
                    } else{
                        evaluator.rotate_vector(res_temp3[n], 4, gal_keys, ctemp1);
                        evaluator.rotate_vector(res_temp3[n], -4, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp3[n], ctemp1);
                        evaluator.add_inplace(res_temp3[n], ctemp2);
                    }
                    
                    if(i == 0){
                        evaluator.rotate_vector(res_temp3[n], 60, gal_keys, ctemp1);
                        evaluator.add_inplace(res_temp3[n], ctemp1);
                    } else if (i == 7){
                        evaluator.rotate_vector(res_temp3[n], -60, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp3[n], ctemp2);
                    } else{
                        evaluator.rotate_vector(res_temp3[n], 60, gal_keys, ctemp1);
                        evaluator.rotate_vector(res_temp3[n], -60, gal_keys, ctemp2);
                        evaluator.add_inplace(res_temp3[n], ctemp1);
                        evaluator.add_inplace(res_temp3[n], ctemp2);
                    }
                    
                    // Multiply an (1-0) plaintext vector
                    evaluator.multiply_plain_leveled_fast(res_temp3[n], mask_poly3[i][j], res_temp3[n]);
                    evaluator.rescale_to_next_inplace(res_temp3[n]);
                
                    // Merge distinct ciphertext to make them a packed ciphertext
                    if((i == 0) && (j == 0)){
                        res_conv3[n] = res_temp3[n];
                    } else{
                        evaluator.add_inplace(res_conv3[n], res_temp3[n]);
                    }
                }
            }
        }
        MT_EXEC_RANGE_END
         
        // output: res3[0], res3[16], res3[32], ...,
        vector<Ciphertext> res3 (nout_ctxts3);
        
        MT_EXEC_RANGE(nout_ctxts3, first, last);
        for(int n = first; n < last; ++n){
            res3[n] = res_conv3[n * 16];
            for(int i = 1; i < 16; i++){
                evaluator.rotate_vector(res_conv3[n * 16 + i], steps_merge[i-1], gal_keys, res_conv3[n * 16 + i]);
                evaluator.add_inplace(res3[n], res_conv3[n * 16 + i]);
            }
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

        LoLaEval.Eval_BN_ActPoly_Fast_inplace(res3, act_poly_3);

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        getrusage(RUSAGE_SELF, &usage);
        detailed_eval_times.push_back(time_diff.count()/1000000.0);
        detailed_memory.push_back((double) usage.ru_maxrss/(Param_conv2d::memoryscale));
        time_total_eval += time_diff;
        cout << "act... " ;
        
        time_start = chrono::high_resolution_clock::now();

        MT_EXEC_RANGE(res3.size(), first, last);
        for(int i = first; i < last; ++i){
            Ciphertext ctemp1;
            evaluator.rotate_vector(res3[i], 4, gal_keys, ctemp1);
            Ciphertext ctemp2;
            evaluator.rotate_vector(res3[i], 8, gal_keys, ctemp2);

            evaluator.add_inplace(res3[i], ctemp1);
            evaluator.add_inplace(res3[i], ctemp2);

            // Aggreate 8 items (unit dist = 15*4 = 60)
            evaluator.rotate_vector(res3[i], 60, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
            evaluator.rotate_vector(res3[i], 60 * 2, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
            evaluator.rotate_vector(res3[i], 60 * 4, gal_keys, ctemp1);
            evaluator.add_inplace(res3[i], ctemp1);
        }
        MT_EXEC_RANGE_END

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
        
        Ciphertext HEres;
        LoLaEval.Eval_Dense_Fast(HEres, res3, dense_ker_poly, dense_bias_poly, NB_CHANNELS[3]);
        
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
        hecnnenc.decrypt_vector(dmsg, HEres); // the first 16 many 2D-(8*3)
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
    }
    
    // time and memory
    for(int t = 0; t < ntest; t++){
        for(int i = 0; i < eval_times[0].size() - 1; i++){
            outf << eval_times[t][i] << ",";
            outfm << memory[t][i] << ",";
        }
        
        outf << eval_times[t][eval_times[0].size() - 1] << endl;
        outfm << memory[t][eval_times[0].size() - 1] << endl;
    }
    
    outf.close();
    outfm.close();
#endif
}



