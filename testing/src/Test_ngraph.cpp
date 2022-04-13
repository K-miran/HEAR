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
#include "Test_ngraph.h"

#define DEBUG false
using namespace std;

/*
 @param[in] intput, input data (size [nin][ht][wt])
 @param[in] ker, the kernel of the convoluational layers (size [nout][nin][3][3])
 @param[in] batch_size, the number of the output channeles
 @param[out] res, output data (size [nout][ht][wt])
*/
void TestPlain::Eval_Conv2d(vector<vector<vector<double>>> &res,
                 vector<vector<vector<double>>> input, vector<dten> ker, int batch_size)
{
    int nch = input.size();    // number of input channels
    int ht = input[0].size();
    int wt = input[0][0].size();

    res.resize(batch_size, vector<vector<double>> (ht, vector<double>(wt)));
    
    for(int bsize = 0; bsize < batch_size; bsize++){
        for(int i = 0; i < ht; i++){
            for(int j = 0; j < wt; j++){
                // 3*3 convolution
                res[bsize][i][j] = 0.0;
                for(int ch = 0; ch < nch; ch++){
                    for(int k = - 1; k <= 1; k++){
                        for(int l = - 1; l <= 1; l++){
                            int i1 = i + k;
                            int j1 = j + l;
                            if ((i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wt)){
                                double temp = input[ch][i1][j1] * ker[bsize][ch][k+1][l+1];
                                res[bsize][i][j] += temp;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 @param[in] res, input data
 @param[in] poly, the coefficients of polynomials of conv_bias/BN/ACT
*/
void TestPlain::Eval_BN_ActPoly2d(vector<vector<vector<double>>> &res, dmat poly)
{
    int ch = res.size();
    int ht = res[0].size();
    int wt = res[0][0].size();
    
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){        // 32,16,8
            for(int k = 0; k < wt; k++){    // 15,8,3
                double temp2 = res[i][j][k] * res[i][j][k] * poly[i][2];
                res[i][j][k] *= poly[i][1];
                res[i][j][k] += temp2 + poly[i][0];
            }
        }
    }
}

/*
 @param[in] input, input data
 @param[out] res, the result of the averaging pooling of size 2*2
*/
void TestPlain::Eval_Average2d(vector<vector<vector<double>>> &res, vector<vector<vector<double>>> input)
{
    int ch = input.size();
    int ht = input[0].size()/ 2;
    int wt = input[0][0].size()/ 2;
    res.resize(ch, vector<vector<double>> (ht, vector<double>(wt)));
    
    for(int i = 0; i < ch; i++){
        for(int j = 0; j < ht; j++){
            for(int k = 0; k < wt; k++){
                res[i][j][k] = input[i][2*j][2*k];
                res[i][j][k] += input[i][2*j][2*k+1];
                res[i][j][k] += input[i][2*j+1][2*k];
                res[i][j][k] += input[i][2*j+1][2*k+1];
            }
        }
    }
}

/*
 @param[in] input, input data
 @param[out] res, the result of the global averaging pooling
*/
void TestPlain::Eval_Global_Average2d(vector<double> &res, vector<vector<vector<double>>> input)
{
    int ch = input.size();          // 4*nch
    int ht = input[0].size();       // 8
    int wt = input[0][0].size();    // 3
    res.resize(ch);
    
    for(int i = 0; i < ch; i++){
        res[i] = 0.0;
        for(int j = 0; j < ht; j++){
            for(int k = 0; k < wt; k++){
                res[i] += input[i][j][k];
            }
        }
    }
}

/*
 @param[in] input, input data
 @param[in] dense_ker, the dense kernel
 @param[in] dense_bias, the dense bias
 @param[out] output, the output result of the dense layer
*/
void TestPlain::Eval_Dense2d(vector<double> &output, vector<double> &input, dmat dense_ker, dvec dense_bias)
{
    int nclass = dense_bias.size();     // 10
    int nin = input.size();             // 4*nch
    
    for(int i = 0; i < nclass; i++){
        output[i] = 0.0;
        for(int j = 0; j < nin; j++){
            output[i] += dense_ker[i][j] * input[j];
        }
        output[i] += dense_bias[i];
        output[i] /= 10.0;
    }
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
 @param[in] nch, the number of channels at the 1st conv layer
 @param[out] output, the predicted result, [id_end-id_st+1][10]
 */
void TestPlain::hecnn2d(dmat &output, dmat test_input, int id_st, int id_end,
                      vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch)
{
    // test_input[ntest][32*15*2]
    int ntest = id_end + 1 - id_st;
   
    // xPlain[ntest][2][32][15]
    vector<vector<vector<vector<double>>>> xPlain;
    for(int t = 0; t < ntest; t++){
        dten test_sample;
        reshape(test_sample, test_input[t], 2, 32, 15);
        xPlain.push_back(test_sample);
    }
    
    int nclass = dense_bias.size();   // 10
    output.resize(ntest, vector<double> (nclass)); // [ntext][10]
    
    for(int t = 0; t < ntest; t++){
        vector<vector<vector<double>>> res;
        vector<vector<vector<double>>> res1;
        vector<vector<vector<double>>> res2;
        vector<double> res3;
        
        Eval_Conv2d(res, xPlain[t], ker[0], nch);
        Eval_BN_ActPoly2d(res, real_poly[0]);
        Eval_Average2d(res1, res);
        res.clear();
        
        Eval_Conv2d(res, res1, ker[1], 2*nch);
        Eval_BN_ActPoly2d(res, real_poly[1]);
        Eval_Average2d(res2, res);
        res.clear();
        
        Eval_Conv2d(res, res2, ker[2], 4*nch);
        Eval_BN_ActPoly2d(res, real_poly[2]);
        Eval_Global_Average2d(res3, res);
        
        Eval_Dense2d(output[t], res3, dense_ker, dense_bias);
        
        for(int j = 0; j < 10; j++) cout << output[t][j] * 10.0 << ",";
        cout << endl;
    }
}

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
 @param[in] nch, the number of channels at the 1st conv layer
 @param[out] output, the predicted result, [id_end-id_st+1][10]
 This implementatation is to perform secure inference algorithm of nGraph-HE2 over 1D-CNN models.
 */
void TestnGraph::hecnn1d(dmat &output, dmat test_input, int id_st, int id_end,
                      vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_total_eval(0);
    
    string filename_time = "result/time_conv1d_" + to_string(nch) + "_ngraph.txt";
    string filename_memory = "result/memory_conv1d_" + to_string(nch) + "_ngraph.txt";
    
    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);
    
    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    // (q0, qc): FC
    // (qc, q, qc): conv3
    // (qc, q, qc): conv2
    // (qc, q, qc): conv1
    vector<int> bit_sizes_vec;
    get_modulus_chain(bit_sizes_vec,
                          Param_HEAR::logq0, Param_HEAR::logq, Param_HEAR::logqc,
                          Param_HEAR::logp0);
    
    parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
  
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
    //std::cout << "| --->  poly_modulus_degree (n): " << context_data -> parms().poly_modulus_degree() << std::endl;
    //std::cout << "| --->  coeff_modulus size (logQ): ";
    //std::cout << context_data->total_coeff_modulus_bit_count() << endl;
    //print_modulus_chain(bit_sizes_vec);

    CKKSEncoder encoder(context);
    
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

    HEnGraphCNN nGraphEncode(context);

    // --------------------
    // Kernel plaintext
    //  - conv1: [0][128][2][3][lvl+1] = [NB[1]][NB[0]][3][lvl+1]
    //  - conv2: [1][256][128][3][lvl+1] = [NB[2]][NB[1]][3][lvl+1]
    //  - conv3: [2][512][256][3][lvl+1] = [NB[3]][NB[2]][3][lvl+1]
    // Act_poly: act_poly[0][128][3][lvl+1], act_poly[1][256][3][lvl+1], act_poly[2][512][3][lvl+1]
    // --------------------
    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    vector<vector<vector<vector<vector<uint64_t>>>>> ker_plain;
    vector<vector<vector<vector<uint64_t>>>> act_plain;
    vector<vector<vector<uint64_t>>> dense_ker_plain;    // [10][4*nch][lvl+1] = [10][NB[3]][lvl+1]
    vector<vector<uint64_t>> dense_bias_plain;           // [10][1]
    
    nGraphEncode.prepare_network1d(ker_plain, act_plain, dense_ker_plain, dense_bias_plain,
                                 ker, real_poly, dense_ker, dense_bias, NB_CHANNELS);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    vector<vector<Ciphertext>> xCipher(2, vector<Ciphertext> (32*15));
    
    int ntest = id_end + 1 - id_st;     // ntest samples in one ciphertext (default = 608)
    int tensor_size = 32 * 15 * 2;
    int tensor_half_size = 32 * 15;     // = 480
    int nslots = encoder.slot_count();
    
    if(ntest > nslots) {
        throw invalid_argument("cannot encode all the samples into a single ciphertext");
    }
    
#pragma omp parallel for
    for(int t = 0; t < tensor_size; ++t){
        vector<double> msg (nslots, 0.0);   // take the t-th samples
        for(int i = id_st; i <= id_end; i++){
            msg[i - id_st] = (test_input[i][t]);
        }

        Plaintext plain;
        encoder.encode(msg, Param_HEAR::qscale, plain);
        int ch = (t < tensor_half_size? 0 : 1);
        int height = (ch == 0? t : (t - tensor_half_size));
        encryptor.encrypt(plain, xCipher[ch][height]); // [2][32]
    }
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB), xCtxt=[" << xCipher.size() << "][" << xCipher[0].size() << "][" << xCipher[0][0].size() << "]" << endl;

    //cout << "| --->  Modulus chain index for ct: q[" << context->get_context_data(xCipher[0][0][0].parms_id()) -> chain_index() << "]" << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B1): " ;
//    cout << "+------------------------------------+" << endl;
    
    HEnGraphEval nGrapheval(evaluator, relin_keys);
    
    // Conv1: xCipher[2][32*15], ker_plain[0][nch][2][3][lvl+1] -> res[nch][32*15]
    time_start = chrono::high_resolution_clock::now();
    
    vector<vector<Ciphertext>> res;
    nGrapheval.Eval_Conv1d(res, xCipher, ker_plain[0]);

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Act1: res[nch][32*15], act_plain[0][nch][3]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly1d(res, act_plain[0]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";

    // Pool1: res[nch][32*15=480] (valid ciphertexts = [nch][240])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Average1d(res);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B2): ";
//    cout << "+------------------------------------+" << endl;

    // Conv2: output a puctured ciphertext res[2*nch][240]
    time_start = chrono::high_resolution_clock::now();

    xCipher.clear();
    nGrapheval.Eval_Puctured_Conv1d(xCipher, res, ker_plain[1]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";

    // Act2: res[2*nch][240]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly1d(xCipher, act_plain[1]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
   
    // Pool2: res[2*nch][240] (valid ciphertexts = [2*nch][120])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Average1d(xCipher);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;
    

//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B3): ";
//    cout << "+------------------------------------+" << endl;

    // Conv3: output a puctured ciphertext res[4*nch][120]
    time_start = chrono::high_resolution_clock::now();
    
    res.clear();
    nGrapheval.Eval_Puctured_Conv1d(res, xCipher, ker_plain[2]); // res[512][8][3]
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Act3: res[4*nch][120]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly1d(res, act_plain[2]); // input: res[512][8][3]
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Pool3: res[4*nch][120] (valid ciphertexts = [4*nch][1])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Global_Average1d(res); //  output: xCipher[512][0][0]
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Dense... " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    xCipher.clear();
    nGrapheval.Eval_Dense1d(xCipher, res, dense_ker_plain, dense_bias_plain);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    cout << ">> Total Eval Time = [" << time_total_eval.count()/1000000.0 << " s] " << endl;
    outf << time_total_eval.count()/1000000.0 << "," ;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Decryption: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();

    int nclass = dense_bias.size();
    output.resize(ntest, vector<double> (nclass));
    for(int j = 0; j < nclass; j++){
        vector<double> dmsg;
        Plaintext pmsg;
        decryptor.decrypt(xCipher[j][0], pmsg);
        encoder.decode(pmsg, dmsg);
        
        for(int i = 0; i < ntest; i++){
            output[i][j] = dmsg[i];
            if(i == 0) cout << dmsg[i] * 10 << "," ;
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << endl ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << endl;
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
    outf.close();
    outfm.close();
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
 @param[in] nch, the number of channels at the 1st conv layer
 @param[out] output, the predicted result, [id_end-id_st+1][10]
 This implementatation is to perform secure inference algorithm of nGraph-HE2 over 2D-CNN models.
 */
void TestnGraph::hecnn2d(dmat &output, dmat test_input, int id_st, int id_end,
                      vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_total_eval(0);
    
    string filename_time = "result/time_conv2d_" + to_string(nch) + "_ngraph.txt";
    string filename_memory = "result/memory_conv2d_" + to_string(nch) + "_ngraph.txt";

    fstream outf;
    outf.open(filename_time.c_str(), fstream::in | fstream::out | fstream::app);
    
    fstream outfm;
    outfm.open(filename_memory.c_str(), fstream::in | fstream::out | fstream::app);
    
    struct rusage usage;
    
    EncryptionParameters parms(scheme_type::CKKS);
    
    // (q0, qc): FC
    // (qc, q, qc): conv3
    // (qc, q, qc): conv2
    // (qc, q, qc): conv1
    vector<int> bit_sizes_vec;
    get_modulus_chain(bit_sizes_vec,
                          Param_HEAR::logq0, Param_HEAR::logq, Param_HEAR::logqc,
                          Param_HEAR::logp0);
    
    parms.set_poly_modulus_degree(Param_HEAR::poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(Param_HEAR::poly_modulus_degree, bit_sizes_vec));
  
//    cout << "+------------------------------------+" << endl;
    cout << "> Key Generation: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    auto context = SEALContext::Create(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    
    auto context_data = context->key_context_data();
    //std::cout << "| --->  poly_modulus_degree (n): " << context_data -> parms().poly_modulus_degree() << std::endl;
    //std::cout << "| --->  coeff_modulus size (logQ): ";
    //std::cout << context_data->total_coeff_modulus_bit_count() << endl;
    //print_modulus_chain(bit_sizes_vec);

    CKKSEncoder encoder(context);
    
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

    HEnGraphCNN nGraphEncode(context);

    // --------------------
    // Kernel plaintext
    //  - conv1:[0][128][2][3][3][lvl+1] = [NB[1]][NB[0]][3][3][lvl+1]
    //  - conv2:[1][256][128][3][3][lvl+1] = [NB[2]][NB[1]][3][3][lvl+1]
    //  - conv3:[2][512][256][3][3][lvl+1] = [NB[3]][NB[2]][3][3][lvl+1]
    // Act_poly: act_poly[0][128][3][lvl+1], act_poly[1][256][3][lvl+1], act_poly[2][512][3][lvl+1]
    // --------------------
    vector<int> NB_CHANNELS = {2, nch, 2*nch, 4*nch};
    vector<vector<vector<vector<vector<vector<uint64_t>>>>>> ker_plain;
    vector<vector<vector<vector<uint64_t>>>> act_plain;
    vector<vector<vector<uint64_t>>> dense_ker_plain;    // [10][4*nch][lvl+1] = [10][NB[3]][]
    vector<vector<uint64_t>> dense_bias_plain;           // [10][1]
    
    nGraphEncode.prepare_network2d(ker_plain, act_plain, dense_ker_plain, dense_bias_plain,
                                 ker, real_poly, dense_ker, dense_bias, NB_CHANNELS);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Encryption " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    vector<vector<vector<Ciphertext>>> xCipher(2, vector<vector<Ciphertext>> (32, vector<Ciphertext>(15)));
    
    int ntest = id_end + 1 - id_st;     // ntest samples in one ciphertext (default = 608)
    int tensor_size = 32 * 15 * 2;
    int tensor_half_size = 32 * 15;     // 480
    int nslots = encoder.slot_count();
    
    if(ntest > nslots) {
        throw invalid_argument("cannot encode all the samples into a single ciphertext");
    }
    
#pragma omp parallel for
    for(int t = 0; t < tensor_size; ++t){
        vector<double> msg (nslots, 0.0); // take the t-th samples
        for(int i = id_st; i <= id_end; i++){
            msg[i - id_st] = (test_input[i][t]);
        }

        Plaintext plain;
        encoder.encode(msg, Param_HEAR::qscale, plain);
        int ch = (t < tensor_half_size? 0 : 1);
        int height = (ch == 0? t / 15 : (t - tensor_half_size) / 15);
        int width = (t % 15);
        encryptor.encrypt(plain, xCipher[ch][height][width]); // [2][32][15]
    }
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    
    cout << "(" << ntest << ") [" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv2d::memoryscale)  << "(GB), xCtxt=[" << xCipher.size() << "][" << xCipher[0].size() << "][" << xCipher[0][0].size() << "]" << endl;

    //cout << "| --->  Modulus chain index for ct: q[" << context->get_context_data(xCipher[0][0][0].parms_id()) -> chain_index() << "]" << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B1): " ;
//    cout << "+------------------------------------+" << endl;
    
    HEnGraphEval nGrapheval(evaluator, relin_keys);
    
    // Conv1: xCipher[2][32*15], ker_plain[0][nch][2][3][3][lvl+1] -> res[nch][32][15]
    time_start = chrono::high_resolution_clock::now();
    
    vector<vector<vector<Ciphertext>>> res;
    nGrapheval.Eval_Conv2d(res, xCipher, ker_plain[0]);

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Act1: res[nch][32][15], act_plain[0][nch][3]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly2d(res, act_plain[0]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";

    // Pool1: res[nch][32][15] (valid ciphertexts = [nch][16][7])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Average2d(res);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;
    
    for(int i = 0; i < 2; i++){
        vector<double> dmsg;
        Plaintext pmsg;
        decryptor.decrypt(res[0][0][2*i], pmsg);
        encoder.decode(pmsg, dmsg);
        cout << dmsg[0] << "\t";
    }
    cout << endl;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B2): ";
//    cout << "+------------------------------------+" << endl;

    // Conv2: output a puctured ciphertext res[2*nch][16][7]
    time_start = chrono::high_resolution_clock::now();

    xCipher.clear();
    nGrapheval.Eval_Puctured_Conv2d(xCipher, res, ker_plain[1]); // res[256][16][17]
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Act2: res[2*nch][16][7]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly2d(xCipher, act_plain[1]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Pool2: res[2*nch][16][7] (valid ciphertexts = [2*nch][8][3])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Average2d(xCipher);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Evaluation (B3): ";
//    cout << "+------------------------------------+" << endl;

    // Conv3: output a puctured ciphertext res[4*nch][8][3]
    time_start = chrono::high_resolution_clock::now();
    
    res.clear();
    nGrapheval.Eval_Puctured_Conv2d(res, xCipher, ker_plain[2]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "conv(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Act3: res[4*nch][8][3]
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_BN_ActPoly2d(res, act_plain[2]);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "act(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... ";
    
    // Pool3: res[4*nch][8][3] (valid ciphertexts = [4*nch][1])
    time_start = chrono::high_resolution_clock::now();
    
    nGrapheval.Eval_Global_Average2d(res);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "avg(" << time_diff.count()/1000000.0 << "," <<(double) usage.ru_maxrss/(Param_conv2d::memoryscale) << "GB)... " << endl;

//    cout << "+------------------------------------+" << endl;
    cout << "> Dense... " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();
    
    xCipher.clear();
    nGrapheval.Eval_Dense2d(xCipher, res, dense_ker_plain, dense_bias_plain);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    time_total_eval += time_diff;
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << "," ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << ",";
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    cout << ">> Total Eval Time = [" << time_total_eval.count()/1000000.0 << " s] " << endl;
    outf << time_total_eval.count()/1000000.0 << "," ;
    
//    cout << "+------------------------------------+" << endl;
    cout << "> Decryption: " ;
//    cout << "+------------------------------------+" << endl;
    
    time_start = chrono::high_resolution_clock::now();

    int nclass = dense_bias.size();   // 10
    output.resize(ntest, vector<double> (nclass));
    for(int j = 0; j < nclass; j++){
        vector<double> dmsg;
        Plaintext pmsg;
        decryptor.decrypt(xCipher[j][0][0], pmsg);
        encoder.decode(pmsg, dmsg);
        
        for(int i = 0; i < ntest; i++){
            output[i][j] = dmsg[i];
            if(i == 0) cout << dmsg[i] * 10 << "," ;
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    getrusage(RUSAGE_SELF, &usage);
    outf << time_diff.count()/1000000.0 << endl ;
    outfm << (double) usage.ru_maxrss/(Param_conv2d::memoryscale) << endl;
    cout << "[" << time_diff.count()/1000000.0 << " s] w/ " << (double) usage.ru_maxrss/(Param_conv1d::memoryscale)  << "(GB)" << endl;
    
    outf.close();
    outfm.close();
}
