#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "thread.h"
#include "param.h"
#include "utils.h"
#include "Test_ngraph.h"
#include "Test_lola.h"
#include <omp.h>

using namespace std;

// example: ./sota ngraph 2 16 0 681 128
int main(int argc, char **argv)
{
    string method = argv[1];                // ngraph, lola
    int dim = atoi(argv[2]);                 // 1 or 2
    size_t NUM_THREADS = atoi(argv[3]);     // number of threads
    int id_st = atoi(argv[4]);              // starting id
    int id_end = atoi(argv[5]);             // ending id
    int nch = atoi(argv[6]);                // number of channels at the 1st convolutional layer
    
    Thread::initThreadPool(NUM_THREADS);
    omp_set_num_threads(NUM_THREADS);
    
    cout << "method=" << method << ", id=(" << id_st << "~" << id_end << "), #(threads)=" <<  omp_get_max_threads() << ", #(channels)=(" << nch << "," << 2*nch << "," << 4*nch << ")" << endl;
     
//    cout << "+------------------------------------+" << endl;
//    cout << "|      0. read the input dataset     |" << endl;
//    cout << "+------------------------------------+" << endl;

    // 0.1. parameters
    vector<dten> ker1d (3);
    vector<ften> ker2d (3);
    
    vector<dmat> real_poly (3);
    dmat dense_ker;
    dvec dense_bias;
    string param_dir = "../../dataset/parameters_conv" + to_string(dim) + "d_" + to_string(nch) + "/";
    
    if(dim == 1){
        read_parameters_conv1d(ker1d, real_poly, dense_ker, dense_bias, nch, param_dir);
    } else if(dim == 2){
        read_parameters_conv2d(ker2d, real_poly, dense_ker, dense_bias, nch, param_dir);
    }
    
    // 0.2. testing data and labels
    string input_filename = "../../dataset/test_input.csv";
    string label_filename = "../../dataset/test_label.csv";
    string prob_filename = "../../dataset/test_prob_conv" + to_string(dim) + "d_" +  to_string(nch) + ".csv";

    char split_char = 0x2C;

    // read the test input data and take the id-th row as the actual testing sample
    vector<vector<double>> test_input;   // [608][3*32*15], testing set
    read_matrix(test_input, split_char, input_filename);
    if(id_end >= test_input.size()){
        id_end = test_input.size() - 1;
    }

    vector<double> test_labels;
    read_onecol(test_labels, label_filename);
    
    vector<vector<double>> pred_probs;// read the estimated probabilities
    read_matrix(pred_probs, split_char, prob_filename);
   
//    cout << "+------------------------------------+" << endl;
//    cout << "|               HE-CNN               |" << endl;
//    cout << "+------------------------------------+" << endl;

    dmat HE_prob;
    
    if(method == "ngraph"){
        TestnGraph test;
        if(dim == 1){
            test.hecnn1d(HE_prob, test_input, id_st, id_end, ker1d, real_poly, dense_ker, dense_bias, nch);
        } else if(dim == 2){
            test.hecnn2d(HE_prob, test_input, id_st, id_end, ker2d, real_poly, dense_ker, dense_bias, nch);
        }
    } else if(method == "lola"){
        TestLoLA test;
        if(dim == 1){
            test.hecnn1d(HE_prob, test_input, id_st, id_end, ker1d, real_poly, dense_ker, dense_bias, nch);
        } else if(dim == 2){
            test.hecnn2d(HE_prob, test_input, id_st, id_end, ker2d, real_poly, dense_ker, dense_bias, nch);
        }
    } else if(method == "plain"){
        TestPlain test;
        test.hecnn2d(HE_prob, test_input, id_st, id_end, ker2d, real_poly, dense_ker, dense_bias, nch);
    }

//    cout << "+------------------------------------+" << endl;
//    cout << "|         Store the Results          |" << endl;
//    cout << "+------------------------------------+" << endl;

    string HE_prob_filename;

    if(method == "ngraph"){
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + ".txt";
    } else if(method == "lola"){
        int resid = id_st/50; // compute 50 many samples at one time
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + "_" + to_string(resid) + ".txt";
    } else if(method == "plain"){
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + ".txt";
    }
    
    // store the results as a tex file
    fstream outff;
    outff.open(HE_prob_filename.c_str(), fstream::in | fstream::out | fstream::app);   // open the file

    for (int id = id_st; id <= id_end; id++){
        for(int j = 0; j < 9; ++j){
            outff << HE_prob[id-id_st][j] * 10.0 << "," ;
        }
        outff << HE_prob[id-id_st][9] * 10.0 << endl;
    }
    outff.close();
    
    return 0;
}
