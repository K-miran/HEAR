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
#include "Test_HEAR.h"

using namespace std;

#define unit 200

// conv1: [32][15][3] -> [32][15][128] -> avg1: [16][7][128]
// conv2: [16][7][128] -> [16][7][256] -> avg2: [8][3][256]
// conv3: [8][3][256] -> [8][3][512] -> avg3: [512]

int main(int argc, char **argv)
{
    string method = argv[1];                // hear, fhear
    int dim = atoi(argv[2]);                 // 1 or 2 (conv1d or conv2d)
    size_t NUM_THREADS = atoi(argv[3]);     // number of threads
    int id_st = atoi(argv[4]);               // starting id
    int id_end = atoi(argv[5]);              // ending id
    int nch = atoi(argv[6]);                 // number of channels at the 1st convolutional layer
    string mode1 = argv[7];                 // fully, baby
    string mode2 = argv[8];                 // fully, baby, giant
    string mode3 = argv[9];                 // fully, baby, giant
    
    Thread::initThreadPool(NUM_THREADS);
    cout << "id=(" << id_st << "~" << id_end << "), #(threads)=" << NUM_THREADS << ", #(channels)=(" << nch << "," << 2*nch << "," << 4*nch << ")" << endl;
    cout << "Method=" << method << ", Mode=(" << mode1 << "," << mode2 << "," << mode3 << ")"  << endl;
     
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
    } else if(dim ==2){
        read_parameters_conv2d(ker2d, real_poly, dense_ker, dense_bias, nch, param_dir);
    }
    
    // 0.2. testing data and labels
    string input_filename = "../../dataset/test_input.csv";
    string label_filename = "../../dataset/test_label.csv";
    string prob_filename = "../../dataset/test_prob_conv" + to_string(dim) + "d_" +  to_string(nch) + ".csv";
    
    char split_char = 0x2C;
    
    // read the test input data and take the id-th row as the testing sample
    vector<vector<double>> test_input;          // [608][3*32*15], testing set
    read_matrix(test_input, split_char, input_filename);
    if(id_end >= test_input.size()){
        id_end = test_input.size() - 1;
    }
    
    vector<double> test_labels;
    read_onecol(test_labels, label_filename);

   
//    cout << "+------------------------------------+" << endl;
//    cout << "|               HE-CNN               |" << endl;
//    cout << "+------------------------------------+" << endl;
    
    dmat HE_prob;

    if(dim == 1){
        TestHEAR1d test;
        test.hecnn(HE_prob, test_input, id_st, id_end,
                   ker1d, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, nch, method);
    } else if(dim == 2){
        TestHEAR2d test;
        test.hecnn(HE_prob, test_input, id_st, id_end,
                   ker2d, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, nch, method);
        
        //This is to explore the effect of multi-threading implementation.
        //test.hecnn_threading(HE_prob, test_input, id_st, id_end,
            //ker2d, real_poly, dense_ker, dense_bias, mode1, mode2, mode3, nch, method, NUM_THREADS);
    }

//    cout << "+------------------------------------+" << endl;
//    cout << "|         Store the Results          |" << endl;
//    cout << "+------------------------------------+" << endl;

    string HE_prob_filename;

    if((method == "hear")&&(dim == 2) && (nch == 128) && (mode2 == "fully")){
        int resid = id_st/unit;
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + "_" + mode2 + "_" + to_string(resid) + ".txt";
    } else {
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
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
