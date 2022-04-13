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

#include "param.h"
#include "utils.h"

using namespace std;

// example: ./accuracy fhear 1 64 giant
// example: ./accuracy ngraph 1 128 giant
// example: ./accuracy ngraph 2 64 giant

int main(int argc, char **argv)
{
    string method = argv[1];                // hear, fhear
    int dim = atoi(argv[2]);                 // 1 or 2 (conv1d or conv2d)
    int nch = atoi(argv[3]);                // number of channels at the 1st convolutional layer
    string mode2 = argv[4];                 // fully, baby, giant
    
//    cout << "+------------------------------------+" << endl;
//    cout << "|      0. read the input dataset     |" << endl;
//    cout << "+------------------------------------+" << endl;

    string input_filename = "../../dataset/test_input.csv";
    string label_filename = "../../dataset/test_label.csv";
    string prob_filename = "../../dataset/test_prob_conv" + to_string(dim) + "d_" +  to_string(nch) + ".csv";
    
    char split_char = 0x2C;
    
    // read the test input data and take the id-th row as the testing sample
    vector<vector<double>> test_input;      // [608][3*32*15], testing set
    read_matrix(test_input, split_char, input_filename);
    
    vector<double> test_labels;
    read_onecol(test_labels, label_filename);
    
    vector<vector<double>> pred_probs;      // read the estimated probabilities
    read_matrix(pred_probs, split_char, prob_filename);
    
//    cout << "+------------------------------------+" << endl;
//    cout << "|               HE-CNN               |" << endl;
//    cout << "+------------------------------------+" << endl;

    string HE_prob_filename;
     
    if(method == "lola"){
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_lola.txt";
    } else if(method == "ngraph"){
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_ngraph.txt";
    } else if((method == "hear")|| (method == "fhear")) {
        HE_prob_filename = "result/test_prob_conv" + to_string(dim) + "d_" + to_string(nch) + "_" + method + "_" + mode2 + ".txt";
    }
   
    // correctness
    vector<int> HE_labels;
    vector<int> plain_labels;
    vector<int> true_labels;
    vector<vector<double>> HE_output;
    int ntest = test_labels.size();
    
    // read the probabilities for all samples
    read_matrix(HE_output, split_char, HE_prob_filename);

    if(HE_output.size() != ntest){
        throw invalid_argument("Error: HE_output is not ready for performance calculation");
    }
    
    // compute the labels
    for (int id = 0; id < ntest; id++){
        //cout << "===========(" << id << ")===========" << endl;
        int HE_label = argmax(HE_output[id]);           // label from an encrypted computation with the trained model
        int plain_label = argmax(pred_probs[id]);       // label from an unencrypted computation with the trained model
        int true_label = int(test_labels[id]);           // actual label

        HE_labels.push_back(HE_label);
        plain_labels.push_back(plain_label);
        true_labels.push_back(true_label);
    }

    int fall_label = 9;
    double accuracy, sensitivity, specificity, precision, F1score;

    // compare HE_labels and true_labels
    get_performance(accuracy, sensitivity, specificity, precision, F1score, HE_labels, true_labels, fall_label);

    cout << "(acc) HE vs true: acc=" << accuracy << ", sens=" << sensitivity << ", spec=" << specificity << ", prec=" << precision << ", F1score=" << F1score << endl;
     
    // compare plain_labels and true_labels
    get_performance(accuracy, sensitivity, specificity, precision, F1score, plain_labels, true_labels, fall_label);
    
    cout << "(acc) plain vs true: acc=" << accuracy << ", sens=" << sensitivity << ", spec=" << specificity << ", prec=" << precision << ", F1score=" << F1score << endl;
    return 0;
}









