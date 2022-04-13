#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <vector>

#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include "time.h"
#include "utils.h"

using namespace std;

/*
 @param[in] data, a double-type  vector
 @param[in] mode, the input name
*/
void getshape(dvec data, string mode)
{
    cout << mode << ": [" << data.size() << "]" << endl;
}

/*
 @param[in] data, double-type matrix
 @param[in] mode, the input name
*/
void getshape(dmat data, string mode)
{
    cout << mode << ": [" << data.size() << "][" << data[0].size() << "]" << endl;
}

/*
 @param[in] data, a double-type tensor
 @param[in] mode, the input name
*/
void getshape(dten data, string mode)
{
    cout << mode <<  ": [" << data.size() << "][" << data[0].size() << "][" << data[0][0].size() << "]" << endl;
}

/*
 @param[in] data, a double-type tensor in a four-dimensional spacetime.
 @param[in] mode, the input name
*/
void getshape(ften data, string mode)
{
    cout << mode <<  ": [" << data.size() << "][" << data[0].size() << "][" ;
    cout << data[0][0].size() << "][" << data[0][0][0].size() << "]" << endl;
}


/*
 @param[in] filepath, the location of a file
 @param[out] res, the double-type value
 */
double read_oneval(string filepath)
{
    ifstream openFile(filepath.data());
    string line;
    char split_char = 0x0A; // new line
    double res = 0.0;
    
    if(openFile.is_open()){
        while(getline(openFile, line, split_char)){
            res = (stod(line));
        }
    }
    return res;
}

/*
 @param[in] filepath, the location of an one-column input file
 @param[out] res, the double-type vector
*/
void read_onecol(dvec& res, string filepath)
{
    string line;
    
    ifstream myfile (filepath);
    if (myfile.is_open())
    {
      while (getline (myfile,line))
      {
          //res.push_back(line);
          res.push_back(stod(line)); 
      }
      myfile.close();
    }
    else{
        throw invalid_argument("Error: cannot read file");
    }
}

/*
 @param[in] filepath, the location of a two-dimensional input file
 @param[out] res, the double-type matrix
*/
void read_matrix(dmat& res, char split_char, string filepath)
{
    ifstream openFile(filepath.data());
    
    vector<vector<string>> res_str;
    if(openFile.is_open()) {
        string line;
        
        /*  read each line */
        while(getline(openFile, line)){
            vector<string> vecsline;
            istringstream split(line);
            vector<string> tokens;
            for (string each; getline(split, each, split_char); tokens.push_back(each));
                
            res_str.push_back(tokens);
        }
    } else {
        throw invalid_argument("Error: cannot read file");
    }
    
    for(int i = 0; i < (int) res_str.size(); ++i){
        vector<double> dtemp;
        for(int j = 0; j < (int) res_str[i].size(); ++j){
            dtemp.push_back(atof(res_str[i][j].c_str()));
        }
        res.push_back(dtemp);
    }
}

/*
 @param[in] data, a double-type input data
 @param[in] filename, the location of an output file
 @param[in] dim, a dimension that specifies a line-break
*/
void output_with_file(vector<double> data, string filename, size_t dim)
{
    ofstream outf_tmp(filename);
    outf_tmp.close();
    
    fstream outf;
    outf.open(filename.c_str(), fstream::in | fstream::out | fstream::app);   // open the file
    
    outf << "[";
    for(size_t i = 0; i < data.size(); ++i){
        outf << data[i];
        if(((i + 1) % dim == 0)){
            outf << "]" << endl;
            outf << "-------------------------------------------------------------" << endl;
            outf << "[" ;
        }
        else if(((i + 1) % (dim * dim) == 0)){
            outf << "]" << endl;
            outf << "==============================================================" << endl;
            outf << "[" ;
        }
        else{
            outf << "\t";
        }
    }
    outf.close();
}

/*
 @param[in] data, a double-type input data
 @param[in] filename, the location of an output file
*/
void output_with_file(vector<double> data, string filename)
{
    ofstream outf_tmp(filename);
    outf_tmp.close();
    
    fstream outf;
    outf.open(filename.c_str(), fstream::in | fstream::out | fstream::app);   // open the file
    
    for(size_t i = 0; i < data.size(); ++i){
        outf << data[i] << endl; 
    }
     outf.close();
}

/*
 @param[in] input, the double-type vector
 @param[in] height, the height
 @param[in] width, the weight
 @param[out] res, a two-dimensional matrix of size [ht][wt]
 */
void reshape(dmat &res, dvec input, int height, int width)
{
    res.resize(height, vector<double> (width));
       
    int index = 0;
    for(int k = 0; k < height; ++k){
        for(int l = 0; l < width; ++l){
            res[k][l] = input[index];
            index++;
        }
    }
}

/*
 @param[in] input, the double-type vector
 @param[in] nchannels, the number of channels
 @param[in] height, the height
 @param[in] width, the weight
 @param[out] res, a three-dimensional tensor of size [nchannels][ht][wt]
*/
void reshape(dten &res, dvec input, int nchannels, int filter_height, int filter_width)
{
    res.resize(nchannels, vector<vector<double>> (filter_height, vector<double>(filter_width)));
    
    int index = 0;
    for(int j = 0; j < nchannels; ++j){
        for(int k = 0; k < filter_height; ++k){
            for(int l = 0; l < filter_width; ++l){
                res[j][k][l] = input[index];
                index++;
            }
        }
    }
}

/*
 @param[in] input, the double-type vector
 @param[in] out_nchannels, the number of output channels
 @param[in] in_nchannels, the number of input channels
 @param[in] filter_height, the height of filters
 @param[in] filter_width, the width of filters
 @param[out] res, a four-dimensional tensor of size [out_nchannels][in_channels][ht][wt]
 */
void reshape(ften &res, dvec input, int out_nchannels, int in_nchannels, int filter_height, int filter_width)
{
    res.resize(out_nchannels, vector<vector<vector<double>>> (in_nchannels,
                                                              vector<vector<double>>(filter_height, vector<double>(filter_width))));
    int index = 0;
    for(int i = 0; i < out_nchannels; ++i){
        for(int j = 0; j < in_nchannels; ++j){
            for(int k = 0; k < filter_height; ++k){
                for(int l = 0; l < filter_width; ++l){
                    res[i][j][k][l] = input[index];
                    index++;
                }
            }
        }
    }
}

/*
 @param[in] input, the double-type vector
 @param[in] out_nchannels, the number of output channels
 @param[in] in_nchannels, the number of input channels
 @param[in] filter_height, the height of filters
 @param[in] filter_width, the width of filters
 @param[out] res, a four-dimensional tensor of size [out_nchannels][in_channels][ht][wt]
 The input is for [out_nchannels][in_nchannels + 1][hegith]][width], so carefully reshape it
*/
void reshape_special(ften &res, dvec input, int out_nchannels, int in_nchannels, int filter_height, int filter_width)
{
    res.resize(out_nchannels, vector<vector<vector<double>>> (in_nchannels,
                                                              vector<vector<double>>(filter_height, vector<double>(filter_width))));
    int index = 0;
    for(int i = 0; i < out_nchannels; ++i){
        for(int j = 0; j < in_nchannels; ++j){
            for(int k = 0; k < filter_height; ++k){
                for(int l = 0; l < filter_width; ++l){
                    res[i][j][k][l] = input[index];
                    index++;
                }
            }
        }
        for(int k = 0; k < filter_height; ++k){
            for(int l = 0; l < filter_width; ++l){
                index++;
            }
        }
    }
}

/*
 @param[in] input, the double-type vector
 @param[in] out_nchannels, the number of output channels
 @param[in] in_nchannels, the number of input channels
 @param[in] filter_size, the size of filters
 @param[out] res, a three-dimensional tensor of size [out_nchannels][in_channels][size]
 */
void reshape_conv1d(dten &res, dvec input, int out_nchannels, int in_nchannels, int filter_size)
{
    res.resize(out_nchannels, vector<vector<double>> (in_nchannels, vector<double>(filter_size)));
                                                             
    int index = 0;
    for(int i = 0; i < out_nchannels; ++i){
        for(int j = 0; j < in_nchannels; ++j){
            for(int k = 0; k < filter_size; ++k){
                res[i][j][k] = input[index];
                index++;
            }
        }
    }
}

/*
 @param[in] input, the double-type vector
 @param[in] out_nchannels, the number of output channels
 @param[in] in_nchannels, the number of input channels
 @param[in] filter_size, the size of filters
 @param[out] res, a three-dimensional tensor of size [out_nchannels][in_channels][size]
 The input is for [out_nchannels][in_nchannels + 1][size], so carefully reshape it
*/
void reshape_special_conv1d(dten &res, dvec input, int out_nchannels, int in_nchannels, int filter_size)
{
    res.resize(out_nchannels, vector<vector<double>> (in_nchannels, vector<double>(filter_size)));
   
    int index = 0;
    for(int i = 0; i < out_nchannels; ++i){
        for(int j = 0; j < in_nchannels; ++j){
            for(int k = 0; k < filter_size; ++k){
                res[i][j][k] = input[index];
                index++;
            }
        }
        for(int k = 0; k < filter_size; ++k){
            index++;
        }
    }
}

/*
 @param[in] id, the considered row index of an input data
 @param[in] split_char, a delimiting character
 @param[in] filpath, the location of an input data
 @param[out] res, [3][32][15]
*/
void read_input_data(dten &res, int id, char split_char, string filpath)
{
    // Step 1: first read the whole data: n * (3*32*15)
    dmat res_temp;
    read_matrix(res_temp, split_char, filpath);
    
    // Step 2: take the id-th row in the whold dataset and reshape
    reshape(res, res_temp[id], 3, 32, 15);
}


/*
 @param[in] filpath, the location of an input data
 @param[out] res, [3][32][15]
 Read the all input data
*/
void read_input_data(dten &res, string filepath)
{
    vector<double> res_temp;
    read_onecol(res_temp, filepath);

    reshape(res, res_temp, 3, 32, 15);
    //getshape(res, "> input data");
}


/*
 @param[in] bias, the bias of the convolution
 @param[in] gamma, the gamma of BN
 @param[in] beta, the beta of BN
 @param[in] mean, the mean of BN
 @param[in] var, the var of BN
 @param[in] act_poly[3], a vector of (a[0], a[1], a[2])
 @param[in] pool_size, the pool size (e.g., 4)
 @param[in] epsilon, the epsilon of BN
 @param[out] real_poly[3], the double-type precomputed values for bias, BN, poly-act, and divide by pool_size
 1. y <= x + bias
 2. BN: gamma * (y - mu)/sqrt(var + epsilon) + beta = (gamma/sqrt(var + epsilon)) * y + (beta - (gamma * mean)/sqrt(var + epsilon))
       = d0 * x + d1 = z
 3. poly-act: a[0] + a[1] * z + a[2] * z^2 = a[0] + a[1] * (d0 * x + d1) + a[2] * (d0 * x + d1)^2
              = (a[2] * d1^2 + a[1] * d1 + a[0]) + (2 * a[2] * d0 * d1 + a[1] * d0) * x + (a[2] * d0^2) * x^2
              = (coeff[0] + coeff[1] * x + coeff[2] * x^2)
 4. real_poly[i] = coeff[i] / pool_size for 0 <= i < 3
 */
void aggregate_bias_BN_actpoly_avg(dmat &real_poly, dvec bias,
                               dvec gamma, dvec beta, dvec mean, dvec var,
                               dvec act_poly, double pool_size, double epsilon)
{
    if(real_poly.size() != bias.size()){
        throw invalid_argument("Error: initialization of real_poly");
    }
    
    for(size_t i = 0; i < bias.size(); ++i){
        double d0 = gamma[i] / sqrt(var[i] + epsilon);
        double d1 = d0 * (bias[i] - mean[i]) + beta[i];
        
        real_poly[i][0] = ((act_poly[2] * d1 * d1) + (act_poly[1] * d1) + act_poly[0])/pool_size;   // constant-term
        real_poly[i][1] = (2 * act_poly[2] * d0 * d1 + act_poly[1] * d0)/pool_size;   // deg=1-term
        real_poly[i][2] = (act_poly[2] * d0 * d0)/pool_size;  // deg=2-term
    }
}


/*
 @param[in] nch, the number of channels at the first convolutional layer
 @param[om] param_dir, the location of the parameter file
 @param[out] ker[3][out_nchannels][in_nchannels][3][3]
 in_nchannels is the number of in-channles of filters/kernels tensors
 out_nchannels is the number of out-channles of filters/kernels tensors
 @param[out] real_poly, the coefficients of polynomials of conv_bias + BN + ACT, [3][*][*]
 @param[out] dense_ker, the kernel of dense layer, [10][*]
 @param[out] dense_bias, the bias of the dense layer, [10]
 Read the model parameters and store it 
 */
void read_parameters_conv2d(vector<ften> &ker, vector<dmat> &real_poly, dmat &dense_ker, dvec &dense_bias,
                            int nch, string param_dir)
{
    vector<int> NUM_CHANNELS = {2, nch, 2*nch, 4*nch};
    
    for(int l = 1; l < 4; ++l){
        string in_nchannels;
        if(l == 1){
            in_nchannels = to_string(NUM_CHANNELS[l - 1] + 1);
        }
        else{
            in_nchannels = to_string(NUM_CHANNELS[l - 1]);
        }
        string out_nchannels = to_string(NUM_CHANNELS[l]);
        
        // 1. read the weight of convolution and reshape to 4D-tensor
        dvec conv_weight;
        string conv_filename = param_dir + "conv" + to_string(l) + ".weight_torch.Size([" + out_nchannels + ", " + in_nchannels + ", 3, 3]).csv";
        read_onecol(conv_weight, conv_filename);    // = [128][3][3][3]
        
        if(l == 1){
            reshape_special(ker[l - 1], conv_weight, NUM_CHANNELS[l], NUM_CHANNELS[l - 1], 3, 3);
        }
        else{
            reshape(ker[l - 1], conv_weight, NUM_CHANNELS[l], NUM_CHANNELS[l - 1], 3, 3);
        }
        //getshape(ker[l - 1], "> ker" + to_string(l));

        // 2. read the bias of convolution
        dvec conv_bias;
        read_onecol(conv_bias, param_dir + "conv" + to_string(l) + ".bias_torch.Size([" + out_nchannels + "]).csv");
        //getshape(conv_bias, "> bias" + to_string(l));

        // read the weights for Batch-normalization
        dvec mean;
        dvec var;
        dvec gamma;
        dvec beta;

        read_onecol(mean, param_dir + "bn" + to_string(l) + ".running_mean_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(var, param_dir + "bn" + to_string(l) + ".running_var_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(gamma, param_dir + "bn" + to_string(l) + ".weight_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(beta, param_dir + "bn" + to_string(l) + ".bias_torch.Size([" + out_nchannels + "]).csv");
     
        // act_poly = act_poly[0] + act_poly[1] * x + act_poly[2] * x ** 2
        dvec act_poly;
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".c_torch.Size([1]).csv"));       // const term
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".beta_torch.Size([1]).csv"));    // deg=1
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".alpha_torch.Size([1]).csv"));   // deg=2

        real_poly[l - 1].resize(NUM_CHANNELS[l], vector<double>(3));  // [128][3], [256][3], [512][3]
        aggregate_bias_BN_actpoly_avg(real_poly[l - 1], conv_bias,
                                      gamma, beta, mean, var, act_poly, Param_conv2d::POOL_SIZE[l - 1], Param_conv2d::epsilon);
        //getshape(real_poly[l - 1], "> real_poly" + to_string(l));
    }
    
    // Dense layer
    dvec dense_ker_temp;
    
    read_onecol(dense_ker_temp, param_dir + "linear.weight_torch.Size([10, " + to_string(NUM_CHANNELS[3]) + "]).csv");
    reshape(dense_ker, dense_ker_temp, 10, NUM_CHANNELS[3]);
    //getshape(dense_ker, "> dense_ker");

    read_onecol(dense_bias, param_dir + "linear.bias_torch.Size([10]).csv");
    //getshape(dense_bias, "> dense_bias");
}

/*
 @param[in] nch, the number of channels at the first convolutional layer
 @param[om] param_dir, the location of the parameter file
 @param[out] ker[3][out_nchannels][in_nchannels][3]
 in_nchannels is the number of in-channles of filters/kernels tensors
 out_nchannels is the number of out-channles of filters/kernels tensors
 @param[out] real_poly, the coefficients of polynomials of conv_bias + BN + ACT, [3][*][*]
 @param[out] dense_ker, the kernel of dense layer, [10][*]
 @param[out] dense_bias, the bias of the dense layer, [10]
 Read the model parameters and store it
 */
void read_parameters_conv1d(vector<dten> &ker, vector<dmat> &real_poly, dmat &dense_ker, dvec &dense_bias,
                            int nch, string param_dir)
{
    vector<int> NUM_CHANNELS = {2, nch, 2*nch, 4*nch};
    for(int l = 1; l < 4; ++l){
        string in_nchannels;
        if(l == 1){
            in_nchannels = to_string(NUM_CHANNELS[l - 1] + 1);    // 2 -> 3
        }
        else{
            in_nchannels = to_string(NUM_CHANNELS[l - 1]);
        }
        string out_nchannels = to_string(NUM_CHANNELS[l]);
        
        // 1. read the weight of convolution and reshape to 4D-tensor
        dvec conv_weight;
        string conv_filename = param_dir + "conv" + to_string(l) + ".weight_torch.Size([" + out_nchannels + ", " + in_nchannels + ", 3]).csv";
        read_onecol(conv_weight, conv_filename);    // = [128][3][3]
        
        if(l == 1){
            reshape_special_conv1d(ker[l - 1], conv_weight, NUM_CHANNELS[l], NUM_CHANNELS[l - 1], 3);
        }
        else{
            reshape_conv1d(ker[l - 1], conv_weight, NUM_CHANNELS[l], NUM_CHANNELS[l - 1], 3);
        }
        //getshape(ker[l - 1], "> ker" + to_string(l));
 
        // 2. read the bias of convolution
        dvec conv_bias;
        read_onecol(conv_bias, param_dir + "conv" + to_string(l) + ".bias_torch.Size([" + out_nchannels + "]).csv");
        //getshape(conv_bias, "> bias" + to_string(l));

        // read the weights for Batch-normalization
        dvec mean;
        dvec var;
        dvec gamma;
        dvec beta;

        read_onecol(mean, param_dir + "bn" + to_string(l) + ".running_mean_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(var, param_dir + "bn" + to_string(l) + ".running_var_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(gamma, param_dir + "bn" + to_string(l) + ".weight_torch.Size([" + out_nchannels + "]).csv");
        read_onecol(beta, param_dir + "bn" + to_string(l) + ".bias_torch.Size([" + out_nchannels + "]).csv");
        
        // act_poly = act_poly[0] + act_poly[1] * x + act_poly[2] * x ** 2
        dvec act_poly;
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".c_torch.Size([1]).csv"));       // const term
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".beta_torch.Size([1]).csv"));    // deg=1
        act_poly.push_back(read_oneval(param_dir + "act" + to_string(l) + ".alpha_torch.Size([1]).csv"));   // deg=2

    
        real_poly[l - 1].resize(NUM_CHANNELS[l], vector<double>(3));  // [128][3], [256][3], [512][3]
        aggregate_bias_BN_actpoly_avg(real_poly[l - 1], conv_bias,
                                      gamma, beta, mean, var, act_poly, Param_conv1d::POOL_SIZE[l - 1], Param_conv1d::epsilon);
        
        //getshape(real_poly[l - 1], "> real_poly" + to_string(l));
    }
 
    // Dense layer
    dvec dense_ker_temp;

    read_onecol(dense_ker_temp, param_dir + "linear.weight_torch.Size([10, " + to_string(NUM_CHANNELS[3]) + "]).csv");
    reshape(dense_ker, dense_ker_temp, 10, NUM_CHANNELS[3]);
    //getshape(dense_ker, "> dense_ker");

    read_onecol(dense_bias, param_dir + "linear.bias_torch.Size([10]).csv");
    //getshape(dense_bias, "> dense_bias");

}

//-------------------
// Functions
//-------------------

/*
@param[in] vals, The input vector
@param[in] steps, The number of steps to rotate
@param[out] res, Left rotation by "steps"
*/
void msgleftRotate(dvec& res, dvec vals, int steps){
    int dim = vals.size();
    int nshift = steps % dim;
    int k = dim - nshift;
    
    for(int j = 0; j < k; ++j){
        res[j] = vals[j + nshift];
    }
    for(int j = k; j < dim; ++j){
        res[j] = vals[j - k];
    }
}

/*
@param[in] vals, The input vector
@param[in] steps, The number of steps to rotate
@param[out] res, Right rotation by "steps"
*/
void msgrightRotate(dvec& res, dvec vals, int steps){
    int dim = vals.size();
    int nshift = steps % dim;
    int k = dim - nshift;
    
    for(int j = 0; j < dim - k; ++j){
        res[j] = vals[j + k];
    }

    for(int j = k; j < dim; ++j){
        res[j] = vals[j - k];
    }
}

/*
@param[in] vals, The vectors to rotate
@param[in] steps, The number of steps to (left) rotate
*/
void msgleftRotate_inplace(dvec& vals, int steps){
    dvec res;
    int dim = vals.size();
    int nshift = steps % dim;
    int k = dim - nshift;
    
    for(int j = 0; j < k; ++j){
        res.push_back(vals[j + nshift]);
    }
    for(int j = k; j < dim; ++j){
        res.push_back(vals[j - k]);
    }
    
    // update
    for(int j = 0; j < dim; ++j){
        vals[j] = res[j];
    }
}

/*
@param[in] vals, The vectors to rotate 
@param[in] steps, The number of steps to (right) rotate
*/
void msgrightRotate_inplace(dvec& vals, int steps){
    dvec res;
    int dim = vals.size();
    int nshift = steps % dim;
    int k = dim - nshift;
    
    for(int j = 0; j < dim - k; ++j){
        res.push_back(vals[j + k]);
    }
    for(int j = 0; j < k; ++j){
        res.push_back(vals[j]);
    }
    

    for(int j = 0; j < dim; ++j){
        vals[j] = res[j];
    }
}

/*
 @param[in] kernel[3][3], a matrix of size 3*3
 @param[in] option, 0 <= option < 9; each number specifies the locations of zeros in the output
 @param[out] res, an one-dimensional vector of size (32 * 16)
 This function is used for encoding kernels at the first convolution layer over 2D-CNN.
 */
void ker1_to_vector(dvec &res, dmat kernel, int option){
    int len = 32 * 16;          // length of res
    int lastrow_st = 31 * 16;   // length of res
    
    switch(option){
        case 0: // (l, r, t)
            res.assign(len, kernel[0][0]);
            for(int i = 0; i < len; i+=16){
                res[i] = 0.0;
                res[i + 15] = 0.0;
            }
            for(int i = 1; i < 15; i++){
                res[i] = 0.0;
            }
            break;
        case 1: // (r, t)
            res.assign(len, kernel[0][1]);
            for(int i = 15; i < len; i+=16){
                res[i] = 0.0;
            }
            for(int i = 0; i < 15; i++){
                res[i] = 0.0;
            }
            break;
        case 2: // (rr, t)
            res.assign(len, kernel[0][2]);
            for(int i = 14; i < len; i+=16){
                res[i] = 0.0;
                res[i + 1] = 0.0;
            }
            for(int i = 0; i < 14; i++){
                res[i] = 0.0;
            }
            break;
        case 3: // (l, r)
            res.assign(len, kernel[1][0]);
            for(int i = 0; i < len; i+=16){
                res[i] = 0.0;
                res[i + 15] = 0.0;
            }
            break;
        case 4: // (r)
            res.assign(len, kernel[1][1]);
            for(int i = 15; i < len; i+=16){
                res[i] = 0.0;
            }
            break;
        case 5: // (rr)
            res.assign(len, kernel[1][2]);
            for(int i = 14; i < len; i+=16){
                res[i] = 0.0;
                res[i + 1] = 0.0;
            }
            break;
        case 6: // (l, r, b)
            res.assign(len, kernel[2][0]);
            for(int i = 0; i < len; i+=16){
                res[i] = 0.0;
                res[i + 15] = 0.0;
            }
            for(int i = lastrow_st + 1; i < len - 1; i++){
                res[i] = 0.0;
            }
            break;
        case 7: // (r, b)
            res.assign(len, kernel[2][1]);
            for(int i = 15; i < len; i+=16){
                res[i] = 0.0;
            }
            for(int i = lastrow_st; i < len - 1; i++){
                res[i] = 0.0;
            }
            break;
        case 8: // (rr, b)
            res.assign(len, kernel[2][2]);
            for(int i = 14; i < len; i+=16){
                res[i] = 0.0;       // the column before last
                res[i + 1] = 0.0;   // the last column
            }
            for(int i = lastrow_st; i < len - 2; i++){
                res[i] = 0.0;       // bottom
            }
            break;
    }
}

/*
 @param[in] kernel[-][3][3], a tensor
 @param[in] option, 0 <= option < 9; each number specifies the locations of zeros in the output
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, one-dimensional vector of size (32 * 16)
 This function is used for encoding kernels at the second/third convolution layers over 2D-CNN.
 */
void ker2_to_vector(dvec &res, dmat kernel, int option, int dist){
    int nrows = 32;
    int ncols = 16;
    int ncols1 = ncols - dist;    // the actual number of columns of input data of conv2 (16 * 7) while multiplying by two
    res.assign(nrows * ncols, 0ULL);
    
    switch(option){
        case 0:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[0][0];
                }
            }
            break;
        case 1:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[0][1];
                }
            }
            break;
        case 2:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    res[i * ncols + j] = kernel[0][2];
                }
            }
            break;
        case 3:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[1][0];
                }
            }
            break;
        case 4:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[1][1];
                }
            }
            break;
        case 5:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    res[i * ncols + j] = kernel[1][2];
                }
            }
            break;
        case 6:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[2][0];
                }
            }
            break;
        case 7:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    res[i * ncols + j] = kernel[2][1];
                }
            }
            break;
        case 8:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    res[i * ncols + j] = kernel[2][2];
                }
            }
            break;
    }
}

/*
 @param[in] kernel[4][3][3] or kernel[16][3][3], kernel[8][3][3] with dist=4
 @param[in] option, 0 <= option < 9; each number specifies the locations of zeros in the output
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, one-dimensional vector of size (32 * 16)
 This function is used for encoding kernels at the second/third convolution layers over 2D-CNN.
 We take as input multiple channels, which are packed together and encoded as a single plaintext.
 */
void ker2_to_interlaced_vector(dvec &res, dten kernels, int option, int dist){
    int nrows = 32;
    int ncols = 16;
    int ncols1 = ncols - dist;    // the actual number of columns of input data of conv2 (16 * 7) while multiplying by two
    res.assign(nrows * ncols, 0ULL);
    
    int packed_nrows = kernels.size()/dist;    // default = dist
    int packed_ncols = dist;
    
    switch(option){
        case 0:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][0];
                        }
                    }
                }
            }
            break;
        case 1:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][1];
                        }
                    }
                }
            }
            break;
        case 2:
            for(int i = dist; i < nrows; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][2];
                        }
                    }
                }
            }
            break;
        case 3:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][0];
                        }
                    }
                }
            }
            break;
        case 4:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][1];
                        }
                    }
                }
            }
            break;
        case 5:
            for(int i = 0; i < nrows; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][2];
                        }
                    }
                }
            }
            break;
        case 6:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = dist; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][0];
                        }
                    }
                }
            }
            break;
        case 7:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = 0; j < ncols1; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][1];
                        }
                    }
                }
            }
            break;
        case 8:
            for(int i = 0; i < nrows - dist; i+=dist){
                for(int j = 0; j < ncols1 - dist; j+=dist){
                    for(int k = 0; k < packed_nrows; k++){
                        for(int l = 0; l < packed_ncols; l++){
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][2];
                        }
                    }
                }
            }
            break;
    }
}


/*
 @param[in] kernel[3], a one-dimensinal vector
 @param[in] option, 0 <= option < 3;
 @param[out] res, a one-dimensional vector of size (32 * 16)
 This is used for encoding kernels at the first convolution layer over 1D-CNN.
 */
void ker1_to_vector_conv1d(dvec &res, dvec kernel, int option){
    int len = 32 * 16; // length of res
    int len_actual = 32 * 15; // length of res
    
    switch(option){
        case 0:
            res.assign(len_actual, kernel[0]);
            res[0] = 0.0;
            break;
        case 1:
            res.assign(len_actual, kernel[1]);
            break;
        case 2:
            res.assign(len_actual, kernel[2]);
            res[len_actual - 1] = 0.0;
            break;
    }
    
    vector<double> zeros (len - len_actual, 0ULL);
    res.insert(res.end(), zeros.begin(), zeros.end());
}

/*
 @param[in] kernel[3], a one-dimensinal vector
 @param[in] option, 0 <= option < 3
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, an one-dimensional vector of size (32 * 16)
 This is used for encoding kernels at the second/third convolution layers over 1D-CNN.
 */
void ker2_to_vector_conv1d(dvec &res, dvec kernel, int option, int dist){
    int len = 32 * 16;
    int actual_len = 32 * 15;
    res.assign(len, 0ULL);
    
    switch(option){
        case 0:
            for(int i = dist; i < actual_len; i+=dist){
                res[i] = kernel[0];
            }
            break;
        case 1:
            for(int i = 0; i < actual_len; i+=dist){
                res[i] = kernel[1];
            }
            break;
        case 2:
            for(int i = 0; i < actual_len - dist; i+=dist){
                res[i] = kernel[2];
            }
            break;
    }
}


/*
 @param[in] kernel[4][3][3] or kernel[16][3][3]
 @param[in] option, 0 <= option < 3
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, an one-dimensional vector of size (32 * 16)
 This is used for encoding kernels at the second/third convolution layers over 1D-CNN.
 We take as input multiple channels, which are packed together and encoded as a single plaintext.
 */
void ker2_to_interlaced_vector_conv1d(dvec &res, dmat kernels, int option, int dist){
    int len = 32 * 16;
    int len_actual = 32 * 15;
    res.assign(len, 0ULL);
 
    switch(option){
        case 0:
            for(int i = dist; i < len_actual; i+=dist){
                for(int l = 0; l < dist; l++){
                    res[i + l] = kernels[l][0];
                }
            }
            break;
        case 1:
            for(int i = 0; i < len_actual; i+=dist){
                for(int l = 0; l < dist; l++){
                    res[i + l] = kernels[l][1];
                }
            }
            break;
        case 2:
            for(int i = 0; i < len_actual - dist; i+=dist){
                for(int l = 0; l < dist; l++){
                    res[i + l] = kernels[l][2];
                }
            }
            break;
    }
}


/*
 @param[in] val, the double-type value
 @param[in] nrows, the number of output rows
 @param[in] ncols, the number of output columns
 @param[out] res, a one-dimensional vector (for 2D-CNN)
 This function is to generate a matrix of size nrows*ncols that have zeros at the last column.
 Then the matrix is converted to "res" using the row-major ordering method.
 */
void val1_to_vector(dvec &res, double val, int nrows, int ncols)
{
    int len = nrows * ncols;
    res.assign(len, val);
    
    // put zeros on the right column
    for(int i = ncols - 1; i < len; i+=ncols){
        res[i] = 0.0;
    }
}

/*
 @param[in] val, the double-type value
 @param[in] nrows, the number of output rows = 32
 @param[in] ncols, the number of output columns = 16
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, a puctured vector (for 2D-CNN)
 */
void val_to_vector(dvec &res, double val, int nrows, int ncols, int dist)
{
    int len = nrows * ncols;
    res.assign(len, 0ULL);
    
    // put zeros on the right column
    for(int i = 0; i < nrows; i+=dist){
        for(int j = 0; j < ncols - dist; j+=dist){
            res[i * ncols + j] = val;
        }
    }
    if(res[1]!= 0){
        throw invalid_argument("Error: val_to_vector");
    }
}

/*
 @param[in] val, the double-type value
 @param[in] nrows, the number of output rows
 @param[in] ncols, the number of output columns
 @param[in] ncols_actual,
 @param[out] res, a one-dimensional vector (for 1D-CNN)
 This function is to generate a matrix of size nrows*ncols that have zeros from (ncols_actual) <= xxx < (ncols)
 Then the matrix is converted to "res" using the row-major ordering method.
 */
void val1_to_vector_conv1d(dvec &res, double val, int len, int len_actual)
{
    res.assign(len_actual, val);
    vector<double> zeros (len - len_actual, 0ULL); // put zeros on the right columns
    res.insert(res.end(), zeros.begin(), zeros.end());
}

/*
 @param[in] val, the double-type value
 @param[in] nrows, the number of output rows = 32
 @param[in] ncols, the number of output columns = 16
 @param[in] ncols_actual,
 @param[in] dist, 2 for the second convolution; 4 for for the third convolution
 @param[out] res, a puctured vector (for 1D-CNN)
 */
void val_to_vector_conv1d(dvec &res, double val, int len, int len_actual, int dist)
{
    res.assign(len, 0ULL);
    
    for(int i = 0; i < len_actual; i+=dist){
        res[i] = val;
    }
}

/*
 @param[in] input, the input vector
 compute the argmax and return the label with the maximum value
 */
int argmax(dvec input)
{
    int label = 0;
    double maximum = input[0];
    for(int i = 1; i < input.size(); ++i){
        if(maximum < input[i]){
            maximum = input[i];
            label = i;
        }
    }
    return label;
}

/*
 @param[in] pred_labels, the predicted label vector
 @param[in] act_labels, the actual label vector
 @param[in] event_label, the true label
 @param[out] accuracy
 @param[out] sensitivity
 @param[out] specificity
 @param[out] precision
 @param[out] F1score
 */

void get_performance(double &accuracy, double &sensitivity, double &specificity, double &precision, double &F1score,
                     vector<int> pred_labels, vector<int> act_labels, int event_label)
{
    // compare pred_labels and act_labels
    if(pred_labels.size() != act_labels.size()){
        throw invalid_argument("Error: cannot read file");
    }
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    int acc = 0;

    for (int id = 0; id < pred_labels.size(); id++){
        int pred_label = pred_labels[id];
        int act_label = act_labels[id];

        if(pred_label == act_label){
            acc++;
        }

        if(act_label == event_label){
            if(pred_label == event_label){
                TP++;
            } else{
                FN++;
            }
        } else{
            if(pred_label == event_label){
                FP++;
            } else{
                TN++;
            }
        }
    }
    //cout << acc << "," << TP << "," << TN << "," << FP << "," << FN << endl;
    
    accuracy = double(acc)/double(pred_labels.size());
    sensitivity = double(TP)/ double(TP + FN);
    specificity = double(TN)/ double(TN + FP);
    precision = double(TP)/ double(TP + FP);
    F1score = 2.0 * precision * sensitivity / (precision + sensitivity);
}
