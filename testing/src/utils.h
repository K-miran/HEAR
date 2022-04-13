#include <iostream>
#include <vector>
#include <string>
#include "param.h"

using namespace std;

// data type
typedef vector<double>   dvec;
typedef vector< vector<double> >  dmat;
typedef vector< vector< vector<double> > >  dten;
typedef vector< vector< vector< vector<double> > > > ften;

// get the shape of an input tensor
void getshape(dvec data, const string mode = "data");
void getshape(dmat data, const string mode = "data");
void getshape(dten data, const string mode = "data");
void getshape(ften data, const string mode = "data");

// read the input value, column vector, or matrix
double read_oneval(string filepath);
void read_onecol(dvec& res, string filepath);
void read_matrix(dmat& res, char split_char, string filepath);

// read the input data
void read_input_data(dten &ker, string filepath);
void read_input_data(dten &res, int id, char split_char, string filepath);

void output_with_file(vector<double> data, string filename, size_t nrows);
void output_with_file(vector<double> data, string filename);

// reshape an one-dimensional vector to a tensor
void reshape(dmat &res, dvec input, int height, int width);
void reshape(dten &res, dvec input, int nchannels, int filter_height, int filter_width);
void reshape(ften &res, dvec input, int out_nchannels, int in_nchannels, int filter_height, int filter_width);
void reshape_special(ften &res, dvec input, int out_nchannels, int in_nchannels, int filter_height, int filter_width);

void reshape_conv1d(dten &res, dvec input, int out_nchannels, int in_nchannels, int filter_size);
void reshape_special_conv1d(dten &res, dvec input, int out_nchannels, int in_nchannels, int filter_size);

void aggregate_bias_BN_actpoly_avg(dmat &real_poly, dvec bias,
                               dvec gamma, dvec beta, dvec mean, dvec var, dvec act_poly, double pool_size, double epsilon);

// read the trainde model parameters
void read_parameters_conv2d(vector<ften> &ker, vector<dmat> &real_poly, dmat &dense_ker, dvec &dense_bias,
                            int nch, string param_dir);

void read_parameters_conv1d(vector<dten> &ker, vector<dmat> &real_poly, dmat &dense_ker, dvec &dense_bias,
                            int nch, string param_dir);

// rotation
void msgleftRotate(dvec& res, dvec vals, int steps);
void msgrightRotate(dvec& res, dvec vals, int steps);
void msgleftRotate_inplace(dvec& vals, int steps);
void msgrightRotate_inplace(dvec& vals, int steps);

// Kernels To Vector
void ker1_to_vector(dvec &res, dmat kernel, int option);
void ker2_to_vector(dvec &res, dmat kernel, int option, int dist);
void ker2_to_interlaced_vector(dvec &res, dten kernels, int option, int dist);

void ker1_to_vector_conv1d(dvec &res, dvec kernel, int option);
void ker2_to_vector_conv1d(dvec &res, dvec kernel, int option, int dist);
void ker2_to_interlaced_vector_conv1d(dvec &res, dmat kernels, int option, int dist); // need to be checked

// Value To Vector
void val1_to_vector(dvec &res, double val, int nrows, int ncols);
void val_to_vector(dvec &res, double val, int nrows, int ncols, int dist);

void val1_to_vector_conv1d(dvec &res, double val, int len, int len_actual);
void val_to_vector_conv1d(dvec &res, double val,  int len, int len_actual, int dist);


int argmax(dvec input);
void get_performance(double &accuracy, double &sensitivity, double &specificity, double &precision, double &F1score,
                     vector<int> HE_labels, vector<int> plain_labels, int event_label);
